"""
Command-line module, for use with a ``__main__`` entrypoint.

This module contains the primary functions used to create or resume a training session, to start a
visualization session, or to start an annotation session. The basic argument that needs to be provided
by the user to create any kind of session is a configuration dictionary. For sessions that produce
outputs, the path to a directory where to save the data is also needed.
"""

import argparse
import json
import logging
import os

import torch
import tqdm

import thelper


def create_session(config, save_dir):
    """Creates a session to train a model.

    All generated outputs (model checkpoints and logs) will be saved in a directory named after the
    session (the name itself is specified in ``config``), and located in ``save_dir``.

    Args:
        config: a dictionary that provides all required data configuration and trainer parameters; see
            :class:`thelper.train.base.Trainer` and :func:`thelper.data.utils.create_loaders` for more information.
            Here, it is only expected to contain a ``name`` field that specifies the name of the session.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.

    .. seealso::
        | :class:`thelper.train.base.Trainer`
    """
    logger = thelper.utils.get_func_logger()
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    logger.info("creating new training session '%s'..." % session_name)
    thelper.utils.setup_globals(config)
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config)
    logger.debug("session will be saved at '%s'" % os.path.abspath(save_dir).replace("\\", "/"))
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(config, save_dir)
    model = thelper.nn.create_model(config, task, save_dir=save_dir)
    loaders = (train_loader, valid_loader, test_loader)
    trainer = thelper.train.create_trainer(session_name, save_dir, config, model, task, loaders)
    logger.debug("starting trainer")
    if train_loader:
        trainer.train()
    else:
        trainer.eval()
    logger.debug("all done")
    return trainer.outputs


def resume_session(ckptdata, save_dir, config=None, eval_only=False):
    """Resumes a previously created training session.

    Since the saved checkpoints contain the original session's configuration, the ``config`` argument
    can be set to ``None`` if the session should simply pick up where it was interrupted. Otherwise,
    the ``config`` argument can be set to a new configuration that will override the older one. This is
    useful when fine-tuning a model, or when testing on a new dataset.

    .. warning::
        If a session is resumed with an overriding configuration, the user must make sure that the
        inputs/outputs of the older model are compatible with the new parameters. For example, with
        classifiers, this means that the number of classes parsed by the dataset (and thus to be
        predicted by the model) should remain the same. This is a limitation of the framework that
        should be addressed in a future update.

    .. warning::
        A resumed session will not be compatible with its original RNG states if the number of workers
        used is changed. To get 100% reproducible results, make sure you run with the same worker count.

    Args:
        ckptdata: raw checkpoint data loaded via ``torch.load()``; it will be parsed by the various
            parts of the framework that need to reload their previous state.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.
        config: a dictionary that provides all required data configuration and trainer parameters; see
            :class:`thelper.train.base.Trainer` and :func:`thelper.data.utils.create_loaders` for more information.
            Here, it is only expected to contain a ``name`` field that specifies the name of the session.
        eval_only: specifies whether training should be resumed or the model should only be evaluated.

    .. seealso::
        | :class:`thelper.train.base.Trainer`
    """
    logger = thelper.utils.get_func_logger()
    if ckptdata is None or not ckptdata:
        raise AssertionError("must provide valid checkpoint data to resume a session!")
    if not config:
        if "config" not in ckptdata or not ckptdata["config"]:
            raise AssertionError("checkpoint data missing 'config' field")
        config = ckptdata["config"]
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    logger.info("loading training session '%s' objects..." % session_name)
    thelper.utils.setup_globals(config)
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config, resume=True)
    logger.debug("session will be saved at '%s'" % os.path.abspath(save_dir).replace("\\", "/"))
    new_task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(config, save_dir)
    if "task" not in ckptdata or not ckptdata["task"] or not isinstance(ckptdata["task"], (thelper.tasks.Task, str)):
        raise AssertionError("invalid checkpoint, cannot reload previous model task")
    old_task = thelper.tasks.create_task(ckptdata["task"]) if isinstance(ckptdata["task"], str) else ckptdata["task"]
    if not old_task.check_compat(new_task, exact=True):
        compat_task = None if not old_task.check_compat(new_task) else old_task.get_compat(new_task)
        choice = thelper.utils.query_string("Found discrepancy between old task from checkpoint and new task from config; "+
                                            "which one would you like to resume the session with?\n" +
                                            f"\told: {str(old_task)}\n\tnew: {str(new_task)}\n" +
                                            (f"\tcompat: {str(compat_task)}\n\n" if compat_task is not None else "\n") +
                                            "WARNING: if resuming with new or compat, some weights might be discarded!",
                                            choices=["old", "new", "compat"])
        task = old_task if choice == "old" else new_task if choice == "new" else compat_task
        if choice != "old":
            # saved optimizer state might cause issues with mismatched tasks, let's get rid of it
            logger.warning("dumping optimizer state to avoid issues when resuming with modified task")
            ckptdata["optimizer"], ckptdata["scheduler"] = None, None
    else:
        task = new_task
    assert task is not None, "invalid task"
    model = thelper.nn.create_model(config, task, save_dir=save_dir, ckptdata=ckptdata)
    loaders = (None if eval_only else train_loader, valid_loader, test_loader)
    trainer = thelper.train.create_trainer(session_name, save_dir, config, model, task, loaders, ckptdata=ckptdata)
    if eval_only:
        logger.info("evaluating session '%s' checkpoint @ epoch %d" % (trainer.name, trainer.current_epoch))
        trainer.eval()
    else:
        logger.info("resuming training session '%s' @ epoch %d" % (trainer.name, trainer.current_epoch))
        trainer.train()
    logger.debug("all done")
    return trainer.outputs


def visualize_data(config):
    """Displays the images used in a training session.

    This mode does not generate any output, and is only used to visualize the (transformed) images used
    in a training session. This is useful to debug the data augmentation and base transformation pipelines
    and make sure the modified images are valid. It does not attempt to load a model or instantiate a
    trainer, meaning the related fields are not required inside ``config``.

    If the configuration dictionary includes a 'loaders' field, it will be parsed and used. Otherwise,
    if only a 'datasets' field is available, basic loaders will be instantiated to load the data. The
    'loaders' field can also be ignored if 'ignore_loaders' is found within the 'viz' section of the config
    and set to ``True``. Each minibatch will be displayed via pyplot or OpenCV. The display will block and
    wait for user input, unless 'block' is set within the 'viz' section's 'kwargs' config as ``False``.

    Args:
        config: a dictionary that provides all required data configuration parameters; see
            :func:`thelper.data.utils.create_loaders` for more information.

    .. seealso::
        | :func:`thelper.data.utils.create_loaders`
        | :func:`thelper.data.utils.create_parsers`
    """
    logger = thelper.utils.get_func_logger()
    logger.info("creating visualization session...")
    thelper.utils.setup_globals(config)
    viz_config = thelper.utils.get_key_def("viz", config, default={})
    if not isinstance(viz_config, dict):
        raise AssertionError("unexpected viz config type")
    ignore_loaders = thelper.utils.get_key_def("ignore_loaders", viz_config, default=False)
    viz_kwargs = thelper.utils.get_key_def("kwargs", viz_config, default={})
    if not isinstance(viz_kwargs, dict):
        raise AssertionError("unexpected viz kwargs type")
    if thelper.utils.get_key_def(["data_config", "loaders"], config, default=None) is None or ignore_loaders:
        datasets, task = thelper.data.create_parsers(config)
        loader_map = {dataset_name: thelper.data.DataLoader(dataset,) for dataset_name, dataset in datasets.items()}
        # we assume no transforms were done in the parser, and images are given as read by opencv
        viz_kwargs["ch_transpose"] = thelper.utils.get_key_def("ch_transpose", viz_kwargs, False)
        viz_kwargs["flip_bgr"] = thelper.utils.get_key_def("flip_bgr", viz_kwargs, False)
    else:
        task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(config)
        loader_map = {"train": train_loader, "valid": valid_loader, "test": test_loader}
        # we assume transforms were done in the loader, and images are given as expected by pytorch
        viz_kwargs["ch_transpose"] = thelper.utils.get_key_def("ch_transpose", viz_kwargs, True)
        viz_kwargs["flip_bgr"] = thelper.utils.get_key_def("flip_bgr", viz_kwargs, False)
    redraw = None
    viz_kwargs["block"] = thelper.utils.get_key_def("block", viz_kwargs, default=True)
    assert "quit" not in loader_map
    choices = list(loader_map.keys()) + ["quit"]
    while True:
        choice = thelper.utils.query_string("Which loader would you like to visualize?", choices=choices)
        if choice == "quit":
            break
        loader = loader_map[choice]
        if loader is None:
            logger.info("loader '%s' is empty" % choice)
            continue
        batch_count = len(loader)
        logger.info("initializing loader '%s' with %d batches..." % (choice, batch_count))
        for samples in tqdm.tqdm(loader):
            redraw = thelper.utils.draw_minibatch(samples, task, redraw=redraw, **viz_kwargs)
        logger.info("all done")


def annotate_data(config, save_dir):
    """Launches an annotation session for a dataset using a specialized GUI tool.

    Note that the annotation type must be supported by the GUI tool. The annotations created by the user
    during the session will be saved in the session directory.

    Args:
        config: a dictionary that provides all required dataset and GUI tool configuration parameters; see
            :func:`thelper.data.utils.create_parsers` and :func:`thelper.gui.utils.create_annotator` for more
            information.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.

    .. seealso::
        | :func:`thelper.gui.annotators.Annotator`
        | :func:`thelper.gui.annotators.ImageSegmentAnnotator`
    """
    # import gui here since it imports packages that will cause errors in CLI-only environments
    import thelper.gui
    logger = thelper.utils.get_func_logger()
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    logger.info("creating annotation session '%s'..." % session_name)
    thelper.utils.setup_globals(config)
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config)
    logger.debug("session will be saved at '%s'" % os.path.abspath(save_dir).replace("\\", "/"))
    datasets, _ = thelper.data.create_parsers(config)
    annotator = thelper.gui.create_annotator(session_name, save_dir, config, datasets)
    logger.debug("starting annotator")
    annotator.run()
    logger.debug("all done")


def split_data(config, save_dir):
    """Launches a dataset splitting session.

    This mode will generate an HDF5 archive that contains the split datasets defined in the session
    configuration file. This archive can then be reused in a new training session to guarantee a fixed
    distribution of training, validation, and testing samples. It can also be used outside the framework
    in order to reproduce an experiment.

    The configuration dictionary must minimally contain two sections: 'datasets' and 'loaders'. A third
    section, 'split', can be used to provide settings regarding the archive packing and compression
    approaches to use.

    The HDF5 archive will be saved in the session's output directory.

    Args:
        config: a dictionary that provides all required data configuration parameters; see
            :func:`thelper.data.utils.create_loaders` for more information.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.

    .. seealso::
        | :func:`thelper.data.utils.create_loaders`
        | :func:`thelper.data.utils.create_hdf5`
        | :class:`thelper.data.parsers.HDF5Dataset`
    """
    logger = thelper.utils.get_func_logger()
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    split_config = thelper.utils.get_key_def("split", config, default={})
    if not isinstance(split_config, dict):
        raise AssertionError("unexpected split config type")
    compression = thelper.utils.get_key_def("compression", split_config, default={})
    if not isinstance(compression, dict):
        raise AssertionError("compression params should be given as dictionary")
    archive_name = thelper.utils.get_key_def("archive_name", split_config, default=(session_name + ".hdf5"))
    logger.info("creating new splitting session '%s'..." % session_name)
    thelper.utils.setup_globals(config)
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config)
    logger.debug("session will be saved at '%s'" % os.path.abspath(save_dir).replace("\\", "/"))
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(config, save_dir)
    archive_path = os.path.join(save_dir, archive_name)
    thelper.data.create_hdf5(archive_path, task, train_loader, valid_loader, test_loader, compression, config)
    logger.debug("all done")


def export_model(config, save_dir):
    """Launches a model exportation session.

    This function will export a model defined via a configuration file into a new checkpoint that can be
    loaded elsewhere. The model can be built using the framework, or provided via its type, construction
    parameters, and weights. Its exported format will be compatible with the framework, and may also be an
    optimized/compiled version obtained using PyTorch's JIT tracer.

    The configuration dictionary must minimally contain a 'model' section that provides details on the model
    to be exported. A section named 'export' can be used to provide settings regarding the exportation
    approaches to use, and the task interface to save with the model. If a task is not explicitly defined in
    the 'export' section, the session configuration will be parsed for a 'datasets' section that can be used
    to define it. Otherwise, it must be provided through the model.

    The exported checkpoint containing the model will be saved in the session's output directory.

    Args:
        config: a dictionary that provides all required data configuration parameters; see
            :func:`thelper.nn.utils.create_model` for more information.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.

    .. seealso::
        | :func:`thelper.nn.utils.create_model`
    """
    logger = thelper.utils.get_func_logger()
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    export_config = thelper.utils.get_key_def("export", config, default={})
    if not isinstance(export_config, dict):
        raise AssertionError("unexpected export config type")
    ckpt_name = thelper.utils.get_key_def("ckpt_name", export_config, default=(session_name + ".export.pth"))
    trace_name = thelper.utils.get_key_def("trace_name", export_config, default=(session_name + ".trace.zip"))
    save_raw = thelper.utils.get_key_def("save_raw", export_config, default=True)
    trace_input = thelper.utils.get_key_def("trace_input", export_config, default=None)
    task = thelper.utils.get_key_def("task", export_config, default=None)
    if isinstance(task, (str, dict)):
        task = thelper.tasks.create_task(task)
    if task is None and "datasets" in config:
        _, task = thelper.data.create_parsers(config)  # try to load via datasets...
    if isinstance(trace_input, str):
        trace_input = eval(trace_input)
    logger.info("exporting model '%s'..." % session_name)
    thelper.utils.setup_globals(config)
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config)
    logger.debug("exported checkpoint will be saved at '%s'" % os.path.abspath(save_dir).replace("\\", "/"))
    model = thelper.nn.create_model(config, task, save_dir=save_dir)
    if task is None:
        assert hasattr(model, "task"), "model should have task attrib if not provided already"
        task = model.task
    log_stamp = thelper.utils.get_log_stamp()
    model_type = model.get_name()
    model_params = model.config if model.config else {}
    # the saved state below should be kept compatible with the one in thelper.train.base._save
    export_state = {
        "name": session_name,
        "source": log_stamp,
        "git_sha1": thelper.utils.get_git_stamp(),
        "version": thelper.__version__,
        "task": str(task) if save_raw else task,
        "model_type": model_type,
        "model_params": model_params,
        "config": config
    }
    if trace_input is not None:
        trace_path = os.path.join(save_dir, trace_name)
        torch.jit.trace(model, trace_input).save(trace_path)
        export_state["model"] = trace_name  # will be loaded in thelper.utils.load_checkpoint
    else:
        export_state["model"] = model.state_dict() if save_raw else model
    torch.save(export_state, os.path.join(save_dir, ckpt_name))
    logger.debug("all done")


def main(args=None):
    """Main entrypoint to use with console applications.

    This function parses command line arguments and dispatches the execution based on the selected
    operating mode. Run with ``--help`` for information on the available arguments.

    .. warning::
        If you are trying to resume a session that was previously executed using a now unavailable GPU,
        you will have to force the checkpoint data to be loaded on CPU using ``--map-location=cpu`` (or
        using ``-m=cpu``).

    .. seealso::
        | :func:`thelper.cli.create_session`
        | :func:`thelper.cli.resume_session`
        | :func:`thelper.cli.visualize_data`
        | :func:`thelper.cli.annotate_data`
        | :func:`thelper.cli.split_data`
    """
    ap = argparse.ArgumentParser(description='thelper model trainer application')
    ap.add_argument("--version", default=False, action="store_true", help="prints the version of the library and exits")
    ap.add_argument("-l", "--log", default=None, type=str, help="path to the top-level log file (default: None)")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="set logging terminal verbosity level (additive)")
    ap.add_argument("--silent", action="store_true", default=False, help="deactivates all console logging activities")
    ap.add_argument("--force-stdout", action="store_true", default=False, help="force logging output to stdout instead of stderr")
    subparsers = ap.add_subparsers(title="Operating mode", dest="mode")
    new_ap = subparsers.add_parser("new", help="creates a new session from a config file")
    new_ap.add_argument("cfg_path", type=str, help="path to the session configuration file")
    new_ap.add_argument("save_dir", type=str, help="path to the root directory where checkpoints should be saved")
    cl_new_ap = subparsers.add_parser("cl_new", help="creates a new session from a config file for the cluster")
    cl_new_ap.add_argument("cfg_path", type=str, help="path to the session configuration file")
    cl_new_ap.add_argument("save_dir", type=str, help="path to the root directory where checkpoints should be saved")
    resume_ap = subparsers.add_parser("resume", help="resume a session from a checkpoint file")
    resume_ap.add_argument("ckpt_path", type=str, help="path to the checkpoint (or save directory) to resume training from")
    resume_ap.add_argument("-s", "--save-dir", default=None, type=str, help="path to the root directory where checkpoints should be saved")
    resume_ap.add_argument("-m", "--map-location", default=None, help="map location for loading data (default=None)")
    resume_ap.add_argument("-c", "--override-cfg", default=None, help="override config file path (default=None)")
    resume_ap.add_argument("-e", "--eval-only", default=False, action="store_true", help="only run evaluation pass (valid+test)")
    viz_ap = subparsers.add_parser("viz", help="visualize the loaded data for a training/eval session")
    viz_ap.add_argument("cfg_path", type=str, help="path to the session configuration file (or session save directory)")
    annot_ap = subparsers.add_parser("annot", help="launches a dataset annotation session with a GUI tool")
    annot_ap.add_argument("cfg_path", type=str, help="path to the session configuration file (or session save directory)")
    annot_ap.add_argument("save_dir", type=str, help="path to the root directory where annotations should be saved")
    split_ap = subparsers.add_parser("split", help="launches a dataset splitting session from a config file")
    split_ap.add_argument("cfg_path", type=str, help="path to the session configuration file (or session save directory)")
    split_ap.add_argument("save_dir", type=str, help="path to the root directory where the split hdf5 dataset archive should be saved")
    split_ap = subparsers.add_parser("export", help="launches a model exportation session from a config file")
    split_ap.add_argument("cfg_path", type=str, help="path to the session configuration file (or session save directory)")
    split_ap.add_argument("save_dir", type=str, help="path to the root directory where the exported checkpoint should be saved")
    args = ap.parse_args(args=args)
    if args.version:
        print(thelper.__version__)
        return 0
    if args.mode is None:
        ap.print_help()
        return 1
    if args.silent and args.verbose > 0:
        raise AssertionError("contradicting verbose/silent arguments provided")
    log_level = logging.INFO if args.verbose < 1 else logging.DEBUG if args.verbose < 2 else logging.NOTSET
    thelper.utils.init_logger(log_level, args.log, args.force_stdout)
    if args.mode == "new" or args.mode == "cl_new":
        thelper.logger.debug("parsing config at '%s'" % args.cfg_path)
        with open(args.cfg_path) as fd:
            config = json.load(fd)
        if args.mode == "cl_new":
            trainer_config = thelper.utils.get_key_def("trainer", config, {})
            device = thelper.utils.get_key_def("device", trainer_config, None)
            if device is not None:
                raise AssertionError("cannot specify device in config for cluster sessions, it is determined at runtime")
        create_session(config, args.save_dir)
    elif args.mode == "resume":
        ckptdata = thelper.utils.load_checkpoint(args.ckpt_path, map_location=args.map_location,
                                                 always_load_latest=(not args.eval_only))
        override_config = None
        if args.override_cfg:
            thelper.logger.debug("parsing override config at '%s'" % args.override_cfg)
            with open(args.override_cfg) as fd:
                override_config = json.load(fd)
        save_dir = args.save_dir
        if save_dir is None:
            ckpt_dir_path = os.path.dirname(os.path.abspath(args.ckpt_path)) \
                if not os.path.isdir(args.ckpt_path) else os.path.abspath(args.ckpt_path)
            # find session dir by looking for 'logs' directory
            if os.path.isdir(os.path.join(ckpt_dir_path, "logs")):
                save_dir = os.path.abspath(os.path.join(ckpt_dir_path, ".."))
            elif os.path.isdir(os.path.join(ckpt_dir_path, "../logs")):
                save_dir = os.path.abspath(os.path.join(ckpt_dir_path, "../.."))
            else:
                save_dir = thelper.utils.query_string("Please provide the path to where the resumed session output should be saved:")
                save_dir = thelper.utils.get_save_dir(save_dir, dir_name="", config=override_config)
        resume_session(ckptdata, save_dir, config=override_config, eval_only=args.eval_only)
    else:
        thelper.logger.debug("parsing config at '%s'" % args.cfg_path)
        with open(args.cfg_path) as fd:
            config = json.load(fd)
        if args.mode == "viz":
            visualize_data(config)
        elif args.mode == "annot":
            annotate_data(config, args.save_dir)
        elif args.mode == "export":
            export_model(config, args.save_dir)
        else:  # if args.mode == "split":
            split_data(config, args.save_dir)
    return 0


if __name__ == "__main__":
    main()
