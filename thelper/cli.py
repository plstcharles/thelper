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
from typing import Any, Union

import torch
import tqdm

import thelper

TASK_COMPAT_CHOICES = frozenset(["old", "new", "compat"])


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
    session_name = thelper.utils.get_config_session_name(config)
    assert session_name is not None, "config missing 'name' field required for output directory"
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


def resume_session(ckptdata, save_dir, config=None, eval_only=False, task_compat=None):
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
        task_compat: specifies how to handle discrepancy between old task from checkpoint and new task from config

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
    session_name = thelper.utils.get_config_session_name(config)
    assert session_name is not None, "config missing 'name' field required for output directory"
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
        if task_compat in TASK_COMPAT_CHOICES:
            logger.warning("discrepancy between old task from checkpoint and new task from config resolved by " +
                           f"input argument: task_compat={task_compat}")
            task_compat_mode = task_compat
        else:
            loaders_config = thelper.utils.get_key(["data_config", "loaders"], config)
            task_compat_mode = thelper.utils.get_key_def("task_compat_mode", loaders_config, default=None)
            if task_compat_mode in TASK_COMPAT_CHOICES:
                logger.warning("discrepancy between old task from checkpoint and new task from config resolved by " +
                               f"config: task_compat_mode={task_compat_mode}")
        if task_compat_mode not in TASK_COMPAT_CHOICES:
            task_compat_mode = thelper.utils.query_string(
                "Found discrepancy between old task from checkpoint and new task from config; " +
                "which one would you like to resume the session with?\n" +
                f"\told: {str(old_task)}\n\tnew: {str(new_task)}\n" +
                (f"\tcompat: {str(compat_task)}\n\n" if compat_task is not None else "\n") +
                "WARNING: if resuming with new or compat, some weights might be discarded!",
                choices=list(TASK_COMPAT_CHOICES))
        task = old_task if task_compat_mode == "old" else new_task if task_compat_mode == "new" else compat_task
        if task_compat_mode != "old":
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
    viz_kwargs = thelper.utils.get_key_def(["params", "kwargs"], viz_config, default={})
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
        for sample in tqdm.tqdm(loader):
            input = sample[task.input_key]
            target = sample[task.gt_key] if task.gt_key in sample else None
            redraw = thelper.draw.draw(task=task, input=input, target=target, redraw=redraw, **viz_kwargs)
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
    session_name = thelper.utils.get_config_session_name(config)
    assert session_name is not None, "config missing 'name' field required for output directory"
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
    session_name = thelper.utils.get_config_session_name(config)
    assert session_name is not None, "config missing 'name' field required for output directory"
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


def inference_session(config, save_dir=None, ckpt_path=None):
    """Executes an inference session on samples with a trained model checkpoint.

    In order to run inference, a model is mandatory and therefore expected to be provided in the configuration.
    Similarly, a list of input sample file paths are expected for which to run inference on. These inputs can provide
    additional data according to how they are being parsed by lower level operations.

    The session will save the current configuration as `config-infer.json` and the employed model's training
    configuration as `config-train.json`. Other outputs depend on the specific implementation of the session runner.

    Args:
        config: a dictionary that provides all required data configuration.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories. If not provided, will use the best value extracted from either the
            configuration path or the configuration dictionary itself.
        ckpt_path: explicit checkpoint path to use for loading a model to execute inference. Otherwise look for a
            model definition in the configuration.

    .. seealso::
        | :func:`thelper.data.geo.utils.prepare_raster_metadata`
        | :func:`thelper.data.geo.utils.sliding_window_inference`
    """
    import thelper.data.geo
    logger = thelper.utils.get_func_logger()

    session_name = thelper.utils.get_config_session_name(config)
    thelper.utils.setup_globals(config)

    def is_defined_dict(container, section):
        # type: (thelper.typedefs.ConfigDict, str) -> bool
        return section in container and isinstance(container[section], dict) and bool(len(container[section]))

    if ckpt_path is not None and os.path.exists(ckpt_path):
        logger.warning("Overriding model definition with explicit checkpoint path argument")
        config["model"] = {"ckpt_path": ckpt_path, "params": {"pretrained": True}}
    if not is_defined_dict(config, "model") or "ckpt_path" not in config["model"]:
        raise RuntimeError("Missing a model checkpoint definition with which to run inference")
    ckpt_path = config["model"]["ckpt_path"]
    if not is_defined_dict(config["model"], "params") or "pretrained" not in config["model"]["params"]:
        logger.warning("Adding model pretrained flag to definition")
        config["model"].setdefault("params", {})
        config["model"]["params"].setdefault("pretrained", True)
    if not config["model"]["params"]["pretrained"]:
        raise RuntimeError("Model checkpoint was explicitly defined as not pretrained. It must be for inference.")

    if not os.path.exists(ckpt_path):
        logger.fatal(f"Model not found: {ckpt_path}")
        raise AssertionError("Model checkpoint missing to run inference")
    ckptdata = thelper.utils.load_checkpoint(ckpt_path, map_location=None, always_load_latest=False)
    if "task" not in ckptdata or not isinstance(ckptdata["task"], (thelper.tasks.Task, str)):
        raise AssertionError("invalid checkpoint, cannot reload model task")
    task = ckptdata["task"]
    task = thelper.tasks.create_task(task)  # make sure the task is instantiated if string

    if save_dir is None:
        save_dir = thelper.utils.get_checkpoint_session_root(ckpt_path)
    save_dir = os.path.join(save_dir, session_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not is_defined_dict(config, "datasets"):
        raise RuntimeError("Missing at least one dataset definition in configuration.")
    logger.info("Updating configuration with loaders inputs...")
    if not is_defined_dict(config, "loaders"):
        logger.warning("Missing loaders definition in configuration. Using default.")
        config["loaders"] = dict()
    loaders = config["loaders"]
    if "batch_size" not in loaders or not (isinstance(loaders["batch_size"], int) and loaders["batch_size"] > 0):
        logger.warning("Missing loader 'batch_size' definition in configuration. Using default (batch_size=1).")
        loaders["batch_size"] = 1
    if not is_defined_dict(loaders, "test_split"):
        logger.warning("Missing loader 'test_split' definition in configuration. Using all found datasets.")
        loaders["test_split"] = {dataset_name: 1.0 for dataset_name in config["datasets"].keys()}
    if "task_compat_mode" not in loaders:
        logger.warning("Missing loader 'task_compat_mode' definition in configuration. Using compatible mode.")
        loaders["task_compat_mode"] = "compat"
    if "shuffle" not in loaders:
        logger.warning("Missing loader 'shuffle' definition in configuration. Using 'false' for inference.")
        loaders["shuffle"] = False
    # Because loaders require a 'task' but that we are doing inference instead of training, all class-names are
    # enforced by the model's task definition (cannot adapt). Override datasets' task to make sure of this and to
    # avoid having the user needing to plug us a dummy task just to meet the loader requirement.
    all_datasets = config["datasets"]
    for dataset_name in all_datasets:
        dataset = all_datasets[dataset_name]
        if is_defined_dict(dataset, "task"):
            logger.warning("Overriding task of dataset '%s' with model's task", dataset_name)
        all_datasets[dataset_name]["task"] = str(task)  # use str to allow both json dump and parsing by dataset loader
    _, _, _, test_loader = thelper.data.create_loaders(config, save_dir)
    if not test_loader or not len(test_loader):
        raise RuntimeError("Could not define a test loader for model inference")

    model = thelper.nn.create_model(config, task, save_dir=save_dir, ckptdata=ckptdata)
    loaders = (None, None, test_loader)
    # Avoid passing any checkpoint data so that any argument don't get incorrectly used by the session runner.
    # Since we only call test/eval, these values don't matter but they still get generated otherwise and this leads
    # to weird internal values such as test starting at 'current_epoch' == 'best_epoch' from checkpoint training or
    # optimizers being instantiated.
    tester = thelper.infer.create_tester(session_name, save_dir, config, model, task, loaders)  # ckptdata=ckptdata)

    config_file = "config-train.json"
    config_file_path = os.path.join(save_dir, config_file)
    train_config = ckptdata['config']
    logger.debug("Writing employed train config: [%s]", config_file_path)
    with open(config_file_path, 'w') as f:
        json.dump(train_config, f, indent=4)
    config_name = "config-infer.json"
    config_file_path = os.path.join(save_dir, config_name)
    logger.debug("Writing employed infer config: [%s]", config_file_path)
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    tester.test()


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
    session_name = thelper.utils.get_config_session_name(config)
    assert session_name is not None, "config missing 'name' field required for output directory"
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
    assert task is not None, "could not get proper task object from export config or data parsers"
    if isinstance(trace_input, str):
        trace_input = eval(trace_input)
    logger.info("exporting model '%s'..." % session_name)
    thelper.utils.setup_globals(config)
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config)
    logger.debug("exported checkpoint will be saved at '%s'" % os.path.abspath(save_dir).replace("\\", "/"))
    model = thelper.nn.create_model(config, task, save_dir=save_dir)
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


def make_argparser():
    # type: () -> argparse.ArgumentParser
    """Creates the (default) argument parser to use for the main entrypoint.

    The argument parser will contain different "operating modes" that dictate the high-level behavior of the CLI. This
    function may be modified in branches of the framework to add project-specific features.
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
    new_ap.add_argument("save_dir", type=str, help="path to the session output root directory")
    cl_new_ap = subparsers.add_parser("cl_new", help="creates a new session from a config file for the cluster")
    cl_new_ap.add_argument("cfg_path", type=str, help="path to the session configuration file")
    cl_new_ap.add_argument("save_dir", type=str, help="path to the session output root directory")
    resume_ap = subparsers.add_parser("resume", help="resume a session from a checkpoint file")
    resume_ap.add_argument("ckpt_path", type=str, help="path to the checkpoint (or directory) to resume training from")
    resume_ap.add_argument("-s", "--save-dir", default=None, type=str, help="path to the session output root directory")
    resume_ap.add_argument("-m", "--map-location", default=None, help="map location for loading data (default=None)")
    resume_ap.add_argument("-c", "--override-cfg", default=None, help="override config file path (default=None)")
    resume_ap.add_argument("-e", "--eval-only", default=False, action="store_true", help="only run evaluation pass (valid+test)")
    resume_ap.add_argument("-t", "--task-compat", default=None, type=str, choices=TASK_COMPAT_CHOICES,
                           help="task compatibility mode to use to resolve any discrepancy between loaded tasks")
    viz_ap = subparsers.add_parser("viz", help="visualize the loaded data for a training/eval session")
    viz_ap.add_argument("cfg_path", type=str, help="path to the session configuration file (or session directory)")
    annot_ap = subparsers.add_parser("annot", help="launches a dataset annotation session with a GUI tool")
    annot_ap.add_argument("cfg_path", type=str, help="path to the session configuration file (or session directory)")
    annot_ap.add_argument("save_dir", type=str, help="path to the session output root directory")
    split_ap = subparsers.add_parser("split", help="launches a dataset splitting session from a config file")
    split_ap.add_argument("cfg_path", type=str, help="path to the session configuration file (or session directory)")
    split_ap.add_argument("save_dir", type=str, help="path to the session output root directory")
    split_ap = subparsers.add_parser("export", help="launches a model exportation session from a config file")
    split_ap.add_argument("cfg_path", type=str, help="path to the session configuration file (or session directory)")
    split_ap.add_argument("save_dir", type=str, help="path to the session output root directory")
    infer_ap = subparsers.add_parser("infer", help="creates a inference session from a config file")
    infer_ap.add_argument("cfg_path", type=str, help="path to the session configuration file (or session directory)")
    infer_ap.add_argument("save_dir", type=str, help="path to the session output root directory")
    infer_ap.add_argument("--ckpt-path", type=str,
                          help="path to the checkpoint (or directory) to use for inference "
                               "(otherwise uses model checkpoint from configuration)")
    return ap


def setup(args=None, argparser=None):
    # type: (Any, argparse.ArgumentParser) -> Union[int, argparse.Namespace]
    """Sets up the argument parser (if not already done externally) and parses the input CLI arguments.

    This function may return an error code (integer) if the program should exit immediately. Otherwise, it will return
    the parsed arguments to use in order to redirect the execution flow of the entrypoint.
    """
    argparser = argparser or make_argparser()
    args = argparser.parse_args(args=args)
    if args.version:
        print(thelper.__version__)
        return 0
    if args.mode is None:
        argparser.print_help()
        return 1
    if args.silent and args.verbose > 0:
        raise AssertionError("contradicting verbose/silent arguments provided")
    log_level = logging.INFO if args.verbose < 1 else logging.DEBUG if args.verbose < 2 else logging.NOTSET
    thelper.utils.init_logger(log_level, args.log, args.force_stdout)
    return args


def main(args=None, argparser=None):
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
        | :func:`thelper.cli.inference_session`
    """
    args = setup(args=args, argparser=argparser)
    if isinstance(args, int):
        return args  # CLI must exit immediately with provided error code
    if args.mode == "new" or args.mode == "cl_new":
        thelper.logger.debug("parsing config at '%s'" % args.cfg_path)
        config = thelper.utils.load_config(args.cfg_path)
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
            override_config = thelper.utils.load_config(args.override_cfg)
        save_dir = args.save_dir
        if save_dir is None:
            save_dir = thelper.utils.get_checkpoint_session_root(args.ckpt_path)
        if save_dir is None:
            save_dir = thelper.utils.get_save_dir(out_root=None, dir_name=None, config=override_config)
        resume_session(ckptdata, save_dir, config=override_config, eval_only=args.eval_only, task_compat=args.task_compat)
    elif args.mode == "infer":
        thelper.logger.debug("parsing config at '%s'" % args.cfg_path)
        config = thelper.utils.load_config(args.cfg_path)
        inference_session(config, save_dir=args.save_dir, ckpt_path=args.ckpt_path)
    else:
        thelper.logger.debug("parsing config at '%s'" % args.cfg_path)
        config = thelper.utils.load_config(args.cfg_path)
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
