"""
Command-line module, for use with a ``__main__`` entrypoint.

This module contains the primary functions used to create or resume a training visualization, or
annotation session. The three basic arguments that need to be provided by the user to create a session
are its configuration (dict), the dataset root directory (string), and the save root directory (string).
"""

import argparse
import glob
import json
import logging
import os

import torch

import thelper

logging.basicConfig(level=logging.INFO)


def create_session(config, data_root, save_dir):
    """Creates a session to train a model.

    All generated outputs (model checkpoints and logs) will be saved in a directory named after the
    session (the name itself is specified in ``config``), and located in ``save_dir``.

    Args:
        config: a dictionary that provides all required data configuration and trainer parameters; see
            :class:`thelper.train.Trainer` and :func:`thelper.data.load` for more information. Here, it
            is only expected to contain a ``name`` field that specifies the name of the session.
        data_root: the path to the dataset root directory that will be passed to the dataset interfaces
            for them to figure out where the training/validation/testing data is located. This path may
            be unused if the dataset interfaces already know where to look via config parameters.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.

    .. seealso::
        :class:`thelper.train.Trainer`
    """
    logger = thelper.utils.get_func_logger()
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    logger.info("creating new training session '%s'..." % session_name)
    if "cudnn_benchmark" in config and thelper.utils.str2bool(config["cudnn_benchmark"]):
        logger.debug("activating benchmark mode for cudnn")
        torch.backends.cudnn.benchmark = True
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config)
    logger.debug("session will be saved at '%s'" % save_dir)
    task, train_loader, valid_loader, test_loader = thelper.data.load(config, data_root, save_dir)
    model = thelper.modules.load_model(config, task, save_dir)
    loaders = (train_loader, valid_loader, test_loader)
    trainer = thelper.train.load_trainer(session_name, save_dir, config, model, loaders)
    logger.debug("starting trainer")
    trainer.train()
    logger.debug("all done")
    return 0


def resume_session(ckptdata, data_root, save_dir, config=None, eval_only=False):
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

    Args:
        ckptdata: raw checkpoint data loaded via ``torch.load()``; it will be parsed by the various
            parts of the framework that need to reload their previous state.
        data_root: the path to the dataset root directory that will be passed to the dataset interfaces
            for them to figure out where the training/validation/testing data is located. This path may
            be unused if the dataset interfaces already know where to look via config parameters.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.
        config: a dictionary that provides all required data configuration and trainer parameters; see
            :class:`thelper.train.Trainer` and :func:`thelper.data.load` for more information. Here, it
            is only expected to contain a ``name`` field that specifies the name of the session.
        eval_only: specifies whether training should be resumed or the model should only be evaluated.

    .. seealso::
        :class:`thelper.train.Trainer`
    """
    logger = thelper.utils.get_func_logger()
    if not config:
        if "config" not in ckptdata or not ckptdata["config"]:
            raise AssertionError("checkpoint data missing 'config' field")
        config = ckptdata["config"]
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    logger.info("loading training session '%s' objects..." % session_name)
    if "cudnn_benchmark" in config and thelper.utils.str2bool(config["cudnn_benchmark"]):
        logger.debug("activating benchmark mode for cudnn")
        torch.backends.cudnn.benchmark = True
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config, resume=True)
    logger.debug("session will be saved at '%s'" % save_dir)
    task, train_loader, valid_loader, test_loader = thelper.data.load(config, data_root, save_dir)
    if "task" not in ckptdata:
        logger.warning("cannot verify that checkpoint task is same as current task, might cause key or class mapping issues")
    elif task != ckptdata["task"]:
        raise AssertionError("checkpoint task mismatch with current task")
    model = thelper.modules.load_model(config, task, save_dir)
    loaders = (None if eval_only else train_loader, valid_loader, test_loader)
    trainer = thelper.train.load_trainer(session_name, save_dir, config, model, loaders, ckptdata=ckptdata)
    if eval_only:
        logger.info("evaluating session '%s' checkpoint @ epoch %d" % (trainer.name, trainer.current_epoch))
        trainer.eval()
    else:
        logger.info("resuming training session '%s' @ epoch %d" % (trainer.name, trainer.current_epoch))
        trainer.train()
    logger.debug("all done")
    return 0


def visualize_data(config, data_root):
    """Displays the images used in a training session.

    This mode does not generate any output, and is only used to visualize the transformed images used
    in a training session. This is useful to debug the data augmentation and base transformation pipelines
    and make sure the modified images are valid. It does not attempt to load a model or instantiate a
    trainer, meaning the related fields are not required inside ``config``.

    Args:
        config: a dictionary that provides all required data configuration parameters; see
            :func:`thelper.data.load` for more information.
        data_root: the path to the dataset root directory that will be passed to the dataset interfaces
            for them to figure out where the training/validation/testing data is located. This path may
            be unused if the dataset interfaces already know where to look via config parameters.
    """
    logger = thelper.utils.get_func_logger()
    logger.info("creating visualization session...")
    task, train_loader, valid_loader, test_loader = thelper.data.load(config, data_root)
    if not isinstance(task, thelper.tasks.Classification):
        raise AssertionError("missing impl, viz mode expects images + labels")
    loader_map = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }
    choice = thelper.utils.query_string("Which loader would you like to visualize?", choices=list(loader_map.keys()), default="train")
    loader = loader_map[choice]
    if loader is None:
        logger.info("loader is empty, all done")
        return 0
    image_key = task.get_input_key()
    label_key = task.get_gt_key()
    batch_count = len(loader)
    logger.info("initializing '%s' loader with %d batches..." % (choice, batch_count))
    for batch_idx, samples in enumerate(loader):
        logger.debug("at batch = %d / %d" % (batch_idx, batch_count))
        if "idx" in samples:
            indices = samples["idx"]
            if isinstance(indices, torch.Tensor):
                logger.debug("(indices = %s)" % indices.tolist())
            else:
                logger.debug("(indices = %s)" % indices)
        thelper.utils.draw_sample(samples, image_key=image_key, label_key=label_key, block=True)
    logger.info("all done")
    return 0


def annotate_data(config, data_root, save_dir):
    """Launches an annotation session for a dataset using a specialized GUI tool.

    Note that the annotation type must be supported by the GUI tool. The annotations created by the user
    during the session will be saved in the session directory.

    Args:
        config: a dictionary that provides all required dataset and GUI tool configuration parameters; see
            :func:`thelper.data.load_datasets` and :class:`thelper.gui.XXXX` for more information.
        data_root: the path to the dataset root directory that will be passed to the dataset interfaces
            for them to figure out where the data to annotate is located. This path may be unused if the
            dataset interfaces already know where to look via config parameters.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.
    """
    logger = thelper.utils.get_func_logger()
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    logger.info("creating annotation session '%s'..." % session_name)
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config)
    logger.debug("session will be saved at '%s'" % save_dir)
    logger.info("parsing datasets configuration")
    if "datasets" not in config or not config["datasets"]:
        raise AssertionError("config missing 'datasets' field (can be dict or str)")
    datasets_config = config["datasets"]
    if isinstance(datasets_config, str):
        if os.path.isfile(datasets_config) and os.path.splitext(datasets_config)[1] == ".json":
            datasets_config = json.load(open(datasets_config))
        else:
            raise AssertionError("'datasets' string should point to valid json file")
    logger.debug("loading datasets templates")
    if not isinstance(datasets_config, dict):
        raise AssertionError("invalid datasets config type")
    datasets, _ = thelper.data.load_datasets(datasets_config, data_root)
    annotator = thelper.gui.load_annotator(session_name, save_dir, config, datasets)
    logger.debug("starting annotator")
    annotator.run()
    logger.debug("all done")
    return 0


def main(args=None):
    """Main entrypoint to use with console applications.

    This function parses command line arguments and dispatches the execution based on the selected
    operating mode (new session, resume session, or visualize). Run with ``--help`` for information
    on the available arguments.

    .. warning::
        If you are trying to resume a session that was previously executed using a now unavailable GPU,
        you will have to force the checkpoint data to be loaded on CPU using ``--map-location=cpu`` (or
        using ``-m=cpu``).

    .. seealso::
        :func:`thelper.cli.create_session`
        :func:`thelper.cli.resume_session`
        :func:`thelper.cli.visualize_data`
    """
    ap = argparse.ArgumentParser(description='thelper model trainer application')
    ap.add_argument("--version", default=False, action="store_true", help="prints the version of the library and exits")
    ap.add_argument("-l", "--log", default="thelper.log", type=str, help="path to the top-level log file (default: 'thelper.log')")
    ap.add_argument("-v", "--verbose", action="count", default=3, help="set logging terminal verbosity level (additive)")
    ap.add_argument("-d", "--data-root", default=None, type=str, help="path to the root directory passed to dataset interfaces for parsing")
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
    args = ap.parse_args(args=args)
    if args.verbose > 2:
        log_level = logging.NOTSET
    elif args.verbose > 1:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING
    if args.version:
        print(thelper.__version__)
        return 0
    if args.mode is None:
        ap.print_help()
        return 1
    logging.getLogger().setLevel(logging.NOTSET)
    thelper.logger.propagate = 0
    logger_format = logging.Formatter("[%(asctime)s - %(name)s - %(process)s:%(thread)s] %(levelname)s : %(message)s")
    if args.log:
        logger_fh = logging.FileHandler(args.log)
        logger_fh.setLevel(logging.DEBUG)
        logger_fh.setFormatter(logger_format)
        thelper.logger.addHandler(logger_fh)
    logger_ch = logging.StreamHandler()
    logger_ch.setLevel(log_level)
    logger_ch.setFormatter(logger_format)
    thelper.logger.addHandler(logger_ch)
    if args.data_root:
        thelper.logger.debug("checking dataset root '%s'..." % args.data_root)
        if not os.path.exists(args.data_root) or not os.path.isdir(args.data_root):
            raise AssertionError("invalid data root folder at '%s'; please specify a valid path via --data-root=PATH")
    if args.mode == "new" or args.mode == "cl_new":
        thelper.logger.debug("parsing config at '%s'" % args.cfg_path)
        config = json.load(open(args.cfg_path))
        if args.mode == "cl_new":
            trainer_config = config["trainer"] if "trainer" in config else None
            if trainer_config is not None:
                if ("train_device" in trainer_config or "valid_device" in trainer_config or
                        "test_device" in trainer_config or "device" in trainer_config):
                    raise AssertionError("cannot specify device in config for cluster sessions, it is determined at runtime")
        return create_session(config, args.data_root, args.save_dir)
    elif args.mode == "resume":
        if os.path.isdir(args.ckpt_path):
            thelper.logger.debug("will search directory '%s' for a checkpoint to load..." % args.ckpt_path)
            search_ckpt_dir = os.path.join(args.ckpt_path, "checkpoints")
            if os.path.isdir(search_ckpt_dir):
                search_dir = search_ckpt_dir
            else:
                search_dir = args.ckpt_path
            ckpt_paths = glob.glob(os.path.join(search_dir, "ckpt.*.pth"))
            if not ckpt_paths:
                raise AssertionError("could not find any valid checkpoint files in directory '%s'" % search_dir)
            latest_checkpoint_epoch = 0
            for ckpt_path in ckpt_paths:
                # note: the 2nd field in the name should be the epoch index, or 'best' if final checkpoint
                tag = os.path.basename(ckpt_path).split(".")[1]
                if tag == "best" and (args.eval_only or latest_checkpoint_epoch == 0):
                    # if eval-only, always pick the best checkpoint; otherwise, only pick if nothing else exists
                    args.ckpt_path = ckpt_path
                    if args.eval_only:
                        break
                elif tag != "best" and int(tag) > latest_checkpoint_epoch:  # otherwise, pick latest
                    # note: if several sessions are merged, this will pick the latest checkpoint of the first...
                    args.ckpt_path = ckpt_path
                    latest_checkpoint_epoch = int(tag)
        if not os.path.isfile(args.ckpt_path):
            raise AssertionError("could not find valid checkpoint at '%s'" % args.ckpt_path)
        thelper.logger.debug("parsing checkpoint at '%s'" % args.ckpt_path)
        ckptdata = torch.load(args.ckpt_path, map_location=args.map_location)
        override_config = None
        if args.override_cfg:
            thelper.logger.debug("parsing override config at '%s'" % args.override_cfg)
            override_config = json.load(open(args.override_cfg))
        save_dir = args.save_dir
        if save_dir is None:
            save_dir = os.path.abspath(os.path.join(os.path.dirname(args.ckpt_path), "../.."))
        return resume_session(ckptdata, args.data_root, save_dir, config=override_config, eval_only=args.eval_only)
    elif args.mode == "viz" or args.mode == "annot":
        if os.path.isdir(args.cfg_path):
            thelper.logger.debug("will search directory '%s' for a config to load..." % args.cfg_path)
            search_cfg_names = ["config.json", "config.latest.json", "test.json"]
            cfg_path = None
            for search_cfg_name in search_cfg_names:
                search_cfg_path = os.path.join(args.cfg_path, search_cfg_name)
                if os.path.isfile(search_cfg_path):
                    cfg_path = search_cfg_path
                    break
            if cfg_path is None or not os.path.isfile(cfg_path):
                raise AssertionError("no valid config found in dir '%s'" % args.cfg_path)
            config = json.load(open(cfg_path))
        else:
            if not os.path.isfile(args.cfg_path):
                raise AssertionError("no config file found at path '%s'" % args.cfg_path)
            config = json.load(open(args.cfg_path))
        if args.mode == "viz":
            return visualize_data(config, args.data_root)
        elif args.mode == "annot":
            return annotate_data(config, args.data_root, args.save_dir)


if __name__ == "__main__":
    main()
