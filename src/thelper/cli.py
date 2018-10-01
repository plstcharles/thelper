"""
Command-line module, for use with __main__ entrypoint or external apps.
"""

import argparse
import json
import glob
import logging
import os
import torch

import thelper

logging.basicConfig(level=logging.INFO)


def create_session(config, data_root, save_dir):
    logger = thelper.utils.get_func_logger()
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    if "cudnn_benchmark" in config and thelper.utils.str2bool(config["cudnn_benchmark"]):
        torch.backends.cudnn.benchmark = True
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config)
    logger.info("Creating new training session '%s'..." % session_name)
    task, train_loader, valid_loader, test_loader = thelper.data.load(config, data_root, save_dir)
    model = thelper.modules.load_model(config, task, save_dir)
    loaders = (train_loader, valid_loader, test_loader)
    trainer = thelper.train.load_trainer(session_name, save_dir, config, model, loaders)
    logger.debug("starting trainer")
    trainer.train()
    logger.debug("all done")
    return 0


def visualize_data(config, data_root):
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


def resume_session(ckptdata, data_root, save_dir, config=None, eval_only=False):
    logger = thelper.utils.get_func_logger()
    if not config:
        if "config" not in ckptdata or not ckptdata["config"]:
            raise AssertionError("checkpoint data missing 'config' field")
        config = ckptdata["config"]
    if "name" not in config or not config["name"]:
        raise AssertionError("config missing 'name' field")
    session_name = config["name"]
    if "cudnn_benchmark" in config and thelper.utils.str2bool(config["cudnn_benchmark"]):
        torch.backends.cudnn.benchmark = True
    save_dir = thelper.utils.get_save_dir(save_dir, session_name, config, resume=True)
    logger.info("loading training session '%s' objects..." % session_name)
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


def main(args=None):
    ap = argparse.ArgumentParser(description='thelper model trainer application')
    ap.add_argument("--version", default=False, action="store_true", help="prints the version of the library and exits")
    ap.add_argument("-l", "--log", default="thelper.log", type=str, help="path to the top-level log file (default: 'thelper.log')")
    ap.add_argument("-v", "--verbose", action="count", default=3, help="set logging terminal verbosity level (additive)")
    ap.add_argument("-d", "--data-root", default=None, type=str, help="path to the root directory passed to dataset interfaces for parsing")
    subparsers = ap.add_subparsers(title="Operating mode", dest="mode")
    new_session_ap = subparsers.add_parser("new", help="creates a new session from a config file")
    new_session_ap.add_argument("cfg_path", type=str, help="path to the training configuration file")
    new_session_ap.add_argument("save_dir", type=str, help="path to the root directory where checkpoints should be saved")
    cl_new_session_ap = subparsers.add_parser("cl_new", help="creates a new session from a config file for the cluster")
    cl_new_session_ap.add_argument("cfg_path", type=str, help="path to the training configuration file")
    cl_new_session_ap.add_argument("save_dir", type=str, help="path to the root directory where checkpoints should be saved")
    resume_session_ap = subparsers.add_parser("resume", help="resume a session from a checkpoint file")
    resume_session_ap.add_argument("ckpt_path", type=str, help="path to the checkpoint (or save directory) to resume training from")
    resume_session_ap.add_argument("-s", "--save-dir", default=None, type=str, help="path to the root directory where checkpoints should be saved")
    resume_session_ap.add_argument("-m", "--map-location", default=None, help="map location for loading data (default=None)")
    resume_session_ap.add_argument("-c", "--override-cfg", default=None, help="override config file path (default=None)")
    resume_session_ap.add_argument("-e", "--eval-only", default=False, action="store_true", help="only run evaluation pass (valid+test)")
    viz_session_ap = subparsers.add_parser("viz", help="visualize the loaded data for a training/eval session")
    viz_session_ap.add_argument("cfg_path", type=str, help="path to the training configuration file (or session save directory)")
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
                if "train_device" in trainer_config or "valid_device" in trainer_config or \
                    "test_device" in trainer_config or "device" in trainer_config:
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
    elif args.mode == "viz":
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
        return visualize_data(config, args.data_root)
