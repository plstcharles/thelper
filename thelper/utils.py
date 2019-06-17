"""General utilities module.

This module only contains non-ML specific functions, i/o helpers,
and matplotlib/pyplot drawing calls.
"""
import copy
import errno
import functools
import glob
import importlib
import inspect
import io
import itertools
import json
import logging
import math
import os
import pickle
import platform
import re
import sys
import time
from typing import AnyStr, Callable, List, Optional  # noqa: F401

import cv2 as cv
import lz4
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch

import thelper.typedefs  # noqa: F401

logger = logging.getLogger(__name__)
bypass_queries = False


class Struct:
    """Generic runtime-defined C-like data structure (maps constructor elements to fields)."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            "(" + ", ".join([f"{key}={repr(val)}" for key, val in self.__dict__.items()]) + ")"


def get_available_cuda_devices(attempts_per_device=5):
    # type: (Optional[int]) -> List[int]
    """
    Tests all visible cuda devices and returns a list of available ones.

    Returns:
        List of available cuda device IDs (integers). An empty list means no
        cuda device is available, and the app should fallback to cpu.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return []
    devices_available = [False] * torch.cuda.device_count()
    attempt_broadcast = False
    for attempt in range(attempts_per_device):
        for device_id in range(torch.cuda.device_count()):
            if not devices_available[device_id]:
                if not attempt_broadcast:
                    logger.debug("testing availability of cuda device #%d (%s)" % (
                        device_id, torch.cuda.get_device_name(device_id)
                    ))
                # noinspection PyBroadException
                try:
                    torch.cuda.set_device(device_id)
                    test_val = torch.cuda.FloatTensor([1])
                    if test_val.cpu().item() != 1.0:
                        raise AssertionError("sometime's really wrong")
                    devices_available[device_id] = True
                except Exception:
                    pass
        attempt_broadcast = True
    return [device_id for device_id, available in enumerate(devices_available) if available]


# noinspection PyUnusedLocal
def setup_cv2(config):
    """Parses the provided config for OpenCV flags and sets up its global state accordingly."""
    # https://github.com/pytorch/pytorch/issues/1355
    cv.setNumThreads(0)
    cv.ocl.setUseOpenCL(False)
    # todo: add more global opencv flags setups here


def setup_cudnn(config):
    """Parses the provided config for CUDNN flags and sets up PyTorch accordingly."""
    if "cudnn" in config and isinstance(config["cudnn"], dict):
        config = config["cudnn"]
        if "benchmark" in config:
            cudnn_benchmark_flag = str2bool(config["benchmark"])
            logger.debug("cudnn benchmark mode = %s" % str(cudnn_benchmark_flag))
            torch.backends.cudnn.benchmark = cudnn_benchmark_flag
        if "deterministic" in config:
            cudnn_deterministic_flag = str2bool(config["deterministic"])
            logger.debug("cudnn deterministic mode = %s" % str(cudnn_deterministic_flag))
            torch.backends.cudnn.deterministic = cudnn_deterministic_flag
    else:
        if "cudnn_benchmark" in config:
            cudnn_benchmark_flag = str2bool(config["cudnn_benchmark"])
            logger.debug("cudnn benchmark mode = %s" % str(cudnn_benchmark_flag))
            torch.backends.cudnn.benchmark = cudnn_benchmark_flag
        if "cudnn_deterministic" in config:
            cudnn_deterministic_flag = str2bool(config["cudnn_deterministic"])
            logger.debug("cudnn deterministic mode = %s" % str(cudnn_deterministic_flag))
            torch.backends.cudnn.deterministic = cudnn_deterministic_flag


def setup_globals(config):
    """Parses the provided config for global flags and sets up the global state accordingly."""
    if "bypass_queries" in config and config["bypass_queries"]:
        global bypass_queries
        bypass_queries = True
    setup_cv2(config)
    setup_cudnn(config)


def load_checkpoint(ckpt,                      # type: thelper.typedefs.CheckpointLoadingType
                    map_location=None,         # type: Optional[thelper.typedefs.MapLocationType]
                    always_load_latest=False,  # type: Optional[bool]
                    check_version=True,        # type: Optional[bool]
                    ):                         # type: (...) -> thelper.typedefs.CheckpointContentType
    """Loads a session checkpoint via PyTorch, check its compatibility, and returns its data.

    If the ``ckpt`` parameter is a path to a valid directory, then that directly will be searched for
    a checkpoint. If multiple checkpoints are found, the latest will be returned (based on the epoch
    index in its name). iF ``always_load_latest`` is set to False and if a checkpoint named
    ``ckpt.best.pth`` is found, it will be returned instead.

    Args:
        ckpt: a file-like object or a path to the checkpoint file or session directory.
        map_location: a function, string or a dict specifying how to remap storage
            locations. See ``torch.load`` for more information.
        always_load_latest: toggles whether to always try to load the latest checkpoint
            if a session directory is provided (instead of loading the 'best' checkpoint).
        check_version: toggles whether the checkpoint's version should be checked for
            compatibility issues, and query the user for how to proceed.

    Returns:
        Content of the checkpoint (a dictionary).
    """
    if map_location is None and not get_available_cuda_devices():
        map_location = 'cpu'
    if isinstance(ckpt, str) and os.path.isdir(ckpt):
        logger.debug("will search directory '%s' for a checkpoint to load..." % ckpt)
        search_ckpt_dir = os.path.join(ckpt, "checkpoints")
        if os.path.isdir(search_ckpt_dir):
            search_dir = search_ckpt_dir
        else:
            search_dir = ckpt
        ckpt_paths = glob.glob(os.path.join(search_dir, "ckpt.*.pth"))
        if not ckpt_paths:
            raise AssertionError("could not find any valid checkpoint files in directory '%s'" % search_dir)
        latest_epoch, latest_day, latest_time = -1, -1, -1
        for ckpt_path in ckpt_paths:
            # note: the 2nd field in the name should be the epoch index, or 'best' if final checkpoint
            split = os.path.basename(ckpt_path).split(".")
            tag = split[1]
            if tag == "best" and (not always_load_latest or latest_epoch == -1):
                # if eval-only, always pick the best checkpoint; otherwise, only pick if nothing else exists
                ckpt = ckpt_path
                if not always_load_latest:
                    break
            elif tag != "best":
                log_stamp = split[2] if len(split) > 2 else ""
                log_stamp = "fake-0-0" if log_stamp.count("-") != 2 else log_stamp
                epoch_stamp, day_stamp, time_stamp = int(tag), int(log_stamp.split("-")[1]), int(log_stamp.split("-")[2])
                if epoch_stamp > latest_epoch or day_stamp > latest_day or time_stamp > latest_time:
                    ckpt, latest_epoch, latest_day, latest_time = ckpt_path, epoch_stamp, day_stamp, time_stamp
        if not os.path.isfile(ckpt):
            raise AssertionError("could not find valid checkpoint at '%s'" % ckpt)
    basepath = None
    if isinstance(ckpt, str):
        logger.debug("parsing checkpoint at '%s'" % ckpt)
        basepath = os.path.dirname(os.path.abspath(ckpt))
    else:
        if hasattr(ckpt, "name"):
            logger.debug("parsing checkpoint provided via file object")
            basepath = os.path.dirname(os.path.abspath(ckpt.name))
    ckptdata = torch.load(ckpt, map_location=map_location)
    if not isinstance(ckptdata, dict):
        raise AssertionError("unexpected checkpoint data type")
    if check_version:
        good_version = False
        from thelper import __version__ as curr_ver
        if "version" not in ckptdata:
            logger.warning("checkpoint missing internal version tag")
            ckpt_ver_str = "0.0.0"
        else:
            ckpt_ver_str = ckptdata["version"]
            if not isinstance(ckpt_ver_str, str) or len(ckpt_ver_str.split(".")) != 3:
                raise AssertionError("unexpected checkpoint version formatting")
            # by default, checkpoints should be from the same minor version, we warn otherwise
            versions = [curr_ver.split("."), ckpt_ver_str.split(".")]
            if versions[0][0] != versions[1][0]:
                logger.error("incompatible checkpoint, major version mismatch (%s vs %s)" % (curr_ver, ckpt_ver_str))
            elif versions[0][1] != versions[1][1]:
                logger.warning("outdated checkpoint, minor version mismatch (%s vs %s)" % (curr_ver, ckpt_ver_str))
            else:
                good_version = True
        if not good_version:
            answer = query_string("Checkpoint version unsupported (framework=%s, checkpoint=%s); how do you want to proceed?" %
                                  (curr_ver, ckpt_ver_str), choices=["continue", "migrate", "abort"], default="migrate", bypass="migrate")
            if answer == "abort":
                logger.error("checkpoint out-of-date; user aborted")
                sys.exit(1)
            elif answer == "continue":
                logger.warning("will attempt to load checkpoint anyway (might crash later due to incompatibilities)")
            elif answer == "migrate":
                ckptdata = migrate_checkpoint(ckptdata)
    # load model trace if needed (we do it here since we can locate the neighboring file)
    if "model" in ckptdata and isinstance(ckptdata["model"], str):
        trace_path = None
        if os.path.isfile(ckptdata["model"]):
            trace_path = ckptdata["model"]
        elif basepath is not None and os.path.isfile(os.path.join(basepath, ckptdata["model"])):
            trace_path = os.path.join(basepath, ckptdata["model"])
        if trace_path is not None:
            if trace_path.endswith(".pth"):
                ckptdata["model"] = torch.load(trace_path, map_location=map_location)
            elif trace_path.endswith(".zip"):
                ckptdata["model"] = torch.jit.load(trace_path, map_location=map_location)
    return ckptdata


def migrate_checkpoint(ckptdata,  # type: thelper.typedefs.CheckpointContentType
                       ):         # type: (...) -> thelper.typedefs.CheckpointContentType
    """Migrates the content of an incompatible or outdated checkpoint to the current version of the framework.

    This function might not be able to fix all backward compatibility issues (e.g. it cannot fix class interfaces
    that were changed). Perfect reproductibility of tests cannot be guaranteed either if this migration tool is used.

    Args:
        ckptdata: checkpoint data in dictionary form obtained via ``thelper.utils.load_checkpoint``. Note that
            the data contained in this dictionary will be modified in-place.

    Returns:
        An updated checkpoint dictionary that should be compatible with the current version of the framework.
    """
    if not isinstance(ckptdata, dict):
        raise AssertionError("unexpected ckptdata type")
    from thelper import __version__ as curr_ver
    curr_ver = [int(num) for num in curr_ver.split(".")]
    ckpt_ver_str = ckptdata["version"] if "version" in ckptdata else "0.0.0"
    ckpt_ver = [int(num) for num in ckpt_ver_str.split(".")]
    if (ckpt_ver[0] > curr_ver[0] or (ckpt_ver[0] == curr_ver[0] and ckpt_ver[1] > curr_ver[1]) or
       (ckpt_ver[0:2] == curr_ver[0:2] and ckpt_ver[2] > curr_ver[2])):
        raise AssertionError("cannot migrate checkpoints from future versions!")
    if "config" not in ckptdata:
        raise AssertionError("checkpoint migration requires config")
    old_config = ckptdata["config"]
    new_config = migrate_config(copy.deepcopy(old_config), ckpt_ver_str)
    if ckpt_ver == [0, 0, 0]:
        logger.warning("trying to migrate checkpoint data from v0.0.0; all bets are off")
    else:
        logger.info("trying to migrate checkpoint data from v%s" % ckpt_ver_str)
    if ckpt_ver[0] <= 0 and ckpt_ver[1] <= 1:
        # combine 'host' and 'time' fields into 'source'
        if "host" in ckptdata and "time" in ckptdata:
            ckptdata["source"] = ckptdata["host"] + ckptdata["time"]
            del ckptdata["host"]
            del ckptdata["time"]
        # update classif task interface
        if "task" in ckptdata and isinstance(ckptdata["task"], thelper.tasks.classif.Classification):
            ckptdata["task"] = str(thelper.tasks.classif.Classification(class_names=ckptdata["task"].class_names,
                                                                        input_key=ckptdata["task"].input_key,
                                                                        label_key=ckptdata["task"].label_key,
                                                                        meta_keys=ckptdata["task"].meta_keys))
        # move 'state_dict' field to 'model'
        if "state_dict" in ckptdata:
            ckptdata["model"] = ckptdata["state_dict"]
            del ckptdata["state_dict"]
        # create 'model_type' and 'model_params' fields
        if "model" in new_config:
            if "type" in new_config["model"]:
                ckptdata["model_type"] = new_config["model"]["type"]
            else:
                ckptdata["model_type"] = None
            if "params" in new_config["model"]:
                ckptdata["model_params"] = copy.deepcopy(new_config["model"]["params"])
            else:
                ckptdata["model_params"] = {}
        # TODO: create 'scheduler' field to restore previous state? (not so important for early versions)
        # ckpt_ver = [0, 2, 0]  # set ver for next update step
    # if ckpt_ver[0] <= x and ckpt_ver[1] <= y and ckpt_ver[2] <= z:
    #     ... add more compatibility fixes here
    ckptdata["config"] = new_config
    return ckptdata


def migrate_config(config,        # type: thelper.typedefs.ConfigDict
                   cfg_ver_str,   # type: str
                   ):             # type: (...) -> thelper.typedefs.ConfigDict
    """Migrates the content of an incompatible or outdated configuration to the current version of the framework.

    This function might not be able to fix all backward compatibility issues (e.g. it cannot fix class interfaces
    that were changed). Perfect reproductibility of tests cannot be guaranteed either if this migration tool is used.

    Args:
        config: session configuration dictionary obtained e.g. by parsing a JSON file. Note that the data contained
            in this dictionary will be modified in-place.
        cfg_ver_str: string representing the version for which the configuration was created (e.g. "0.2.0").

    Returns:
        An updated configuration dictionary that should be compatible with the current version of the framework.
    """
    if not isinstance(config, dict):
        raise AssertionError("unexpected config type")
    if not isinstance(cfg_ver_str, str) or len(cfg_ver_str.split(".")) != 3:
        raise AssertionError("unexpected checkpoint version formatting")
    from thelper import __version__ as curr_ver
    curr_ver = [int(num) for num in curr_ver.split(".")]
    cfg_ver = [int(num) for num in cfg_ver_str.split(".")]
    if (cfg_ver[0] > curr_ver[0] or (cfg_ver[0] == curr_ver[0] and cfg_ver[1] > curr_ver[1]) or
       (cfg_ver[0:2] == curr_ver[0:2] and cfg_ver[2] > curr_ver[2])):
        raise AssertionError("cannot migrate configs from future versions!")
    if cfg_ver == [0, 0, 0]:
        logger.warning("trying to migrate config from v0.0.0; all bets are off")
    else:
        logger.info("trying to migrate config from v%s" % cfg_ver_str)
    if cfg_ver[0] <= 0 and cfg_ver[1] < 1:
        # must search for name-value parameter lists and convert them to dictionaries
        def name_value_replacer(cfg):
            if isinstance(cfg, dict):
                for key, val in cfg.items():
                    if (key == "params" or key == "parameters") and isinstance(val, list) and \
                       all([isinstance(p, dict) and list(p.keys()) == ["name", "value"] for p in val]):
                        cfg["params"] = {param["name"]: name_value_replacer(param["value"]) for param in val}
                        if key == "parameters":
                            del cfg["parameters"]
                    elif isinstance(val, (dict, list)):
                        cfg[key] = name_value_replacer(val)
            elif isinstance(cfg, list):
                for idx, val in enumerate(cfg):
                    cfg[idx] = name_value_replacer(val)
            return cfg
        config = name_value_replacer(config)
        # must replace "data_config" section by "loaders"
        if "data_config" in config:
            config["loaders"] = config["data_config"]
            del config["data_config"]
        # remove deprecated name attribute for models
        if "model" in config and isinstance(config["model"], dict) and "name" in config["model"]:
            del config["model"]["name"]
        # must update import targets wrt class name refactorings
        def import_refactoring(cfg):  # noqa: E306
            if isinstance(cfg, dict):
                for key, val in cfg.items():
                    cfg[key] = import_refactoring(val)
            elif isinstance(cfg, list):
                for idx, val in enumerate(cfg):
                    cfg[idx] = import_refactoring(val)
            elif isinstance(cfg, str) and cfg.startswith("thelper."):
                cfg = thelper.utils.resolve_import(cfg)
            return cfg
        config = import_refactoring(config)
        if "trainer" in config and isinstance(config["trainer"], dict):
            trainer_cfg = config["trainer"]
            # move 'loss' section to 'optimization' section
            if "loss" in trainer_cfg:
                if "optimization" not in trainer_cfg or not isinstance(trainer_cfg["optimization"], dict):
                    trainer_cfg["optimization"] = {}
                trainer_cfg["optimization"]["loss"] = trainer_cfg["loss"]
                del trainer_cfg["loss"]
            # replace all devices with cuda:all
            if "train_device" in trainer_cfg:
                del trainer_cfg["train_device"]
            if "valid_device" in trainer_cfg:
                del trainer_cfg["valid_device"]
            if "test_device" in trainer_cfg:
                del trainer_cfg["test_device"]
            if "device" not in trainer_cfg:
                trainer_cfg["device"] = "cuda:all"
            # remove params from trainer config
            if "params" in trainer_cfg:
                if not isinstance(trainer_cfg["params"], (dict, list)) or trainer_cfg["params"]:
                    logger.warning("removing non-empty parameter section from trainer config")
                del trainer_cfg["params"]
        cfg_ver = [0, 1, 0]  # set ver for next update step
    if cfg_ver[0] <= 0 and cfg_ver[1] <= 1:
        # remove 'force_convert' flags from all transform pipelines + build augment pipeline wrappers
        def remove_force_convert(cfg):  # noqa: E306
            if isinstance(cfg, list):
                for idx, stage in enumerate(cfg):
                    cfg[idx] = remove_force_convert(stage)
            elif isinstance(cfg, dict):
                if "parameters" in cfg:
                    cfg["params"] = cfg["parameters"]
                    del cfg["parameters"]
                if "operation" in cfg and cfg["operation"] == "thelper.transforms.TransformWrapper":
                    if "params" in cfg and "force_convert" in cfg["params"]:
                        del cfg["params"]["force_convert"]
                for key, stage in cfg.items():
                    cfg[key] = remove_force_convert(stage)
            return cfg
        for pipeline in ["base_transforms", "train_augments", "valid_augments", "test_augments"]:
            if "loaders" in config and isinstance(config["loaders"], dict) and pipeline in config["loaders"]:
                if pipeline.endswith("_augments"):
                    stages = config["loaders"][pipeline]
                    for stage in stages:
                        if "append" in stage:
                            if stage["append"]:
                                logger.warning("overriding augmentation stage ordering")
                            del stage["append"]
                        if "operation" in stage and stage["operation"] == "Augmentor.Pipeline":
                            if "params" in stage:
                                stage["params"] = stage["params"]["operations"]
                            elif "parameters" in stage:
                                stage["params"] = stage["parameters"]["operations"]
                                del stage["parameters"]
                    config["loaders"][pipeline] = {"append": False, "transforms": remove_force_convert(stages)}
                else:
                    config["loaders"][pipeline] = remove_force_convert(config["loaders"][pipeline])
        cfg_ver = [0, 2, 0]  # set ver for next update step
    if cfg_ver[0] <= 0 and cfg_ver[1] <= 2 and cfg_ver[2] < 5:
        # TODO: add scheduler 0-based step fix here? (unlikely to cause serious issues)
        # cfg_ver = [0, 2, 5]  # set ver for next update step
        pass
    # if cfg_ver[0] <= x and cfg_ver[1] <= y and cfg_ver[2] <= z:
    #     ... add more compatibility fixes here
    return config


def download_file(url, root, filename, md5=None):
    """Downloads a file from a given URL to a local destination.

    Args:
        url: path to query for the file (query will be based on urllib).
        root: destination folder where the file should be saved.
        filename: destination name for the file.
        md5: optional, for md5 integrity check.

    Returns:
        The path to the downloaded file.
    """
    # inspired from torchvision.datasets.utils.download_url; no dep check
    from six.moves import urllib
    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    if not os.path.isfile(fpath):
        logger.info("Downloading %s to %s ..." % (url, fpath))
        urllib.request.urlretrieve(url, fpath, reporthook)
        sys.stdout.write("\r")
        sys.stdout.flush()
    if md5 is not None:
        import hashlib
        md5o = hashlib.md5()
        with open(fpath, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                md5o.update(chunk)
        md5c = md5o.hexdigest()
        if md5c != md5:
            raise AssertionError("md5 check failed for '%s'" % fpath)
    return fpath


def extract_tar(filepath, root, flags="r:gz"):
    """Extracts the content of a tar file to a specific location.

    Args:
        filepath: location of the tar archive.
        root: where to extract the archive's content.
        flags: extra flags passed to ``tarfile.open``.
    """
    import tarfile

    class _FileWrapper(io.FileIO):
        def __init__(self, path, *args, **kwargs):
            self.start_time = time.time()
            self._size = os.path.getsize(path)
            super().__init__(path, *args, **kwargs)

        def read(self, *args, **kwargs):
            duration = time.time() - self.start_time
            progress_size = self.tell()
            speed = str(int(progress_size / (1024 * duration))) if duration > 0 else "?"
            percent = min(int(progress_size * 100 / self._size), 100)
            sys.stdout.write("\r\t=> extracted %d%% (%d MB) @ %s KB/s..." %
                             (percent, progress_size / (1024 * 1024), speed))
            sys.stdout.flush()
            return io.FileIO.read(self, *args, **kwargs)

    cwd = os.getcwd()
    tar = tarfile.open(fileobj=_FileWrapper(filepath), mode=flags)
    os.chdir(root)
    tar.extractall()
    tar.close()
    os.chdir(cwd)
    sys.stdout.write("\r")
    sys.stdout.flush()


def reporthook(count, block_size, total_size):
    """Report hook used to display a download progression bar when using urllib requests."""
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = str(int(progress_size / (1024 * duration))) if duration > 0 else "?"
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r\t=> downloaded %d%% (%d MB) @ %s KB/s..." %
                     (percent, progress_size / (1024 * 1024), speed))
    sys.stdout.flush()


def init_logger(log_level=logging.NOTSET, filename=None, force_stdout=False):
    """Initializes the framework logger with a specific filter level, and optional file output."""
    logging.getLogger().setLevel(logging.NOTSET)
    thelper.logger.propagate = 0
    logger_format = logging.Formatter("[%(asctime)s - %(name)s] %(levelname)s : %(message)s")
    if filename is not None:
        logger_fh = logging.FileHandler(filename)
        logger_fh.setLevel(logging.NOTSET)
        logger_fh.setFormatter(logger_format)
        thelper.logger.addHandler(logger_fh)
    stream = sys.stdout if force_stdout else None
    logger_ch = logging.StreamHandler(stream=stream)
    logger_ch.setLevel(log_level)
    logger_ch.setFormatter(logger_format)
    thelper.logger.addHandler(logger_ch)


def resolve_import(fullname):
    # type: (AnyStr) -> AnyStr
    """
    Class name resolver.

    Takes a string corresponding to a module and class fullname to be imported with :func:`thelper.utils.import_class`
    and resolves any back compatibility issues related to renamed or moved classes.

    Args:
        fullname: the fully qualified class name to be resolved.

    Returns:
        The resolved class fullname.
    """
    cases = [
        ('thelper.modules', 'thelper.nn'),
        ('thelper.samplers', 'thelper.data.samplers'),
        ('thelper.transforms.ImageTransformWrapper', 'thelper.transforms.TransformWrapper'),
        ('thelper.transforms.wrappers.ImageTransformWrapper', 'thelper.transforms.wrappers.TransformWrapper'),
    ]
    old_name = fullname
    for old, new in cases:
        fullname = fullname.replace(old, new)
    if old_name != fullname:
        logger.warning("class fullname '{!s}' was resolved to '{!s}'.".format(old_name, fullname))
    return fullname


def import_class(fullname):
    """General-purpose runtime class importer.

    Args:
        fullname: the fully qualified class name to be imported.

    Returns:
        The imported class.
    """
    fullname = resolve_import(fullname)
    module_name, class_name = fullname.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def import_function(fullname, params=None):
    """General-purpose runtime function importer, with support for param binding.

    Args:
        fullname: the fully qualified function name to be imported.
        params: optional params dictionary to bind to the function call via functools.

    Returns:
        The imported function, with optionally bound parameters.
    """
    func = import_class(fullname)
    if params is not None:
        if not isinstance(params, dict):
            raise AssertionError("unexpected params dict type")
        return functools.partial(func, **params)
    return func


def check_func_signature(func,      # type: Callable
                         params     # type: List[AnyStr]
                         ):         # type: (...) -> None
    """Checks whether the signature of a function matches the expected parameter list.

    See ``thelper.typedefs.IterCallbackType`` and ``thelper.typedefs.IterCallbackParams`` for more info.
    """
    if func is None or not callable(func):
        raise AssertionError("invalid function object")
    if params is not None:
        if not isinstance(params, list) or not all([isinstance(p, str) for p in params]):
            raise AssertionError("unexpected param name list format")
        import inspect
        func_sig = inspect.signature(func)
        for p in params:
            if p not in func_sig.parameters:
                raise AssertionError("function missing parameter '%s'" % p)


def encode_data(data, approach="lz4", **kwargs):
    """Encodes a numpy array using a given coding approach.

    Args:
        data: the numpy array to encode.
        approach: the encoding; supports `none`, `lz4`, `jpg`, `png`.

    .. seealso::
        | :func:`thelper.utils.decode_data`
    """
    supported_approaches = ["none", "lz4", "jpg", "png"]
    if approach not in supported_approaches:
        raise AssertionError(f"unexpected approach type (got '{approach}')")
    if approach == "none":
        return data
    elif approach == "lz4":
        return lz4.frame.compress(data, **kwargs)
    elif approach == "jpg" or approach == "jpeg":
        ret, buf = cv.imencode(".jpg", data, **kwargs)
    elif approach == "png":
        ret, buf = cv.imencode(".png", data, **kwargs)
    else:
        raise NotImplementedError
    if not ret:
        raise AssertionError("failed to encode data")
    return buf


def decode_data(data, approach="lz4", **kwargs):
    """Decodes a binary array using a given coding approach.

    Args:
        data: the binary array to decode.
        approach: the encoding; supports `none`, `lz4`, `jpg`, `png`.

    .. seealso::
        | :func:`thelper.utils.encode_data`
    """
    supported_approach_types = ["none", "lz4", "jpg", "png"]
    if approach not in supported_approach_types:
        raise AssertionError(f"unexpected approach type (got '{approach}')")
    if approach == "none":
        return data
    elif approach == "lz4":
        return lz4.frame.decompress(data, **kwargs)
    elif approach in ["jpg", "jpeg", "png"]:
        kwargs = copy.deepcopy(kwargs)
        if isinstance(kwargs["flags"], str):  # required arg by opencv
            kwargs["flags"] = eval(kwargs["flags"])
        return cv.imdecode(data, **kwargs)
    else:
        raise NotImplementedError


def get_class_logger(skip=0):
    """Shorthand to get logger for current class frame."""
    return logging.getLogger(get_caller_name(skip + 1).rsplit(".", 1)[0])


def get_func_logger(skip=0):
    """Shorthand to get logger for current function frame."""
    return logging.getLogger(get_caller_name(skip + 1))


def get_caller_name(skip=2):
    # source: https://gist.github.com/techtonik/2151727
    """Returns the name of a caller in the format module.class.method.

    Args:
        skip: specifies how many levels of stack to skip while getting the caller.

    Returns:
        An empty string is returned if skipped levels exceed stack height; otherwise,
        returns the requested caller name.
    """

    def stack_(frame):
        frame_list = []
        while frame:
            frame_list.append(frame)
            frame = frame.f_back
        return frame_list

    # noinspection PyProtectedMember
    stack = stack_(sys._getframe(1))
    start = 0 + skip
    if len(stack) < start + 1:
        return ""
    parent_frame = stack[start]
    name = []
    module = inspect.getmodule(parent_frame)
    # `modname` can be None when frame is executed directly in console
    if module:
        name.append(module.__name__)
    # detect class name
    if "self" in parent_frame.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        name.append(parent_frame.f_locals["self"].__class__.__name__)
    codename = parent_frame.f_code.co_name
    if codename != "<module>":  # top level usually
        name.append(codename)  # function or a method
    del parent_frame
    return ".".join(name)


def get_key(key, config, msg=None, delete=False):
    """Returns a value given a dictionary key, throwing if not available."""
    if isinstance(key, list):
        if len(key) <= 1:
            if msg is not None:
                raise AssertionError(msg)
            else:
                raise AssertionError("must provide at least two valid keys to test")
        for k in key:
            if k in config:
                val = config[k]
                if delete:
                    del config[k]
                return val
        if msg is not None:
            raise AssertionError(msg)
        else:
            raise AssertionError("config dictionary missing a field named as one of '%s'" % str(key))
    else:
        if key not in config:
            if msg is not None:
                raise AssertionError(msg)
            else:
                raise AssertionError("config dictionary missing '%s' field" % key)
        else:
            val = config[key]
            if delete:
                del config[key]
            return val


def get_key_def(key, config, default=None, msg=None, delete=False):
    """Returns a value given a dictionary key, or the default value if it cannot be found."""
    if isinstance(key, list):
        if len(key) <= 1:
            if msg is not None:
                raise AssertionError(msg)
            else:
                raise AssertionError("must provide at least two valid keys to test")
        for k in key:
            if k in config:
                val = config[k]
                if delete:
                    del config[k]
                return val
        return default
    else:
        if key not in config:
            return default
        else:
            val = config[key]
            if delete:
                del config[key]
            return val


def get_log_stamp():
    """Returns a print-friendly and filename-friendly identification string containing platform and time."""
    return str(platform.node()) + "-" + time.strftime("%Y%m%d-%H%M%S")


def get_git_stamp():
    """Returns a print-friendly SHA signature for the framework's underlying git repository (if found)."""
    try:
        import git
        try:
            repo = git.Repo(path=os.path.abspath(__file__), search_parent_directories=True)
            sha = repo.head.object.hexsha
            return str(sha)
        except (AttributeError, git.InvalidGitRepositoryError):
            return "unknown"
    except (ImportError, AttributeError):
        return "unknown"


def get_env_list():
    """Returns a list of all packages installed in the current environment.

    If the required packages cannot be imported, the returned list will be empty. Note that some
    packages may not be properly detected by this approach, and it is pretty hacky, so use it with
    a grain of salt (i.e. logging is fine).
    """
    try:
        import pip
        # noinspection PyUnresolvedReferences
        pkgs = pip.get_installed_distributions()
        return sorted(["%s %s" % (pkg.key, pkg.version) for pkg in pkgs])
    except (ImportError, AttributeError):
        try:
            import pkg_resources as pkgr
            return sorted([str(pkg) for pkg in pkgr.working_set])
        except (ImportError, AttributeError):
            return []


def str2size(input_str):
    """Returns a (WIDTH, HEIGHT) integer size tuple from a string formatted as 'WxH'."""
    if not isinstance(input_str, str):
        raise AssertionError("unexpected input type")
    display_size_str = input_str.split('x')
    if len(display_size_str) != 2:
        raise AssertionError("bad size string formatting")
    return tuple([max(int(substr), 1) for substr in display_size_str])


def str2bool(s):
    """Converts a string to a boolean.

    If the lower case version of the provided string matches any of 'true', '1', or
    'yes', then the function returns ``True``.
    """
    if isinstance(s, bool):
        return s
    if isinstance(s, (int, float)):
        return s != 0
    if isinstance(s, str):
        positive_flags = ["true", "1", "yes"]
        return s.lower() in positive_flags
    raise AssertionError("unrecognized input type")


def clipstr(s, size, fill=" "):
    """Clips a string to a specific length, with an optional fill character."""
    if len(s) > size:
        s = s[:size]
    if len(s) < size:
        s = fill * (size - len(s)) + s
    return s


def lreplace(string, old_prefix, new_prefix):
    """Replaces a single occurrence of `old_prefix` in the given string by `new_prefix`."""
    return re.sub(r'^(?:%s)+' % re.escape(old_prefix), lambda m: new_prefix * (m.end() // len(old_prefix)), string)


def query_yes_no(question, default=None, bypass=None):
    """Asks the user a yes/no question and returns the answer.

    Args:
        question: the string that is presented to the user.
        default: the presumed answer if the user just hits ``<Enter>``. It must be 'yes',
            'no', or ``None`` (meaning an answer is required).
        bypass: the option to select if the ``bypass_queries`` global variable is set to
            ``True``. Can be ``None``, in which case the function will throw an exception.

    Returns:
        ``True`` for 'yes', or ``False`` for 'no' (or their respective variations).
    """
    valid = {"yes": True, "ye": True, "y": True, "no": False, "n": False}
    if bypass is not None and (not isinstance(bypass, str) or bypass not in valid):
        raise AssertionError("unexpected bypass value")
    if bypass_queries:
        if bypass is None:
            raise AssertionError("cannot bypass interactive query, no default value provided")
        return valid[bypass]
    if (isinstance(default, bool) and default) or \
       (isinstance(default, str) and default.lower() in ["yes", "ye", "y"]):
        prompt = " [Y/n] "
    elif (isinstance(default, bool) and not default) or \
         (isinstance(default, str) and default.lower() in ["no", "n"]):
        prompt = " [y/N] "
    else:
        prompt = " [y/n] "
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(0.25)  # to make sure all debug/info prints are done, and we see the question
    while True:
        sys.stdout.write(question + prompt + "\n>> ")
        choice = input().lower()
        if default is not None and choice == "":
            if isinstance(default, str):
                return valid[default]
            else:
                return default
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes/y' or 'no/n'.\n")


def query_string(question, choices=None, default=None, allow_empty=False, bypass=None):
    """Asks the user a question and returns the answer (a generic string).

    Args:
        question: the string that is presented to the user.
        choices: a list of predefined choices that the user can pick from. If
            ``None``, then whatever the user types will be accepted.
        default: the presumed answer if the user just hits ``<Enter>``. If ``None``,
            then an answer is required to continue.
        allow_empty: defines whether an empty answer should be accepted.
        bypass: the returned value if the ``bypass_queries`` global variable is set to
            ``True``. Can be ``None``, in which case the function will throw an exception.

    Returns:
        The string entered by the user.
    """
    if bypass_queries:
        if bypass is None:
            raise AssertionError("cannot bypass interactive query, no default value provided")
        return bypass
    sys.stdout.flush()
    sys.stderr.flush()
    time.sleep(0.25)  # to make sure all debug/info prints are done, and we see the question
    while True:
        msg = question
        if choices is not None:
            msg += "\n\t(choices=%s)" % str(choices)
        if default is not None:
            msg += "\n\t(default=%s)" % default
        sys.stdout.write(msg + "\n>> ")
        answer = input()
        if answer == "":
            if default is not None:
                return default
            elif allow_empty:
                return answer
        elif choices is not None:
            if answer in choices:
                return answer
        else:
            return answer
        sys.stdout.write("Please respond with a valid string.\n")


def get_save_dir(out_root, dir_name, config=None, resume=False, backup_ext=".json"):
    """Returns a directory path in which the app can save its data.

    If a folder with name ``dir_name`` already exists in the directory ``out_root``, then the user will be
    asked to pick a new name. If the user refuses, ``sys.exit(1)`` is called. If config is not ``None``, it
    will be saved to the output directory as a json file. Finally, a ``logs`` directory will also be created
    in the output directory for writing logger files.

    Args:
        out_root: path to the directory root where the save directory should be created.
        dir_name: name of the save directory to create. If it already exists, a new one will be requested.
        config: dictionary of app configuration parameters. Used to overwrite i/o queries, and will be
            written to the save directory in json format to test writing. Default is ``None``.
        resume: specifies whether this session is new, or resumed from an older one (in the latter
            case, overwriting is allowed, and the user will never have to choose a new folder)
        backup_ext: extension to use when creating configuration file backups.

    Returns:
        The path to the created save directory for this session.
    """
    func_logger = get_func_logger()
    save_dir = out_root
    if save_dir is None:
        time.sleep(0.25)  # to make sure all debug/info prints are done, and we see the question
        save_dir = query_string("Please provide the path to where session directories should be created/saved:")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = os.path.join(save_dir, dir_name)
    if not resume:
        overwrite = str2bool(config["overwrite"]) if config is not None and "overwrite" in config else False
        time.sleep(0.25)  # to make sure all debug/info prints are done, and we see the question
        while os.path.exists(save_dir) and not overwrite:
            abs_save_dir = os.path.abspath(save_dir).replace("\\", "/")
            overwrite = query_yes_no("Training session at '%s' already exists; overwrite?" % abs_save_dir, bypass="y")
            if not overwrite:
                save_dir = query_string("Please provide a new save directory path:")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if config is not None:
            save_config(config, os.path.join(save_dir, "config.latest" + backup_ext))
    else:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if config is not None:
            backup_path = os.path.join(save_dir, "config.latest" + backup_ext)
            if os.path.exists(backup_path):
                with open(backup_path, "r") as fd:
                    config_backup = json.load(fd)
                if config_backup != config:
                    query_msg = f"Config backup in '{backup_path}' differs from config loaded through checkpoint; overwrite?"
                    answer = query_yes_no(query_msg, bypass="y")
                    if answer:
                        func_logger.warning("config mismatch with previous run; "
                                            "will overwrite latest backup in save directory")
                    else:
                        func_logger.error("config mismatch with previous run; user aborted")
                        sys.exit(1)
            save_config(config, backup_path)
    logs_dir = os.path.join(save_dir, "logs")
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    save_config(config, os.path.join(logs_dir, "config." + thelper.utils.get_log_stamp() + backup_ext))
    return save_dir


def save_config(config, path, force_convert=True):
    """Saves the given session/object configuration dictionary to the provided path.

    The type of file that is created is based on the extension specified in the path. If the file
    cannot hold some of the objects within the configuration, they will be converted to strings before
    serialization, unless `force_convert` is set to `False` (in which case the function will raise
    an exception).

    Args:
        config: the session/object configuration dictionary to save.
        path: the path specifying where to create the output file. The extension used will determine
            what type of backup to create (e.g. Pickle = .pkl, JSON = .json).
        force_convert: specifies whether non-serializable types should be converted if necessary.
    """
    if path.endswith(".json"):
        serializer = (lambda x: str(x)) if force_convert else None
        with open(path, "w") as fd:
            json.dump(config, fd, indent=4, sort_keys=False, default=serializer)
    elif path.endswith(".pkl"):
        with open(path, "w") as fd:
            pickle.dump(config, fd)
    else:
        raise AssertionError("unknown output file type")


def save_env_list(path):
    """Saves a list of all packages installed in the current environment to a log file.

    Args:
        path: the path where the log file should be created.
    """
    with open(path, "w") as fd:
        pkgs_list = thelper.utils.get_env_list()
        if pkgs_list:
            for pkg in pkgs_list:
                fd.write("%s\n" % pkg)
        else:
            fd.write("<n/a>\n")


def safe_crop(image, tl, br, bordertype=cv.BORDER_CONSTANT, borderval=0, force_copy=False):
    """Safely crops a region from within an image, padding borders if needed.

    Args:
        image: the image to crop (provided as a numpy array).
        tl: a tuple or list specifying the (x,y) coordinates of the top-left crop corner.
        br: a tuple or list specifying the (x,y) coordinates of the bottom-right crop corner.
        bordertype: border copy type to use when the image is too small for the required crop size.
            See ``cv2.copyMakeBorder`` for more information.
        borderval: border value to use when the image is too small for the required crop size. See
            ``cv2.copyMakeBorder`` for more information.
        force_copy: defines whether to force a copy of the target image region even when it can be
            avoided.

    Returns:
        The cropped image.
    """
    if not isinstance(image, np.ndarray):
        raise AssertionError("expected input image to be numpy array")
    if isinstance(tl, tuple):
        tl = list(tl)
    if isinstance(br, tuple):
        br = list(br)
    if not isinstance(tl, list) or not isinstance(br, list):
        raise AssertionError("expected tl/br coords to be provided as tuple or list")
    if tl[0] < 0 or tl[1] < 0 or br[0] > image.shape[1] or br[1] > image.shape[0]:
        image = cv.copyMakeBorder(image, max(-tl[1], 0), max(br[1] - image.shape[0], 0),
                                  max(-tl[0], 0), max(br[0] - image.shape[1], 0),
                                  borderType=bordertype, value=borderval)
        if tl[0] < 0:
            br[0] -= tl[0]
            tl[0] = 0
        if tl[1] < 0:
            br[1] -= tl[1]
            tl[1] = 0
        return image[tl[1]:br[1], tl[0]:br[0], ...]
    if force_copy:
        return np.copy(image[tl[1]:br[1], tl[0]:br[0], ...])
    return image[tl[1]:br[1], tl[0]:br[0], ...]


def get_bgr_from_hsl(hue, sat, light):
    """Converts a single HSL triplet (0-360 hue, 0-1 sat & lightness) into an 8-bit RGB triplet."""
    # this function is not intended for fast conversions; use OpenCV's cvtColor for large-scale stuff
    if hue < 0 or hue > 360:
        raise AssertionError("invalid hue")
    if sat < 0 or sat > 1:
        raise AssertionError("invalid saturation")
    if light < 0 or light > 1:
        raise AssertionError("invalid lightness")
    if sat == 0:
        return (int(np.clip(round(light * 255), 0, 255)),) * 3
    if light == 0:
        return 0, 0, 0
    if light == 1:
        return 255, 255, 255

    def h2rgb(_p, _q, _t):
        if _t < 0:
            _t += 1
        if _t > 1:
            _t -= 1
        if _t < 1 / 6:
            return _p + (_q - _p) * 6 * _t
        if _t < 1 / 2:
            return _q
        if _t < 2 / 3:
            return _p + (_q - _p) * (2 / 3 - _t) * 6
        return _p

    q = light * (1 + sat) if (light < 0.5) else light + sat - light*sat
    p = 2 * light - q
    h = hue / 360
    return (int(np.clip(round(h2rgb(p, q, h - 1 / 3) * 255), 0, 255)),
            int(np.clip(round(h2rgb(p, q, h) * 255), 0, 255)),
            int(np.clip(round(h2rgb(p, q, h + 1 / 3) * 255), 0, 255)))


def get_displayable_image(image,                # type: thelper.typedefs.ArrayType
                          grayscale=False,      # type: Optional[bool]
                          ):                    # type: (...) -> thelper.typedefs.ArrayType
    """Returns a 'displayable' image that has been normalized and padded to three channels."""
    if image.ndim != 3:
        raise AssertionError("indexing should return a pre-squeezed array")
    if image.shape[2] == 2:
        image = np.dstack((image, image[:, :, 0]))
    elif image.shape[2] > 3:
        image = image[..., :3]
    if grayscale and image.shape[2] != 1:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif not grayscale and image.shape[2] == 1:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    image_normalized = np.empty_like(image, dtype=np.uint8).copy()  # copy needed here due to ocv 3.3 bug
    cv.normalize(image, image_normalized, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return image_normalized


def get_displayable_heatmap(array,              # type: thelper.typedefs.ArrayType
                            convert_rgb=True,   # type: Optional[bool]
                            ):                  # type: (...) -> thelper.typedefs.ArrayType
    """Returns a 'displayable' array that has been min-maxed and mapped to color triplets."""
    if array.ndim != 2:
        array = np.squeeze(array)
    if array.ndim != 2:
        raise AssertionError("indexing should return a pre-squeezed array")
    array_normalized = np.empty_like(array, dtype=np.uint8).copy()  # copy needed here due to ocv 3.3 bug
    cv.normalize(array, array_normalized, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    heatmap = cv.applyColorMap(array_normalized, cv.COLORMAP_JET)
    if convert_rgb:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)
    return heatmap


def is_scalar(val):
    """Returns whether the input value is a scalar according to numpy and PyTorch."""
    if np.isscalar(val):
        return True
    if isinstance(val, torch.Tensor) and (val.dim() == 0 or val.numel() == 1):
        return True
    return False


def to_numpy(array):
    """Converts a list or PyTorch tensor to numpy. Does nothing if already a numpy array."""
    if isinstance(array, list):
        return np.asarray(array)
    elif isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    elif isinstance(array, np.ndarray):
        return array
    else:
        raise AssertionError(f"unexpected input type ({type(array)})")


def draw_histogram(data,                # type: thelper.typedefs.ArrayType
                   bins=50,             # type: Optional[int]
                   xlabel="",           # type: Optional[thelper.typedefs.LabelType]
                   ylabel="Proportion"  # type: Optional[thelper.typedefs.LabelType]
                   ):                   # type: (...) -> thelper.typedefs.DrawingType
    """Draws and returns a histogram figure using pyplot."""
    fig, ax = plt.subplots()
    ax.hist(data, density=True, bins=bins)
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel)
    ax.set_xlim(xmin=0)
    fig.show()
    return fig


def draw_popbars(labels,                # type: thelper.typedefs.LabelList
                 counts,                # type: int
                 xlabel="",             # type: Optional[thelper.typedefs.LabelType]
                 ylabel="Pop. Count",   # type: Optional[thelper.typedefs.LabelType]
                 ):                     # type: (...) -> thelper.typedefs.DrawingType
    """Draws and returns a bar histogram figure using pyplot."""
    fig, ax = plt.subplots()
    xrange = range(len(labels))
    ax.bar(xrange, counts, align="center")
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel)
    ax.set_xticks(xrange)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize="8", labelrotation=45)
    fig.show()
    return fig


def draw_images(images,               # type: thelper.typedefs.OneOrManyArrayType
                captions=None,        # type: Optional[List[str]]
                redraw=None,          # type: Optional[thelper.typedefs.DrawingType]
                block=False,          # type: Optional[bool]
                use_cv2=True,         # type: Optional[bool]
                cv2_flip_bgr=True,    # type: Optional[bool]
                img_shape=None,       # type: Optional[thelper.typedefs.ArrayShapeType]
                max_img_size=None,    # type: Optional[thelper.typedefs.ArrayShapeType]
                grid_size_x=None,     # type: Optional[int]
                grid_size_y=None,     # type: Optional[int]
                caption_opts=None,
                window_name=None,     # type: Optional[str]
                ):                    # type: (...) -> thelper.typedefs.DrawingType
    """Draws a set of images with optional captions."""
    nb_imgs = len(images) if isinstance(images, list) else images.shape[0]
    if nb_imgs < 1:
        return None
    assert captions is None or len(captions) == nb_imgs, "captions count mismatch with image count"
    # for display on typical monitors... (height, width)
    max_img_size = (800, 1600) if max_img_size is None else max_img_size
    grid_size_x = int(math.ceil(math.sqrt(nb_imgs))) if grid_size_x is None else grid_size_x
    grid_size_y = int(math.ceil(nb_imgs / grid_size_x)) if grid_size_y is None else grid_size_y
    assert grid_size_x * grid_size_y >= nb_imgs, f"bad gridding for subplots (need at least {nb_imgs} tiles)"
    if use_cv2:
        if caption_opts is None:
            caption_opts = {
                "org": (10, 40),
                "fontFace": cv.FONT_HERSHEY_SIMPLEX,
                "fontScale": 0.40,
                "color": (255, 255, 255),
                "thickness": 1,
                "lineType": cv.LINE_AA
            }
        if window_name is None:
            window_name = "images"
        img_grid_shape = None
        img_grid = None if redraw is None else redraw[1]
        for img_idx in range(nb_imgs):
            image = images[img_idx] if isinstance(images, list) else images[img_idx, ...]
            if img_shape is None:
                img_shape = image.shape
            if img_grid_shape is None:
                img_grid_shape = (img_shape[0] * grid_size_y, img_shape[1] * grid_size_x, img_shape[2])
            if img_grid is None or img_grid.shape != img_grid_shape:
                img_grid = np.zeros(img_grid_shape, dtype=np.uint8)
            if image.shape[2] != img_shape[2]:
                raise AssertionError(f"unexpected image depth ({image.shape[2]} vs {img_shape[2]})")
            if image.shape != img_shape:
                image = cv.resize(image, (img_shape[1], img_shape[0]), interpolation=cv.INTER_NEAREST)
            if captions is not None and str(captions[img_idx]):
                image = cv.putText(image.copy(), str(captions[img_idx]), **caption_opts)
            offsets = (img_idx // grid_size_x) * img_shape[0], (img_idx % grid_size_x) * img_shape[1]
            np.copyto(img_grid[offsets[0]:(offsets[0] + img_shape[0]), offsets[1]:(offsets[1] + img_shape[1]), :], image)
        win_name = str(window_name) if redraw is None else redraw[0]
        if img_grid is not None:
            display = img_grid[..., ::-1] if cv2_flip_bgr else img_grid
            if display.shape[0] > max_img_size[0] or display.shape[1] > max_img_size[1]:
                if display.shape[0] / max_img_size[0] > display.shape[1] / max_img_size[1]:
                    dsize = (max_img_size[0], int(round(display.shape[1] / (display.shape[0] / max_img_size[0]))))
                else:
                    dsize = (int(round(display.shape[0] / (display.shape[1] / max_img_size[1]))), max_img_size[1])
                display = cv.resize(display, (dsize[1], dsize[0]))
            cv.imshow(win_name, display)
            cv.waitKey(0 if block else 1)
        return win_name, img_grid
    else:
        fig, axes = redraw if redraw is not None else plt.subplots(grid_size_y, grid_size_x)
        plt.tight_layout()
        if nb_imgs == 1:
            axes = np.array(axes)
        for ax_idx, ax in enumerate(axes.reshape(-1)):
            if ax_idx < nb_imgs:
                image = images[ax_idx] if isinstance(images, list) else images[ax_idx, ...]
                if image.shape != img_shape:
                    image = cv.resize(image, (img_shape[1], img_shape[0]), interpolation=cv.INTER_NEAREST)
                ax.imshow(image, interpolation='nearest')
                if captions is not None and str(captions[ax_idx]):
                    ax.set_xlabel(str(captions[ax_idx]))
            ax.set_xticks([])
            ax.set_yticks([])
        fig.show()
        if block:
            plt.show(block=block)
            return None
        plt.pause(0.5)
        return fig, axes


def draw_predicts(images, preds=None, targets=None, swap_channels=False, redraw=None, block=False, **kwargs):
    """Draws and returns a set of generic prediction results."""
    image_list = [get_displayable_image(images[batch_idx, ...]) for batch_idx in range(images.shape[0])]
    image_gray_list = [cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR) for image in image_list]
    nb_imgs = len(image_list)
    caption_list = [""] * nb_imgs
    grid_size_x, grid_size_y = nb_imgs, 1  # all images on one row, by default (add gt and preds as extra rows)
    if targets is not None:
        if not isinstance(targets, list) and not (isinstance(targets, torch.Tensor) and targets.shape[0] == nb_imgs):
            raise AssertionError("expected targets to be in list or tensor format (Bx...)")
        if isinstance(targets, list):
            if all([isinstance(t, list) for t in targets]):
                targets = list(itertools.chain.from_iterable(targets))  # merge all augmented lists together
            targets = torch.cat(targets, 0)  # merge all masks into a single tensor
        if targets.shape[0] != nb_imgs:
            raise AssertionError("images/targets count mismatch")
        targets = targets.numpy()
        if swap_channels:
            if not targets.ndim == 4:
                raise AssertionError("unexpected swap for targets tensor that is not 4-dim")
            targets = np.transpose(targets, (0, 2, 3, 1))  # BxCxHxW to BxHxWxC
        if ((targets.ndim == 4 and targets.shape[1] == 1) or targets.ndim == 3) and targets.shape[-2:] == images.shape[1:3]:
            target_list = [get_displayable_heatmap(targets[batch_idx, ...]) for batch_idx in range(nb_imgs)]
            target_list = [cv.addWeighted(image_gray_list[idx], 0.3, target_list[idx], 0.7, 0) for idx in range(nb_imgs)]
            image_list += target_list
            caption_list += [""] * nb_imgs
            grid_size_y += 1
        elif targets.shape == images.shape:
            image_list += [get_displayable_image(targets[batch_idx, ...]) for batch_idx in range(nb_imgs)]
            caption_list += [""] * nb_imgs
            grid_size_y += 1
        else:
            for idx in range(nb_imgs):
                caption_list[idx] = f"GT={str(targets[idx])}"
    if preds is not None:
        if not isinstance(preds, list) and not (isinstance(preds, torch.Tensor) and preds.shape[0] == nb_imgs):
            raise AssertionError("expected preds to be in list or tensor shape (Bx...)")
        if isinstance(preds, list):
            if all([isinstance(p, list) for p in preds]):
                preds = list(itertools.chain.from_iterable(preds))  # merge all augmented lists together
            preds = torch.cat(preds, 0)  # merge all preds into a single tensor
        if preds.shape[0] != nb_imgs:
            raise AssertionError("images/preds count mismatch")
        preds = preds.numpy()
        if swap_channels:
            if not preds.ndim == 4:
                raise AssertionError("unexpected swap for targets tensor that is not 4-dim")
            preds = np.transpose(preds, (0, 2, 3, 1))  # BxCxHxW to BxHxWxC
        if targets is not None and preds.shape != targets.shape:
            raise AssertionError("preds/targets shape mismatch")
        if ((preds.ndim == 4 and preds.shape[1] == 1) or preds.ndim == 3) and preds.shape[-2:] == images.shape[1:3]:
            pred_list = [get_displayable_heatmap(preds[batch_idx, ...]) for batch_idx in range(nb_imgs)]
            pred_list = [cv.addWeighted(image_gray_list[idx], 0.3, pred_list[idx], 0.7, 0) for idx in range(nb_imgs)]
            image_list += pred_list
            caption_list += [""] * nb_imgs
            grid_size_y += 1
        elif preds.shape == images.shape:
            image_list += [get_displayable_image(preds[batch_idx, ...]) for batch_idx in range(nb_imgs)]
            caption_list += [""] * nb_imgs
            grid_size_y += 1
        else:
            for idx in range(nb_imgs):
                if len(caption_list[idx]) != 0:
                    caption_list[idx] += ", "
                caption_list[idx] = f"Pred={str(preds[idx])}"
    return draw_images(image_list, captions=caption_list, redraw=redraw, window_name="predictions",
                       block=block, grid_size_x=grid_size_x, grid_size_y=grid_size_y, **kwargs)


def draw_segments(images, preds=None, masks=None, color_map=None, redraw=None, block=False, **kwargs):
    """Draws and returns a set of segmentation results."""
    image_list = [get_displayable_image(images[batch_idx, ...]) for batch_idx in range(images.shape[0])]
    image_gray_list = [cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2BGR) for image in image_list]
    nb_imgs = len(image_list)
    grid_size_x, grid_size_y = nb_imgs, 1  # all images on one row, by default (add gt and preds as extra rows)
    if color_map is not None and isinstance(color_map, dict):
        assert len(color_map) <= 256, "too many indices for uint8 map"
        color_map_new = np.zeros((256, 1, 3), dtype=np.uint8)
        for idx, val in color_map.items():
            color_map_new[idx, ...] = val
        color_map = color_map_new
    if masks is not None:
        if not isinstance(masks, list) and not (isinstance(masks, torch.Tensor) and masks.dim() == 3):
            raise AssertionError("expected segmentation masks to be in list or 3-d tensor format (BxHxW)")
        if isinstance(masks, list):
            if all([isinstance(m, list) for m in masks]):
                masks = list(itertools.chain.from_iterable(masks))  # merge all augmented lists together
            masks = torch.cat(masks, 0)  # merge all masks into a single tensor
        if masks.shape[0] != nb_imgs:
            raise AssertionError("images/masks count mismatch")
        if images.shape[0:3] != masks.shape:
            raise AssertionError("images/masks shape mismatch")
        masks = masks.numpy()
        if color_map is not None:
            masks = [apply_color_map(masks[idx], color_map) for idx in range(masks.shape[0])]
        image_list += [cv.addWeighted(image_gray_list[idx], 0.3, masks[idx], 0.7, 0) for idx in range(nb_imgs)]
        grid_size_y += 1
    if preds is not None:
        if not isinstance(preds, list) and not (isinstance(preds, torch.Tensor) and preds.dim() == 4):
            raise AssertionError("expected segmentation preds to be in list or 3-d tensor format (BxCxHxW)")
        if isinstance(preds, list):
            if all([isinstance(p, list) for p in preds]):
                preds = list(itertools.chain.from_iterable(preds))  # merge all augmented lists together
            preds = torch.cat(preds, 0)  # merge all preds into a single tensor
        with torch.no_grad():
            preds = torch.squeeze(preds.topk(k=1, dim=1)[1], dim=1)  # keep top prediction index only
        if preds.shape[0] != nb_imgs:
            raise AssertionError("images/preds count mismatch")
        if images.shape[0:3] != preds.shape:
            raise AssertionError("images/preds shape mismatch")
        preds = preds.numpy()
        if color_map is not None:
            preds = [apply_color_map(preds[idx], color_map) for idx in range(preds.shape[0])]
        image_list += [cv.addWeighted(image_gray_list[idx], 0.3, preds[idx], 0.7, 0) for idx in range(nb_imgs)]
        grid_size_y += 1
    return draw_images(image_list, redraw=redraw, window_name="segments", block=block,
                       grid_size_x=grid_size_x, grid_size_y=grid_size_y, **kwargs)


def draw_classifs(images, preds=None, labels=None, class_names_map=None, redraw=None, block=False, **kwargs):
    """Draws and returns a set of classification results."""
    image_list = [get_displayable_image(images[batch_idx, ...]) for batch_idx in range(images.shape[0])]
    caption_list = [""] * len(image_list)
    if labels is not None:  # convert labels to flat list, if available
        if not isinstance(labels, list) and not (isinstance(labels, torch.Tensor) and labels.dim() == 1):
            raise AssertionError("expected classification labels to be in list or 1-d tensor format")
        if isinstance(labels, list):
            if all([isinstance(l, list) for l in labels]):
                labels = list(itertools.chain.from_iterable(labels))  # merge all augmented lists together
            if all([isinstance(t, torch.Tensor) for t in labels]):
                labels = torch.cat(labels, 0)
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        if images.shape[0] != len(labels):
            raise AssertionError("images/labels count mismatch")
        if class_names_map is not None:
            labels = [class_names_map[lbl] if lbl in class_names_map else lbl for lbl in labels]
        for idx in range(len(image_list)):
            caption_list[idx] = f"GT={labels[idx]}"
    if preds is not None:  # convert predictions to flat list, if available
        if not isinstance(preds, list) and not (isinstance(preds, torch.Tensor) and preds.dim() == 2):
            raise AssertionError("expected classification predictions to be in list or 2-d tensor format (BxC)")
        if isinstance(preds, list):
            if all([isinstance(p, list) for p in preds]):
                preds = list(itertools.chain.from_iterable(preds))  # merge all augmented lists together
            if all([isinstance(t, torch.Tensor) for t in preds]):
                preds = torch.cat(preds, 0)
        with torch.no_grad():
            preds = torch.squeeze(preds.topk(1, dim=1)[1], dim=1)
        if images.shape[0] != preds.shape[0]:
            raise AssertionError("images/predictions count mismatch")
        preds = preds.tolist()
        if class_names_map is not None:
            preds = [class_names_map[lbl] if lbl in class_names_map else lbl for lbl in preds]
        for idx in range(len(image_list)):
            if len(caption_list[idx]) != 0:
                caption_list[idx] += ", "
            caption_list[idx] += f"Pred={preds[idx]}"
    return draw_images(image_list, captions=caption_list, redraw=redraw, window_name="classifs", block=block, **kwargs)


def draw_minibatch(minibatch, task, preds=None, block=False, ch_transpose=True, flip_bgr=False, redraw=None, **kwargs):
    """Draws and returns a figure of a minibatch using pyplot or OpenCV."""
    if not isinstance(minibatch, dict):
        raise AssertionError("expected dict-based sample")
    import thelper.tasks
    if not isinstance(task, thelper.tasks.Task):
        raise AssertionError("invalid task object")
    if task.input_key not in minibatch:
        raise AssertionError("images not found in minibatch with key '%s'" % task.input_key)
    images = minibatch[task.input_key]
    if isinstance(images, list) and all([isinstance(t, torch.Tensor) for t in images]):
        # if we have a list, it must be due to a augmentation stage
        if not all([image.shape == images[0].shape for image in images]):
            raise AssertionError("image shape mismatch throughout list")
        images = torch.cat(images, 0)  # merge all images into a single tensor
    if not isinstance(images, torch.Tensor) or images.dim() != 4:
        raise AssertionError("expected input images to be in 4-d tensor format (BxCxHxW or BxHxWxC)")
    images = images.numpy().copy()
    if ch_transpose:
        images = np.transpose(images, (0, 2, 3, 1))  # BxCxHxW to BxHxWxC
    if flip_bgr:
        images = images[..., ::-1]  # BGR to RGB
    if preds is not None:
        preds = preds.cpu().detach()  # avoid latency for preprocessing on gpu
    if isinstance(task, thelper.tasks.Classification):
        labels = thelper.utils.get_key_def(task.gt_key, minibatch, None)
        class_names_map = {idx: name for name, idx in task.class_indices.items()}
        return draw_classifs(images=images, preds=preds, labels=labels,
                             class_names_map=class_names_map, redraw=redraw, block=block, **kwargs)
    elif isinstance(task, thelper.tasks.Segmentation):
        masks = thelper.utils.get_key_def(task.gt_key, minibatch, None)
        color_map = task.color_map if task.color_map else {idx: get_label_color_mapping(idx) for idx in task.class_indices.values()}
        if task.dontcare is not None and task.dontcare not in color_map:
            color_map[task.dontcare] = np.asarray([0, 0, 0])
        return draw_segments(images=images, preds=preds, masks=masks, color_map=color_map, redraw=redraw, block=block, **kwargs)
    elif isinstance(task, thelper.tasks.Detection):
        bboxes = thelper.utils.get_key_def(task.gt_key, minibatch, None)
        color_map = task.color_map if task.color_map else {idx: get_label_color_mapping(idx) for idx in task.class_indices.values()}
        return draw_bboxes(images=images, preds=preds, bboxes=bboxes, color_map=color_map, redraw=redraw, block=block, **kwargs)
    elif isinstance(task, thelper.tasks.Regression):
        targets = thelper.utils.get_key_def(task.gt_key, minibatch, None)
        swap_channels = isinstance(task, thelper.tasks.SuperResolution)  # must update BxCxHxW to BxHxWxC in targets/preds
        # @@@ todo: cleanup swap_channels above via flag in superres task?
        return draw_predicts(images=images, preds=preds, targets=targets,
                             swap_channels=swap_channels, redraw=redraw, block=block, **kwargs)
    else:
        raise AssertionError("unhandled drawing mode, missing impl")


# noinspection PyUnusedLocal
def draw_errbars(labels,                # type: thelper.typedefs.LabelList
                 min_values,            # type: thelper.typedefs.ArrayType
                 max_values,            # type: thelper.typedefs.ArrayType
                 stddev_values,         # type: thelper.typedefs.ArrayType
                 mean_values,           # type: thelper.typedefs.ArrayType
                 xlabel="",             # type: thelper.typedefs.LabelType
                 ylabel="Raw Value"     # type: thelper.typedefs.LabelType
                 ):                     # type: (...) -> thelper.typedefs.DrawingType
    """Draws and returns an error bar histogram figure using pyplot."""
    if min_values.shape != max_values.shape \
            or min_values.shape != stddev_values.shape \
            or min_values.shape != mean_values.shape:
        raise AssertionError("input dim mismatch")
    if len(min_values.shape) != 1 and len(min_values.shape) != 2:
        raise AssertionError("input dim unexpected")
    if len(min_values.shape) == 1:
        np.expand_dims(min_values, 1)
        np.expand_dims(max_values, 1)
        np.expand_dims(stddev_values, 1)
        np.expand_dims(mean_values, 1)
    nb_subplots = min_values.shape[1]
    fig, axs = plt.subplots(nb_subplots)
    xrange = range(len(labels))
    for ax_idx in range(nb_subplots):
        ax = axs[ax_idx]
        ax.locator_params(nbins=nb_subplots)
        ax.errorbar(xrange, mean_values[:, ax_idx], stddev_values[:, ax_idx], fmt='ok', lw=3)
        ax.errorbar(xrange, mean_values[:, ax_idx], [mean_values[:, ax_idx] - min_values[:, ax_idx],
                                                     max_values[:, ax_idx] - mean_values[:, ax_idx]],
                    fmt='.k', ecolor='gray', lw=1)
        ax.set_xticks(xrange)
        ax.set_xticklabels(labels, visible=(ax_idx == nb_subplots - 1))
        ax.set_title("Band %d" % (ax_idx + 1))
        ax.tick_params(axis="x", labelsize="6", labelrotation=45)
    plt.tight_layout()
    fig.show()
    return fig


def draw_roc_curve(fpr, tpr, labels=None, size_inch=(5, 5), dpi=320):
    """Draws and returns an ROC curve figure using pyplot."""
    if not isinstance(fpr, np.ndarray) or not isinstance(tpr, np.ndarray):
        raise AssertionError("invalid inputs")
    if fpr.shape != tpr.shape:
        raise AssertionError("mismatched input sizes")
    if fpr.ndim == 1:
        fpr = np.expand_dims(fpr, 0)
    if tpr.ndim == 1:
        tpr = np.expand_dims(tpr, 0)
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        if len(labels) != fpr.shape[0]:
            raise AssertionError("should have one label per curve")
    else:
        labels = [None] * fpr.shape[0]
    fig = plt.figure(num="roc", figsize=size_inch, dpi=dpi, facecolor="w", edgecolor="k")
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    for idx, label in enumerate(labels):
        auc = sklearn.metrics.auc(fpr[idx, ...], tpr[idx, ...])
        if label is not None:
            ax.plot(fpr[idx, ...], tpr[idx, ...], "b", label=("%s [auc = %0.3f]" % (label, auc)))
        else:
            ax.plot(fpr[idx, ...], tpr[idx, ...], "b", label=("auc = %0.3f" % auc))
    ax.legend(loc="lower right")
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("True Positive Rate")
    ax.set_xlabel("False Positive Rate")
    fig.set_tight_layout(True)
    return fig


def draw_confmat(confmat, class_list, size_inch=(5, 5), dpi=320, normalize=False, keep_unset=False):
    """Draws and returns an a confusion matrix figure using pyplot."""
    if not isinstance(confmat, np.ndarray) or not isinstance(class_list, list):
        raise AssertionError("invalid inputs")
    if confmat.ndim != 2:
        raise AssertionError("invalid confmat shape")
    if not keep_unset and "<unset>" in class_list:
        unset_idx = class_list.index("<unset>")
        del class_list[unset_idx]
        np.delete(confmat, unset_idx, 0)
        np.delete(confmat, unset_idx, 1)
    if normalize:
        row_sums = confmat.sum(axis=1)[:, np.newaxis]
        confmat = np.nan_to_num(confmat.astype(np.float) / np.maximum(row_sums, 0.0001))
    fig = plt.figure(num="confmat", figsize=size_inch, dpi=dpi, facecolor="w", edgecolor="k")
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(confmat, cmap=plt.cm.Blues)
    labels = [clipstr(label, 9) for label in class_list]
    tick_marks = np.arange(len(labels))
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=4, rotation=-90, ha="center")
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    ax.set_ylabel("Real", fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels, fontsize=4, va="center")
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    thresh = confmat.max() / 2.
    for i, j in itertools.product(range(confmat.shape[0]), range(confmat.shape[1])):
        if not normalize:
            txt = ("%d" % confmat[i, j]) if confmat[i, j] != 0 else "."
        else:
            if confmat[i, j] >= 0.01:
                txt = "%.02f" % confmat[i, j]
            else:
                txt = "~0" if confmat[i, j] > 0 else "."
        color = "white" if confmat[i, j] > thresh else "black"
        ax.text(j, i, txt, horizontalalignment="center", fontsize=4, verticalalignment="center", color=color)
    fig.set_tight_layout(True)
    return fig


def draw_bbox(image, tl, br, text, color, box_thickness=2, font_thickness=1, font_scale=0.4):
    """Draws a single bounding box on a given image (used in :func:`thelper.utils.draw_bboxes`)."""
    text_size, baseline = cv.getTextSize(text, fontFace=cv.FONT_HERSHEY_SIMPLEX,
                                         fontScale=font_scale, thickness=font_thickness)
    text_bl = (tl[0] + box_thickness + 1, tl[1] + text_size[1] + box_thickness + 1)
    # note: text will overflow if box is too small
    text_box_br = (text_bl[0] + text_size[0] + box_thickness, text_bl[1] + box_thickness * 2)
    cv.rectangle(image, (tl[0] - 1, tl[1] - 1), (text_box_br[0] + 1, text_box_br[1] + 1),
                 color=(0, 0, 0), thickness=-1)
    cv.rectangle(image, tl, br, color=(0, 0, 0), thickness=(box_thickness + 1))
    cv.rectangle(image, tl, br, color=color, thickness=box_thickness)
    cv.rectangle(image, tl, text_box_br, color=color, thickness=-1)
    cv.putText(image, text, text_bl, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
               color=(0, 0, 0), thickness=font_thickness + 1)
    cv.putText(image, text, text_bl, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
               color=(255, 255, 255), thickness=font_thickness)


def draw_bboxes(images, preds=None, bboxes=None, color_map=None, redraw=None, block=False, **kwargs):
    """Draws and returns a set of bounding box prediction results."""
    image_list = [get_displayable_image(images[batch_idx, ...]) for batch_idx in range(images.shape[0])]
    if color_map is not None and isinstance(color_map, dict):
        assert len(color_map) <= 256, "too many indices for uint8 map"
        color_map_new = np.zeros((256, 3), dtype=np.uint8)
        for idx, val in color_map.items():
            color_map_new[idx, ...] = val
        color_map = color_map_new.tolist()
    nb_imgs = len(image_list)
    grid_size_x, grid_size_y = nb_imgs, 1  # all images on one row, by default (add gt and preds as extra rows)
    box_thickness = thelper.utils.get_key_def("box_thickness", kwargs, default=2, delete=True)
    font_thickness = thelper.utils.get_key_def("font_thickness", kwargs, default=1, delete=True)
    font_scale = thelper.utils.get_key_def("font_scale", kwargs, default=0.4, delete=True)
    if preds is not None:
        assert len(image_list) == len(bboxes)
        for bboxes_list, image in zip(bboxes, image_list):
            for bbox_idx, bbox in enumerate(bboxes_list):
                assert isinstance(bbox, thelper.data.BoundingBox), "unrecognized bbox type"
                color = get_bgr_from_hsl(bbox_idx / len(bboxes_list) * 360, 1.0, 0.5) \
                    if color_map is None else color_map[bbox.class_id]
                conf = ""
                if thelper.utils.is_scalar(bbox.confidence):
                    conf = f" ({bbox.confidence})"
                elif isinstance(bbox.confidence, (list, tuple, np.ndarray)):
                    conf = f" ({bbox.confidence[bbox.class_id]})"
                draw_bbox(image, bbox.top_left, bbox.bottom_right, f"{bbox.task.class_names[bbox.class_id]}{conf}", color,
                          box_thickness=box_thickness, font_thickness=font_thickness, font_scale=font_scale)
    if bboxes is not None:
        assert len(image_list) == len(bboxes), "mismatched bboxes list and image list sizes"
        clean_image_list = [get_displayable_image(images[batch_idx, ...]) for batch_idx in range(images.shape[0])]
        for bboxes_list, image in zip(bboxes, clean_image_list):
            for bbox_idx, bbox in enumerate(bboxes_list):
                assert isinstance(bbox, thelper.data.BoundingBox), "unrecognized bbox type"
                color = get_bgr_from_hsl(bbox_idx / len(bboxes_list) * 360, 1.0, 0.5) \
                    if color_map is None else color_map[bbox.class_id]
                draw_bbox(image, bbox.top_left, bbox.bottom_right, f"GT: {bbox.task.class_names[bbox.class_id]}", color,
                          box_thickness=box_thickness, font_thickness=font_thickness, font_scale=font_scale)
        grid_size_y += 1
        image_list += clean_image_list
    return draw_images(image_list, redraw=redraw, window_name="detections", block=block,
                       grid_size_x=grid_size_x, grid_size_y=grid_size_y, **kwargs)


def get_label_color_mapping(idx):
    """Returns the PASCAL VOC color triplet for a given label index."""
    # https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    def bitget(byteval, ch):
        return (byteval & (1 << ch)) != 0
    r = g = b = 0
    for j in range(8):
        r = r | (bitget(idx, 0) << 7 - j)
        g = g | (bitget(idx, 1) << 7 - j)
        b = b | (bitget(idx, 2) << 7 - j)
        idx = idx >> 3
    return np.array([r, g, b], dtype=np.uint8)


def apply_color_map(image, colormap, dst=None):
    """Applies a color map to an image of 8-bit color indices; works similarly to cv2.applyColorMap (v3.3.1)."""
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise AssertionError("invalid input image")
    if not isinstance(colormap, np.ndarray) or colormap.shape != (256, 1, 3) or colormap.dtype != np.uint8:
        raise AssertionError("invalid color map")
    out_shape = (image.shape[0], image.shape[1], 3)
    if dst is None:
        dst = np.empty(out_shape, dtype=np.uint8)
    elif not isinstance(dst, np.ndarray) or dst.shape != out_shape or dst.dtype != np.uint8:
        raise AssertionError("invalid output image")
    # using np.take might avoid an extra allocation...
    np.copyto(dst, colormap.squeeze()[image.ravel(), :].reshape(out_shape))
    return dst


def stringify_confmat(confmat, class_list, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """Transforms a confusion matrix array obtained in list or numpy format into a printable string."""
    if not isinstance(confmat, np.ndarray) or not isinstance(class_list, list):
        raise AssertionError("invalid inputs")
    column_width = 9
    empty_cell = " " * column_width
    fst_empty_cell = (column_width - 3) // 2 * " " + "t/p" + (column_width - 3) // 2 * " "
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    res = "\t" + fst_empty_cell + " "
    for label in class_list:
        res += ("%{0}s".format(column_width) % clipstr(label, column_width)) + " "
    res += ("%{0}s".format(column_width) % "total") + "\n"
    for idx_true, label in enumerate(class_list):
        res += ("\t%{0}s".format(column_width) % clipstr(label, column_width)) + " "
        for idx_pred, _ in enumerate(class_list):
            cell = "%{0}d".format(column_width) % int(confmat[idx_true, idx_pred])
            if hide_zeroes:
                cell = cell if int(confmat[idx_true, idx_pred]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if idx_true != idx_pred else empty_cell
            if hide_threshold:
                cell = cell if confmat[idx_true, idx_pred] > hide_threshold else empty_cell
            res += cell + " "
        res += ("%{0}d".format(column_width) % int(confmat[idx_true, :].sum())) + "\n"
    res += ("\t%{0}s".format(column_width) % "total") + " "
    for idx_pred, _ in enumerate(class_list):
        res += ("%{0}d".format(column_width) % int(confmat[:, idx_pred].sum())) + " "
    res += ("%{0}d".format(column_width) % int(confmat.sum())) + "\n"
    return res


def fig2array(fig):
    """Transforms a pyplot figure into a numpy-compatible RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    return buf


def get_glob_paths(input_glob_pattern, can_be_dir=False):
    """Parse a wildcard-compatible file name pattern for valid file paths."""
    glob_file_paths = glob.glob(input_glob_pattern)
    if not glob_file_paths:
        raise AssertionError("invalid input glob pattern '%s' (no matches found)" % input_glob_pattern)
    for file_path in glob_file_paths:
        if not os.path.isfile(file_path) and not (can_be_dir and os.path.isdir(file_path)):
            raise AssertionError("invalid input file at globed path '%s'" % file_path)
    return glob_file_paths


def get_file_paths(input_path, data_root, allow_glob=False, can_be_dir=False):
    """Parse a wildcard-compatible file name pattern at a given root level for valid file paths."""
    if os.path.isabs(input_path):
        if '*' in input_path and allow_glob:
            return get_glob_paths(input_path)
        elif not os.path.isfile(input_path) and not (can_be_dir and os.path.isdir(input_path)):
            raise AssertionError("invalid input file at absolute path '%s'" % input_path)
    else:
        if not os.path.isdir(data_root):
            raise AssertionError("invalid dataset root directory at '%s'" % data_root)
        input_path = os.path.join(data_root, input_path)
        if '*' in input_path and allow_glob:
            return get_glob_paths(input_path)
        elif not os.path.isfile(input_path) and not (can_be_dir and os.path.isdir(input_path)):
            raise AssertionError("invalid input file at path '%s'" % input_path)
    return [input_path]
