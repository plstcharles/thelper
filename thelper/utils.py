"""General utilities module.

This module only contains non-ML specific functions, i/o helpers,
and matplotlib/pyplot drawing calls.
"""
import copy
import errno
import functools
import glob
import hashlib
import importlib
import importlib.util
import inspect
import io
import json
import logging
import os
import pathlib
import pickle
import platform
import re
import sys
import time
from distutils.version import LooseVersion
from typing import TYPE_CHECKING

import numpy as np
import torch
import yaml

import thelper.typedefs  # noqa: F401

if TYPE_CHECKING:
    from typing import Any, AnyStr, Callable, Dict, List, Optional, Tuple, Type, Union  # noqa: F401
    from types import FunctionType  # noqa: F401

logger = logging.getLogger(__name__)
bypass_queries = False
warned_generic_draw = False
fixed_yaml_parsing = False


class Struct:
    """Generic runtime-defined C-like data structure (maps constructor elements to fields)."""

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            "(" + ", ".join([f"{key}={repr(val)}" for key, val in self.__dict__.items()]) + ")"


def test_cuda_device_availability(device_idx):
    # type: (int) -> bool
    """Tests the availability of a single cuda device and returns its status."""
    # noinspection PyBroadException
    try:
        torch.cuda.set_device(device_idx)
        test_val = torch.cuda.FloatTensor([1])
        return test_val.cpu().item() == 1.0
    except Exception:
        return False


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
                devices_available[device_id] = test_cuda_device_availability(device_id)
        attempt_broadcast = True
    return [device_id for device_id, available in enumerate(devices_available) if available]


def setup_plt(config):
    """Parses the provided config for matplotlib flags and sets up its global state accordingly."""
    import matplotlib.pyplot as plt
    config = get_key_def(["plt", "pyplot", "matplotlib"], config, {})
    if "backend" in config:
        import matplotlib
        matplotlib.use(get_key("backend", config))
    plt.interactive(get_key_def("interactive", config, False))


# noinspection PyUnusedLocal
def setup_cv2(config):
    """Parses the provided config for OpenCV flags and sets up its global state accordingly."""
    # https://github.com/pytorch/pytorch/issues/1355
    import cv2 as cv
    cv.setNumThreads(0)
    cv.ocl.setUseOpenCL(False)
    # todo: add more global opencv flags setups here


def setup_gdal(config):
    """Parses the provided config for GDAL flags and sets up its global state accordingly."""
    config = get_key_def("gdal", config, {})
    if "proj_search_path" in config:
        import osr
        osr.SetPROJSearchPath(config["proj_search_path"])


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


def setup_sys(config):
    """Parses the provided config for PYTHON sys paths and sets up its global state accordingly."""
    paths_to_add = []
    if "sys" in config:
        if isinstance(config["sys"], list):
            paths_to_add = config["sys"]
        elif isinstance(config["sys"], str):
            paths_to_add = [config["sys"]]
    for dir_path in paths_to_add:
        if os.path.isdir(dir_path):
            logger.debug(f"will append path to python's syspaths: {dir_path}")
            sys.path.append(dir_path)
        else:
            logger.warning(f"could not append to syspaths, invalid dir: {dir_path}")


def setup_globals(config):
    """Parses the provided config for global flags and sets up the global state accordingly."""
    if "bypass_queries" in config and config["bypass_queries"]:
        global bypass_queries
        bypass_queries = True

    setup_sys(config)
    setup_plt(config)
    setup_cv2(config)
    setup_gdal(config)
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


def check_version(version_check, version_required):
    # type: (AnyStr, AnyStr) -> Tuple[bool, List[Union[int, AnyStr]], List[Union[int, AnyStr]]]
    """Verifies that the checked version is not greater than the required one (ie: not a future version).

    Version format is ``MAJOR[.MINOR[.PATCH[[-]<RELEASE>]]]``.

    Note that for ``RELEASE`` part, comparison depends on alphabetical order if all other previous parts were equal
    (i.e.: ``alpha`` will be lower than ``beta``, which in turn is lower than ``rc`` and so on). The ``-`` is optional
    and will be removed for comparison (i.e.: ``0.5.0-rc`` is exactly the same as ``0.5.0rc`` and the additional ``-``
    will not result in evaluating ``0.5.0a0`` as a greater version because of ``-`` being lower ascii than ``a``).

    Args:
        version_check: the version string that needs to be verified and compared for lower than the required version.
        version_required: the control version against which the check is done.

    Returns:
        Tuple of the validated check, and lists of both parsed version parts as ``[MAJOR, MINOR, PATCH, 'RELEASE']``.
        The returned lists are *guaranteed* to be formed of 4 elements, adding 0 or '' as applicable for missing parts.
    """
    v_check = LooseVersion(version_check)
    v_req = LooseVersion(version_required)
    l_check = [0, 0, 0, '']
    l_req = [0, 0, 0, '']
    for ver_list, ver_parse in [(l_check, v_check), (l_req, v_req)]:
        for v in [0, 1, 2]:
            ver_list[v] = 0 if len(ver_parse.version) < v + 1 else ver_parse.version[v]
        if len(ver_parse.version) >= 4:
            release_idx = 4 if len(ver_parse.version) >= 5 and ver_parse.version[3] == '-' else 3
            ver_list[3] = ''.join(str(v) for v in ver_parse.version[release_idx:])
    # check with re-parsed version after fixing release dash
    v_ok = LooseVersion('.'.join(str(v) for v in l_check)) <= LooseVersion('.'.join(str(v) for v in l_req))
    return v_ok, l_check, l_req


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
    from thelper import __version__ as curr_ver_str
    ckpt_ver_str = ckptdata["version"] if "version" in ckptdata else "0.0.0"
    ok_ver, ckpt_ver, curr_ver = check_version(ckpt_ver_str, curr_ver_str)
    if not ok_ver:
        raise AssertionError("cannot migrate checkpoints from future versions! You need to update your thelper package")
    if "config" not in ckptdata:
        raise AssertionError("checkpoint migration requires config")
    old_config = ckptdata["config"]
    new_config = migrate_config(copy.deepcopy(old_config), ckpt_ver_str)
    if ckpt_ver[:3] == [0, 0, 0]:
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
    that were changed). Perfect reproducibility of tests cannot be guaranteed either if this migration tool is used.

    Args:
        config: session configuration dictionary obtained e.g. by parsing a JSON file. Note that the data contained
            in this dictionary will be modified in-place.
        cfg_ver_str: string representing the version for which the configuration was created (e.g. "0.2.0").

    Returns:
        An updated configuration dictionary that should be compatible with the current version of the framework.
    """
    if not isinstance(config, dict):
        raise AssertionError("unexpected config type")
    from thelper import __version__ as curr_ver_str
    ok_ver, cfg_ver, curr_ver = check_version(cfg_ver_str, curr_ver_str)
    if not ok_ver:
        raise AssertionError("cannot migrate checkpoints from future versions! You need to update your thelper package")
    if cfg_ver[:3] == [0, 0, 0]:
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
            trainer_cfg = config["trainer"]    # type: thelper.typedefs.ConfigDict
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
        cfg_ver = [0, 2, 5]  # set ver for next update step
    if cfg_ver[0] <= 0 and cfg_ver[1] <= 3 and cfg_ver[2] < 6:
        if "trainer" in config:
            if "eval_metrics" in config["trainer"]:
                assert "valid_metrics" not in config["trainer"]
                config["trainer"]["valid_metrics"] = config["trainer"]["eval_metrics"]
                del config["trainer"]["eval_metrics"]
            for mset in ["train_metrics", "valid_metrics", "test_metrics", "metrics"]:
                if mset in config["trainer"]:
                    metrics_config = config["trainer"][mset]
                    for mname, mcfg in metrics_config.items():
                        if "type" in mcfg and mcfg["type"].endswith("ExternalMetric"):
                            assert "params" in mcfg
                            assert "goal" in mcfg["params"]
                            mcfg["params"]["metric_goal"] = mcfg["params"]["goal"]
                            del mcfg["params"]["goal"]
                            if "metric_params" in mcfg["params"]:
                                if isinstance(mcfg["params"]["metric_params"], list):
                                    assert not mcfg["params"]["metric_params"], "cannot fill in kw names"
                                    mcfg["params"]["metric_params"] = {}
                        elif "type" in mcfg and mcfg["type"].endswith("ROCCurve"):
                            assert "params" in mcfg
                            if "log_params" in mcfg["params"]:
                                logger.warning("disabling logging via ROCCurve metric")
                                del mcfg["params"]["log_params"]
        cfg_ver = [0, 3, 6]  # set ver for next update step
    if cfg_ver[0] <= 0 and cfg_ver[1] <= 4 and cfg_ver[2] < 2:
        if "model" in config and isinstance(config, dict):
            model_config = thelper.utils.get_key("model", config)
            model_type = thelper.utils.get_key_def("type", model_config, None)
            if model_type == "thelper.nn.resnet.ResNet":
                model_params = thelper.utils.get_key_def("params", model_config, {})
                coordconv_flag = thelper.utils.get_key_def("coordconv", model_params, False)
                if coordconv_flag:
                    logger.warning("coordconv implementation for resnets changed in v0.4.2; "
                                   "beware if reloading old model weights!")
        cfg_ver = [0, 4, 2]
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
    if getattr(thelper, "LOGGER_INITIALIZED", None) is None:
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
        setattr(thelper, "LOGGER_INITIALIZED", True)


def resolve_import(fullname):
    # type: (str) -> str
    """
    Class name resolver.

    Takes a string corresponding to a module and class fullname to be imported with :func:`thelper.utils.import_class`
    and resolves any back compatibility issues related to renamed or moved classes.

    Args:
        fullname: the fully qualified class name to be resolved.

    Returns:
        The resolved class fullname.
    """
    removed_cases = [
        'thelper.optim.metrics.RawPredictions',  # removed in 0.3.5
    ]
    if fullname in removed_cases:
        raise AssertionError(f"class {repr(fullname)} was deprecated and removed in a previous version")
    refactor_cases = [
        ('thelper.modules', 'thelper.nn'),
        ('thelper.samplers', 'thelper.data.samplers'),
        ('thelper.optim.BinaryAccuracy', 'thelper.optim.metrics.Accuracy'),
        ('thelper.optim.CategoryAccuracy', 'thelper.optim.metrics.Accuracy'),
        ('thelper.optim.ClassifLogger', 'thelper.train.utils.ClassifLogger'),
        ('thelper.optim.ClassifReport', 'thelper.train.utils.ClassifReport'),
        ('thelper.optim.ConfusionMatrix', 'thelper.train.utils.ConfusionMatrix'),
        ('thelper.optim.metrics.BinaryAccuracy', 'thelper.optim.metrics.Accuracy'),
        ('thelper.optim.metrics.CategoryAccuracy', 'thelper.optim.metrics.Accuracy'),
        ('thelper.optim.metrics.ClassifLogger', 'thelper.train.utils.ClassifLogger'),
        ('thelper.optim.metrics.ClassifReport', 'thelper.train.utils.ClassifReport'),
        ('thelper.optim.metrics.ConfusionMatrix', 'thelper.train.utils.ConfusionMatrix'),
        ('thelper.transforms.ImageTransformWrapper', 'thelper.transforms.TransformWrapper'),
        ('thelper.transforms.wrappers.ImageTransformWrapper', 'thelper.transforms.wrappers.TransformWrapper'),
    ]
    old_name = fullname
    for old, new in refactor_cases:
        fullname = fullname.replace(old, new)
    if old_name != fullname:
        logger.warning(f"class fullname '{str(old_name)}' was resolved to '{str(fullname)}'")
    return fullname


def import_class(fullname):
    # type: (str) -> Type
    """General-purpose runtime class importer.

    Supported syntax:
        1. ``module.package.Class`` will import the fully qualified ``Class`` located
           in ``package`` from the *installed* ``module``
        2. ``/some/path/mod.pkg.Cls`` will import ``Cls`` as fully qualified ``mod.pkg.Cls`` from
           ``/some/path`` directory

    Args:
        fullname: the fully qualified class name to be imported.

    Returns:
        The imported class.
    """
    if inspect.isclass(fullname):
        return fullname  # useful shortcut for hacky configs
    assert isinstance(fullname, str), "should specify class by its (fully qualified) name"
    fullname = pathlib.Path(fullname).as_posix()
    if "/" in fullname:
        mod_path, mod_cls_name = fullname.rsplit("/", 1)
        pkg_name = mod_cls_name.rsplit(".", 1)[0]
        pkg_file = os.path.join(mod_path, pkg_name.replace(".", "/")) + ".py"
        mod_cls_name = resolve_import(mod_cls_name)
        spec = importlib.util.spec_from_file_location(mod_cls_name, pkg_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        class_name = mod_cls_name.rsplit('.', 1)[-1]
    else:
        fullname = resolve_import(fullname)
        module_name, class_name = fullname.rsplit('.', 1)
        module = importlib.import_module(module_name)
    return getattr(module, class_name)


def import_function(func,           # type: Union[Callable, AnyStr, List, Dict]
                    params=None     # type: Optional[thelper.typedefs.ConfigDict]
                    ):              # type: (...) -> FunctionType
    """General-purpose runtime function importer, with support for parameter binding.

    Args:
        func: the fully qualified function name to be imported, or a dictionary with
            two members (a ``type`` and optional ``params``), or a list of any of these.
        params: optional params dictionary to bind to the function call via functools.
            If a dictionary of parameters is also provided in ``func``, both will be merged.

    Returns:
        The imported function, with optionally bound parameters.
    """
    assert isinstance(func, (str, dict, list)) or callable(func), "invalid target function type"
    assert params is None or isinstance(params, dict), "invalid target function parameters"
    params = {} if params is None else params
    if isinstance(func, list):
        def multi_caller(funcs, *args, **kwargs):
            return [fn(*args, **kwargs) for fn in funcs]
        return functools.partial(multi_caller, [import_function(fn, params) for fn in func])
    if isinstance(func, dict):
        errmsg = "dynamic function import via dictionary must provide 'type' and 'params' members"
        fn_type = thelper.utils.get_key(["type", "func", "function", "op", "operation", "name"], func, msg=errmsg)
        fn_params = thelper.utils.get_key_def(["params", "param", "parameters", "kwargs"], func, None)
        fn_params = {} if fn_params is None else fn_params
        fn_params = {**params, **fn_params}
        return import_function(fn_type, params=fn_params)
    if isinstance(func, str):
        func = import_class(func)
    assert callable(func), f"unsupported function type ({type(func)})"
    if params:
        return functools.partial(func, **params)
    return func


def check_func_signature(func,      # type: FunctionType
                         params     # type: List[str]
                         ):         # type: (...) -> None
    """Checks whether the signature of a function matches the expected parameter list."""
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
        import lz4
        return lz4.frame.compress(data, **kwargs)
    elif approach == "jpg" or approach == "jpeg":
        import cv2 as cv
        ret, buf = cv.imencode(".jpg", data, **kwargs)
    elif approach == "png":
        import cv2 as cv
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
        import lz4
        return lz4.frame.decompress(data, **kwargs)
    elif approach in ["jpg", "jpeg", "png"]:
        kwargs = copy.deepcopy(kwargs)
        if isinstance(kwargs["flags"], str):  # required arg by opencv
            kwargs["flags"] = eval(kwargs["flags"])
        import cv2 as cv
        return cv.imdecode(data, **kwargs)
    else:
        raise NotImplementedError


def get_class_logger(skip=0, base=False):
    """Shorthand to get logger for current class frame."""
    return logging.getLogger(get_caller_name(skip + 1, base_class=base).rsplit(".", 1)[0])


def get_func_logger(skip=0):
    """Shorthand to get logger for current function frame."""
    return logging.getLogger(get_caller_name(skip + 1))


def get_caller_name(skip=2, base_class=False):
    # source: https://gist.github.com/techtonik/2151727
    """Returns the name of a caller in the format module.class.method.

    Args:
        skip: specifies how many levels of stack to skip while getting the caller.
        base_class: specified if the base class should be returned or the top-most class in case of inheritance
                    If the caller is not a class, this doesn't do anything.

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
        # frame module in case of inherited classes will point to base class
        # but frame local will still refer to top-most class when checking for 'self'
        # (stack: top(mid).__init__ -> mid(base).__init__ -> base.__init__)
        name.append(module.__name__)
    # detect class name
    if "self" in parent_frame.f_locals:
        # I don't know any way to detect call from the object method
        # XXX: there seems to be no way to detect static method call - it will
        #      be just a function call
        cls = parent_frame.f_locals["self"].__class__
        if not base_class and module and inspect.isclass(cls):
            name[0] = cls.__module__
        name.append(cls.__name__)
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
    clean_q = clipstr(question.replace("\n", " ").replace("\t", " ").replace("'", "`"), 45)
    if bypass_queries:
        if bypass is None:
            raise AssertionError("cannot bypass interactive query, no default value provided")
        logger.debug(f"bypassed query '{clean_q}...' with {valid[bypass]}")
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
                sys.stdout.write("\n")
                logger.debug(f"defaulted query '{clean_q}...' with {valid[default]}")
                return valid[default]
            else:
                sys.stdout.write("\n")
                logger.debug(f"defaulted query '{clean_q}...' with {default}")
                return default
        elif choice in valid:
            sys.stdout.write("\n")
            logger.debug(f"answered query '{clean_q}...' with {valid[choice]}")
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
    clean_q = clipstr(question.replace("\n", " ").replace("\t", " ").replace("'", "`"), 45)
    if bypass_queries:
        if bypass is None:
            raise AssertionError("cannot bypass interactive query, no default value provided")
        logger.debug(f"bypassed query '{clean_q}...' with {bypass}")
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
                sys.stdout.write("\n")
                logger.debug(f"defaulted query '{clean_q}...' with {default}")
                return default
            elif allow_empty:
                sys.stdout.write("\n")
                logger.debug(f"answered query '{clean_q}...' with empty string")
                return answer
        elif choices is not None:
            if answer in choices:
                sys.stdout.write("\n")
                logger.debug(f"answered query '{clean_q}...' with choice '{answer}'")
                return answer
        else:
            sys.stdout.write("\n")
            logger.debug(f"answered query '{clean_q}...' with {answer}")
            return answer
        sys.stdout.write("Please respond with a valid string.\n")


def get_config_session_name(config):
    # type: (thelper.typedefs.ConfigDict) -> Optional[str]
    """Returns the 'name' of a session as defined inside a configuration dictionary.

    The current implementation will scan for multiple keywords and return the first value found. If no
    keyword is matched, the function will return None.

    Args:
        config: the configuration dictionary to parse for a name.

    Returns:
        The name that should be given to the session (or 'None' if unknown/unavailable).
    """
    return thelper.utils.get_key_def(["output_dir_name", "output_directory_name",
                                      "session_name", "name"], config, None)


def get_config_output_root(config):
    # type: (thelper.typedefs.ConfigDict) -> Optional[str]
    """Returns the output root directory as defined inside a configuration dictionary.

    The current implementation will scan for multiple keywords and return the first value found. If no
    keyword is matched, the function will return None.

    Args:
        config: the configuration dictionary to parse for a root output directory.

    Returns:
        The path to the output root directory. Can point to a non-existing directory, or be None.
    """
    return thelper.utils.get_key_def(["output_root_dir", "output_root_directory"], config, None)


def get_checkpoint_session_root(ckpt_path):
    # type: (str) -> Optional[str]
    """Returns the session root directory associated with a checkpoint path.

    The given path can point to a checkpoint file or to a directory that contains checkpoints. The
    returned output directory will be the top-level of the session that created the checkpoint, or
    None if it cannot be deduced.

    Args:
        ckpt_path: the path to a checkpoint or to an exisiting directory that contains checkpoints.

    Returns:
        The path to the session root directory. Will always point to an existing directory, or be None.
    """
    assert os.path.exists(ckpt_path), "input path should point to valid filesystem node"
    ckpt_dir_path = os.path.dirname(os.path.abspath(ckpt_path)) \
        if not os.path.isdir(ckpt_path) else os.path.abspath(ckpt_path)
    # find session dir by looking for 'logs' directory
    if os.path.isdir(os.path.join(ckpt_dir_path, "logs")):
        return os.path.abspath(os.path.join(ckpt_dir_path, ".."))
    elif os.path.isdir(os.path.join(ckpt_dir_path, "../logs")):
        return os.path.abspath(os.path.join(ckpt_dir_path, "../.."))
    return None  # cannot be found... giving up


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
    if config is not None:
        config_out_root = thelper.utils.get_config_output_root(config)
        if out_root is not None and config_out_root is not None and out_root != config_out_root:
            answer = query_string("Received conflicting output root directory paths; which one should be used?\n"
                                  f"\t[config] = {config_out_root}\n\t[cli] = {out_root}", choices=["config", "cli"],
                                  default="config", bypass="config")
            if answer == "config":
                out_root = config_out_root
        elif out_root is None and config_out_root is not None:
            out_root = config_out_root
        config_dir_name = thelper.utils.get_config_session_name(config)
        if config_dir_name is not None:
            assert isinstance(config_dir_name, str), "config session/directory name should be given as string"
            assert not os.path.isabs(config_dir_name), "config session/directory name should never be full (abs) path"
            if dir_name is not None and dir_name != config_dir_name:
                func_logger.warning(f"overriding output session directory name '{dir_name}' to '{config_dir_name}'")
            dir_name = config_dir_name
    if out_root is None:
        time.sleep(0.25)  # to make sure all debug/info prints are done, and we see the question
        out_root = query_string("Please provide the path to where session directories should be created/saved:")
    func_logger.info(f"output root directory = {os.path.abspath(out_root)}")
    os.makedirs(out_root, exist_ok=True)
    save_dir = os.path.join(out_root, dir_name) if dir_name is not None else out_root
    if not resume:
        overwrite = str2bool(config["overwrite"]) if config is not None and "overwrite" in config else False
        time.sleep(0.25)  # to make sure all debug/info prints are done, and we see the question
        while os.path.exists(save_dir) and not overwrite:
            abs_save_dir = os.path.abspath(save_dir).replace("\\", "/")
            overwrite = query_yes_no("Training session at '%s' already exists; overwrite?" % abs_save_dir, bypass="y")
            if not overwrite:
                save_dir = query_string("Please provide a new save directory path:")
    func_logger.info(f"output session directory = {os.path.abspath(save_dir)}")
    os.makedirs(save_dir, exist_ok=True)
    logs_dir = os.path.join(save_dir, "logs")
    func_logger.info(f"output logs directory = {os.path.abspath(logs_dir)}")
    os.makedirs(logs_dir, exist_ok=True)
    if config is not None:
        common_backup_path = os.path.join(save_dir, "config.latest" + backup_ext)
        if resume and os.path.exists(common_backup_path):
            config_backup = thelper.utils.load_config(common_backup_path, add_name_if_missing=False)
            if config_backup != config:  # TODO make config dict comparison smarter...?
                query_msg = f"Config backup in '{common_backup_path}' differs from config loaded through checkpoint; overwrite?"
                answer = query_yes_no(query_msg, bypass="y")
                if answer:
                    func_logger.warning("config mismatch with previous run; "
                                        "will overwrite latest backup in save directory")
                else:
                    func_logger.error("config mismatch with previous run; user aborted")
                    sys.exit(1)
        save_config(config, common_backup_path)
        tagged_backup_path = os.path.join(logs_dir, "config." + thelper.utils.get_log_stamp() + backup_ext)
        save_config(config, tagged_backup_path)
    return save_dir


def load_config(path, as_json=False, add_name_if_missing=True, **kwargs):
    # type: (str, bool, bool, **Any) -> thelper.typedefs.ConfigDict
    """Loads the configuration dictionary from the provided path.

    The type of file that is loaded is based on the extension in the path.

    If the loaded configuration dictionary does not contain a 'name' field, the name of
    the file itself will be inserted as a value.

    Args:
        path: the path specifying which configuration to be loaded.
            only supported types are loaded unless `as_json` is `True`.
        as_json: specifies if an alternate extension should be considered as JSON format.
        add_name_if_missing: specifies whether the file name should be added to the config
            dictionary if it is missing a 'name' field.
        kwargs: forwarded to the extension-specific importer.
    """
    global fixed_yaml_parsing
    if not fixed_yaml_parsing:
        # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        fixed_yaml_parsing = True
    ext = os.path.splitext(path)[-1]
    if ext in [".json", ".yml", ".yaml"] or as_json:
        with open(path) as fd:
            assert not kwargs, "yaml safe load takes no extra args"
            config = yaml.safe_load(fd)  # also supports json
    elif ext == ".pkl":
        with open(path, "rb") as fd:
            config = pickle.load(fd, **kwargs)
    else:
        raise AssertionError(f"unknown input file type: {ext}")
    if add_name_if_missing and thelper.utils.get_config_session_name(config) is None:
        config["name"] = os.path.splitext(os.path.basename(path))[0]
    return config


def save_config(config, path, force_convert=True, as_json=False, **kwargs):
    # type: (thelper.typedefs.ConfigDict, str, bool, bool, **Any) -> None
    """Saves the given session/object configuration dictionary to the provided path.

    The type of file that is created is based on the extension specified in the path. If the file
    cannot hold some of the objects within the configuration, they will be converted to strings before
    serialization, unless `force_convert` is set to `False` (in which case the function will raise
    an exception).

    Args:
        config: the session/object configuration dictionary to save.
        path: the path specifying where to create the output file. The extension used will determine
            what type of backup to create (e.g. Pickle = .pkl, JSON = .json, YAML = .yml/.yaml).
            if `as_json` is `True`, then any specified extension will be preserved bump dumped as JSON.
        force_convert: specifies whether non-serializable types should be converted if necessary.
        as_json: specifies if an alternate extension should be considered as JSON format.
        kwargs: forwarded to the extension-specific exporter.
    """
    ext = os.path.splitext(path)[-1]
    if ext in [".json", ".yml", ".yaml"] or as_json:
        with open(path, "w") as fd:
            kwargs.setdefault("indent", 4)
            if ext == ".json" or as_json:
                serializer = (lambda x: str(x)) if force_convert else None
                kwargs.setdefault("default", serializer)
                kwargs.setdefault("sort_keys", False)
                json.dump(config, fd, **kwargs)
            else:
                yaml.dump(config, fd, **kwargs)
    elif ext == ".pkl":
        with open(path, "wb") as fd:
            pickle.dump(config, fd, **kwargs)
    else:
        raise AssertionError(f"unknown output file type: {ext}")


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


def get_params_hash(*args, **kwargs):
    """Returns a sha1 hash for the given list of parameters (useful for caching)."""
    # by default, will use the repr of all params but remove the 'at 0x00000000' addresses
    clean_str = re.sub(r" at 0x[a-fA-F\d]+", "", str(args) + str(kwargs))
    return hashlib.sha1(clean_str.encode()).hexdigest()


def check_installed(package_name):
    """Attempts to import a specified package by name, returning a boolean indicating success."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def set_matplotlib_agg():
    """Sets the matplotlib backend to Agg."""
    import matplotlib
    matplotlib.use('Agg')
