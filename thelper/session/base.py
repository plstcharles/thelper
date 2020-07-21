import functools
import json
import logging
import os
import pickle
import platform
import random
import time
from copy import deepcopy
from typing import Any, AnyStr, Optional

import cv2 as cv
import numpy as np
import torch

import thelper.data
import thelper.nn
import thelper.optim
import thelper.tasks
import thelper.typedefs
import thelper.utils
import thelper.viz

logger = logging.getLogger(__name__)


class SessionRunner:
    """Abstract session runner interface that defines basic session i/o and setup operations.

    This class offers the most basic methods that can be employed by more specialized training or inference sessions.
    By itself, it doesn't actually run anything.

    Attributes:
        checkpoint_dir: session checkpoint output directory (located within the 'session directory').
        config: session configuration dictionary holding all original settings, including trainer configuration.
        devices: list of (cuda) device IDs to upload the model/tensors to; can be empty if only the CPU is available.
        epochs: number of epochs to train the model for.
        logger: used to output debug/warning/error messages to session log.
        model: reference to the model being trained or used for evaluation/prediction.
        monitor: name of the training/validation metric that should be monitored for model improvement.
        name: name of the session, used for printing and creating log folders.
        optimization_config: dictionary of optim-related parameters, parsed at training time.
        output_paths: map of session output paths where training/evaluation results should be saved.
        save_freq: frequency of checkpoint saves while training (i.e. save every X epochs).
        save_raw: specifies whether to save raw types or thelper objects in checkpoints.
        skip_eval_iter: number of evaluation iterations to skip (useful for resuming a session).
        skip_tbx_histograms: flag used to skip the generation of graph histograms in tbx (useful for large models).
        task: reference to the object used to specialize the model and that holds task metainformation.
        tbx_histogram_freq: frequency of tbx histogram saves while training (i.e. save every X epochs).
        use_tbx: defines whether to use tensorboardX writers for logging or not.
        writers: map of tbx writers used to save training/evaluation events.

    .. seealso::
        | :class:`thelper.train.base.Trainer`
    """
    def __init__(self,
                 session_name,    # type: AnyStr
                 session_dir,     # type: AnyStr
                 model,           # type: thelper.typedefs.ModelType
                 task,            # type: thelper.tasks.Task
                 loaders,         # type: thelper.typedefs.MultiLoaderType
                 config,          # type: thelper.typedefs.ConfigDict
                 ckptdata=None    # type: Optional[thelper.typedefs.CheckpointContentType]
                 ):
        """Receives the trainer configuration dictionary, parses it, and sets up the session."""
        assert isinstance(model, (thelper.nn.Module, torch.nn.Module)), "unknown model object type"
        assert isinstance(task, thelper.tasks.Task), "unknown task object type"
        assert isinstance(loaders, (list, tuple, np.ndarray)) and len(loaders) == 3, "invalid loaders array"
        assert isinstance(config, dict), "invalid config type"
        self.task = task
        self.model = model
        self.config = config

        # parse basic training config args
        # use 'trainer' key first for backward compatibility and to prioritize it - most configs will define it as so
        trainer_config = thelper.utils.get_key(["trainer", "runner", "tester"], config)
        os.makedirs(session_dir, exist_ok=True)
        logs_dir = os.path.join(session_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        thelper.utils.init_logger()  # make sure all logging is initialized before attaching this part
        thelper.utils.save_env_list(os.path.join(logs_dir, "packages.log"))
        train_logger_path = os.path.join(logs_dir, "trainer.log")
        train_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        train_logger_fh = logging.FileHandler(train_logger_path)
        train_logger_fh.setLevel(logging.NOTSET)
        train_logger_fh.setFormatter(train_logger_format)
        self.logger = thelper.utils.get_class_logger()
        self.logger.addHandler(train_logger_fh)
        self.logger.info(f"created training log for session '{session_name}'")
        self.logger.debug(f"session directory = {os.path.abspath(session_dir)}")
        self.logger.debug(f"logs directory = {os.path.abspath(logs_dir)}")
        logstamp = thelper.utils.get_log_stamp()
        repover = thelper.__version__ + ":" + thelper.utils.get_git_stamp()
        self.logger.debug(f"logstamp = {logstamp}")
        self.logger.debug(f"version = {repover}")
        self.name = session_name
        self.epochs = 1
        self.save_freq = int(thelper.utils.get_key_def("save_freq", trainer_config, 1))
        assert self.save_freq >= 1, "checkpoint save frequency should be strictly positive integer"
        self.save_raw = thelper.utils.str2bool(thelper.utils.get_key_def("save_raw", trainer_config, True))
        self.checkpoint_dir = os.path.join(session_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        output_root_dir = thelper.utils.get_key_def("output_dir", trainer_config)
        if not output_root_dir:
            # append session name for cleaner TBX folder merging
            output_root_dir = os.path.join(session_dir, "output", self.name)
        assert isinstance(output_root_dir, str) and len(output_root_dir), "invalid output directory path"
        self.logger.debug(f"output directory = {os.path.abspath(output_root_dir)}")
        os.makedirs(output_root_dir, exist_ok=True)
        unique_output_dir = thelper.utils.get_key_def("unique_output_dir", trainer_config, True)
        assert isinstance(unique_output_dir, bool), "invalid unique_output_dir flag (should be bool)"
        self.logger.debug(f"output subdirectories {'will' if unique_output_dir else 'will not'} have unique names")
        devices_str = thelper.utils.get_key_def(["device", "devices", "train_device"], trainer_config, None)
        self.devices = self._load_devices(devices_str)
        self.skip_eval_iter = thelper.utils.get_key_def("skip_eval_iter", trainer_config, 0)

        # parse and prepare tbx stuff
        self.use_tbx = thelper.utils.str2bool(thelper.utils.get_key_def(["use_tbx", "tbx", "use_tb", "tb", "tensorboard"],
                                                                        trainer_config, False))
        if self.use_tbx:
            try:
                import tensorboardX
                self.tbx = tensorboardX
                logger.debug("using external tensorboard")
            except ImportError:
                import torch.utils.tensorboard as tensorboard
                self.tbx = tensorboard
                logger.debug("using PyTorch's tensorboard")
            self.logger.debug(
                f"tensorboard init : tensorboard --logdir {os.path.abspath(output_root_dir)} --port <your_port>")

        self.skip_tbx_histograms = thelper.utils.str2bool(
            thelper.utils.get_key_def("skip_tbx_histograms", trainer_config, False))
        self.tbx_histogram_freq = int(thelper.utils.get_key_def("tbx_histogram_freq", trainer_config, 5))
        assert self.tbx_histogram_freq >= 1, "histogram output frequency should be strictly positive integer"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.writers, self.output_paths = {}, {}
        for cname, loader in zip(["train", "valid", "test"], loaders):
            if loader:
                folder_name = f"{cname}-{str(platform.node())}-{timestr}" if unique_output_dir else cname
                self.output_paths[cname] = os.path.join(output_root_dir, folder_name)
                self.logger.debug(f"output {cname} directory = {os.path.abspath(self.output_paths[cname])}")
                os.makedirs(self.output_paths[cname], exist_ok=True)
            else:
                self.output_paths[cname] = None
            self.writers[cname] = None  # will be instantiated only when needed based on above path

        # split loaders
        train_loader, valid_loader, test_loader = loaders
        assert (train_loader or valid_loader or test_loader), "must provide at least one loader with available data"
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader
        if train_loader:
            assert "epochs" in trainer_config and int(trainer_config["epochs"]) > 0, "bad trainer config epoch count"
            self.epochs = int(trainer_config["epochs"])
            # loading optimization stuff later since model needs to be on correct device
            self.optimization_config = thelper.utils.get_key_def("optimization", trainer_config, {})
        else:
            self.logger.info("no training data provided, will run a single epoch on valid/test data")

        # parse metrics
        assert "metrics" not in trainer_config or "base_metrics" not in trainer_config, \
            "trainer config should have only one of 'metrics' and 'base_metrics'"
        metrics = {}
        if "metrics" in trainer_config:
            self.logger.debug("loading metrics defined in trainer config")
            metrics = thelper.train.create_consumers(trainer_config["metrics"])
        elif "base_metrics" in trainer_config:
            self.logger.debug("loading base metrics defined in trainer config")
            metrics = thelper.train.create_consumers(trainer_config["base_metrics"])
        self.train_metrics, self.valid_metrics, self.test_metrics = \
            deepcopy(metrics), deepcopy(metrics), deepcopy(metrics)
        for skey, sval in zip(["train_metrics", "valid_metrics", "test_metrics"],
                              [self.train_metrics, self.valid_metrics, self.test_metrics]):
            if skey in trainer_config:
                new_metrics = thelper.train.create_consumers(trainer_config[skey])
                for mkey, mval in new_metrics.items():
                    assert mkey not in sval, f"metric name '{mkey}' duplicated in set '{skey}'"
                    sval[mkey] = mval
                for mkey, mval in sval.items():
                    self.logger.info(f"parsed metric '{mkey}': {str(mval)}")

        # check for monitored metric
        self.monitor, self.monitor_best, self.monitor_best_epoch = None, None, -1
        if "monitor" in trainer_config and trainer_config["monitor"]:
            self.monitor = trainer_config["monitor"]
            assert any([self.monitor in mset for mset in [self.train_metrics, self.valid_metrics]]), \
                f"metric with name '{self.monitor}' could not be found in training/validation metrics"
            metric = self.valid_metrics[self.monitor] if self.monitor in self.valid_metrics \
                else self.train_metrics[self.monitor]  # makes no sense to search for it in test metrics...
            assert isinstance(metric, thelper.optim.metrics.Metric), \
                "monitoring target should be an actual 'metric' class that returns a scalar!"
            assert metric.goal in [thelper.optim.Metric.minimize, thelper.optim.Metric.maximize], \
                "monitored metric does not return proper optimization goal"
            self.monitor_goal = metric.goal
            self.monitor_best = thelper.optim.Metric.minimize if metric.goal == thelper.optim.Metric.maximize \
                else thelper.optim.Metric.maximize
            self.logger.debug(f"will monitor metric '{self.monitor}' for best state checkpointing/early stopping")

        # parse checkpoint data from previous run (if available)
        ckptdata = {} if ckptdata is None else ckptdata
        self.monitor_best = thelper.utils.get_key_def("monitor_best", ckptdata, self.monitor_best)
        self.monitor_best_epoch = thelper.utils.get_key_def("monitor_best_epoch", ckptdata, -1)
        self.optimizer_state = thelper.utils.get_key_def("optimizer", ckptdata, None)
        self.scheduler_state = thelper.utils.get_key_def("scheduler", ckptdata, None)
        self.current_iter = thelper.utils.get_key_def("iter", ckptdata, 0)
        self.current_epoch = thelper.utils.get_key_def("epoch", ckptdata, 0)
        self.outputs = thelper.utils.get_key_def("outputs", ckptdata, {})

        # parse callbacks (see ``thelper.typedefs.IterCallbackType`` and ``thelper.typedefs.IterCallbackParams``)
        for cname, mset in zip(["train", "valid", "test"], [self.train_metrics, self.valid_metrics, self.test_metrics]):
            # parse user (custom) callback
            user_callback_keys = [f"{cname}_iter_callback", f"{cname}_callback", "callback"]
            user_callback = \
                thelper.utils.get_key_def(user_callback_keys,
                                          trainer_config)  # type: Optional[thelper.typedefs.IterCallbackType]
            if user_callback is not None:
                assert f"{cname}_user_callback" not in mset, f"metrics set already had a '{cname}_user_callback' in it"
                mset[f"{cname}_user_callback"] = thelper.train.utils.PredictionCallback(user_callback)
            # parse display callback
            display_callback_keys = [f"display_{cname}_preds", f"display_{cname}_predictions", f"display_{cname}",
                                     "display_preds", "display_predictions", "display"]
            display_callback = thelper.utils.get_key_def(display_callback_keys, trainer_config)
            if display_callback:
                assert f"{cname}_display_callback" not in mset, \
                    f"metrics set already had a '{cname}_display_callback' in it"
                if isinstance(display_callback, bool):  # if simply toggled on, use default draw function wrapper
                    display_callback = {"type": "thelper.train.utils._draw_wrapper", "params": {"save": False}}
                mset[f"{cname}_display_callback"] = thelper.train.utils.PredictionCallback(display_callback)
            # parse logging callback
            logging_callback_keys = \
                [f"{cname}_logger", f"{cname}_log", f"logger_{cname}", f"log_{cname}", "log", "logger"]
            logging_callback = \
                thelper.utils.get_key_def(logging_callback_keys, trainer_config, self._iter_logger_callback)
            if logging_callback:
                assert f"{cname}_logger_callback" not in mset, \
                    f"metrics set already had a '{cname}_logger_callback' in it"
                logging_kwargs = {"set_name": cname, "writers": self.writers}  # pass writers by ref, fill later
                mset[f"{cname}_logger_callback"] = \
                    thelper.train.utils.PredictionCallback(logging_callback, logging_kwargs)
            else:
                logger.warning("logging is disabled by user, internal iteration count might never be updated")

        # parse visualization config (if any)
        self.viz = thelper.utils.get_key_def(["viz", "visualization", "visualizations"], trainer_config, {})
        assert isinstance(self.viz, dict), "invalid visulaization dictionary config"
        for viz_key, viz_config in self.viz.items():
            assert isinstance(viz_key, str) and viz_key in thelper.viz.supported_types, \
                f"invalid visualization type '{viz_key}' (not in available modules)"
            assert isinstance(viz_config, dict), f"invalid visualization configuration dictionary for type '{viz_key}'"

    def _init_writer(self, writer, path):
        if self.use_tbx and not writer:
            writer = self.tbx.SummaryWriter(path, comment=self.name)
            writer.add_text("config", json.dumps(self.config, indent=4, sort_keys=False, default=lambda x: str(x)))
            thelper.utils.save_config(self.config, os.path.join(path, "config.json"))
        return writer

    @staticmethod
    def _set_rng_state(seeds, epoch):
        if "torch" in seeds:
            torch.manual_seed(seeds["torch"] + epoch)
            torch.cuda.manual_seed_all(seeds["torch"] + epoch)
        if "numpy" in seeds:
            np.random.seed(seeds["numpy"] + epoch)
        if "random" in seeds:
            random.seed(seeds["random"] + epoch)

    @staticmethod
    def _upload_model(model, dev):
        """Uploads a model to a specific device, wrapping it in ``torch.nn.DataParallel`` if needed."""
        if isinstance(dev, list):
            if len(dev) == 0:
                return model.cpu()
            elif len(dev) == 1:
                return model.cuda(dev[0])
            else:
                return torch.nn.DataParallel(model, device_ids=dev).cuda(dev[0])
        else:
            return model.to(dev)

    @staticmethod
    def _move_tensor(tensor, dev, detach=False):
        """Uploads a tensor to a specific device."""
        if isinstance(tensor, (list, tuple)):
            return [SessionRunner._move_tensor(t, dev) for t in tensor]
        if isinstance(tensor, dict):
            return {k: SessionRunner._move_tensor(t, dev) for k, t in tensor.items()}
        if not isinstance(tensor, torch.Tensor):
            return tensor  # ignored (cannot upload)
        if isinstance(dev, list):
            if len(dev) == 0:
                out = tensor.cpu()
            else:
                # no reason to have multiple devices if not cuda-enabled GPUs
                out = tensor.cuda(dev[0])
        else:
            out = tensor.to(dev)
        return out.detach() if detach else out

    def _load_optimization(self, model, dev):
        """Instantiates and returns all optimization objects required for training the model."""
        config = self.optimization_config  # for abbrev only
        assert isinstance(config, dict), "optimization config should be provided as a dictionary"
        assert self.train_loader is not None and self.train_loader, "optimization only useful with training data"
        loss = None  # can be omitted if using custom trainer
        if "loss" in config:
            uploader = functools.partial(self._move_tensor, dev=dev)
            loss = thelper.optim.create_loss_fn(config["loss"], model, self.train_loader, uploader)
        optimizer = None  # can be omitted if using custom trainer
        if "optimizer" in config:
            optimizer = thelper.optim.create_optimizer(config["optimizer"], model)
        scheduler, scheduler_step_metric = None, None
        if "scheduler" in config and config["scheduler"]:  # can always be omitted
            scheduler, scheduler_step_metric = thelper.optim.create_scheduler(config["scheduler"], optimizer)
        return loss, optimizer, scheduler, scheduler_step_metric

    def _load_devices(self, devices_str=None):
        """Validates and returns the list of CUDA devices available on the system."""
        self.logger.debug("loading available devices")
        if devices_str is not None:
            devices = []
            available_cuda_devices = None
            assert isinstance(devices_str, (str, list)), "unexpected device string type"
            if isinstance(devices_str, str):
                assert devices_str, "cannot specify empty device name, use 'None' to auto-detect"
                devices_str = devices_str.split(",")
            elif isinstance(devices_str, list):
                assert devices_str, "cannot specify empty device list, use 'None' to auto-detect"
                assert all([isinstance(dev_str, str) for dev_str in devices_str]), "unexpected type in dev list"
            for dev_idx, dev_str in enumerate(devices_str):
                assert "cuda" in dev_str or dev_str == "cpu", \
                    f"unknown device type '{dev_str}' (expecting 'cpu' or 'cuda:X')"
                if dev_str == "cpu":
                    assert len(devices_str) == 1, "cannot combine cpu with other devices"
                    return []
                if dev_str == "cuda" or dev_str == "cuda:all":
                    assert len(devices_str) == 1, "must specify device index (e.g. 'cuda:0') if combining devices"
                    if available_cuda_devices is None:
                        available_cuda_devices = thelper.utils.get_available_cuda_devices()
                    assert available_cuda_devices, "could not find any available cuda devices"
                    return available_cuda_devices
                assert "cuda:" in dev_str, "expecting cuda device format to be 'cuda:X' (where X is device index)"
                cuda_dev_idx = int(dev_str.rsplit(":", 1)[-1])
                assert thelper.utils.test_cuda_device_availability(cuda_dev_idx), f"cuda device '{dev_str}' unavailable"
                devices.append(cuda_dev_idx)
            return devices
        else:
            return thelper.utils.get_available_cuda_devices()

    def _to_tensor(self, sample):
        """Fetches and returns tensors of input and groundtruth data from a batched sample dictionary.

        The specifics of how to unpack a sample dictionary into usable parts is tied to the trainer, so
        it cannot be defined in a perfectly generic way here. The implementation below is given as a
        baseline to support some visualization techniques (see :mod:`thelper.viz` for more info). Derived
        trainers (both custom and framework-provided) are likely to override this function to properly
        unpack groundtruth data.

        Args:
            sample: the (batched) sample to unpack into tensors, obtained directly from a data loader.

        Returns:
            A tuple of input data and groundtruth data tensors. In this implementation, the groundtruth
            data tensor is always ``None``.
        """
        assert isinstance(sample, dict), "trainer expects samples to come in dicts for key-based usage"
        assert self.task.input_key in sample, f"could not find input key '{self.task.input_key}' in sample dict"
        return torch.FloatTensor(sample[self.task.input_key]), None

    def _iter_logger_callback(self,         # see `thelper.typedefs.IterCallbackParams` for more info
                              task,         # type: thelper.tasks.utils.Task
                              input,        # type: thelper.typedefs.InputType
                              pred,         # type: thelper.typedefs.AnyPredictionType
                              target,       # type: thelper.typedefs.AnyTargetType
                              sample,       # type: thelper.typedefs.SampleType
                              loss,         # type: Optional[float]
                              iter_idx,     # type: int
                              max_iters,    # type: int
                              epoch_idx,    # type: int
                              max_epochs,   # type: int
                              output_path,  # type: AnyStr
                              # note: kwargs must contain two args here: 'set_name' and 'writers'
                              **kwargs,     # type: Any
                              ):            # type: (...) -> None
        """Receives callback data for logging loss/monitored metric values each training/eval iteration."""
        # NOTE: THIS FUNCTION IS RESPONSIBLE FOR INCREASING THE INTERNAL ITERATION COUNTER.
        set_name = thelper.utils.get_key("set_name", kwargs, "missing set name in iter logger args")
        assert set_name in ["train", "valid", "test"], "unrecognized iter logger set name"
        metrics = self.train_metrics if set_name == "train" else self.valid_metrics if set_name == "valid" \
            else self.test_metrics
        monitor_val = None
        monitor_str = ""
        if self.monitor is not None and self.monitor in metrics:
            assert isinstance(metrics[self.monitor], thelper.optim.metrics.Metric), "unexpected metric type"
            if metrics[self.monitor].live_eval:
                monitor_val = metrics[self.monitor].eval()
                monitor_str = f"   {self.monitor}: {monitor_val:.2f}"
        loss_str = ""
        if loss is not None:
            loss_str = f"   loss: {loss:.6f}"
        assert self.current_epoch == epoch_idx, "something's messed up"
        self.logger.info(
            f"{set_name} epoch#{epoch_idx}  (iter#{self.current_iter})" +
            f"   batch: {iter_idx + 1}/{max_iters} ({((iter_idx + 1) / max_iters) * 100.0:.0f}%)" +
            f"{loss_str}{monitor_str}"
        )
        writers = thelper.utils.get_key("writers", kwargs, msg="missing writers dict in iter logger args")
        if (set_name == "train" or iter_idx == max_iters - 1) and writers[set_name]:
            if loss is not None:
                writers[set_name].add_scalar("iter/loss", loss, self.current_iter)
            for metric_name, metric in metrics.items():
                if isinstance(metric, thelper.optim.metrics.Metric):
                    if metric_name == self.monitor and monitor_val is not None:
                        writers[set_name].add_scalar(f"iter/{self.monitor}", monitor_val, self.current_iter)
                    elif metric.live_eval:
                        # if live eval is not true, metric might be too heavy to compute at each iteration
                        writers[set_name].add_scalar(f"iter/{metric_name}", metric.eval(), self.current_iter)
        if set_name == "train":
            self.current_iter += 1

    def _write_data(self, data, writer_prefix, file_suffix, writer, output_path, idx=None):
        """Writes a generic chunk of data passed as a dictionary to the specified output path."""
        os.makedirs(output_path, exist_ok=True)
        assert isinstance(data, dict) and all([isinstance(key, str) for key in data]), \
            "unexpected data chunk formatting (should be dict with str-based keys)"
        reserved_keys = ["/image", "/extension", "/json", "/text", "/pickle"]
        for key, val in data.items():
            if thelper.utils.is_scalar(val) and not any([key.endswith(s) for s in reserved_keys]):
                if writer is not None:
                    if isinstance(val, str):
                        writer.add_text(f"{writer_prefix}{key}", val, idx)
                    else:
                        writer.add_scalar(f"{writer_prefix}{key}", val, idx)
            if key.endswith("/image") and val is not None:  # some metrics got the callable but return None
                assert isinstance(val, np.ndarray) and len(val.shape) == 3 and val.shape[2] == 3, \
                    "unexpected image format (should be numpy array with RGB channels)"
                image_ext = thelper.utils.get_key_def(key + "/extension", data, "png")
                image_path = os.path.join(output_path, f"{''.join(key.rsplit('/image', 1))}{file_suffix}.{image_ext}")
                self.logger.debug(f"writing {key} to {os.path.abspath(image_path)}")
                cv.imwrite(image_path, val[..., ::-1])  # flip to BGR for opencv compat
                if writer is not None:
                    writer.add_image(f"{writer_prefix}{key}", val, idx, dataformats="HWC")
            if key.endswith("/json"):
                json_ext = thelper.utils.get_key_def(key + "/extension", data, "json")
                json_path = os.path.join(output_path, f"{''.join(key.rsplit('/json', 1))}{file_suffix}.{json_ext}")
                self.logger.debug(f"writing {key} to {os.path.abspath(json_path)}")
                with open(json_path, "w") as fd:
                    json.dump(val, fd)
            if key.endswith("/text"):
                txt_ext = thelper.utils.get_key_def(key + "/extension", data, "txt")
                txt_path = os.path.join(output_path, f"{''.join(key.rsplit('/text', 1))}{file_suffix}.{txt_ext}")
                self.logger.debug(f"writing {key} to {os.path.abspath(txt_path)}")
                with open(txt_path, "w") as fd:
                    fd.write(val)
            if key.endswith("/pickle"):
                pkl_ext = thelper.utils.get_key_def(key + "/extension", data, "pkl")
                pkl_path = os.path.join(output_path, f"{''.join(key.rsplit('/pickle', 1))}{file_suffix}.{pkl_ext}")
                self.logger.debug(f"writing {key} to {os.path.abspath(pkl_path)}")
                with open(pkl_path, "wb") as fd:
                    pickle.dump(val, fd)

    def _write_metrics_data(self, epoch, metrics, tbx_writer, output_path, loss=None, optimizer=None, use_suffix=True):
        """Writes the cumulative evaluation result of all metrics using a specific writer."""
        os.makedirs(output_path, exist_ok=True)
        if tbx_writer is not None:
            if loss is not None:
                tbx_writer.add_scalar("epoch/loss", loss, epoch)
            if optimizer is not None:
                tbx_writer.add_scalar("epoch/lr", thelper.optim.get_lr(optimizer), epoch)
        writer_prefix = "epoch/"
        file_suffix = f"-{epoch:04d}" if use_suffix else ""
        for metric_name, metric in metrics.items():
            output = {}
            if hasattr(metric, "render") and callable(metric.render):
                output[f"{metric_name}/image"] = metric.render()
                output[f"{metric_name}/image/extension"] = "png"
            if hasattr(metric, "report") and callable(metric.report):
                output[f"{metric_name}/text"] = metric.report()
                output[f"{metric_name}/text/extension"] = getattr(metric, "ext", "txt")
            if hasattr(metric, "eval") and callable(metric.eval):
                eval_res = metric.eval()
                if f"{metric_name}/text" not in output and eval_res is not None:
                    if isinstance(eval_res, float):
                        output[f"{metric_name}/text"] = f"{eval_res:.4f}"
                    else:
                        output[f"{metric_name}/text"] = str(eval_res)
                    output[f"{metric_name}/text/extension"] = getattr(metric, "ext", "txt")
                output[metric_name] = eval_res
            self._write_data(output, writer_prefix, file_suffix, tbx_writer, output_path, epoch)

    def _save(self, epoch, iter, optimizer, scheduler, save_best=False):
        """Saves a session checkpoint containing all the information required to resume training."""
        # logically, this should only be called during training (i.e. with a valid optimizer)
        log_stamp = thelper.utils.get_log_stamp()
        # the saved state below should be kept compatible with the one in thelper.cli.export_model
        curr_state = {
            "name": self.name,
            "epoch": epoch,
            "iter": iter,
            "source": log_stamp,
            "git_sha1": thelper.utils.get_git_stamp(),
            "version": thelper.__version__,
            "task": str(self.task) if self.save_raw else self.task,
            "outputs": self.outputs,
            # we save model type/params here in case those are not in the current config
            "model": self.model.state_dict() if self.save_raw else self.model,
            "model_type": self.model.get_name(),
            "model_params": self.model.config if self.model.config else {},
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if (scheduler is not None and
                                                    hasattr(scheduler, "state_dict")) else None,
            "monitor_best": self.monitor_best,
            "monitor_best_epoch": self.monitor_best_epoch,
            "config": self.config  # note: this is the global app config
        }
        filename = f"ckpt.{epoch:04d}.{log_stamp}.pth"
        filename = os.path.join(self.checkpoint_dir, filename)
        self.logger.debug(f"writing checkpoint to {os.path.abspath(filename)}")
        torch.save(curr_state, filename)
        if save_best:
            filename_best = os.path.join(self.checkpoint_dir, "ckpt.best.pth")
            self.logger.debug(f"writing checkpoint to {os.path.abspath(filename_best)}")
            torch.save(curr_state, filename_best)
