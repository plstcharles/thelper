"""Model training/evaluation base interface module.

This module contains the interface required to train and/or evaluate a model based on different tasks. The trainers
based on this interface are instantiated in launched sessions based on configuration dictionaries.
"""
import functools
import json
import logging
import math
import os
import platform
import random
import time
from abc import abstractmethod
from copy import deepcopy
from typing import AnyStr  # noqa: F401

import cv2 as cv
import numpy as np
import torch
import torch.optim

import thelper.typedefs as typ  # noqa: F401
import thelper.utils

logger = logging.getLogger(__name__)


class Trainer:
    """Abstract trainer interface that defines basic session i/o and setup operations.

    This interface defines the general behavior of a training session which includes configuration parsing, tensorboard
    setup, metrics and goal setup, and loss/optimizer setup. It also provides utilities for uploading models and tensors
    on specific devices, and for saving the state of a session. This interface should be specialized for every task by
    implementing the ``train_epoch`` and ``eval_epoch`` functions in a derived class. See
    :class:`thelper.train.classif.ImageClassifTrainer` for an example.

    The main parameters that will be parsed by this interface from a configuration dictionary are the following:

    - ``epochs`` (mandatory if training): number of epochs to train for; one epoch is one iteration over all mini-batches.
    - ``optimization`` (mandatory if training): sub-dictionary containing types and extra parameters required for
      instantiating the loss, optimizer, and scheduler objects. See the code of each related loading function for more
      information on special parameters.
    - ``save_freq`` (optional, default=1): checkpoint save frequency (will save every epoch multiple of given number).
    - ``save_raw`` (optional, default=True): specifies whether to save raw types or thelper objects in checkpoints.
    - ``use_tbx`` (optional, default=False): defines whether to use tensorboardX writers for logging or not.
    - ``device`` (optional): specifies which device to train/evaluate the model on (default=all available).
    - ``metrics``: list of metrics to instantiate and update during training/evaluation; see related loading function for
      more information.
    - ``monitor``: specifies the name of the metric that should be monitored on the validation set for model improvement.

    Example configuration file::

        # ...
        "trainer": {
            # type of trainer to instantiate (linked to task type)
            "type": "thelper.train.ImageClassifTrainer",
            # train for 40 epochs
            "epochs": 40,
            # save every 5 epochs
            "save_freq": 5,
            # monitor validation accuracy and save best model based on that
            "monitor": "accuracy",
            # optimization parameters block
            "optimization": {
                # all types & params below provided by PyTorch
                "loss": {
                    "type": "torch.nn.CrossEntropyLoss"
                },
                "optimizer": {
                    "type": "torch.optim.SGD",
                    "params": {
                        "lr": 0.1,
                        "momentum": 0.9,
                        "weight_decay": 1e-06,
                        "nesterov": true
                    }
                },
                "scheduler": {
                    "type": "torch.optim.lr_scheduler.StepLR",
                    "params": {
                        "step_size": 10,
                        "step_size": 0.1
                    }
                }
            },
            # in this example, we use two metrics in total
            # (one for monitoring, and one for logging only)
            "metrics": {
                "accuracy": {
                    "type": "thelper.optim.CategoryAccuracy"
                },
                "fullreport": {
                    "type": "thelper.optim.ClassifReport"
                }
            }
        }
        # ...

    Attributes:
        logger: used to output debug/warning/error messages to session log.
        name: name of the session, used for printing and creating log folders.
        epochs: number of epochs to train the model for.
        optimization_config: dictionary of optim-related parameters, parsed at training time.
        save_freq: frequency of checkpoint saves while training (i.e. save every X epochs).
        save_raw: specifies whether to save raw types or thelper objects in checkpoints.
        checkpoint_dir: session checkpoint output directory (located within 'save_dir').
        use_tbx: defines whether to use tensorboardX writers for logging or not.
        model: model to train; will be uploaded to target device(s) at runtime.
        config: full configuration dictionary of the session; will be incorporated into all saved checkpoints.
        devices: list of (cuda) device IDs to upload the model/tensors to; can be empty if only the CPU is available.
        monitor: specifies the name of the metric that should be monitored on the validation set for model improvement.

    TODO: move static utils to their related modules

    .. seealso::
        | :class:`thelper.train.classif.ImageClassifTrainer`
        | :class:`thelper.train.segm.ImageSegmTrainer`
        | :class:`thelper.train.detect.ObjDetectTrainer`
        | :class:`thelper.train.regr.RegressionTrainer`
        | :func:`thelper.train.utils.create_trainer`
    """

    def __init__(self,
                 session_name,    # type: AnyStr
                 save_dir,        # type: AnyStr
                 model,           # type: thelper.typedefs.ModelType
                 task,            # type: thelper.tasks.Task
                 loaders,         # type: thelper.typedefs.MultiLoaderType
                 config,          # type: thelper.typedefs.ConfigDict
                 ckptdata=None    # type: typ.Optional[thelper.typedefs.CheckpointContentType]
                 ):
        """Receives the trainer configuration dictionary, parses it, and sets up the session."""
        assert isinstance(model, (thelper.nn.Module, torch.nn.Module)), "unknown model object type"
        train_loader, valid_loader, test_loader = loaders
        assert (train_loader or valid_loader or test_loader), "must provide at least one loader with available data"
        assert "trainer" in config, "session configuration dictionary missing 'trainer' field"
        trainer_config = config["trainer"]
        thelper.utils.save_env_list(os.path.join(save_dir, "logs", "packages.log"))
        train_logger_path = os.path.join(save_dir, "logs", "trainer.log")
        train_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        train_logger_fh = logging.FileHandler(train_logger_path)
        train_logger_fh.setFormatter(train_logger_format)
        self.logger = thelper.utils.get_class_logger()
        self.logger.addHandler(train_logger_fh)
        self.logger.info("created training log for session '%s'" % session_name)
        logstamp = thelper.utils.get_log_stamp()
        repover = thelper.__version__ + ":" + thelper.utils.get_git_stamp()
        self.logger.debug("logstamp = %s" % logstamp)
        self.logger.debug("version = %s" % repover)
        self.name = session_name
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.epochs = 1
        if train_loader:
            if "epochs" not in trainer_config or not trainer_config["epochs"] or int(trainer_config["epochs"]) <= 0:
                raise AssertionError("bad trainer config epoch count")
            self.epochs = int(trainer_config["epochs"])
            if self.epochs <= 0:
                raise AssertionError("should train for at least one epoch")
            # no need to load optimization stuff if not training (i.e. no train_loader)
            # loading optimization stuff later since model needs to be on correct device
            if "optimization" not in trainer_config or not trainer_config["optimization"]:
                raise AssertionError("trainer config missing 'optimization' field")
            self.optimization_config = trainer_config["optimization"]
        else:
            self.logger.info("no training data provided, will run a single epoch on valid/test data")
        self.save_freq = int(trainer_config["save_freq"]) if "save_freq" in trainer_config else 1
        if self.save_freq < 1:
            raise AssertionError("checkpoint save frequency should be integer great or equal to 1")
        self.save_raw = thelper.utils.str2bool(thelper.utils.get_key_def("save_raw", trainer_config, True))
        self.checkpoint_dir = os.path.join(save_dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        output_root_dir = thelper.utils.get_key_def("output_dir", trainer_config, os.path.join(save_dir, "output"))
        output_root_dir = os.path.join(output_root_dir, self.name)
        os.makedirs(output_root_dir, exist_ok=True)
        self.use_tbx = thelper.utils.str2bool(thelper.utils.get_key_def("use_tbx", trainer_config, False))
        if self.use_tbx:
            import tensorboardX
            self.tbx = tensorboardX
            self.logger.debug("tensorboard init : tensorboard --logdir %s --port <your_port>" % output_root_dir)
        self.skip_tbx_histograms = thelper.utils.str2bool(thelper.utils.get_key_def("skip_tbx_histograms", trainer_config, False))
        self.tbx_histogram_freq = thelper.utils.get_key_def("tbx_histogram_freq", trainer_config, 1)
        if self.tbx_histogram_freq < 1:
            raise AssertionError("histogram output frequency should be integer great or equal to 1")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        train_foldername = "train-%s-%s" % (platform.node(), timestr)
        valid_foldername = "valid-%s-%s" % (platform.node(), timestr)
        test_foldername = "test-%s-%s" % (platform.node(), timestr)
        foldernames = [train_foldername, valid_foldername, test_foldername]
        output_paths = [None, None, None]
        for idx, (loader, foldername) in enumerate(zip(loaders, foldernames)):
            output_paths[idx] = os.path.join(output_root_dir, foldername) if loader else None
        self.train_output_path, self.valid_output_path, self.test_output_path = output_paths
        self.task = task
        self.model = model
        self.config = config
        devices_str = None
        if "device" in trainer_config:
            devices_str = trainer_config["device"]
        elif "train_device" in trainer_config:  # for backw compat only
            devices_str = trainer_config["train_device"]
        self.devices = self._load_devices(devices_str)
        if "metrics" in trainer_config and "base_metrics" in trainer_config:
            raise AssertionError("trainer config should have only one of 'metrics' and 'base_metrics'")
        if ("metrics" in trainer_config and trainer_config["metrics"]) or \
           ("base_metrics" in trainer_config and trainer_config["base_metrics"]):
            if "metrics" in trainer_config:
                self.logger.debug("loading metrics defined in trainer config")
                metrics = thelper.optim.create_metrics(trainer_config["metrics"])
            else:
                self.logger.debug("loading base metrics defined in trainer config")
                metrics = thelper.optim.create_metrics(trainer_config["base_metrics"])
        else:
            metrics = {}
        self.train_metrics = deepcopy(metrics)
        if "train_metrics" in trainer_config and trainer_config["train_metrics"]:
            self.train_metrics = {**self.train_metrics, **thelper.optim.create_metrics(trainer_config["train_metrics"])}
        for metric_name, metric in self.train_metrics.items():
            self.logger.info("parsed train metric '%s': %s" % (metric_name, str(metric)))
        self.valid_metrics = deepcopy(metrics)
        if "valid_metrics" in trainer_config and trainer_config["valid_metrics"]:
            self.valid_metrics = {**self.valid_metrics, **thelper.optim.create_metrics(trainer_config["valid_metrics"])}
        for metric_name, metric in self.valid_metrics.items():
            self.logger.info("parsed valid metric '%s': %s" % (metric_name, str(metric)))
        self.test_metrics = deepcopy(metrics)
        if "test_metrics" in trainer_config and trainer_config["test_metrics"]:
            self.test_metrics = {**self.test_metrics, **thelper.optim.create_metrics(trainer_config["test_metrics"])}
        for metric_name, metric in self.test_metrics.items():
            self.logger.info("parsed test metric '%s': %s" % (metric_name, str(metric)))
        self.monitor, self.monitor_best, self.monitor_best_epoch = None, None, -1
        if "monitor" in trainer_config and trainer_config["monitor"]:
            self.monitor = trainer_config["monitor"]
            if self.monitor not in self.train_metrics and self.monitor not in self.valid_metrics:
                raise AssertionError("metric with name '%s' should be declared in training and/or validation metrics"
                                     % self.monitor)
            if self.monitor in self.valid_metrics:
                self.monitor_goal = self.valid_metrics[self.monitor].goal()
            elif self.monitor in self.train_metrics:
                self.monitor_goal = self.train_metrics[self.monitor].goal()
            if self.monitor_goal == thelper.optim.Metric.minimize:
                self.monitor_best = thelper.optim.Metric.maximize
            elif self.monitor_goal == thelper.optim.Metric.maximize:
                self.monitor_best = thelper.optim.Metric.minimize
            else:
                raise AssertionError("monitored metric does not return proper optimization goal")
        if ckptdata is None:
            ckptdata = {}
        self.monitor_best = thelper.utils.get_key_def("monitor_best", ckptdata, self.monitor_best)
        self.monitor_best_epoch = thelper.utils.get_key_def("monitor_best_epoch", ckptdata, -1)
        self.optimizer_state = thelper.utils.get_key_def("optimizer", ckptdata, None)
        self.scheduler_state = thelper.utils.get_key_def("scheduler", ckptdata, None)
        self.current_iter = thelper.utils.get_key_def("iter", ckptdata, 0)
        self.current_epoch = thelper.utils.get_key_def("epoch", ckptdata, 0)
        self.outputs = thelper.utils.get_key_def("outputs", ckptdata, {})
        # callbacks (see ``thelper.typedefs.IterCallbackType`` and ``thelper.typedefs.IterCallbackParams`` definitions)
        self.train_iter_callback = thelper.utils.get_key_def(
            "train_iter_callback", trainer_config, None)    # type: typ.IterCallbackType
        if self.train_iter_callback is not None and isinstance(self.train_iter_callback, str):
            self.train_iter_callback = thelper.utils.import_function(
                self.train_iter_callback)                   # type: typ.IterCallbackType
        self.eval_iter_callback = thelper.utils.get_key_def(
            "eval_iter_callback", trainer_config, None)     # type: typ.IterCallbackType
        if self.eval_iter_callback is not None and isinstance(self.eval_iter_callback, str):
            self.eval_iter_callback = thelper.utils.import_function(
                self.eval_iter_callback)                    # type: typ.IterCallbackType
        self.callback_kwargs = thelper.utils.get_key_def("callback_kwargs", trainer_config, {})
        if not isinstance(self.callback_kwargs, dict):
            raise AssertionError("invalid callback kwargs type")
        display_predictions = thelper.utils.get_key_def(["display_preds", "display_predictions"], trainer_config, False)
        display_train_predictions = thelper.utils.get_key_def(["display_train_preds", "display_train_predictions"], trainer_config, False)
        display_eval_predictions = thelper.utils.get_key_def(["display_eval_preds", "display_eval_predictions"], trainer_config, False)
        if display_predictions or display_train_predictions:
            if self.train_iter_callback is not None:
                raise AssertionError("cannot use 'display_preds' while also using an external callback")
            self.train_iter_callback = thelper.utils.import_function("thelper.train.utils._draw_minibatch_wrapper")
        if display_predictions or display_eval_predictions:
            if self.eval_iter_callback is not None:
                raise AssertionError("cannot use 'display_preds' while also using an external callback")
            self.eval_iter_callback = thelper.utils.import_function("thelper.train.utils._draw_minibatch_wrapper")
        if self.train_iter_callback is not None:
            thelper.utils.check_func_signature(self.train_iter_callback, typ.IterCallbackParams)
        if self.eval_iter_callback is not None:
            thelper.utils.check_func_signature(self.eval_iter_callback, typ.IterCallbackParams)
        self.skip_eval_iter = thelper.utils.get_key_def("skip_eval_iter", trainer_config, 0)

    def _init_writer(self, writer, path):
        if self.use_tbx and not writer:
            writer = self.tbx.SummaryWriter(log_dir=path, comment=self.name)
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
            return [Trainer._move_tensor(t, dev) for t in tensor]
        if isinstance(tensor, dict):
            return {k: Trainer._move_tensor(t, dev) for k, t in tensor.items()}
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
        if not isinstance(config, dict):
            raise AssertionError("config should be provided as a dictionary")
        if self.train_loader is None or not self.train_loader:
            raise AssertionError("optimization only useful if training data is available")
        loss = None  # can now be omitted if using custom trainer
        if "loss" in config:
            uploader = functools.partial(self._move_tensor, dev=dev)
            loss = thelper.optim.create_loss_fn(config["loss"], model, self.train_loader, uploader)
        if "optimizer" not in config or not config["optimizer"]:
            raise AssertionError("optimization config missing 'optimizer' field")
        optimizer = thelper.optim.create_optimizer(config["optimizer"], model)
        scheduler, scheduler_step_metric = None, None
        if "scheduler" in config and config["scheduler"]:
            scheduler, scheduler_step_metric = thelper.optim.create_scheduler(config["scheduler"], optimizer)
        return loss, optimizer, scheduler, scheduler_step_metric

    def _load_devices(self, devices_str=None):
        """Validates and returns the list of CUDA devices available on the system."""
        self.logger.debug("loading available devices")
        available_cuda_devices = thelper.utils.get_available_cuda_devices()
        devices = None
        if devices_str is not None:
            if isinstance(devices_str, str):
                if not devices_str:
                    raise AssertionError("cannot specify empty device name, use default to auto-detect")
                devices_str = devices_str.split(",")
            elif isinstance(devices_str, list):
                if not devices_str:
                    raise AssertionError("cannot specify empty device list, use default to auto-detect")
                if not all([isinstance(dev_str, str) for dev_str in devices_str]):
                    raise AssertionError("unexpected type in device list, should be string")
            else:
                raise AssertionError("unexpected device string type")
            for dev_idx, dev_str in enumerate(devices_str):
                if "cuda" not in dev_str and dev_str != "cpu":
                    raise AssertionError("unknown device type '%s' (expecting 'cpu' or 'cuda:X')" % dev_str)
                elif dev_str == "cpu":
                    if len(devices_str) > 1:
                        raise AssertionError("cannot combine cpu with other devices")
                    return []
                elif dev_str == "cuda" or dev_str == "cuda:all":
                    if len(devices_str) > 1:
                        raise AssertionError("must specify device index (e.g. 'cuda:0') if combining devices")
                    if not available_cuda_devices:
                        raise AssertionError("could not find any available cuda devices")
                    return available_cuda_devices
                elif "cuda:" not in dev_str:
                    raise AssertionError("expecting cuda device format to be 'cuda:X' (where X is device index)")
                cuda_dev_idx = int(dev_str.rsplit(":", 1)[-1])
                if cuda_dev_idx not in available_cuda_devices:
                    raise AssertionError("cuda device '%s' out of range (detected devices = %s)"
                                         % (dev_str, str(available_cuda_devices)))
                if devices is None:
                    devices = [cuda_dev_idx]
                else:
                    devices.append(cuda_dev_idx)
            return devices
        else:
            return available_cuda_devices

    def train(self):
        """Starts the training process.

        This function will train the model until the required number of epochs is reached, and then evaluate it
        on the test data. The setup of loggers, tensorboard writers is done here, so is model improvement tracking
        via monitored metrics. However, the code related to loss computation and back propagation is implemented in
        a derived class via :func:`thelper.train.base.Trainer.train_epoch`.
        """
        if not self.train_loader:
            raise AssertionError("missing training data, invalid loader!")
        if isinstance(self.model, torch.jit.ScriptModule):
            raise AssertionError("cannot train model trace")
        self.logger.debug("uploading model to '%s'..." % str(self.devices))
        model = self._upload_model(self.model, self.devices)
        loss, optimizer, scheduler, scheduler_step_metric = self._load_optimization(model, self.devices)
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
            self.optimizer_state = None
        if self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)
            self.scheduler_state = None
        self.logger.debug("loss: %s" % str(loss))
        self.logger.debug("optimizer: %s" % str(optimizer))
        latest_loss = math.inf
        train_writer, valid_writer = None, None
        while self.current_epoch < self.epochs:
            train_writer = self._init_writer(train_writer, self.train_output_path)
            self.logger.info("at epoch#%d for '%s' (dev=%s)" % (self.current_epoch, self.name, str(self.devices)))
            if scheduler:
                if scheduler_step_metric:
                    if scheduler_step_metric == "loss":
                        # todo: use validation loss instead? more stable?
                        scheduler.step(metrics=latest_loss, epoch=self.current_epoch)
                    else:
                        if self.valid_loader and scheduler_step_metric in self.valid_metrics:
                            metric = self.valid_metrics[scheduler_step_metric]
                        elif self.train_loader and scheduler_step_metric in self.train_metrics:
                            metric = self.train_metrics[scheduler_step_metric]
                        else:
                            raise AssertionError("cannot find metric '%s' for scheduler step" % scheduler_step_metric)
                        if not metric.is_scalar():
                            raise AssertionError("cannot use metric '%s' for scheduler step (not a scalar)" % scheduler_step_metric)
                        metric_val = metric.eval() if self.current_epoch > 0 else metric.anti_goal()
                        scheduler.step(metrics=metric_val, epoch=self.current_epoch)
                else:
                    scheduler.step(epoch=self.current_epoch)
            if train_writer and not self.skip_tbx_histograms and (self.current_epoch % self.tbx_histogram_freq) == 0:
                for pname, param in model.named_parameters():
                    if "bn" in pname:
                        continue  # skip batch norm modules
                    pname = pname.replace(".", "/")  # for proper grouping
                    if pname.startswith("module/"):
                        pname = pname.replace("module/", "", 1)
                    if pname.startswith("model/"):
                        pname = pname.replace("model/", "", 1)
                    data = param.data.cpu().numpy().flatten()
                    train_writer.add_histogram(pname, data, self.current_epoch)
                    if param.grad is not None:
                        grad = param.grad.data.cpu().numpy().flatten()
                        train_writer.add_histogram(pname + '/grad', grad, self.current_epoch)
            self.logger.debug("learning rate at %.8f" % thelper.optim.get_lr(optimizer))
            self._set_rng_state(self.train_loader.seeds, self.current_epoch)
            model.train()
            for metric in self.train_metrics.values():
                if hasattr(metric, "set_max_accum") and callable(metric.set_max_accum):
                    metric.set_max_accum(len(self.train_loader))  # used to make scalar metric evals smoother between epochs
                if metric.needs_reset():
                    metric.reset()  # if a metric needs to be reset between two epochs, do it here
            if hasattr(self.train_loader, "set_epoch") and callable(self.train_loader.set_epoch):
                self.train_loader.set_epoch(self.current_epoch)
            latest_loss, self.current_iter = self.train_epoch(model, self.current_epoch, self.current_iter, self.devices,
                                                              loss, optimizer, self.train_loader, self.train_metrics,
                                                              self.monitor, train_writer)
            self._write_epoch_output(self.current_epoch, self.train_metrics, train_writer, self.train_output_path,
                                     loss=latest_loss, optimizer=optimizer)
            train_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.train_metrics.items()}
            result = {"train/loss": latest_loss, "train/metrics": train_metric_vals}
            monitor_type_key = "train/metrics"  # if we cannot run validation, will monitor progression on training metrics
            if self.valid_loader:
                self._set_rng_state(self.valid_loader.seeds, self.current_epoch)
                model.eval()
                valid_writer = self._init_writer(valid_writer, self.valid_output_path)
                for metric in self.valid_metrics.values():
                    metric.reset()  # force reset here, we always evaluate from a clean state
                if hasattr(self.valid_loader, "set_epoch") and callable(self.valid_loader.set_epoch):
                    self.valid_loader.set_epoch(self.current_epoch)
                self.eval_epoch(model, self.current_epoch, self.devices, self.valid_loader,
                                self.valid_metrics, self.monitor, valid_writer)
                self._write_epoch_output(self.current_epoch, self.valid_metrics, valid_writer, self.valid_output_path)
                valid_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.valid_metrics.items()}
                result = {**result, "valid/metrics": valid_metric_vals}
                monitor_type_key = "valid/metrics"  # since validation is available, use that to monitor progression
            new_best = False
            monitor_val = None
            for key, value in result.items():
                if key == monitor_type_key and self.monitor is not None:
                    if self.monitor not in value:
                        raise AssertionError("not monitoring required variable '%s' in metrics" % self.monitor)
                    monitor_val = value[self.monitor]
                    if (self.monitor_goal == thelper.optim.Metric.minimize and monitor_val < self.monitor_best) or \
                       (self.monitor_goal == thelper.optim.Metric.maximize and monitor_val > self.monitor_best):
                        self.monitor_best = monitor_val
                        self.monitor_best_epoch = self.current_epoch
                        new_best = True
                if not isinstance(value, dict):
                    self.logger.debug(" epoch#{} result =>  {}: {}".format(self.current_epoch, str(key), value))
                else:
                    for subkey, subvalue in value.items():
                        self.logger.debug(" epoch#{} result =>  {}:{}: {}".format(self.current_epoch, str(key), str(subkey), subvalue))
            if self.monitor is not None:
                if monitor_val is None:
                    raise AssertionError("training/validation did not produce required monitoring variable '%s'" % self.monitor)
                if new_best:
                    best_str = "(new best value)"
                else:
                    best_str = ("(previous best = %s @ epoch = %d)" % (self.monitor_best, self.monitor_best_epoch))
                self.logger.info("epoch %d, monitored %s = %s  %s" % (self.current_epoch, self.monitor, monitor_val, best_str))
            self.outputs[self.current_epoch] = result
            if new_best or (self.current_epoch % self.save_freq) == 0:
                self.logger.info("saving checkpoint @ epoch#%d" % self.current_epoch)
                self._save(self.current_epoch, self.current_iter, optimizer, scheduler, save_best=new_best)
            self.current_epoch += 1
        self.logger.info("training for session '%s' done" % self.name)
        return self.outputs

    def eval(self):
        """Starts the evaluation process.

        This function will evaluate the model using the test data (or the validation data, if no test data is available),
        and return the results. Note that the code related to the forwarding of samples inside the model itself is implemented
        in a derived class via :func:`thelper.train.base.Trainer.eval_epoch`.
        """
        if not self.valid_loader and not self.test_loader:
            raise AssertionError("missing validation/test data, invalid loaders!")
        self.logger.debug("uploading model to '%s'..." % str(self.devices))
        model = self._upload_model(self.model, self.devices)
        result = {}
        valid_writer, test_writer = None, None
        if self.test_loader:
            self._set_rng_state(self.test_loader.seeds, self.current_epoch)
            model.eval()
            test_writer = self._init_writer(test_writer, self.test_output_path)
            for metric in self.test_metrics.values():
                metric.reset()  # force reset here, we always evaluate from a clean state
            if hasattr(self.test_loader, "set_epoch") and callable(self.test_loader.set_epoch):
                self.test_loader.set_epoch(self.current_epoch)
            self.eval_epoch(model, self.current_epoch, self.devices, self.test_loader,
                            self.test_metrics, self.monitor, test_writer)
            self._write_epoch_output(self.current_epoch, self.test_metrics, test_writer, self.test_output_path)
            test_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.test_metrics.items()}
            result = {**result, **test_metric_vals}
            output_group = "test/metrics"
        elif self.valid_loader:
            self._set_rng_state(self.valid_loader.seeds, self.current_epoch)
            model.eval()
            valid_writer = self._init_writer(valid_writer, self.valid_output_path)
            for metric in self.valid_metrics.values():
                metric.reset()  # force reset here, we always evaluate from a clean state
            if hasattr(self.valid_loader, "set_epoch") and callable(self.valid_loader.set_epoch):
                self.valid_loader.set_epoch(self.current_epoch)
            self.eval_epoch(model, self.current_epoch, self.devices, self.valid_loader,
                            self.valid_metrics, self.monitor, valid_writer)
            self._write_epoch_output(self.current_epoch, self.valid_metrics, valid_writer, self.valid_output_path)
            valid_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.valid_metrics.items()}
            result = {**result, **valid_metric_vals}
            output_group = "valid/metrics"
        for key, value in result.items():
            if not isinstance(value, dict):
                self.logger.debug(" final result =>  {}: {}".format(str(key), value))
            else:
                for subkey, subvalue in value.items():
                    self.logger.debug(" final result =>  {}:{}: {}".format(str(key), str(subkey), subvalue))
        if self.current_epoch not in self.outputs:
            self.outputs[self.current_epoch] = {}
        self.outputs[self.current_epoch][output_group] = result
        self.logger.info("evaluation for session '%s' done" % self.name)
        return self.outputs

    @abstractmethod
    def train_epoch(self, model, epoch, iter, dev, loss, optimizer, loader, metrics, monitor=None, writer=None):
        """Trains the model for a single epoch using the provided objects.

        Args:
            model: the model to train that is already uploaded to the target device(s).
            epoch: the epoch index we are training for (0-based).
            iter: the iteration count at the start of the current epoch.
            dev: the target device that tensors should be uploaded to.
            loss: the loss function used to evaluate model fidelity.
            optimizer: the optimizer used for back propagation.
            loader: the data loader used to get transformed training samples.
            metrics: the dictionary of metrics to update every iteration.
            monitor: name of the metric to update/monitor for improvements.
            writer: the writer used to store tbx events/messages/metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_epoch(self, model, epoch, dev, loader, metrics, monitor=None, writer=None):
        """Evaluates the model using the provided objects.

        Args:
            model: the model to evaluate that is already uploaded to the target device(s).
            epoch: the epoch index we are training for (0-based).
            dev: the target device that tensors should be uploaded to.
            loader: the data loader used to get transformed valid/test samples.
            metrics: the dictionary of metrics to update every iteration.
            monitor: name of the metric to update/monitor for improvements.
            writer: the writer used to store tbx events/messages/metrics.
        """
        raise NotImplementedError

    def _write_epoch_output(self, epoch, metrics, tbx_writer, output_path, loss=None, optimizer=None):
        """Writes the cumulative evaluation result of all metrics using a specific writer."""
        self.logger.debug("writing epoch metrics to '%s'" % output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if tbx_writer is not None and loss is not None and optimizer is not None:
            tbx_writer.add_scalar("epoch/loss", loss, epoch)
            tbx_writer.add_scalar("epoch/lr", thelper.optim.get_lr(optimizer), epoch)
        for metric_name, metric in metrics.items():
            if metric.is_scalar():
                if tbx_writer is not None:
                    tbx_writer.add_scalar("epoch/%s" % metric_name, metric.eval(), epoch)
            if hasattr(metric, "render") and callable(metric.render):
                img = metric.render()
                if img is not None:
                    if tbx_writer is not None:
                        tbx_writer.add_image(metric_name, img, epoch, dataformats="HWC")
                    raw_filename = "%s-%04d.png" % (metric_name, epoch)
                    raw_filepath = os.path.join(output_path, raw_filename)
                    cv.imwrite(raw_filepath, img[..., [2, 1, 0]])
            txt = metric.print() if hasattr(metric, "print") and callable(metric.print) else None
            if not txt:
                eval_res = metric.eval()
                if eval_res is not None:
                    if isinstance(eval_res, float):
                        txt = "%.4f" % eval_res  # make sure we always have decent precision
                    else:
                        txt = str(eval_res)
            if txt:
                raw_filename = "%s-%04d.txt" % (metric_name, epoch)
                raw_filepath = os.path.join(output_path, raw_filename)
                with open(raw_filepath, "w") as fd:
                    fd.write(txt)

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
        filename = "ckpt.%04d.%s.pth" % (epoch, log_stamp)
        filename = os.path.join(self.checkpoint_dir, filename)
        self.logger.debug("writing checkpoint to '%s'" % filename)
        torch.save(curr_state, filename)
        if save_best:
            filename_best = os.path.join(self.checkpoint_dir, "ckpt.best.pth")
            self.logger.debug("writing checkpoint to '%s'" % filename_best)
            torch.save(curr_state, filename_best)
