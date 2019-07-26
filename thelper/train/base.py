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
from typing import AnyStr, Optional  # noqa: F401

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
            # in this example, we use two consumers in total
            # (one metric for monitoring, and one for logging)
            "metrics": {
                "accuracy": {
                    "type": "thelper.optim.Accuracy"
                },
                "fullreport": {
                    "type": "thelper.train.ClassifReport"
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
        monitor: name of the training/validation metric that should be monitored for model improvement.

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
        assert isinstance(task, thelper.tasks.Task), "unknown task object type"
        assert isinstance(loaders, (list, tuple, np.ndarray)) and len(loaders) == 3, "invalid loaders array"
        assert isinstance(config, dict), "invalid config type"
        self.task = task
        self.model = model
        self.config = config

        # parse basic training config args
        trainer_config = thelper.utils.get_key("trainer", config, msg="session config dictionary missing 'trainer' field")
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
        self.epochs = 1
        self.save_freq = int(thelper.utils.get_key_def("save_freq", trainer_config, 1))
        assert self.save_freq >= 1, "checkpoint save frequency should be strictly positive integer"
        self.save_raw = thelper.utils.str2bool(thelper.utils.get_key_def("save_raw", trainer_config, True))
        self.checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        output_root_dir = thelper.utils.get_key_def("output_dir", trainer_config, os.path.join(save_dir, "output"))
        output_root_dir = os.path.join(output_root_dir, self.name)
        os.makedirs(output_root_dir, exist_ok=True)
        devices_str = thelper.utils.get_key_def(["device", "train_device"], trainer_config, None)
        self.devices = self._load_devices(devices_str)
        self.skip_eval_iter = thelper.utils.get_key_def("skip_eval_iter", trainer_config, 0)

        # parse and prepare tbx stuff
        self.use_tbx = thelper.utils.str2bool(thelper.utils.get_key_def("use_tbx", trainer_config, False))
        if self.use_tbx:
            import tensorboardX
            self.tbx = tensorboardX
            self.logger.debug("tensorboard init : tensorboard --logdir %s --port <your_port>" % output_root_dir)
        self.skip_tbx_histograms = thelper.utils.str2bool(thelper.utils.get_key_def("skip_tbx_histograms", trainer_config, False))
        self.tbx_histogram_freq = int(thelper.utils.get_key_def("tbx_histogram_freq", trainer_config, 1))
        assert self.tbx_histogram_freq >= 1, "histogram output frequency should be strictly positive integer"
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.writers, self.output_paths = {}, {}
        for cname, loader in zip(["train", "valid", "test"], loaders):
            folder_name = f"{cname}-{str(platform.node())}-{timestr}"
            self.output_paths[cname] = os.path.join(output_root_dir, folder_name) if loader else None
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
                    self.logger.info("parsed metric '%s': %s" % (mkey, str(mval)))

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

        # parse checkpoint data from previous run (if available)
        ckptdata = {} if ckptdata is None else ckptdata
        self.monitor_best = thelper.utils.get_key_def("monitor_best", ckptdata, self.monitor_best)
        self.monitor_best_epoch = thelper.utils.get_key_def("monitor_best_epoch", ckptdata, -1)
        self.optimizer_state = thelper.utils.get_key_def("optimizer", ckptdata, None)
        self.scheduler_state = thelper.utils.get_key_def("scheduler", ckptdata, None)
        self.current_iter = thelper.utils.get_key_def("iter", ckptdata, 0)
        self.current_epoch = thelper.utils.get_key_def("epoch", ckptdata, 0)
        self.outputs = thelper.utils.get_key_def("outputs", ckptdata, {})

        # parse callbacks (see ``thelper.typedefs.IterCallbackType`` and ``thelper.typedefs.IterCallbackParams`` definitions)
        for cname, mset in zip(["train", "valid", "test"], [self.train_metrics, self.valid_metrics, self.test_metrics]):
            # parse user (custom) callback
            user_callback_keys = [f"{cname}_iter_callback", f"{cname}_callback", "callback"]
            user_callback = thelper.utils.get_key_def(user_callback_keys, trainer_config)  # type: typ.IterCallbackType
            user_callback_kwargs_keys = [f"{cname}_iter_callback_kwargs", f"{cname}_callback_kwargs", "callback_kwargs"]
            user_callback_kwargs = thelper.utils.get_key_def(user_callback_kwargs_keys, trainer_config, {})
            if user_callback is not None:
                assert "user_callback" not in mset, "metrics set already had a 'user_callback' in it"
                mset["user_callback"] = thelper.train.utils.PredictionCallback(user_callback, user_callback_kwargs)
            # parse display callback
            display_flag_keys = [f"display_{cname}_preds", f"display_{cname}_predictions", f"display_{cname}",
                                 "display_preds", "display_predictions", "display"]
            display_flag = thelper.utils.get_key_def(display_flag_keys, trainer_config, False)
            display_kwargs_keys = [f"display_{cname}_preds_kwargs", f"display_{cname}_predictions_kwargs",
                                   f"display_{cname}_kwargs", "display_preds_kwargs", "display_predictions_kwargs",
                                   "display_kwargs"]
            display_kwargs = thelper.utils.get_key_def(display_kwargs_keys, trainer_config, {})
            if display_flag:
                assert "display_callback" not in mset, "metrics set already had a 'display_callback' in it"
                display_kwargs["output_path"] = self.output_paths[cname]
                display_kwargs["save"] = thelper.utils.get_key_def(["save", "save_draw", "save_draw_output"],
                                                                   display_kwargs, False)
                mset["display_callback"] = thelper.train.utils.PredictionCallback("thelper.train.utils._draw_wrapper",
                                                                                  display_kwargs)
            # add logging callback (will print to console and update iter metric evals)
            logging_kwargs = thelper.utils.get_key_def("logging_kwargs", trainer_config, {})
            logging_kwargs["set_name"] = cname
            logging_kwargs["writers"] = self.writers  # pass by ref, will be filled later
            display_kwargs["output_path"] = self.output_paths[cname]
            mset["logging_callback"] = thelper.train.utils.PredictionCallback(self._iter_logger_callback,
                                                                              logging_kwargs)

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
                if available_cuda_devices is None:
                    available_cuda_devices = thelper.utils.get_available_cuda_devices()
                assert cuda_dev_idx in available_cuda_devices, \
                    f"cuda device '{dev_str}' unavailable (detected devices = {str(available_cuda_devices)})"
                devices.append(cuda_dev_idx)
            return devices
        else:
            return thelper.utils.get_available_cuda_devices()

    def train(self):
        """Starts the training process.

        This function will train the model until the required number of epochs is reached, and then evaluate it
        on the test data. The setup of loggers, tensorboard writers is done here, so is model improvement tracking
        via monitored metrics. However, the code related to loss computation and back propagation is implemented in
        a derived class via :func:`thelper.train.base.Trainer.train_epoch`.
        """
        assert self.train_loader, "missing training data, invalid loader!"
        assert not isinstance(self.model, torch.jit.ScriptModule), "current impl cannot train model traces"  # TODO
        self.logger.debug("uploading model to '%s'..." % str(self.devices))
        model = self._upload_model(self.model, self.devices)
        loss, optimizer, scheduler, scheduler_step_metric = self._load_optimization(model, self.devices)
        if optimizer is not None and self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
            self.optimizer_state = None
        if scheduler is not None and self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)
            self.scheduler_state = None
        self.logger.debug(f"loss: {str(loss)}")
        self.logger.debug(f"optimizer: {str(optimizer)}")
        latest_loss = math.inf
        while self.current_epoch < self.epochs:
            self.writers["train"] = self._init_writer(self.writers["train"], self.output_paths["train"])
            self.logger.info("at epoch#%d for '%s' (dev=%s)" % (self.current_epoch, self.name, str(self.devices)))
            if scheduler:
                if scheduler_step_metric:
                    if scheduler_step_metric == "loss":
                        # todo: use validation loss instead? more stable?
                        scheduler.step(metrics=latest_loss, epoch=self.current_epoch)
                    else:
                        metric = None
                        if self.valid_loader and scheduler_step_metric in self.valid_metrics:
                            metric = self.valid_metrics[scheduler_step_metric]
                        elif self.train_loader and scheduler_step_metric in self.train_metrics:
                            metric = self.train_metrics[scheduler_step_metric]
                        # note: makes no sense to look for it in test metrics
                        assert metric is not None, f"cannot find metric '{scheduler_step_metric}' for scheduler step"
                        assert isinstance(metric, thelper.optim.metrics.Metric), "monitoring consumer must be metric"
                        metric_anti_goal = thelper.optim.Metric.maximize if metric.goal == thelper.optim.Metric.minimize \
                            else thelper.optim.Metric.minimize
                        metric_val = metric.eval() if self.current_epoch > 0 else metric_anti_goal
                        scheduler.step(metrics=metric_val, epoch=self.current_epoch)
                else:
                    scheduler.step(epoch=self.current_epoch)
            if self.writers["train"] and not self.skip_tbx_histograms and \
                    (self.current_epoch % self.tbx_histogram_freq) == 0:
                for pname, param in model.named_parameters():
                    if "bn" in pname:
                        continue  # skip batch norm modules
                    pname = pname.replace(".", "/")  # for proper grouping
                    if pname.startswith("module/"):
                        pname = pname.replace("module/", "", 1)
                    if pname.startswith("model/"):
                        pname = pname.replace("model/", "", 1)
                    data = param.data.cpu().numpy().flatten()
                    self.writers["train"].add_histogram(pname, data, self.current_epoch)
                    if param.grad is not None:
                        grad = param.grad.data.cpu().numpy().flatten()
                        self.writers["train"].add_histogram(pname + '/grad', grad, self.current_epoch)
            self.logger.debug("learning rate at %.8f" % thelper.optim.get_lr(optimizer))
            self._set_rng_state(self.train_loader.seeds, self.current_epoch)
            model.train()
            if hasattr(self.train_loader, "set_epoch") and callable(self.train_loader.set_epoch):
                self.train_loader.set_epoch(self.current_epoch)
            latest_loss = self.train_epoch(model, self.current_epoch, self.devices,
                                           loss, optimizer, self.train_loader, self.train_metrics)
            self._write_epoch_output(self.current_epoch, self.train_metrics,
                                     self.writers["train"], self.output_paths["train"],
                                     loss=latest_loss, optimizer=optimizer)
            train_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.train_metrics.items()
                                 if isinstance(metric, thelper.optim.metrics.Metric)}
            result = {"train/loss": latest_loss, "train/metrics": train_metric_vals}
            monitor_type_key = "train/metrics"  # if we cannot run validation, will monitor progression on training metrics
            if self.valid_loader:
                self._set_rng_state(self.valid_loader.seeds, self.current_epoch)
                model.eval()
                self.writers["valid"] = self._init_writer(self.writers["valid"], self.output_paths["valid"])
                for metric in self.valid_metrics.values():
                    metric.reset()  # force reset here, we always evaluate from a clean state
                if hasattr(self.valid_loader, "set_epoch") and callable(self.valid_loader.set_epoch):
                    self.valid_loader.set_epoch(self.current_epoch)
                self.eval_epoch(model, self.current_epoch, self.devices, self.valid_loader, self.valid_metrics)
                self._write_epoch_output(self.current_epoch, self.valid_metrics,
                                         self.writers["valid"], self.output_paths["valid"])
                valid_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.valid_metrics.items()
                                     if isinstance(metric, thelper.optim.metrics.Metric)}
                result = {**result, "valid/metrics": valid_metric_vals}
                monitor_type_key = "valid/metrics"  # since validation is available, use that to monitor progression
            new_best = False
            monitor_val = None
            for key, value in result.items():
                if key == monitor_type_key and self.monitor is not None:
                    assert self.monitor in value, f"not monitoring required variable '{self.monitor}' in metrics"
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
                assert monitor_val is not None, f"training/validation did not evaluate required metric '{self.monitor}'"
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
        assert self.valid_loader or self.test_loader, "missing validation/test data, invalid loaders!"
        self.logger.debug("uploading model to '%s'..." % str(self.devices))
        model = self._upload_model(self.model, self.devices)
        result = {}
        output_group = None, None
        if self.test_loader:
            self._set_rng_state(self.test_loader.seeds, self.current_epoch)
            model.eval()
            self.writers["test"] = self._init_writer(self.writers["test"], self.output_paths["test"])
            for metric in self.test_metrics.values():
                metric.reset()  # force reset here, we always evaluate from a clean state
            if hasattr(self.test_loader, "set_epoch") and callable(self.test_loader.set_epoch):
                self.test_loader.set_epoch(self.current_epoch)
            self.eval_epoch(model, self.current_epoch, self.devices, self.test_loader, self.test_metrics)
            self._write_epoch_output(self.current_epoch, self.test_metrics,
                                     self.writers["test"], self.output_paths["test"])
            test_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.test_metrics.items()
                                if isinstance(metric, thelper.optim.metrics.Metric)}
            result = {**result, **test_metric_vals}
            output_group = "test/metrics"
        elif self.valid_loader:
            self._set_rng_state(self.valid_loader.seeds, self.current_epoch)
            model.eval()
            self.writers["valid"] = self._init_writer(self.writers["valid"], self.output_paths["valid"])
            for metric in self.valid_metrics.values():
                metric.reset()  # force reset here, we always evaluate from a clean state
            if hasattr(self.valid_loader, "set_epoch") and callable(self.valid_loader.set_epoch):
                self.valid_loader.set_epoch(self.current_epoch)
            self.eval_epoch(model, self.current_epoch, self.devices, self.valid_loader, self.valid_metrics)
            self._write_epoch_output(self.current_epoch, self.valid_metrics,
                                     self.writers["valid"], self.output_paths["valid"])
            valid_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.valid_metrics.items()
                                 if isinstance(metric, thelper.optim.metrics.Metric)}
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
    def train_epoch(self, model, epoch, dev, loss, optimizer, loader, metrics):
        """Trains the model for a single epoch using the provided objects.

        Args:
            model: the model to train that is already uploaded to the target device(s).
            epoch: the epoch index we are training for (0-based).
            dev: the target device that tensors should be uploaded to.
            loss: the loss function used to evaluate model fidelity.
            optimizer: the optimizer used for back propagation.
            loader: the data loader used to get transformed training samples.
            metrics: the dictionary of metrics/consumers to update every iteration.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_epoch(self, model, epoch, dev, loader, metrics):
        """Evaluates the model using the provided objects.

        Args:
            model: the model to evaluate that is already uploaded to the target device(s).
            epoch: the epoch index we are training for (0-based).
            dev: the target device that tensors should be uploaded to.
            loader: the data loader used to get transformed valid/test samples.
            metrics: the dictionary of metrics/consumers to update every iteration.
        """
        raise NotImplementedError

    def _iter_logger_callback(self,  # see `thelper.typedefs.IterCallbackParams` for more info
                              task,  # type: thelper.tasks.utils.Task
                              input,  # type: thelper.typedefs.InputType
                              pred,  # type: thelper.typedefs.PredictionType
                              target,  # type: thelper.typedefs.TargetType
                              sample,  # type: thelper.typedefs.SampleType
                              loss,  # type: Optional[float]
                              iter_idx,  # type: int
                              max_iters,  # type: int
                              epoch_idx,  # type: int
                              max_epochs,  # type: int
                              **kwargs):  # type: (...) -> None
        """Receives callback data for logging loss/monitored metric values each training/eval iteration."""
        set_name = thelper.utils.get_key("set_name", kwargs, "missing set name in iter logger args")
        assert set_name in ["train", "valid", "test"], "unrecognized iter logger set name"
        metrics = self.train_metrics if set_name == "train" else self.valid_metrics if set_name == "valid" \
            else self.test_metrics
        writers = thelper.utils.get_key("writers", kwargs, "missing writers dict in iter logger args")
        assert set_name in writers, "expected set name writer match in kwargs"
        writer = writers[set_name]
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
        if writer:
            if loss is not None:
                writer.add_scalar("iter/loss", loss, self.current_iter)
            for metric_name, metric in metrics.items():
                if isinstance(metric, thelper.optim.metrics.Metric):
                    if metric_name == self.monitor and monitor_val is not None:
                        writer.add_scalar("iter/%s" % self.monitor, monitor_val, self.current_iter)
                    elif metric.live_eval:
                        writer.add_scalar("iter/%s" % metric_name, metric.eval(), self.current_iter)
        if set_name == "train":
            self.current_iter += 1

    def _write_epoch_output(self, epoch, metrics, tbx_writer, output_path, loss=None, optimizer=None):
        """Writes the cumulative evaluation result of all metrics using a specific writer."""
        self.logger.debug("writing epoch metrics to '%s'" % output_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if tbx_writer is not None and loss is not None and optimizer is not None:
            tbx_writer.add_scalar("epoch/loss", loss, epoch)
            tbx_writer.add_scalar("epoch/lr", thelper.optim.get_lr(optimizer), epoch)
        for metric_name, metric in metrics.items():
            if isinstance(metric, thelper.optim.metrics.Metric) and tbx_writer is not None:
                tbx_writer.add_scalar("epoch/%s" % metric_name, metric.eval(), epoch)
            if hasattr(metric, "render") and callable(metric.render):
                img = metric.render()
                if img is not None:
                    if tbx_writer is not None:
                        tbx_writer.add_image(metric_name, img, epoch, dataformats="HWC")
                    raw_filename = "%s-%04d.png" % (metric_name, epoch)
                    raw_filepath = os.path.join(output_path, raw_filename)
                    cv.imwrite(raw_filepath, img[..., [2, 1, 0]])
            txt = metric.report() if hasattr(metric, "report") and callable(metric.report) else None
            if not txt and isinstance(metric, thelper.optim.metrics.Metric):
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
