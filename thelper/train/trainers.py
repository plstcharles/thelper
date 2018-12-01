"""Model training/evaluation module.

This module contains the classes required to train and/or evaluate a model based on different tasks. These are
instantiated in launched sessions based on configuration dictionaries, like many other classes of the framework.
"""
import logging
import math
import os
import platform
import time
from abc import abstractmethod
from copy import deepcopy

import cv2 as cv
import torch
import torch.optim

import thelper.data.samplers
import thelper.utils

logger = logging.getLogger(__name__)


class Trainer:
    """Abstract trainer interface that defines basic session i/o and setup operations.

    This interface defines the general behavior of a training session which includes configuration parsing, tensorboard
    setup, metrics and goal setup, and loss/optimizer setup. It also provides utilities for uploading models and tensors
    on specific devices, and for saving the state of a session. This interface should be specialized for every task by
    implementing the ``_train_epoch`` and ``_train_epoch`` functions in a derived class. See
    :class:`thelper.train.trainers.ImageClassifTrainer` for an example.

    The parameters that will be parsed by this interface from a configuration dictionary are the following:

    - ``epochs`` (mandatory if training): number of epochs to train for; one epoch is one iteration over all samples.
    - ``optimization`` (mandatory if training): subdictionary containing types and extra parameters required for
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
            # turn on tensorboardX logging for real-time monitoring
            "use_tbx": true,
            # optimization parameters block
            "optimization": {
                # all types & params below provided by PyTorch
                "loss": {
                    "type": "torch.nn.CrossEntropyLoss"
                },
                "optimizer": {
                    "type": "torch.optim.SGD",
                    "params": [
                        {"name": "lr", "value": 0.1},
                        {"name": "momentum", "value": 0.9},
                        {"name": "weight_decay", "value": 1e-06},
                        {"name": "nesterov", "value": true}
                    ]
                },
                "scheduler": {
                    "type": "torch.optim.lr_scheduler.StepLR",
                    "params": [
                        {"name": "step_size", "value": 10},
                        {"name": "step_size", "value": 0.1}
                    ]
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
        tbx_root_dir: top-level tensorboard log/event output directory (located within 'save_dir').
        model: model to train; will be uploaded to target device(s) at runtime.
        config: full configuration dictionary of the session; will be incorporated into all saved checkpoints.
        devices: list of (cuda) device IDs to upload the model/tensors to; can be empty if only the CPU is available.
        monitor: specifies the name of the metric that should be monitored on the validation set for model improvement.

    TODO: move static utils to their related modules

    .. seealso::
        | :class:`thelper.train.trainers.ImageClassifTrainer`
        | :func:`thelper.train.utils.create_trainer`
    """

    def __init__(self, session_name, save_dir, model, loaders, config, ckptdata=None):
        """Receives the trainer configuration dictionary, parses it, and sets up the session."""
        if not model or not loaders or not config:
            raise AssertionError("missing input args")
        train_loader, valid_loader, test_loader = loaders
        if not (train_loader or valid_loader or test_loader):
            raise AssertionError("must provide at least one loader with available data")
        if "trainer" not in config or not config["trainer"]:
            raise AssertionError("config missing 'trainer' field")
        trainer_config = config["trainer"]
        packages_log_path = os.path.join(save_dir, "logs", "packages.log")
        with open(packages_log_path, "w") as fd:
            pkgs_list = thelper.utils.get_env_list()
            if pkgs_list:
                for pkg in pkgs_list:
                    fd.write("%s\n" % pkg)
            else:
                fd.write("<n/a>\n")
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
        if train_loader:
            if "epochs" not in trainer_config or not trainer_config["epochs"] or int(trainer_config["epochs"]) <= 0:
                raise AssertionError("bad trainer config epoch count")
            self.epochs = int(trainer_config["epochs"])
            # no need to load optimization stuff if not training (i.e. no train_loader)
            # loading optimization stuff later since model needs to be on correct device
            if "optimization" not in trainer_config or not trainer_config["optimization"]:
                raise AssertionError("trainer config missing 'optimization' field")
            self.optimization_config = trainer_config["optimization"]
        else:
            self.epochs = 1
            self.logger.info("no training data provided, will run a single epoch on valid/test data")
        self.save_freq = int(trainer_config["save_freq"]) if "save_freq" in trainer_config else 1
        if self.save_freq < 1:
            raise AssertionError("checkpoint save frequency should be integer great or equal to 1")
        self.save_raw = thelper.utils.str2bool(thelper.utils.get_key_def("save_raw", trainer_config, True))
        self.checkpoint_dir = os.path.join(save_dir, "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        self.use_tbx = False
        if "use_tbx" in trainer_config:
            self.use_tbx = thelper.utils.str2bool(trainer_config["use_tbx"])
        writer_paths = [None, None, None]
        if self.use_tbx:
            self.tbx_root_dir = os.path.join(save_dir, "output", self.name)
            if not os.path.exists(self.tbx_root_dir):
                os.makedirs(self.tbx_root_dir)
            timestr = time.strftime("%Y%m%d-%H%M%S")
            train_foldername = "train-%s-%s" % (platform.node(), timestr)
            valid_foldername = "valid-%s-%s" % (platform.node(), timestr)
            test_foldername = "test-%s-%s" % (platform.node(), timestr)
            foldernames = [train_foldername, valid_foldername, test_foldername]
            for idx, (loader, foldername) in enumerate(zip(loaders, foldernames)):
                if loader:
                    tbx_dir = os.path.join(self.tbx_root_dir, foldername)
                    if os.path.exists(tbx_dir):
                        raise AssertionError("tbx session paths should be unique")
                    writer_paths[idx] = tbx_dir
            import tensorboardX
            self.tbx = tensorboardX
            self.logger.debug("tensorboard init : tensorboard --logdir %s --port <your_port>" % self.tbx_root_dir)
        self.train_writer_path, self.valid_writer_path, self.test_writer_path = writer_paths
        self.model = model
        self.config = config
        devices_str = None
        if "device" in trainer_config:
            devices_str = trainer_config["device"]
        elif "train_device" in trainer_config:  # for backw compat only
            devices_str = trainer_config["train_device"]
        self.devices = self._load_devices(devices_str)
        if "loss" in trainer_config:  # warning for older configs only
            self.logger.warning("trainer config has 'loss' field, but it should now be moved inside the 'optimization' field")
        if "metrics" in trainer_config and "base_metrics" in trainer_config:
            raise AssertionError("trainer config should have only one of 'metrics' and 'base_metrics'")
        if ("metrics" in trainer_config and trainer_config["metrics"]) or \
           ("base_metrics" in trainer_config and trainer_config["base_metrics"]):
            self.logger.debug("loading base metrics")
            if "metrics" in trainer_config:
                metrics = thelper.optim.create_metrics(trainer_config["metrics"])
            else:
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
        self.monitor, self.monitor_best = None, None
        if "monitor" in trainer_config and trainer_config["monitor"]:
            self.monitor = trainer_config["monitor"]
            if self.monitor not in self.train_metrics:
                raise AssertionError("monitored metric with name '%s' should be declared in config 'metrics' field" % self.monitor)
            self.monitor_goal = self.train_metrics[self.monitor].goal()
            if self.monitor_goal == thelper.optim.Metric.minimize:
                self.monitor_best = thelper.optim.Metric.maximize
            elif self.monitor_goal == thelper.optim.Metric.maximize:
                self.monitor_best = thelper.optim.Metric.minimize
            else:
                raise AssertionError("monitored metric does not return proper optimization goal")
        if ckptdata is not None:
            self.monitor_best = ckptdata["monitor_best"]
            self.optimizer_state = ckptdata["optimizer"]
            self.current_iter = ckptdata["iter"] if "iter" in ckptdata else 0
            self.current_epoch = ckptdata["epoch"]
            self.outputs = ckptdata["outputs"]
        else:
            self.optimizer_state = None
            self.current_iter = 0
            self.current_epoch = 0
            self.outputs = {}

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
    def _upload_tensor(tensor, dev):
        """Uploads a tensor to a specific device."""
        if isinstance(dev, list):
            if len(dev) == 0:
                return tensor.cpu()
            else:
                return tensor.cuda(dev[0])
        else:
            return tensor.to(dev)

    def _load_optimization(self, model):
        """Instantiates and returns all optimization objects required for training the model."""
        config = self.optimization_config  # for abbrev only
        if not isinstance(config, dict):
            raise AssertionError("config should be provided as a dictionary")
        if self.train_loader is None or not self.train_loader:
            raise AssertionError("optimization only useful if training data is available")
        if "loss" not in config or not config["loss"]:
            raise AssertionError("optimization config missing 'loss' field\n"
                                 "(is it still located in 'trainer'? just move it to 'optimization'!)")
        loss = self._load_loss(config["loss"])
        if "optimizer" not in config or not config["optimizer"]:
            raise AssertionError("optimization config missing 'optimizer' field")
        optimizer = self._load_optimizer(config["optimizer"], model)
        scheduler, scheduler_step_metric = None, None
        if "scheduler" in config and config["scheduler"]:
            scheduler, scheduler_step_metric = self._load_scheduler(config["scheduler"], optimizer)
        return loss, optimizer, scheduler, scheduler_step_metric

    def _load_loss(self, config):
        """Instantiates and returns the loss function to use for training.

        This function supports an extra special parameter if the task is related to classification : ``weight_classes``.
        If this parameter is found and positive (boolean), then the loss function will apply weights to the computed
        loss of each class. The strategy used to compute these weights is related to the one in
        :class:`thelper.data.samplers.WeightedSubsetRandomSampler`. The exact parameters that are expected for class
        reweighting are the following:

        - ``weight_param_name`` (optional, default="weight"): name of the loss constructor parameter that expects the weight list.
        - ``weight_param_pass_tensor`` (optional, default=True): specifies whether the weights should be passed as a tensor or list.
        - ``weight_distribution`` (mandatory): the dictionary of weights assigned to each class, or the rebalancing strategy to use.
        - ``weight_max`` (optional, default=inf): the maximum weight that can be assigned to a class.
        - ``weight_min`` (optional, default=0): the minimum weight that can be assigned to a class.
        - ``weight_norm`` (optional, default=True): specifies whether the weights should be normalized or not.

        """
        # todo: add flag to toggle loss comp in validation?
        self.logger.debug("loading loss")
        if not isinstance(config, dict):
            raise AssertionError("config should be provided as a dictionary")
        if "type" not in config or not config["type"]:
            raise AssertionError("loss config missing 'type' field")
        loss_type = thelper.utils.import_class(config["type"])
        loss_params = thelper.utils.get_key_def("params", config, {})
        if isinstance(self.model.task, thelper.tasks.Classification) and "weight_classes" in config:
            weight_classes = thelper.utils.str2bool(config["weight_classes"])
            if weight_classes:
                if self.train_loader is None or not self.train_loader:
                    raise AssertionError("cannot get class sizes, no training data available")
                samples_map = self.model.task.get_class_sample_map(self.train_loader.dataset.samples)
                weight_param_name = "weight"
                if "weight_param_name" in config:
                    weight_param_name = config["weight_param_name"]
                weight_param_pass_tensor = True
                if "weight_param_pass_tensor" in config:
                    weight_param_pass_tensor = thelper.utils.str2bool(config["weight_param_pass_tensor"])
                if "weight_distribution" not in config:
                    raise AssertionError("missing 'weight_distribution' field in loss config")
                weight_distrib = config["weight_distribution"]
                if isinstance(weight_distrib, dict):
                    for label, weight in weight_distrib.items():
                        if label not in samples_map:
                            raise AssertionError("weight distribution label '%s' not in dataset class list" % label)
                        if not isinstance(weight, float):
                            raise AssertionError("expected weight distrib map to provide weights as floats directly")
                elif isinstance(weight_distrib, str):
                    weight_max = float("inf")
                    if "weight_max" in config:
                        weight_max = float(config["weight_max"])
                    weight_min = 0
                    if "weight_min" in config:
                        weight_min = float(config["weight_min"])
                    weight_norm = True
                    if "weight_norm" in config:
                        weight_norm = thelper.utils.str2bool(config["weight_norm"])
                    weight_distrib = thelper.data.utils.get_class_weights(samples_map, weight_distrib, invmax=True,
                                                                          maxw=weight_max, minw=weight_min, norm=weight_norm)
                else:
                    raise AssertionError("unexpected weight distribution strategy (should be map or string)")
                weight_list = [weight_distrib[label] if label in weight_distrib else 1.0 for label in samples_map]
                if weight_param_pass_tensor:
                    loss_params[weight_param_name] = self._upload_tensor(torch.FloatTensor(weight_list), dev=self.devices)
                else:
                    loss_params[weight_param_name] = weight_list
        loss = loss_type(**loss_params)
        return loss

    def _load_optimizer(self, config, model):
        """Instantiates and returns the optimizer to use for training.

        By default, the optimizer will be instantiated with the model parameters given as the first argument of its constructor.
        All supplementary arguments are expected to be handed in through the configuration via a dictionary named 'params'.
        """
        self.logger.debug("loading optimizer")
        if not isinstance(config, dict):
            raise AssertionError("config should be provided as a dictionary")
        if "type" not in config or not config["type"]:
            raise AssertionError("optimizer config missing 'type' field")
        optimizer_type = thelper.utils.import_class(config["type"])
        optimizer_params = thelper.utils.get_key_def("params", config, {})
        optimizer = optimizer_type(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)
        return optimizer

    def _load_scheduler(self, config, optimizer):
        """Instantiates and returns the learning rate scheduler to use for training.

        All arguments are expected to be handed in through the configuration via a dictionary named 'params'.
        """
        self.logger.debug("loading scheduler")
        if not isinstance(config, dict):
            raise AssertionError("config should be provided as a dictionary")
        if "type" not in config or not config["type"]:
            raise AssertionError("scheduler config missing 'type' field")
        scheduler_type = thelper.utils.import_class(config["type"])
        scheduler_params = thelper.utils.get_key_def("params", config, {})
        scheduler = scheduler_type(optimizer, **scheduler_params)
        scheduler_step_metric = None
        if "step_metric" in config:
            scheduler_step_metric = config["step_metric"]
        return scheduler, scheduler_step_metric

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
                    raise AssertionError("cuda device '%s' out of range (detected devices = %s)" % (dev_str, str(available_cuda_devices)))
                if devices is None:
                    devices = [cuda_dev_idx]
                else:
                    devices.append(cuda_dev_idx)
            return devices
        else:
            return available_cuda_devices

    def train(self):
        """Starts the training process.

        This function will train the model until the required number of epochs is reached, and then evaluate it on the test data. The
        setup of loggers, tensorboard writers is done here, so is model improvement tracking via monitored metrics. However, the code
        related to loss computation and backpropagation is implemented in a derived class via :func:`thelper.train.trainers.Trainer._train_epoch`.
        """
        if not self.train_loader:
            raise AssertionError("missing training data, invalid loader!")
        self.logger.debug("uploading model to '%s'..." % str(self.devices))
        model = self._upload_model(self.model, self.devices)
        loss, optimizer, scheduler, scheduler_step_metric = self._load_optimization(model)
        if self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
            self.optimizer_state = None
        self.logger.debug("loss: %s" % str(loss))
        self.logger.debug("optimizer: %s" % str(optimizer))
        start_epoch = self.current_epoch + 1
        latest_loss = math.inf
        train_writer, valid_writer, test_writer = None, None, None
        for epoch in range(start_epoch, self.epochs + 1):
            self.logger.info("launching training epoch %d for '%s' (dev=%s)" % (epoch, self.name, str(self.devices)))
            if scheduler:
                # epoch idx is 1-based, scheduler expects 0-based
                if scheduler_step_metric:
                    if scheduler_step_metric == "loss":
                        # todo: use validation loss instead? more stable?
                        scheduler.step(metrics=latest_loss, epoch=(epoch - 1))
                    else:
                        if self.valid_loader and scheduler_step_metric in self.valid_metrics:
                            scheduler.step(metrics=self.valid_metrics[scheduler_step_metric].eval(), epoch=(epoch - 1))
                        elif self.train_loader and scheduler_step_metric in self.train_metrics:
                            scheduler.step(metrics=self.train_metrics[scheduler_step_metric].eval(), epoch=(epoch - 1))
                        else:
                            raise AssertionError("cannot use metric '%s' for scheduler step" % scheduler_step_metric)
                else:
                    scheduler.step(epoch=(epoch - 1))  # epoch idx is 1-based, scheduler expects 0-based
            self.logger.debug("learning rate at %.8f" % self._get_lr(optimizer))
            model.train()
            if self.use_tbx and not train_writer:
                train_writer = self.tbx.SummaryWriter(log_dir=self.train_writer_path, comment=self.name)
                setattr(train_writer, "path", self.train_writer_path)  # for external usage, if needed
                setattr(train_writer, "prefix", "train")  # to prefix data added to tbx logs (if needed)
            for metric in self.train_metrics.values():
                if hasattr(metric, "set_max_accum") and callable(metric.set_max_accum):
                    metric.set_max_accum(len(self.train_loader))  # used to make scalar metric evals smoother between epochs
                if metric.needs_reset():
                    metric.reset()  # if a metric needs to be reset between two epochs, do it here
            latest_loss, self.current_iter = self._train_epoch(model, epoch, self.current_iter, self.devices, loss, optimizer,
                                                               self.train_loader, self.train_metrics, train_writer)
            self._write_epoch_metrics(epoch, self.train_metrics, train_writer)
            train_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.train_metrics.items()}
            result = {"train/loss": latest_loss, "train/metrics": train_metric_vals}
            monitor_type_key = "train/metrics"  # if we cannot run validation, will monitor progression on training metrics
            if self.valid_loader:
                model.eval()
                if self.use_tbx and not valid_writer:
                    valid_writer = self.tbx.SummaryWriter(log_dir=self.valid_writer_path, comment=self.name)
                    setattr(valid_writer, "path", self.valid_writer_path)  # for external usage, if needed
                    setattr(valid_writer, "prefix", "valid")  # to prefix data added to tbx logs (if needed)
                for metric in self.valid_metrics.values():
                    metric.reset()  # force reset here, we always evaluate from a clean state
                self._eval_epoch(model, epoch, self.current_iter, self.devices,
                                 self.valid_loader, self.valid_metrics, valid_writer)
                self._write_epoch_metrics(epoch, self.valid_metrics, valid_writer)
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
                        new_best = True
                if not isinstance(value, dict):
                    self.logger.debug(" epoch {} result =>  {}: {}".format(epoch, str(key), value))
                else:
                    for subkey, subvalue in value.items():
                        self.logger.debug(" epoch {} result =>  {}:{}: {}".format(epoch, str(key), str(subkey), subvalue))
            if self.monitor is not None:
                if monitor_val is None:
                    raise AssertionError("training/validation did not produce required monitoring variable '%s'" % self.monitor)
                self.logger.info("epoch %d, %s = %s  (best = %s)" % (epoch, self.monitor, monitor_val, self.monitor_best))
            self.outputs[epoch] = result
            if new_best:
                self.logger.info("(new best checkpoint)")
            if new_best or (epoch % self.save_freq) == 0:
                self.logger.info("saving checkpoint @ epoch %d" % epoch)
                self._save(epoch, optimizer, save_best=new_best)
        self.logger.info("training for session '%s' done" % self.name)
        if self.test_loader:
            # reload 'best' model checkpoint on cpu (will remap to current device setup)
            filename_best = os.path.join(self.checkpoint_dir, "ckpt.best.pth")
            self.logger.info("loading best model & initializing final test run")
            ckptdata = thelper.utils.load_checkpoint(filename_best, map_location="cpu")
            if self.config != ckptdata["config"]:  # todo: dig into members and check only critical ones
                raise AssertionError("could not load compatible best checkpoint to run test eval")
            if self.save_raw:
                self.model.load_state_dict(ckptdata["model"])
            else:
                self.model = ckptdata["model"]
            best_epoch = ckptdata["epoch"]
            best_iter = ckptdata["iter"] if "iter" in ckptdata else None
            model = self._upload_model(self.model, self.devices)
            model.eval()
            if self.use_tbx and not test_writer:
                test_writer = self.tbx.SummaryWriter(log_dir=self.test_writer_path, comment=self.name)
                setattr(test_writer, "path", self.test_writer_path)  # for external usage, if needed
                setattr(test_writer, "prefix", "test")  # to prefix data added to tbx logs (if needed)
            for metric in self.test_metrics.values():
                metric.reset()  # force reset here, we always evaluate from a clean state
            self._eval_epoch(model, best_epoch, best_iter, self.devices,
                             self.test_loader, self.test_metrics, test_writer)
            self._write_epoch_metrics(best_epoch, self.test_metrics, test_writer)
            test_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.test_metrics.items()}
            self.outputs[best_epoch] = {**ckptdata["outputs"], "test/metrics": test_metric_vals}
            for key, value in self.outputs[best_epoch].items():
                if not isinstance(value, dict):
                    self.logger.debug(" final result =>  {}: {}".format(str(key), value))
                else:
                    for subkey, subvalue in value.items():
                        self.logger.debug(" final result =>  {}:{}: {}".format(str(key), str(subkey), subvalue))
            self.logger.info("evaluation for session '%s' done" % self.name)
        return self.outputs

    def eval(self):
        """Starts the evaluation process.

        This function will evaluate the model using the test data (or the validation data, if no test data is available), and return the
        results. Note that the code related to the forwarding of samples inside the model itself is implemented in a derived class via
        :func:`thelper.train.trainers.Trainer._train_epoch`.
        """
        if not self.valid_loader and not self.test_loader:
            raise AssertionError("missing validation/test data, invalid loaders!")
        self.logger.debug("uploading model to '%s'..." % str(self.devices))
        model = self._upload_model(self.model, self.devices)
        model.eval()
        result = {}
        valid_writer, test_writer = None, None
        if self.test_loader:
            if self.use_tbx and not test_writer:
                test_writer = self.tbx.SummaryWriter(log_dir=self.test_writer_path, comment=self.name)
                setattr(test_writer, "path", self.test_writer_path)  # for external usage, if needed
                setattr(test_writer, "prefix", "test")  # to prefix data added to tbx logs (if needed)
            for metric in self.test_metrics.values():
                metric.reset()  # force reset here, we always evaluate from a clean state
            self._eval_epoch(model, self.current_epoch, self.current_iter, self.devices,
                             self.test_loader, self.test_metrics, test_writer)
            self._write_epoch_metrics(self.current_epoch, self.test_metrics, test_writer)
            test_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.test_metrics.items()}
            result = {**result, **test_metric_vals}
        elif self.valid_loader:
            if self.use_tbx and not valid_writer:
                valid_writer = self.tbx.SummaryWriter(log_dir=self.valid_writer_path, comment=self.name)
                setattr(valid_writer, "path", self.valid_writer_path)  # for external usage, if needed
                setattr(valid_writer, "prefix", "valid")  # to prefix data added to tbx logs (if needed)
            for metric in self.valid_metrics.values():
                metric.reset()  # force reset here, we always evaluate from a clean state
            self._eval_epoch(model, self.current_epoch, self.current_iter, self.devices,
                             self.valid_loader, self.valid_metrics, valid_writer)
            self._write_epoch_metrics(self.current_epoch, self.valid_metrics, valid_writer)
            valid_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.valid_metrics.items()}
            result = {**result, **valid_metric_vals}
        for key, value in result.items():
            if not isinstance(value, dict):
                self.logger.debug(" final result =>  {}: {}".format(str(key), value))
            else:
                for subkey, subvalue in value.items():
                    self.logger.debug(" final result =>  {}:{}: {}".format(str(key), str(subkey), subvalue))
        self.outputs[self.current_epoch] = result
        self.logger.info("evaluation for session '%s' done" % self.name)
        return result

    @abstractmethod
    def _train_epoch(self, model, epoch, iter, dev, loss, optimizer, loader, metrics, writer=None):
        """Trains the model for a single epoch using the provided objects.

        Args:
            model: the model to train that is already uploaded to the target device(s).
            epoch: the index of the epoch we are training for.
            iter: the index of the iteration at the start of the current epoch.
            dev: the target device that tensors should be uploaded to.
            loss: the loss function used to evaluate model fidelity.
            optimizer: the optimizer used for back propagation.
            loader: the data loader used to get transformed training samples.
            metrics: the list of metrics to evaluate after every iteration.
            writer: the writer used to store tbx events/messages/metrics.
        """
        raise NotImplementedError

    @abstractmethod
    def _eval_epoch(self, model, epoch, iter, dev, loader, metrics, writer=None):
        """Evaluates the model using the provided objects.

        Args:
            model: the model to evaluate that is already uploaded to the target device(s).
            epoch: the index of the epoch we are evaluating for.
            iter: the index of the iteration at the start of the current epoch.
            dev: the target device that tensors should be uploaded to.
            loader: the data loader used to get transformed valid/test samples.
            metrics: the list of metrics to evaluate after every iteration.
            writer: the writer used to store tbx events/messages/metrics.
        """
        raise NotImplementedError

    def _get_lr(self, optimizer):
        """Returns the optimizer's learning rate, or 0 if not found."""
        for param_group in optimizer.param_groups:
            if "lr" in param_group:
                return param_group["lr"]
        return 0

    def _write_epoch_metrics(self, epoch, metrics, writer):
        """Writes the cumulative evaluation result of all metrics using a specific writer."""
        if not self.use_tbx or writer is None:
            return
        self.logger.debug("writing epoch metrics")
        for metric_name, metric in metrics.items():
            if metric.is_scalar():
                writer.add_scalar("epoch/%s" % metric_name, metric.eval(), epoch)
            if hasattr(metric, "render") and callable(metric.render):
                img = metric.render()
                if img is not None:
                    writer.add_image(writer.prefix + "/%s" % metric_name, img, epoch)
                    raw_filename = "%s-%s-%04d.png" % (writer.prefix, metric_name, epoch)
                    raw_filepath = os.path.join(writer.path, raw_filename)
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
                raw_filename = "%s-%s-%04d.txt" % (writer.prefix, metric_name, epoch)
                raw_filepath = os.path.join(writer.path, raw_filename)
                with open(raw_filepath, "w") as fd:
                    fd.write(txt)

    def _save(self, epoch, optimizer, save_best=False):
        """Saves a session checkpoint containing all the information required to resume training."""
        # logically, this should only be called during training (i.e. with a valid optimizer)
        log_stamp = thelper.utils.get_log_stamp()
        curr_state = {
            "name": self.name,
            "epoch": epoch,
            "iter": self.current_iter,
            "source": log_stamp,
            "sha1": thelper.utils.get_git_stamp(),
            "version": thelper.__version__,
            "task": str(self.model.task) if self.save_raw else self.model.task,
            "outputs": self.outputs[epoch],
            # we save model type/params here in case those are not in the current config
            "model": self.model.state_dict() if self.save_raw else self.model,
            "model_type": self.model.get_name(),
            "model_params": self.model.config if self.model.config else {},
            "optimizer": optimizer.state_dict(),
            "monitor_best": self.monitor_best,
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


class ImageClassifTrainer(Trainer):
    """Trainer interface specialized for image classification.

    This class implements the abstract functions of :class:`thelper.train.trainers.Trainer` required to train/evaluate
    a model for image classification or recognition. It also provides a utility function for fetching i/o packets
    (images, class labels) from a sample, and that converts those into tensors for forwarding and loss estimation.

    .. seealso::
        | :class:`thelper.train.trainers.Trainer`
    """

    def __init__(self, session_name, save_dir, model, loaders, config, ckptdata=None):
        """Receives session parameters, parses image/label keys from task object, and sets up metrics."""
        super().__init__(session_name, save_dir, model, loaders, config, ckptdata=ckptdata)
        if not isinstance(self.model.task, thelper.tasks.Classification):
            raise AssertionError("expected task to be classification")
        self.input_key = self.model.task.get_input_key()
        self.label_key = self.model.task.get_gt_key()
        self.class_names = self.model.task.get_class_names()
        self.meta_keys = self.model.task.get_meta_keys()
        self.class_idxs_map = self.model.task.get_class_idxs_map()
        metrics = list(self.train_metrics.values()) + list(self.valid_metrics.values()) + list(self.test_metrics.values())
        for metric in metrics:  # check all metrics for classification-specific attributes, and set them
            if hasattr(metric, "set_class_names") and callable(metric.set_class_names):
                metric.set_class_names(self.class_names)
        self.warned_no_shuffling_augments = False

    def _to_tensor(self, sample):
        """Fetches and returns tensors of input images and class labels from a batched sample dictionary."""
        if not isinstance(sample, dict):
            raise AssertionError("trainer expects samples to come in dicts for key-based usage")
        if self.input_key not in sample:
            raise AssertionError("could not find input key '%s' in sample dict" % self.input_key)
        input = sample[self.input_key]
        if isinstance(input, list):
            for idx in range(len(input)):
                input[idx] = torch.FloatTensor(input[idx])
        else:
            input = torch.FloatTensor(input)
        label_idx = None
        if self.label_key in sample:
            label = sample[self.label_key]
            label_idx = []
            for class_name in label:
                if isinstance(class_name, (int, torch.Tensor)):
                    if isinstance(class_name, torch.Tensor):
                        if torch.numel(class_name) != 1:
                            raise AssertionError("unexpected label name type, got vector")
                        class_name = class_name.item()
                    # dataset must already be using indices, we will forgive this...
                    if class_name < 0 or class_name >= len(self.class_names):
                        raise AssertionError("class name given as out-of-range index (%d) for class list" % class_name)
                    class_name = self.class_names[class_name]
                elif not isinstance(class_name, str):
                    raise AssertionError("expected label to be in str format (task will convert to proper index)")
                if class_name not in self.class_names:
                    raise AssertionError("got unexpected label '%s' for a sample (unknown class)" % class_name)
                label_idx.append(self.class_idxs_map[class_name])
            label_idx = torch.LongTensor(label_idx)
        return input, label_idx

    def _train_epoch(self, model, epoch, iter, dev, loss, optimizer, loader, metrics, writer=None):
        """Trains the model for a single epoch using the provided objects.

        Args:
            model: the model to train that is already uploaded to the target device(s).
            epoch: the index of the epoch we are training for.
            iter: the index of the iteration at the start of the current epoch.
            dev: the target device that tensors should be uploaded to.
            loss: the loss function used to evaluate model fidelity.
            optimizer: the optimizer used for back propagation.
            loader: the data loader used to get transformed training samples.
            metrics: the list of metrics to evaluate after every iteration.
            writer: the writer used to store tbx events/messages/metrics.
        """
        if not optimizer:
            raise AssertionError("missing optimizer")
        if not loader:
            raise AssertionError("no available data to load")
        if not isinstance(metrics, dict):
            raise AssertionError("expect metrics as dict object")
        epoch_loss = 0
        epoch_size = len(loader)
        self.logger.debug("fetching data loader samples...")
        for sample_idx, sample in enumerate(loader):
            input, label = self._to_tensor(sample)
            optimizer.zero_grad()
            if label is None:
                raise AssertionError("groundtruth required when training a model")
            label = self._upload_tensor(label, dev)
            if isinstance(input, list):  # training samples got augmented, we need to backprop in multiple steps
                if not input:
                    raise AssertionError("cannot train with empty post-augment sample lists")
                if not self.warned_no_shuffling_augments:
                    self.logger.warning("using training augmentation without global shuffling, gradient steps might be affected")
                    self.warned_no_shuffling_augments = True
                iter_loss = None
                iter_pred = None
                augs_count = len(input)
                for input_idx in range(augs_count):
                    aug_pred = model(self._upload_tensor(input[input_idx], dev))
                    aug_loss = loss(aug_pred, label)
                    aug_loss.backward()
                    if iter_pred is None:
                        iter_loss = aug_loss.clone().detach()
                        iter_pred = aug_pred.clone().detach()
                    else:
                        iter_loss += aug_loss.detach()
                        iter_pred += aug_pred.detach()
                iter_loss /= augs_count
                iter_pred /= augs_count
            else:
                iter_pred = model(self._upload_tensor(input, dev))
                iter_loss = loss(iter_pred, label)
                iter_loss.backward()
            epoch_loss += iter_loss.item()
            optimizer.step()
            if iter is not None:
                iter += 1
            if metrics:
                meta = {key: sample[key] if key in sample else None for key in self.meta_keys}
                for metric in metrics.values():
                    metric.accumulate(iter_pred.cpu(), label.cpu(), meta=meta)
            if self.monitor is not None:
                monitor_output = "{}: {:.2f}".format(self.monitor, metrics[self.monitor].eval())
            else:
                monitor_output = "(not monitoring)"
            self.logger.info(
                "train epoch: {}   iter: {}   batch: {}/{} ({:.0f}%)   loss: {:.6f}   {}".format(
                    epoch,
                    iter,
                    sample_idx,
                    epoch_size,
                    (sample_idx / epoch_size) * 100.0,
                    iter_loss.item(),
                    monitor_output
                )
            )
            if writer and iter is not None:
                writer.add_scalar("iter/loss", iter_loss.item(), iter)
                writer.add_scalar("iter/lr", self._get_lr(optimizer), iter)
                for metric_name, metric in metrics.items():
                    if metric.is_scalar():  # only useful assuming that scalar metrics are smoothed...
                        writer.add_scalar("iter/%s" % metric_name, metric.eval(), iter)
        epoch_loss /= epoch_size
        if writer:
            writer.add_scalar("epoch/loss", epoch_loss, epoch)
            writer.add_scalar("epoch/lr", self._get_lr(optimizer), epoch)
        return epoch_loss, iter

    def _eval_epoch(self, model, epoch, iter, dev, loader, metrics, writer=None):
        """Evaluates the model using the provided objects.

        Args:
            model: the model to evaluate that is already uploaded to the target device(s).
            epoch: the index of the epoch we are evaluating for.
            iter: the index of the iteration at the start of the current epoch.
            dev: the target device that tensors should be uploaded to.
            loader: the data loader used to get transformed valid/test samples.
            metrics: the list of metrics to evaluate after every iteration.
            writer: the writer used to store tbx events/messages/metrics.
        """
        if not loader:
            raise AssertionError("no available data to load")
        with torch.no_grad():
            epoch_size = len(loader)
            self.logger.debug("fetching data loader samples...")
            for idx, sample in enumerate(loader):
                input, label = self._to_tensor(sample)
                if label is not None:
                    label = self._upload_tensor(label, dev)
                if isinstance(input, list):  # evaluation samples got augmented, we need to get the mean prediction
                    if not input:
                        raise AssertionError("cannot eval with empty post-augment sample lists")
                    preds = None
                    for input_idx in range(len(input)):
                        pred = model(self._upload_tensor(input[input_idx], dev))
                        if preds is None:
                            preds = torch.unsqueeze(pred.clone(), 0)
                        else:
                            preds = torch.cat((preds, torch.unsqueeze(pred, 0)), 0)
                    pred = torch.mean(preds, dim=0)
                else:
                    pred = model(self._upload_tensor(input, dev))
                if metrics:
                    if self.meta_keys:
                        meta = {key: sample[key] if key in sample else None for key in self.meta_keys}
                    else:
                        meta = None
                    for metric in metrics.values():
                        metric.accumulate(pred.cpu(), label.cpu() if label is not None else None, meta=meta)
                if self.monitor is not None:
                    monitor_output = "{}: {:.2f}".format(self.monitor, metrics[self.monitor].eval())
                else:
                    monitor_output = "(not monitoring)"
                self.logger.info(
                    "eval epoch: {}   batch: {}/{} ({:.0f}%)   {}".format(
                        epoch,
                        idx,
                        epoch_size,
                        (idx / epoch_size) * 100.0,
                        monitor_output
                    )
                )
