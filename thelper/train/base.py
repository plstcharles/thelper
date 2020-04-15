"""Model training/evaluation base interface module.

This module contains the interface required to train and/or evaluate a model based on different tasks. The trainers
based on this interface are instantiated in launched sessions based on configuration dictionaries.
"""
import functools
import logging
import math
from abc import abstractmethod
from typing import AnyStr, Optional

import torch

import thelper.optim
import thelper.tasks
from thelper.session.base import SessionRunner

logger = logging.getLogger(__name__)


class Trainer(SessionRunner):
    """Abstract trainer interface that defines basic session i/o and setup operations.

    This interface defines the general behavior of a training session which includes configuration parsing, tensorboard
    setup, metrics and goal setup, and loss/optimizer setup. It also provides utilities for uploading models and tensors
    on specific devices, and for saving the state of a session. This interface should be specialized for every task by
    implementing the ``train_epoch`` and ``eval_epoch`` functions in a derived class. For better support from visualization
    utilities, the derived class should also implement ``to_tensor``. See :class:`thelper.train.classif.ImageClassifTrainer`
    for a complete example.

    Any trainer derived from this class will alternate between training and validation epochs. It will also support
    post-training (final) evaluation using a separate test set. If requested, visualizations can be computed after
    the validation epoch during training (e.g. sample activation maps, or t-SNE plots). See :mod:`thelper.viz` for
    more information on these.

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
            # visualization block (optional)
            "viz": {
                # multiple visualization techniques can be toggled by name
                "tsne": {
                    # visualization parameters would be provided here
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
        config: session configuration dictionary holding all original settings, including trainer configuration.
        model: reference to the model being trained or used for evaluation/prediction.
        task: reference to the object used to specialize the model and that holds task meta-information.

    TODO: move static utils to their related modules

    .. seealso::
        | :class:`thelper.session.base.SessionRunner`
        | :class:`thelper.train.classif.ImageClassifTrainer`
        | :class:`thelper.train.segm.ImageSegmTrainer`
        | :class:`thelper.train.detect.ObjDetectTrainer`
        | :class:`thelper.train.regr.RegressionTrainer`
        | :func:`thelper.train.utils.create_trainer`
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
        super(Trainer, self).__init__(session_name, session_dir, model, task, loaders, config, ckptdata=ckptdata)

    def train(self):
        """Starts the training process.

        This function will train the model until the required number of epochs is reached, and then evaluate it
        on the test data. The setup of loggers, tensorboard writers is done here, so is model improvement tracking
        via monitored metrics. However, the code related to loss computation and back propagation is implemented in
        a derived class via :func:`thelper.train.base.Trainer.train_epoch`.
        """
        assert self.train_loader, "missing training data, invalid loader!"
        assert not isinstance(self.model, torch.jit.ScriptModule), "current impl cannot train model traces"  # TODO
        self.logger.debug(f"uploading model to '{str(self.devices)}'...")
        model = self._upload_model(self.model, self.devices)
        loss, optimizer, scheduler, scheduler_step_metric = self._load_optimization(model, self.devices)
        if optimizer is not None and self.optimizer_state is not None:
            optimizer.load_state_dict(self.optimizer_state)
            self.optimizer_state = None
        if scheduler is not None and self.scheduler_state is not None:
            scheduler.load_state_dict(self.scheduler_state)
            self.scheduler_state = None
        self.logger.info(f"loss: {str(loss)}")
        self.logger.info(f"optimizer: {str(optimizer)}")
        latest_loss = math.inf
        while self.current_epoch < self.epochs:
            self.writers["train"] = self._init_writer(self.writers["train"], self.output_paths["train"])
            self.logger.info(f"at epoch#{self.current_epoch} for '{self.name}' (dev={str(self.devices)})")
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
                        metric_anti_goal = thelper.optim.Metric.maximize \
                            if metric.goal == thelper.optim.Metric.minimize \
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
            self.logger.debug(f"learning rate at {thelper.optim.get_lr(optimizer):.8f}")
            self._set_rng_state(self.train_loader.seeds, self.current_epoch)
            model.train()
            if hasattr(self.train_loader, "set_epoch") and callable(self.train_loader.set_epoch):
                self.train_loader.set_epoch(self.current_epoch)
            latest_loss = self.train_epoch(model, self.current_epoch, self.devices, loss, optimizer,
                                           self.train_loader, self.train_metrics, self.output_paths["train"])
            self._write_metrics_data(self.current_epoch, self.train_metrics,
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
                self.eval_epoch(model, self.current_epoch, self.devices, self.valid_loader,
                                self.valid_metrics, self.output_paths["valid"])
                self._write_metrics_data(self.current_epoch, self.valid_metrics,
                                         self.writers["valid"], self.output_paths["valid"])
                valid_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.valid_metrics.items()
                                     if isinstance(metric, thelper.optim.metrics.Metric)}
                result = {**result, "valid/metrics": valid_metric_vals}
                monitor_type_key = "valid/metrics"  # since validation is available, use that to monitor progression
                uploader = functools.partial(self._move_tensor, dev=self.devices, detach=True)
                wrapped_loader = thelper.data.DataLoaderWrapper(self.valid_loader, uploader)
                for viz, kwargs in self.viz.items():
                    viz_data = thelper.viz.visualize(model, self.task, wrapped_loader, viz_type=viz, **kwargs)
                    self._write_data(viz_data, "epoch/", f"-{self.current_epoch:04d}", self.writers["valid"],
                                     self.output_paths["valid"], self.current_epoch)
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
                    self.logger.info(f" epoch#{self.current_epoch} result =>  {str(key)}: {value}")
                else:
                    for subkey, subvalue in value.items():
                        self.logger.info(f" epoch#{self.current_epoch} result =>  {str(key)}:{str(subkey)}: {subvalue}")
            if self.monitor is not None:
                assert monitor_val is not None, f"training/validation did not evaluate required metric '{self.monitor}'"
                if new_best:
                    best_str = "(new best value)"
                else:
                    best_str = f"(previous best = {self.monitor_best} @ epoch = {self.monitor_best_epoch})"
                self.logger.info(f"epoch {self.current_epoch}, monitored {self.monitor} = {monitor_val}  {best_str}")
            self.outputs[self.current_epoch] = result
            if new_best or (self.current_epoch % self.save_freq) == 0:
                self.logger.info(f"saving checkpoint @ epoch#{self.current_epoch}")
                self._save(self.current_epoch, self.current_iter, optimizer, scheduler, save_best=new_best)
            self.current_epoch += 1
        self.logger.info(f"training for session '{self.name}' done")
        return self.outputs

    def eval(self):
        """Starts the evaluation process.

        This function will evaluate the model using the test data (or the validation data, if no test data is available),
        and return the results. Note that the code related to the forwarding of samples inside the model itself is implemented
        in a derived class via :func:`thelper.train.base.Trainer.eval_epoch`.
        """
        assert self.valid_loader or self.test_loader, "missing validation/test data, invalid loaders!"
        self.logger.debug(f"uploading model to '{str(self.devices)}'...")
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
            self.eval_epoch(model, self.current_epoch, self.devices, self.test_loader,
                            self.test_metrics, self.output_paths["test"])
            self._write_metrics_data(self.current_epoch, self.test_metrics,
                                     self.writers["test"], self.output_paths["test"], use_suffix=False)
            test_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.test_metrics.items()
                                if isinstance(metric, thelper.optim.metrics.Metric)}
            result = {**result, **test_metric_vals}
            output_group = "test/metrics"
            uploader = functools.partial(self._move_tensor, dev=self.devices, detach=True)
            wrapped_loader = thelper.data.DataLoaderWrapper(self.test_loader, uploader)
            for viz, kwargs in self.viz.items():
                viz_data = thelper.viz.visualize(model, self.task, wrapped_loader, viz_type=viz, **kwargs)
                self._write_data(viz_data, "epoch/", "", self.writers["test"], self.output_paths["test"], self.current_epoch)
        elif self.valid_loader:
            self._set_rng_state(self.valid_loader.seeds, self.current_epoch)
            model.eval()
            self.writers["valid"] = self._init_writer(self.writers["valid"], self.output_paths["valid"])
            for metric in self.valid_metrics.values():
                metric.reset()  # force reset here, we always evaluate from a clean state
            if hasattr(self.valid_loader, "set_epoch") and callable(self.valid_loader.set_epoch):
                self.valid_loader.set_epoch(self.current_epoch)
            self.eval_epoch(model, self.current_epoch, self.devices, self.valid_loader,
                            self.valid_metrics, self.output_paths["valid"])
            self._write_metrics_data(self.current_epoch, self.valid_metrics,
                                     self.writers["valid"], self.output_paths["valid"], use_suffix=False)
            valid_metric_vals = {metric_name: metric.eval() for metric_name, metric in self.valid_metrics.items()
                                 if isinstance(metric, thelper.optim.metrics.Metric)}
            result = {**result, **valid_metric_vals}
            output_group = "valid/metrics"
            uploader = functools.partial(self._move_tensor, dev=self.devices, detach=True)
            wrapped_loader = thelper.data.DataLoaderWrapper(self.valid_loader, uploader)
            for viz, kwargs in self.viz.items():
                viz_data = thelper.viz.visualize(model, self.task, wrapped_loader, viz_type=viz, **kwargs)
                self._write_data(viz_data, "epoch/", "", self.writers["valid"], self.output_paths["valid"], self.current_epoch)
        for key, value in result.items():
            if not isinstance(value, dict):
                self.logger.info(f" final result =>  {str(key)}: {value}")
            else:
                for subkey, subvalue in value.items():
                    self.logger.info(f" final result =>  {str(key)}:{str(subkey)}: {subvalue}")
        if self.current_epoch not in self.outputs:
            # probably using an 'untrained model' (such as a FCN adapted from a classifier)
            self.outputs[self.current_epoch] = {}
        self.outputs[self.current_epoch][output_group] = result
        self.logger.info(f"evaluation for session '{self.name}' done")
        return self.outputs

    @abstractmethod
    def train_epoch(self, model, epoch, dev, loss, optimizer, loader, metrics, output_path):
        """Trains the model for a single epoch using the provided objects.

        Args:
            model: the model to train that is already uploaded to the target device(s).
            epoch: the epoch index we are training for (0-based).
            dev: the target device that tensors should be uploaded to.
            loss: the loss function used to evaluate model fidelity.
            optimizer: the optimizer used for back propagation.
            loader: the data loader used to get transformed training samples.
            metrics: the dictionary of metrics/consumers to update every iteration.
            output_path: directory where output files should be written, if necessary.
        """
        raise NotImplementedError

    @abstractmethod
    def eval_epoch(self, model, epoch, device, loader, metrics, output_path):
        """Evaluates the model using the provided objects.

        Args:
            model: the model to evaluate that is already uploaded to the target device(s).
            epoch: the epoch index we are training for (0-based).
            device: the target device that tensors should be uploaded to.
            loader: the data loader used to get transformed valid/test samples.
            metrics: the dictionary of metrics/consumers to update every iteration.
            output_path: directory where output files should be written, if necessary.
        """
        raise NotImplementedError
