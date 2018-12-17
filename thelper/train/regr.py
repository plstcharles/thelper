"""Regression trainer/evaluator implementation module."""
import logging

import numpy as np
import torch
import torch.optim

import thelper.utils
from thelper.train.base import Trainer

logger = logging.getLogger(__name__)


class RegressionTrainer(Trainer):
    """Trainer interface specialized for generic (n-dim) regression.

    This class implements the abstract functions of :class:`thelper.train.base.Trainer` required to train/evaluate
    a model for generic regression (i.e. n-dim target value prediction). It also provides a utility function
    for fetching i/o packets (input tensors, target values) from a sample, and that converts those into tensors for
    forwarding and loss estimation.

    .. seealso::
        | :class:`thelper.train.base.Trainer`
    """

    def __init__(self, session_name, save_dir, model, loaders, config, ckptdata=None):
        """Receives session parameters, parses tensor/target keys from task object, and sets up metrics."""
        super().__init__(session_name, save_dir, model, loaders, config, ckptdata=ckptdata)
        if not isinstance(self.model.task, thelper.tasks.Regression):
            raise AssertionError("expected task to be regression")
        self.input_key = self.model.task.get_input_key()
        self.target_key = self.model.task.get_gt_key()
        self.input_shape = self.model.task.get_input_shape()
        self.target_shape = self.model.task.get_target_shape()
        self.target_type = self.model.task.get_target_type()
        self.target_min = self.model.task.get_target_min()
        if isinstance(self.target_min, np.ndarray):
            self.target_min = torch.from_numpy(self.target_min)
        self.target_max = self.model.task.get_target_max()
        if isinstance(self.target_max, np.ndarray):
            self.target_max = torch.from_numpy(self.target_max)
        self.meta_keys = self.model.task.get_meta_keys()

    def _to_tensor(self, sample):
        """Fetches and returns tensors of inputs and targets from a batched sample dictionary."""
        if not isinstance(sample, dict):
            raise AssertionError("trainer expects samples to come in dicts for key-based usage")
        if self.input_key not in sample:
            raise AssertionError("could not find input key '%s' in sample dict" % self.input_key)
        input = sample[self.input_key]
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        if not isinstance(input, torch.Tensor):
            raise AssertionError("unexpected input type; should be torch.Tensor")
        if self.input_shape is not None:
            if input.dim() != len(self.input_shape) + 1:
                raise AssertionError("expected input as Nx[shape] where N = batch size")
            if self.input_shape != input.shape[1:]:
                raise AssertionError("invalid input shape; got '%s', expected '%s'" % (input.shape[1:], self.input_shape))
        target = None
        if self.target_key in sample:
            target = sample[self.target_key]
            if isinstance(target, np.ndarray):
                if self.target_type is not None and target.dtype != self.target_type:
                    raise AssertionError("unexpected target type, should be %s" % str(self.target_type))
                target = torch.from_numpy(target)
            if not isinstance(target, torch.Tensor):
                raise AssertionError("unexpected target type; should be torch.Tensor")
            if self.target_shape is not None:
                if target.dim() != len(self.target_shape) + 1:
                    raise AssertionError("expected target as Nx[shape] where N = batch size")
                if self.target_shape != target.shape[1:]:
                    raise AssertionError("invalid target shape; got '%s', expected '%s'" % (target.shape[1:], self.target_shape))
        return input, target

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
        if not loss:
            raise AssertionError("missing loss function")
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
            input, target = self._to_tensor(sample)
            # todo: add support to fraction samples that are too big for a single iteration
            # (e.g. when batching non-image data that would be too inefficient one sample at a time)
            optimizer.zero_grad()
            if target is None:
                raise AssertionError("groundtruth required when training a model")
            target = self._upload_tensor(target, dev)
            iter_pred = model(self._upload_tensor(input, dev))
            iter_loss = loss(iter_pred, target.float())
            iter_loss.backward()
            if metrics:
                meta = {key: sample[key] if key in sample else None for key in self.meta_keys}
                for metric in metrics.values():
                    metric.accumulate(iter_pred.detach().cpu(), target.detach().cpu(), meta=meta)
            epoch_loss += iter_loss.item()
            optimizer.step()
            if iter is not None:
                iter += 1
                monitor_output = ""
                if self.monitor is not None and self.monitor in metrics:
                    monitor_output = "   {}: {:.2f}".format(self.monitor, metrics[self.monitor].eval())
                self.logger.info(
                    "train epoch: {}   iter: {}   batch: {}/{} ({:.0f}%)   loss: {:.6f}{}".format(
                        epoch,
                        iter,
                        sample_idx,
                        epoch_size,
                        (sample_idx / epoch_size) * 100.0,
                        iter_loss.item(),
                        monitor_output
                    )
                )
                if writer:
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
                input, target = self._to_tensor(sample)
                if target is not None:
                    target = self._upload_tensor(target, dev)
                pred = model(self._upload_tensor(input, dev))
                if metrics:
                    if self.meta_keys:
                        meta = {key: sample[key] if key in sample else None for key in self.meta_keys}
                    else:
                        meta = None
                    for metric in metrics.values():
                        metric.accumulate(pred.cpu(), target.cpu() if target is not None else None, meta=meta)
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
