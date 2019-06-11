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

    def __init__(self, session_name, save_dir, model, task, loaders, config, ckptdata=None):
        """Receives session parameters, parses tensor/target keys from task object, and sets up metrics."""
        super().__init__(session_name, save_dir, model, task, loaders, config, ckptdata=ckptdata)
        if not isinstance(self.task, thelper.tasks.Regression):
            raise AssertionError("expected task to be regression")
        # @@@@@ todo: use target_min/target_max and other props from task?

    def _to_tensor(self, sample):
        """Fetches and returns tensors of inputs and targets from a batched sample dictionary."""
        if not isinstance(sample, dict):
            raise AssertionError("trainer expects samples to come in dicts for key-based usage")
        if self.task.input_key not in sample:
            raise AssertionError("could not find input key '%s' in sample dict" % self.task.input_key)
        input_val = sample[self.task.input_key]
        if isinstance(input_val, np.ndarray):
            input_val = torch.from_numpy(input_val)
        if not isinstance(input_val, torch.Tensor):
            raise AssertionError("unexpected input type; should be torch.Tensor")
        if self.task.input_shape is not None:
            if input_val.dim() != len(self.task.input_shape) + 1:
                raise AssertionError("expected input as Nx[shape] where N = batch size")
            if self.task.input_shape != input_val.shape[1:]:
                raise AssertionError("invalid input shape; got '%s', expected '%s'" % (input_val.shape[1:], self.task.input_shape))
        target = None
        if self.task.gt_key in sample:
            target = sample[self.task.gt_key]
            if isinstance(target, np.ndarray):
                if self.task.target_type is not None and target.dtype != self.task.target_type:
                    raise AssertionError("unexpected target type, should be %s" % str(self.task.target_type))
                target = torch.from_numpy(target)
            if not isinstance(target, torch.Tensor):
                raise AssertionError("unexpected target type; should be torch.Tensor")
            if self.task.target_shape is not None:
                if target.dim() != len(self.task.target_shape) + 1:
                    raise AssertionError("expected target as Nx[shape] where N = batch size")
                if self.task.target_shape != target.shape[1:]:
                    raise AssertionError("invalid target shape; got '%s', expected '%s'" % (target.shape[1:], self.task.target_shape))
        return input_val, target

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
            metrics: the list of metrics to evaluate after every iteration.
            monitor: name of the metric to update/monitor for improvements.
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
        for idx, sample in enumerate(loader):
            input_val, target = self._to_tensor(sample)
            # todo: add support to fraction samples that are too big for a single iteration
            # (e.g. when batching non-image data that would be too inefficient one sample at a time)
            if target is None:
                raise AssertionError("groundtruth required when training a model")
            if isinstance(input_val, list):
                raise AssertionError("missing regr trainer support for duped minibatches")  # todo
            optimizer.zero_grad()
            target = self._move_tensor(target, dev)
            iter_pred = model(self._move_tensor(input_val, dev))
            iter_loss = loss(iter_pred, target.float())
            iter_loss.backward()
            optimizer.step()
            if metrics:
                meta = {key: sample[key] if key in sample else None
                        for key in self.task.meta_keys} if self.task.meta_keys else None
                iter_pred_cpu = self._move_tensor(iter_pred, dev="cpu", detach=True)
                target_cpu = self._move_tensor(target, dev="cpu", detach=True)
                for metric in metrics.values():
                    metric.accumulate(iter_pred_cpu, target_cpu, meta=meta)
            if self.train_iter_callback is not None:
                self.train_iter_callback(sample=sample, task=self.task, pred=iter_pred,
                                         iter_idx=iter, max_iters=epoch_size,
                                         epoch_idx=epoch, max_epochs=self.epochs,
                                         **self.callback_kwargs)
            epoch_loss += iter_loss.item()
            monitor_output = ""
            if monitor is not None and monitor in metrics:
                monitor_output = "   {}: {:.2f}".format(monitor, metrics[monitor].eval())
            self.logger.info(
                "train epoch#{}  (iter#{})   batch: {}/{} ({:.0f}%)   loss: {:.6f}{}".format(
                    epoch,
                    iter,
                    idx + 1,
                    epoch_size,
                    ((idx + 1) / epoch_size) * 100.0,
                    iter_loss.item(),
                    monitor_output
                )
            )
            if writer:
                writer.add_scalar("iter/loss", iter_loss.item(), iter)
                for metric_name, metric in metrics.items():
                    if metric.is_scalar():  # only useful assuming that scalar metrics are smoothed...
                        writer.add_scalar("iter/%s" % metric_name, metric.eval(), iter)
            iter += 1
        epoch_loss /= epoch_size
        return epoch_loss, iter

    def eval_epoch(self, model, epoch, dev, loader, metrics, monitor=None, writer=None):
        """Evaluates the model using the provided objects.

        Args:
            model: the model to evaluate that is already uploaded to the target device(s).
            epoch: the epoch number we are evaluating for (0-based).
            dev: the target device that tensors should be uploaded to.
            loader: the data loader used to get transformed valid/test samples.
            metrics: the dictionary of metrics to update every iteration.
            monitor: name of the metric to update/monitor for improvements.
            writer: the writer used to store tbx events/messages/metrics.
        """
        if not loader:
            raise AssertionError("no available data to load")
        with torch.no_grad():
            epoch_size = len(loader)
            self.logger.debug("fetching data loader samples...")
            for idx, sample in enumerate(loader):
                if idx < self.skip_eval_iter:
                    continue  # skip until previous iter count (if set externally; no effect otherwise)
                input_val, target = self._to_tensor(sample)
                if isinstance(input_val, list):
                    raise AssertionError("missing regr trainer support for duped minibatches")  # todo
                pred = model(self._move_tensor(input_val, dev))
                if metrics:
                    meta = {key: sample[key] if key in sample else None
                            for key in self.task.meta_keys} if self.task.meta_keys else None
                    pred_cpu = self._move_tensor(pred, dev="cpu", detach=True)
                    target_cpu = self._move_tensor(target, dev="cpu", detach=True)
                    for metric in metrics.values():
                        metric.accumulate(pred_cpu, target_cpu if target is not None else None, meta=meta)
                if self.eval_iter_callback is not None:
                    self.eval_iter_callback(sample=sample, task=self.task, pred=pred,
                                            iter_idx=idx, max_iters=epoch_size,
                                            epoch_idx=epoch, max_epochs=self.epochs,
                                            **self.callback_kwargs)
                self.logger.info(
                    "eval epoch#{}   batch: {}/{} ({:.0f}%){}".format(
                        epoch,
                        idx + 1,
                        epoch_size,
                        ((idx + 1) / epoch_size) * 100.0,
                        "   {}: {:.2f}".format(monitor, metrics[monitor].eval()) if monitor is not None else ""
                    )
                )
