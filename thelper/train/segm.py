"""Segmentation trainer/evaluator implementation module."""
import logging

import torch
import torch.optim

import thelper.utils
from thelper.train.base import Trainer

logger = logging.getLogger(__name__)


class ImageSegmTrainer(Trainer):
    """Trainer interface specialized for image segmentation.

    This class implements the abstract functions of :class:`thelper.train.base.Trainer` required to train/evaluate
    a model for image segmentation (i.e. pixel-level classification/labeling). It also provides a utility function
    for fetching i/o packets (images, class labels) from a sample, and that converts those into tensors for forwarding
    and loss estimation.

    .. seealso::
        | :class:`thelper.train.base.Trainer`
    """

    def __init__(self, session_name, save_dir, model, task, loaders, config, ckptdata=None):
        """Receives session parameters, parses image/label keys from task object, and sets up metrics."""
        super().__init__(session_name, save_dir, model, task, loaders, config, ckptdata=ckptdata)
        if not isinstance(self.task, thelper.tasks.Segmentation):
            raise AssertionError("expected task to be segmentation")
        trainer_config = thelper.utils.get_key_def("params", config["trainer"], {})
        self.scale_preds = thelper.utils.get_key_def("scale_preds", trainer_config, default=False)
        metrics = list(self.train_metrics.values()) + list(self.valid_metrics.values()) + list(self.test_metrics.values())
        for metric in metrics:  # check all metrics for classification-specific attributes, and set them
            if hasattr(metric, "set_class_names") and callable(metric.set_class_names):
                metric.set_class_names(self.task.class_names)
        self.warned_no_shuffling_augments = False

    def _to_tensor(self, sample):
        """Fetches and returns tensors of input images and label maps from a batched sample dictionary."""
        if not isinstance(sample, dict):
            raise AssertionError("trainer expects samples to come in dicts for key-based usage")
        if self.task.input_key not in sample:
            raise AssertionError("could not find input key '%s' in sample dict" % self.task.input_key)
        input_val, label_map = sample[self.task.input_key], None
        if isinstance(input_val, list):
            if self.task.gt_key in sample and sample[self.task.gt_key] is not None:
                label_map = sample[self.task.gt_key]
                if not isinstance(label_map, list) or len(label_map) != len(input_val):
                    raise AssertionError("label_map should also be a list of the same length as input")
                for idx in range(len(input_val)):
                    input_val[idx], label_map[idx] = self._to_tensor({self.task.input_key: input_val[idx],
                                                                      self.task.gt_key: label_map[idx]})
            else:
                for idx in range(len(input_val)):
                    input_val[idx] = torch.FloatTensor(input_val[idx])
        else:
            input_val = torch.FloatTensor(input_val)
            if self.task.gt_key in sample and sample[self.task.gt_key] is not None:
                label_map = sample[self.task.gt_key]
                if isinstance(label_map, list):
                    raise AssertionError("unexpected label map type")
                label_map = torch.ByteTensor(label_map)
        return input_val, label_map

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
            input_val, label_map = self._to_tensor(sample)
            if label_map is None:
                raise AssertionError("groundtruth required when training a model")
            optimizer.zero_grad()
            if isinstance(input_val, list):
                # training samples got augmented, we need to backprop in multiple steps
                if not input_val:
                    raise AssertionError("cannot train with empty post-augment sample lists")
                if not isinstance(label_map, list) or len(label_map) != len(input_val):
                    raise AssertionError("label maps should also be provided via a list of the same length as the input_val")
                if not self.warned_no_shuffling_augments:
                    self.logger.warning("using training augmentation without global shuffling, gradient steps might be affected")
                    # see the docstring of thelper.transforms.operations.Duplicator for more information
                    self.warned_no_shuffling_augments = True
                iter_loss = None
                iter_pred = None
                augs_count = len(input_val)
                for aug_idx in range(augs_count):
                    aug_pred = model(self._move_tensor(input_val[aug_idx], dev))
                    if self.scale_preds:
                        aug_pred = torch.nn.functional.interpolate(aug_pred, size=input_val[aug_idx].shape[-2:], mode="bilinear")
                    aug_loss = loss(aug_pred, label_map[aug_idx].long())
                    aug_loss.backward()  # test backprop all at once? might not fit in memory...
                    if iter_pred is None:
                        iter_loss = aug_loss.clone().detach()
                        iter_pred = aug_pred.clone().detach()
                    else:
                        iter_loss += aug_loss.detach()
                        iter_pred = torch.cat((aug_pred.detach(), iter_pred), dim=0)
                iter_loss /= augs_count
                label_map = torch.cat(label_map, dim=0)
            else:
                iter_pred = model(self._move_tensor(input_val, dev))
                if self.scale_preds:
                    iter_pred = torch.nn.functional.interpolate(iter_pred, size=input_val.shape[-2:], mode="bilinear")
                # todo: find a more efficient way to compute loss w/ byte vals directly?
                iter_loss = loss(iter_pred, self._move_tensor(label_map, dev).long())
                iter_loss.backward()
            optimizer.step()
            if metrics:
                meta = {key: sample[key] if key in sample else None
                        for key in self.task.meta_keys} if self.task.meta_keys else None
                iter_pred_cpu = self._move_tensor(iter_pred, dev="cpu", detach=True)
                label_map_cpu = self._move_tensor(label_map, dev="cpu", detach=True)
                for metric in metrics.values():
                    metric.accumulate(iter_pred_cpu, label_map_cpu, meta=meta)
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
                input_val, label_map = self._to_tensor(sample)
                if isinstance(input_val, list):
                    # evaluation samples got augmented, we need to get the mean prediction
                    if not input_val:
                        raise AssertionError("cannot eval with empty post-augment sample lists")
                    if not isinstance(label_map, list) or len(label_map) != len(input_val):
                        raise AssertionError("label maps should also be provided via a list of the same length as the input_val")
                    # this might be costly for nothing, we could remove the check and assume user is not dumb
                    if any([not torch.eq(l, label_map[0]).all() for l in label_map]):
                        raise AssertionError("all label maps should be identical! (why do eval-time augment otherwise?)")
                    label_map = label_map[0]  # since all identical, just pick the first one and pretend its the only one
                    preds = None
                    for input_idx in range(len(input_val)):
                        pred = model(self._move_tensor(input_val[input_idx], dev))
                        if preds is None:
                            preds = torch.unsqueeze(pred.clone(), 0)
                        else:
                            preds = torch.cat((preds, torch.unsqueeze(pred, 0)), 0)
                    pred = torch.mean(preds, dim=0)
                else:
                    pred = model(self._move_tensor(input_val, dev))
                if self.scale_preds:
                    pred = torch.nn.functional.interpolate(pred, size=input_val.shape[-2:], mode="bilinear")
                if metrics:
                    meta = {key: sample[key] if key in sample else None
                            for key in self.task.meta_keys} if self.task.meta_keys else None
                    pred_cpu = self._move_tensor(pred, dev="cpu", detach=True)
                    label_map_cpu = self._move_tensor(label_map, dev="cpu", detach=True)
                    for metric in metrics.values():
                        metric.accumulate(pred_cpu, label_map_cpu if label_map is not None else None, meta=meta)
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
