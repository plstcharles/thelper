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

    def __init__(self, session_name, save_dir, model, loaders, config, ckptdata=None):
        """Receives session parameters, parses image/label keys from task object, and sets up metrics."""
        super().__init__(session_name, save_dir, model, loaders, config, ckptdata=ckptdata)
        if not isinstance(self.model.task, thelper.tasks.Segmentation):
            raise AssertionError("expected task to be segmentation")
        self.input_key = self.model.task.get_input_key()
        self.label_map_key = self.model.task.get_gt_key()
        self.class_names = self.model.task.get_class_names()
        self.meta_keys = self.model.task.get_meta_keys()
        self.class_idxs_map = self.model.task.get_class_idxs_map()
        self.dontcare_val = self.model.task.get_dontcare_val()
        metrics = list(self.train_metrics.values()) + list(self.valid_metrics.values()) + list(self.test_metrics.values())
        for metric in metrics:  # check all metrics for classification-specific attributes, and set them
            if hasattr(metric, "set_class_names") and callable(metric.set_class_names):
                metric.set_class_names(self.class_names)
        self.warned_no_shuffling_augments = False

    def _to_tensor(self, sample):
        """Fetches and returns tensors of input images and label maps from a batched sample dictionary."""
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
        label_map = None
        if self.label_map_key in sample:
            label_map = sample[self.label_map_key]
            if isinstance(label_map, list):
                for idx in range(len(label_map)):
                    label_map[idx] = torch.ByteTensor(label_map[idx])
            else:
                label_map = torch.ByteTensor(label_map)
        return input, label_map

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
            input, label_map = self._to_tensor(sample)
            optimizer.zero_grad()
            if label_map is None:
                raise AssertionError("groundtruth required when training a model")
            label_map = self._upload_tensor(label_map, dev)
            meta = None
            if metrics:
                meta = {key: sample[key] if key in sample else None for key in self.meta_keys}
            if isinstance(input, list):
                # training samples got augmented, we need to backprop in multiple steps
                # note: we do NOT assume all samples are registered; each input must have its own label map
                if not input:
                    raise AssertionError("cannot train with empty post-augment sample lists")
                if not isinstance(label_map, list) or len(input) != len(label_map):
                    raise AssertionError("unexpected augmented input image / label map list lengths")
                if not self.warned_no_shuffling_augments:
                    self.logger.warning("using training augmentation without global shuffling, gradient steps might be affected")
                    self.warned_no_shuffling_augments = True
                iter_loss = None
                iter_pred = None
                augs_count = len(input)
                for aug_idx in range(augs_count):
                    aug_pred = model(self._upload_tensor(input[aug_idx], dev))
                    aug_loss = loss(aug_pred, label_map[aug_idx].long())
                    aug_loss.backward()  # test backprop all at once? @@@
                    if iter_pred is None:
                        iter_loss = aug_loss.clone().detach()
                        iter_pred = aug_pred.clone().detach()
                    else:
                        iter_loss += aug_loss.detach()
                        iter_pred += aug_pred.detach()
                    if metrics:
                        for metric in metrics.values():
                            metric.accumulate(aug_pred.detach().cpu(), label_map[aug_idx].detach().cpu(), meta=meta)
                iter_loss /= augs_count
            else:
                iter_pred = model(self._upload_tensor(input, dev))
                # todo: find a more efficient way to compute loss w/ byte vals directly?
                iter_loss = loss(iter_pred, label_map.long())
                iter_loss.backward()
                if metrics:
                    for metric in metrics.values():
                        metric.accumulate(iter_pred.detach().cpu(), label_map.detach().cpu(), meta=meta)
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
                input, label_map = self._to_tensor(sample)
                if label_map is not None:
                    label_map = self._upload_tensor(label_map, dev)
                if isinstance(input, list):
                    # evaluation samples got augmented, we need to get the mean prediction
                    # note: we assume all samples are registered images with a common label map
                    # (this differs from the training loop!)
                    if not input:
                        raise AssertionError("cannot eval with empty post-augment sample lists")
                    if isinstance(label_map, list):
                        raise AssertionError("unexpected augmented input image / label map type")
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
                        metric.accumulate(pred.cpu(), label_map.cpu() if label_map is not None else None, meta=meta)
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
