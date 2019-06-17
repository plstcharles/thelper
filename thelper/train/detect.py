"""Object detection trainer/evaluator implementation module."""
import collections
import logging

import numpy as np
import torch
import torch.optim
import torchvision

import thelper.utils
from thelper.train.base import Trainer

logger = logging.getLogger(__name__)


class ObjDetectTrainer(Trainer):
    """Trainer interface specialized for object detection.

    This class implements the abstract functions of :class:`thelper.train.base.Trainer` required to train/evaluate
    a model for object detection (i.e. 2D bounding box regression). It also provides a utility function for fetching
    i/o packets (input images, bounding boxes) from a sample, and that converts those into tensors for forwarding
    and loss estimation.

    .. seealso::
        | :class:`thelper.train.base.Trainer`
    """

    def __init__(self, session_name, save_dir, model, task, loaders, config, ckptdata=None):
        """Receives session parameters, parses tensor/target keys from task object, and sets up metrics."""
        super().__init__(session_name, save_dir, model, task, loaders, config, ckptdata=ckptdata)
        if not isinstance(self.task, thelper.tasks.Detection):
            raise AssertionError("expected task to be object detection")

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
        assert input_val.dim() == 4, "input image stack should be 4-dim to be decomposed into list of images"
        # unpack input images into list (as required by torchvision preproc)
        input_val = [input_val[i] for i in range(input_val.shape[0])]
        bboxes = None
        if self.task.gt_key in sample:
            bboxes = sample[self.task.gt_key]
            if not isinstance(bboxes, list) or not all([isinstance(bset, list) for bset in bboxes]):
                raise AssertionError("bboxes should be provided as a list of lists (dims = batch x bboxes-per-image)")
            if not all([all([isinstance(box, thelper.data.BoundingBox) for box in bset]) for bset in bboxes]):
                raise AssertionError("bboxes should be provided as a thelper.data.BoundingBox-compat object")
            assert all([len(np.unique([b.image_id for b in bset if b.image_id is not None])) <= 1 for bset in bboxes]), \
                "some bboxes tied to a single image have different reference ids"
            # here, we follow the format used in torchvision (>=0.3) for forwarding targets to detection models
            # (see https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html for more info)
            bboxes = [{
                "boxes": torch.as_tensor([[*b.bbox] for b in bset], dtype=torch.float32),
                "labels": torch.as_tensor([b.class_id for b in bset], dtype=torch.int64),
                "image_id": torch.as_tensor([b.image_id for b in bset]),
                "area": torch.as_tensor([b.area for b in bset], dtype=torch.float32),
                "iscrowd": torch.as_tensor([b.iscrowd for b in bset], dtype=torch.int64),
                "refs": bset
            } for bset in bboxes]
        return input_val, bboxes

    def _train_epoch(self, model, epoch, iter, dev, loss, optimizer, loader, metrics, monitor=None, writer=None):
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
        assert loss is None, "current implementation assumes that loss is computed inside the model"
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
            images, targets = self._to_tensor(sample)
            if targets is None or any([not bset for bset in targets]):
                raise AssertionError("groundtruth required when training a model")
            optimizer.zero_grad()
            targets = self._move_tensor(targets, dev)
            images = self._move_tensor(images, dev)
            if isinstance(model, thelper.nn.utils.ExternalModule):
                model = model.model  # temporarily unwrap to simplify code below
            if isinstance(model, torchvision.models.detection.generalized_rcnn.GeneralizedRCNN):
                # unfortunately, the default generalized RCNN model forward does not return predictions while training...
                # loss_dict = model(images=images, targets=targets)  # we basically reimplement this call below
                original_image_sizes = [img.shape[-2:] for img in images]
                images, targets = model.transform(images, targets)
                features = model.backbone(images.tensors)
                if isinstance(features, torch.Tensor):
                    features = collections.OrderedDict([(0, features)])
                proposals, proposal_losses = model.rpn(images, features, targets)
                iter_pred, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
                iter_pred = model.transform.postprocess(iter_pred, images.image_sizes, original_image_sizes)
                iter_loss = sum(loss for loss in {**detector_losses, **proposal_losses}.values())
                iter_loss.backward()
            else:
                raise AssertionError("unknown/unhandled detection model type")
            optimizer.step()
            if metrics:
                meta = {key: sample[key] if key in sample else None
                        for key in self.task.meta_keys} if self.task.meta_keys else None
                iter_pred_cpu = self._move_tensor(iter_pred, dev="cpu", detach=True)
                targets_cpu = self._move_tensor(targets, dev="cpu", detach=True)
                for metric in metrics.values():
                    metric.accumulate(iter_pred_cpu, targets_cpu, meta=meta)
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

    def _eval_epoch(self, model, epoch, dev, loader, metrics, monitor=None, writer=None):
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
                images, targets = self._to_tensor(sample)
                pred = model(self._move_tensor(images, dev))
                if metrics:
                    meta = {key: sample[key] if key in sample else None
                            for key in self.task.meta_keys} if self.task.meta_keys else None
                    pred_cpu = self._move_tensor(pred, dev="cpu", detach=True)
                    targets_cpu = self._move_tensor(targets, dev="cpu", detach=True)
                    for metric in metrics.values():
                        metric.accumulate(pred_cpu, targets_cpu, meta=meta)
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
