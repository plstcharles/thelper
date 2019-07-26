"""Object detection trainer/evaluator implementation module."""
import collections
import logging
from typing import AnyStr  # noqa: F401

import numpy as np
import torch
import torch.optim
import torchvision

import thelper.typedefs as typ  # noqa: F401
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

    def __init__(self,
                 session_name,    # type: AnyStr
                 save_dir,        # type: AnyStr
                 model,           # type: thelper.typedefs.ModelType
                 task,            # type: thelper.tasks.Task
                 loaders,         # type: thelper.typedefs.MultiLoaderType
                 config,          # type: thelper.typedefs.ConfigDict
                 ckptdata=None    # type: typ.Optional[thelper.typedefs.CheckpointContentType]
                 ):
        """Receives session parameters, parses tensor/target keys from task object, and sets up metrics."""
        super().__init__(session_name, save_dir, model, task, loaders, config, ckptdata=ckptdata)
        assert isinstance(self.task, thelper.tasks.Detection), "expected task to be object detection"

    def _to_tensor(self, sample):
        """Fetches and returns tensors of inputs and targets from a batched sample dictionary."""
        assert isinstance(sample, dict), "trainer expects samples to come in dicts for key-based usage"
        assert self.task.input_key in sample, f"could not find input key '{self.task.input_key}' in sample dict"
        input_val = sample[self.task.input_key]
        if isinstance(input_val, np.ndarray):
            input_val = torch.from_numpy(input_val)
        assert isinstance(input_val, torch.Tensor), "unexpected input type; should be torch.Tensor"
        if self.task.input_shape is not None:
            assert input_val.dim() == len(self.task.input_shape) + 1, \
                "expected input as Nx[shape] where N = batch size"
            assert self.task.input_shape == input_val.shape[1:], \
                f"invalid input shape; got '{input_val.shape[1:]}', expected '{self.task.input_shape}'"
        assert input_val.dim() == 4, "input image stack should be 4-dim to be decomposed into list of images"
        # unpack input images into list (as required by torchvision preproc)
        input_val = [input_val[i] for i in range(input_val.shape[0])]
        bboxes = None
        if self.task.gt_key in sample:
            bboxes = sample[self.task.gt_key]
            assert isinstance(bboxes, list) and all([isinstance(bset, list) for bset in bboxes]), \
                "bboxes should be provided as a list of lists (dims = batch x bboxes-per-image)"
            assert all([all([isinstance(box, thelper.data.BoundingBox) for box in bset]) for bset in bboxes]), \
                "bboxes should be provided as a thelper.data.BoundingBox-compat object"
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

    def _from_tensor(self, bboxes, sample=None):
        """Fetches and returns a list of bbox objects from a model-specific representation."""
        # for now, we can only unpack torchvision-format bbox dictionary lists (everything else will throw)
        assert isinstance(bboxes, list), "input should be list since we do batch predictions"
        if all([isinstance(d, dict) and len(d) == 3 and
                all([k in ["boxes", "labels", "scores"] for k in d]) for d in bboxes]):
            outputs = []
            for batch_idx, d in enumerate(bboxes):
                boxes = d["boxes"].detach().cpu()
                labels = d["labels"].detach().cpu()
                scores = d["scores"].detach().cpu()
                assert boxes.shape[0] == labels.shape[0] and boxes.shape[0] == scores.shape[0], "mismatched tensor dims"
                curr_output = []
                for box_idx, box in enumerate(boxes):
                    if sample is not None and self.task.gt_key in sample and sample[self.task.gt_key][batch_idx]:
                        gt_box = sample[self.task.gt_key][batch_idx][0]  # use first gt box to get image-level props
                        out = thelper.data.BoundingBox(labels[box_idx].item(), box, confidence=scores[box_idx].item(),
                                                       image_id=gt_box.image_id, task=self.task)

                    elif sample is not None and "idx" in sample:
                        out = thelper.data.BoundingBox(labels[box_idx].item(), box, confidence=scores[box_idx].item(),
                                                       image_id=sample["idx"][batch_idx], task=self.task)
                    else:
                        out = thelper.data.BoundingBox(labels[box_idx].item(), box, confidence=scores[box_idx].item(),
                                                       task=self.task)
                    curr_output.append(out)
                outputs.append(curr_output)
            return outputs
        raise AssertionError("unrecognized packed bboxes vector format")

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
        assert loss is None, "current implementation assumes that loss is computed inside the model"
        assert optimizer is not None, "missing optimizer"
        assert loader, "no available data to load"
        assert isinstance(metrics, dict), "expect metrics as dict object"
        epoch_loss = 0
        epoch_size = len(loader)
        self.logger.debug("fetching data loader samples...")
        for idx, sample in enumerate(loader):
            images, targets = self._to_tensor(sample)
            assert targets is not None and not any([not bset for bset in targets]), \
                "groundtruth required when training a model"
            optimizer.zero_grad()
            targets_dev = self._move_tensor(targets, dev)
            images_dev = self._move_tensor(images, dev)
            if isinstance(model, thelper.nn.utils.ExternalModule):
                model = model.model  # temporarily unwrap to simplify code below
            assert isinstance(model, torchvision.models.detection.generalized_rcnn.GeneralizedRCNN), \
                "unknown/unhandled detection model type"  # user should probably implement their own trainer
            # unfortunately, the default generalized RCNN model forward does not return predictions while training...
            # loss_dict = model(images=images_dev, targets=targets)  # we basically reimplement this call below
            original_image_sizes = [img.shape[-2:] for img in images_dev]
            images_dev, targets_dev = model.transform(images_dev, targets_dev)
            features = model.backbone(images_dev.tensors)
            if isinstance(features, torch.Tensor):
                features = collections.OrderedDict([(0, features)])
            proposals, proposal_losses = model.rpn(images_dev, features, targets_dev)
            pred, pred_losses = model.roi_heads(features, proposals, images_dev.image_sizes, targets_dev)
            pred = model.transform.postprocess(pred, images_dev.image_sizes, original_image_sizes)
            iter_loss = sum(loss for loss in {**pred_losses, **proposal_losses}.values())
            iter_loss.backward()
            optimizer.step()
            pred = self._from_tensor(pred, sample)
            target_bboxes = [target["refs"] for target in targets]
            # pack image list back into 4d tensor
            images = torch.cat(images) if len(images) > 1 else torch.unsqueeze(images[0], 0)
            iter_loss = iter_loss.item()
            for metric in metrics.values():
                metric.update(task=self.task, input=images, pred=pred, target=target_bboxes,
                              sample=sample, loss=iter_loss, iter_idx=idx, max_iters=epoch_size,
                              epoch_idx=epoch, max_epochs=self.epochs)
            epoch_loss += iter_loss
        epoch_loss /= epoch_size
        return epoch_loss

    def eval_epoch(self, model, epoch, dev, loader, metrics):
        """Evaluates the model using the provided objects.

        Args:
            model: the model to evaluate that is already uploaded to the target device(s).
            epoch: the epoch index we are training for (0-based).
            dev: the target device that tensors should be uploaded to.
            loader: the data loader used to get transformed valid/test samples.
            metrics: the dictionary of metrics/consumers to update every iteration.
        """
        assert loader, "no available data to load"
        assert isinstance(metrics, dict), "expect metrics as dict object"
        with torch.no_grad():
            epoch_size = len(loader)
            self.logger.debug("fetching data loader samples...")
            for idx, sample in enumerate(loader):
                if idx < self.skip_eval_iter:
                    continue  # skip until previous iter count (if set externally; no effect otherwise)
                images, targets = self._to_tensor(sample)
                pred = model(self._move_tensor(images, dev))
                pred = self._from_tensor(pred, sample)
                target_bboxes = [target["refs"] for target in targets]
                # pack image list back into 4d tensor
                images = torch.cat(images) if len(images) > 1 else torch.unsqueeze(images[0], 0)
                for metric in metrics.values():
                    metric.update(task=self.task, input=images, pred=pred, target=target_bboxes,
                                  sample=sample, loss=None, iter_idx=idx, max_iters=epoch_size,
                                  epoch_idx=epoch, max_epochs=self.epochs)
