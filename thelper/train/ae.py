import functools
import logging
import typing

import cv2 as cv
import kornia
import numpy as np
import torch
import torch.optim

import thelper.concepts
import thelper.typedefs as typ  # noqa: F401
import thelper.utils
from thelper.train.base import Trainer

logger = logging.getLogger(__name__)


@thelper.concepts.classification
@thelper.concepts.segmentation
class AutoEncoderTrainer(Trainer):

    def __init__(self,
                 session_name,    # type: typing.AnyStr
                 session_dir,     # type: typing.AnyStr
                 model,           # type: thelper.typedefs.ModelType
                 task,            # type: thelper.tasks.Task
                 loaders,         # type: thelper.typedefs.MultiLoaderType
                 config,          # type: thelper.typedefs.ConfigDict
                 ckptdata=None    # type: typ.Optional[thelper.typedefs.CheckpointContentType]
                 ):
        """Receives session parameters, parses image/label keys from task object, and sets up metrics."""
        super().__init__(session_name, session_dir, model, task, loaders, config, ckptdata=ckptdata)
        assert isinstance(self.task, (thelper.tasks.Classification, thelper.tasks.Segmentation)), \
            "expected task to be classification/segmentation only"
        self.warned_no_shuffling_augments = False
        self.reconstr_display_count = thelper.utils.get_key("reconstr_display_count", config["trainer"])
        self.reconstr_display_mean = thelper.utils.get_key("reconstr_display_mean", config["trainer"])
        self.reconstr_display_stddev = thelper.utils.get_key("reconstr_display_stddev", config["trainer"])
        self.reconstr_scale = thelper.utils.get_key("reconstr_scale", config["trainer"])
        self.reconstr_edges_layer = thelper.utils.get_key("reconstr_edges", config["trainer"])
        if self.reconstr_edges_layer:
            self.reconstr_edges_layer = kornia.filters.SpatialGradient()
        self.reconstr_l2_loss, self.reconstr_l1_loss = torch.nn.MSELoss(), torch.nn.L1Loss()
        classif_loss_config = thelper.utils.get_key("classif_loss", config["trainer"])
        uploader = functools.partial(self._move_tensor, dev=self.devices)
        self.classif_loss = thelper.optim.utils.create_loss_fn(classif_loss_config, model, uploader=uploader)

    def _to_tensor(self, sample):
        """Fetches and returns tensors of input images and class labels from a batched sample dictionary."""
        assert isinstance(sample, dict), "trainer expects samples to come in dicts for key-based usage"
        assert self.task.input_key in sample, f"could not find input key '{self.task.input_key}' in sample dict"
        input_val, target_val = sample[self.task.input_key].float(), None
        if self.task.gt_key in sample and sample[self.task.gt_key] is not None:
            gt_tensor = sample[self.task.gt_key]
            assert len(gt_tensor) == len(input_val), \
                "target tensor should be an array of the same length as input (== batch size)"
            if isinstance(gt_tensor, torch.Tensor) and gt_tensor.dtype == torch.int64:
                target_val = gt_tensor  # shortcut with less checks (dataset is already using tensor'd indices)
            else:
                if isinstance(self.task, thelper.tasks.Classification):
                    if self.task.multi_label:
                        assert isinstance(gt_tensor, torch.Tensor) and \
                            gt_tensor.shape == (len(input_val), len(self.task.class_names)), \
                            "gt tensor for multi-label classification should be 2d array (batch size x nbclasses)"
                        target_val = gt_tensor.float()
                    else:
                        target_val = []
                        for class_name in gt_tensor:
                            assert isinstance(class_name, (int, torch.Tensor, str)), \
                                "expected gt tensor to be an array of names (string) or indices (int)"
                            if isinstance(class_name, (int, torch.Tensor)):
                                if isinstance(class_name, torch.Tensor):
                                    assert torch.numel(class_name) == 1, "unexpected scalar label, got vector"
                                    class_name = class_name.item()
                                # dataset must already be using indices, we will forgive this...
                                assert 0 <= class_name < len(self.task.class_names), \
                                    "class name given as out-of-range index (%d) for class list" % class_name
                                target_val.append(class_name)
                            else:
                                assert class_name in self.task.class_names, \
                                    "got unexpected label '%s' for a sample (unknown class)" % class_name
                                target_val.append(self.task.class_indices[class_name])
                        target_val = torch.LongTensor(target_val)
                elif isinstance(self.task, thelper.tasks.Segmentation):
                    assert not isinstance(gt_tensor, list), "unexpected label map type"
                    if gt_tensor.ndim == 4:
                        assert gt_tensor.shape[1] == 1, "unexpected channel count (should be index map)"
                        gt_tensor = gt_tensor.squeeze(1)
                    target_val = gt_tensor.long()  # long instead of bytes to support large/negative values for dontcare
        return input_val, target_val

    def train_epoch(self, model, epoch, dev, classif_loss, optimizer, loader, metrics, output_path):
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
        assert classif_loss is None, "loss function defined by trainer"
        assert optimizer is not None, "missing optimizer"
        assert loader, "no available data to load"
        assert isinstance(metrics, dict), "expect metrics as dict object"
        epoch_loss = 0
        epoch_size = len(loader)
        self.logger.debug("fetching data loader samples...")
        for idx, sample in enumerate(loader):
            input_val, target_val = self._to_tensor(sample)
            input_val_dev = self._move_tensor(input_val, dev)
            target_val_dev = self._move_tensor(target_val, dev)
            assert target_val is not None, "groundtruth required when training a model"
            optimizer.zero_grad()
            class_logits, reconstr = model(input_val_dev)
            classif_loss = self.classif_loss(class_logits, target_val_dev)
            reconstr_loss = self.reconstr_l2_loss(reconstr, input_val_dev)
            if self.reconstr_edges_layer:
                target_edges_shape = (
                    reconstr.shape[0],
                    reconstr.shape[1] * 2,  # for gradX/gradY
                    reconstr.shape[2],
                    reconstr.shape[3],
                )
                reconstr_gradients = self.reconstr_edges_layer(reconstr).view(target_edges_shape)
                input_gradients = self.reconstr_edges_layer(input_val_dev).view(target_edges_shape)
                reconstr_edge_loss = self.reconstr_l1_loss(reconstr_gradients, input_gradients)
                reconstr_loss += reconstr_edge_loss
            iter_loss = classif_loss + self.reconstr_scale * reconstr_loss
            iter_loss.backward()
            optimizer.step()
            iter_loss = iter_loss.item()
            for metric in metrics.values():
                metric.update(task=self.task, input=input_val, pred=class_logits,
                              target=target_val, sample=sample, loss=iter_loss, iter_idx=idx,
                              max_iters=epoch_size, epoch_idx=epoch, max_epochs=self.epochs,
                              output_path=output_path)
            epoch_loss += iter_loss
        epoch_loss /= epoch_size
        return epoch_loss

    def eval_epoch(self, model, epoch, dev, loader, metrics, output_path):
        """Evaluates the model using the provided objects.

        Args:
            model: the model to evaluate that is already uploaded to the target device(s).
            epoch: the epoch index we are training for (0-based).
            dev: the target device that tensors should be uploaded to.
            loader: the data loader used to get transformed valid/test samples.
            metrics: the dictionary of metrics/consumers to update every iteration.
            output_path: directory where output files should be written, if necessary.
        """
        assert loader, "no available data to load"
        assert isinstance(metrics, dict), "expect metrics as dict object"
        with torch.no_grad():
            epoch_size = len(loader)
            self.logger.debug("fetching data loader samples...")
            display_array = []
            for idx, sample in enumerate(loader):
                if idx < self.skip_eval_iter:
                    continue  # skip until previous iter count (if set externally; no effect otherwise)
                input_val, target_val = self._to_tensor(sample)
                input_val_dev = self._move_tensor(input_val, dev)
                target_val_dev = self._move_tensor(target_val, dev)
                class_logits, reconstr = model(input_val_dev)
                classif_loss = self.classif_loss(class_logits, target_val_dev)
                reconstr_loss = self.reconstr_l2_loss(reconstr, input_val_dev)
                if self.reconstr_edges_layer:
                    target_edges_shape = (
                        reconstr.shape[0],
                        reconstr.shape[1] * 2,  # for gradX/gradY
                        reconstr.shape[2],
                        reconstr.shape[3],
                    )
                    reconstr_gradients = self.reconstr_edges_layer(reconstr).view(target_edges_shape)
                    input_gradients = self.reconstr_edges_layer(input_val_dev).view(target_edges_shape)
                    reconstr_edge_loss = self.reconstr_l1_loss(reconstr_gradients, input_gradients)
                    reconstr_loss += reconstr_edge_loss
                iter_loss = (classif_loss + self.reconstr_scale * reconstr_loss).item()
                for metric in metrics.values():
                    metric.update(task=self.task, input=input_val, pred=class_logits,
                                  target=target_val, sample=sample, loss=iter_loss, iter_idx=idx,
                                  max_iters=epoch_size, epoch_idx=epoch, max_epochs=self.epochs,
                                  output_path=output_path)
                if self.use_tbx:
                    if isinstance(self.reconstr_display_mean, str):
                        display_mean = eval(self.reconstr_display_mean)
                    else:
                        display_mean = np.asarray(self.reconstr_display_mean)
                    if isinstance(self.reconstr_display_stddev, str):
                        display_stddev = eval(self.reconstr_display_stddev)
                    else:
                        display_stddev = np.asarray(self.reconstr_display_stddev)
                    # make sure not to shuffle if you want to get the same images each epoch...
                    while len(display_array) < self.reconstr_display_count:
                        for input_img, reconstr_img in zip(input_val_dev, reconstr):
                            display = []
                            for img in [input_img, reconstr_img]:
                                # move back to HxWxC format
                                img = np.transpose(img.cpu().numpy(), (1, 2, 0))
                                # de-normalize w/ provided vals
                                img = (img * display_stddev) + display_mean
                                clip_max = display_mean + display_stddev * 3,
                                clip_min = display_mean - display_stddev * 3
                                img = (img - clip_min) / (clip_max - clip_min)
                                img = np.minimum(np.maximum(0, img), 1)
                                display.append(thelper.draw.get_displayable_image(img))
                            display_array.append(cv.vconcat(display))
                            if len(display_array) >= self.reconstr_display_count:
                                break
            if self.use_tbx:
                writer_prefix = "epoch/"
                file_suffix = f"-{epoch:04d}"
                output = {"reconstr/image": cv.hconcat(display_array)}
                self._write_data(output, writer_prefix, file_suffix,
                                 self.writers["valid"], self.output_paths["valid"], epoch)
