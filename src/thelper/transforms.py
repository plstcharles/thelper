import logging

import Augmentor
import cv2 as cv
import numpy as np
import PIL
import torchvision.transforms
import torchvision.utils

import thelper.utils

logger = logging.getLogger(__name__)


def fixup_augmentor_list(sample):
    # augmentor sometimes returns single-element lists following some transforms...
    if isinstance(sample, list):
        if len(sample) != 1:
            raise AssertionError("not the case we expected to catch here")
        return sample[0]
    return sample


def load_transforms(config):
    logger.debug("loading transforms from config")
    transforms = []
    append_transf = True
    for transform_name, transform_config in config.items():
        if transform_name == "Augmentor.Pipeline":
            augp = Augmentor.Pipeline()
            if "input_tensor" not in transform_config:
                raise AssertionError("missing mandatory augmentor pipeline config 'input_tensor' field")
            if "output_tensor" not in transform_config:
                raise AssertionError("missing mandatory augmentor pipeline config 'output_tensor' field")
            if "stages" not in transform_config:
                raise AssertionError("missing mandatory augmentor pipeline config 'stages' field")
            stages = transform_config["stages"]
            if not isinstance(stages, dict):
                raise AssertionError("augmentor pipeline 'stages' field should contain dictionary")
            for stage_name, stage_config in stages.items():
                getattr(augp, stage_name)(**stage_config)
            if transform_config["input_tensor"]:
                transforms.append(torchvision.transforms.ToPILImage())
            transforms.append(augp.torch_transform())
            transforms.append(fixup_augmentor_list)
            if transform_config["output_tensor"]:
                transforms.append(torchvision.transforms.ToTensor())
        elif transform_name == "append":
            append_transf = thelper.utils.str2bool(transform_config)
        else:
            transform_type = thelper.utils.import_class(transform_name)
            transform = transform_type(**transform_config)
            transforms.append(transform)
    if len(transforms) > 1:
        return thelper.transforms.Compose(transforms), append_transf
    elif len(transforms) == 1:
        return transforms[0], append_transf
    else:
        return None, False


class Compose(torchvision.transforms.Compose):

    """Composes several transforms together (with support for invert ops)."""

    def __init__(self, transforms):
        super().__init__(transforms)

    def invert(self, img):
        for t in reversed(self.transforms):
            if not hasattr(t, "invert"):
                raise AssertionError("missing invert op for transform = %s" % repr(t))
            img = t.invert(img)
        return img


class OpenCVCenterCrop(object):

    def __init__(self, size, bordertype=cv.BORDER_CONSTANT, borderval=0):
        if isinstance(size, (tuple, list)):
            if len(size) != 2:
                raise AssertionError("expected center crop dim input as 2-item tuple (width,height)")
            self.size = tuple(size)
        elif isinstance(size, (int, float)):
            self.size = (size, size)
        else:
            raise AssertionError("unexpected center crop dim input type (need tuple or int/float)")
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise AssertionError("crop dimensions must be positive")
        if isinstance(self.size[0], float):
            self.relative = True
        elif isinstance(self.size[0], int):
            self.relative = False
        else:
            raise AssertionError("unexpected center crop dim input type (need tuple or int/float)")
        self.bordertype = thelper.utils.import_class(bordertype) if isinstance(bordertype, str) else bordertype
        self.borderval = borderval

    def __call__(self, sample):
        if sample.ndim < 2 or sample.ndim > 3:
            raise AssertionError("bad input dimensions; must be 2-d, or 3-d (with channels)")
        crop_height = int(round(self.size[1] * sample.shape[0])) if self.relative else self.size[1]
        crop_width = int(round(self.size[0] * sample.shape[1])) if self.relative else self.size[0]
        tl = [sample.shape[1] // 2 - crop_width // 2, sample.shape[0] // 2 - crop_height // 2]
        br = [tl[0] + crop_width, tl[1] + crop_height]
        if tl[0] < 0 or tl[1] < 0 or br[0] > sample.shape[1] or br[1] > sample.shape[0]:
            sample = cv.copyMakeBorder(sample, max(-tl[1], 0), max(br[1] - sample.shape[0], 0),
                                       max(-tl[0], 0), max(br[0] - sample.shape[1], 0),
                                       borderType=self.bordertype, value=self.borderval)
            if tl[0] < 0:
                br[0] -= tl[0]
                tl[0] = 0
            if tl[1] < 0:
                br[1] -= tl[1]
                tl[1] = 0
        return sample[tl[1]:br[1], tl[0]:br[0], ...]

    def invert(self, sample):
        raise AssertionError("cannot be inverted")

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, bordertype={1}, bordervalue={2})".format(self.size, self.bordertype, self.borderval)


class OpenCVResize(object):

    def __init__(self, dsize, fx=0, fy=0, interp=cv.INTER_LINEAR, buffer=False):
        self.interp = thelper.utils.import_class(interp) if isinstance(interp, str) else interp
        self.buffer = buffer
        if isinstance(dsize, tuple):
            self.dsize = dsize
        else:
            self.dsize = tuple(dsize)
        self.dst = None
        self.fx = fx
        self.fy = fy
        if fx < 0 or fy < 0:
            raise AssertionError("scale factors should be null (ignored) or positive")
        if dsize[0] < 0 or dsize[1] < 0:
            raise AssertionError("destination image size should be null (ignored) or positive")
        if fx == 0 and fy == 0 and dsize[0] == 0 and dsize[1] == 0:
            raise AssertionError("need to specify either destination size or scale factor(s)")

    def __call__(self, sample):
        if sample.ndim < 2 or sample.ndim > 3:
            raise AssertionError("bad input dimensions; must be 2-d, or 3-d (with channels)")
        if sample.ndim < 3 or sample.shape[2] <= 4:
            if self.buffer:
                cv.resize(sample, self.dsize, dst=self.dst, fx=self.fx, fy=self.fy, interpolation=self.interp)
                if self.dst.ndim == 2:
                    return np.expand_dims(self.dst, 2)
                return self.dst
            else:
                dst = cv.resize(sample, self.dsize, fx=self.fx, fy=self.fy, interpolation=self.interp)
                if dst.ndim == 2:
                    dst = np.expand_dims(dst, 2)
                return dst
        else:  # too many channels, need to split-resize
            slices_dst = self.dst if self.buffer else None
            slices = np.split(sample, sample.shape[2], 2)
            if slices_dst is None or not isinstance(slices_dst, list) or len(slices_dst) != len(slices):
                slices_dst = [None] * len(slices)
            for idx in range(len(slices)):
                cv.resize(slice[idx], self.dsize, dst=slices_dst[idx], fx=self.fx, fy=self.fy, interpolation=self.interp)
            if self.buffer:
                self.dst = slices_dst
            return np.stack(slices_dst, 2)

    def invert(self, sample):
        raise AssertionError("missing implementation")  # todo

    def __repr__(self):
        return self.__class__.__name__ + "(dsize={0}, fx={1}, fy={2}, interp={3})".format(self.dsize, self.fx, self.fy, self.interp)


class NumpyTranspose(object):

    def __init__(self, axes):
        axes = np.asarray(axes)
        if axes.ndim > 1:
            raise AssertionError("tranpose param should be 1-d")
        if np.any(axes >= axes.size):
            raise AssertionError("oob dim in axes")
        self.axes = axes
        self.axes_inv = np.asarray([None] * axes.size)
        for i in list(range(len(axes))):
            self.axes_inv[self.axes[i]] = i

    def __call__(self, sample):
        return np.transpose(sample, self.axes)

    def invert(self, sample):
        return np.transpose(sample, self.axes_inv)

    def __repr__(self):
        return self.__class__.__name__ + "(axes={0})".format(self.axes)


class NumpyNormalizeZeroMeanUnitVar(object):

    def __init__(self, mean, std, out_type=np.float32):
        self.out_type = out_type
        self.mean = np.asarray(mean).astype(out_type)
        self.std = np.asarray(std).astype(out_type)
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise AssertionError("normalization params should be 1-d")
        if self.mean.size != self.std.size:
            raise AssertionError("normalization params size mismatch")
        if any([d == 0 for d in self.std]):
            raise AssertionError("normalization std must be non-null")

    def __call__(self, sample):
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return ((sample - self.mean) / self.std).astype(self.out_type)

    def invert(self, sample):
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return (sample * self.std + self.mean).astype(self.out_type)

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1}, out_type={2})".format(self.mean, self.std, self.out_type)


class NumpyNormalizeMinMax(object):

    def __init__(self, min, max, out_type=np.float32):
        self.out_type = out_type
        self.min = np.asarray(min).astype(out_type)
        if self.min.ndim == 0:
            self.min = np.expand_dims(self.min, 0)
        self.max = np.asarray(max).astype(out_type)
        if self.max.ndim == 0:
            self.max = np.expand_dims(self.max, 0)
        if self.min.ndim != 1 or self.max.ndim != 1:
            raise AssertionError("normalization params should be a 1-d array (one value per channel)")
        if self.min.size != self.max.size:
            raise AssertionError("normalization params size mismatch")
        self.diff = self.max - self.min
        if any([d == 0 for d in self.diff]):
            raise AssertionError("normalization diff must be non-null")

    def __call__(self, sample):
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return ((sample - self.min) / self.diff).astype(self.out_type)

    def invert(self, sample):
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return (sample * self.diff + self.min).astype(self.out_type)

    def __repr__(self):
        return self.__class__.__name__ + "(min={0}, max={1}, out_type={2})".format(self.min, self.max, self.out_type)
