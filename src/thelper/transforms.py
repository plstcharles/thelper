import logging
import random

import Augmentor
import cv2 as cv
import numpy as np
import PIL.Image
import torchvision.transforms
import torchvision.utils

import thelper.utils

logger = logging.getLogger(__name__)


class AugmentorWrapper(object):
    # interface wrapper last updated for augmentor v0.2.2
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, sample):
        cvt_array = False
        if isinstance(sample, np.ndarray):
            sample = PIL.Image.fromarray(sample)
            cvt_array = True
        elif not isinstance(sample, PIL.Image.Image):
            raise AssertionError("unexpected input sample type (must be np.ndarray or PIL.Image)")
        sample = [sample]
        for operation in self.pipeline.operations:
            r = round(random.uniform(0, 1), 1)
            if r <= operation.probability:
                sample = operation.perform_operation(sample)
        if not isinstance(sample, list) or len(sample) != 1:
            raise AssertionError("not the fixup we expected to catch here")
        sample = sample[0]
        if cvt_array:
            sample = np.asarray(sample)
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
            transforms.append(AugmentorWrapper(augp))
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


class CenterCrop(object):

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


class Resize(object):

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


class Affine(object):

    def __init__(self, transf, out_size=None, flags=None, border_mode=None, border_val=None):
        if isinstance(transf, np.ndarray):
            if transf.size != 6:
                raise AssertionError("transformation matrix must be 2x3")
            self.transf = transf.reshape((2, 3)).astype(np.float32)
        elif isinstance(transf, list):
            if not len(transf) == 6:
                raise AssertionError("transformation matrix must be 6 elements (2x3)")
            self.transf = np.asarray(transf).reshape((2, 3)).astype(np.float32)
        else:
            raise AssertionError("unexpected transformation matrix type")
        self.out_size = None
        if out_size is not None:
            if isinstance(out_size, list):
                if len(out_size) != 2:
                    raise AssertionError("output image size should be 2-elem list or tuple")
                self.out_size = tuple(out_size)
            elif isinstance(out_size, tuple):
                if len(out_size) != 2:
                    raise AssertionError("output image size should be 2-elem list or tuple")
                self.out_size = out_size
            else:
                raise AssertionError("unexpected output size type")
        self.flags = flags
        if self.flags is None:
            self.flags = cv.INTER_LINEAR
        self.border_mode = border_mode
        if self.border_mode is None:
            self.border_mode = cv.BORDER_CONSTANT
        self.border_val = border_val
        if self.border_val is None:
            self.border_val = 0

    def __call__(self, sample):
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        out_size = self.out_size
        if out_size is None:
            out_size = (sample.shape[1], sample.shape[0])
        return cv.warpAffine(sample, self.transf, dsize=out_size, flags=self.flags, borderMode=self.border_mode, borderValue=self.border_val)

    def invert(self, sample):
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        out_size = self.out_size
        if out_size is None:
            out_size = (sample.shape[1], sample.shape[0])
        else:
            raise AssertionError("unknown original image size, cannot invert affine transform")
        return cv.warpAffine(sample, self.transf, dsize=out_size, flags=self.flags ^ cv.WARP_INVERSE_MAP, borderMode=self.border_mode, borderValue=self.border_val)

    def __repr__(self):
        return self.__class__.__name__ + "(transf={0}, out_size={1})".format(np.array2string(self.transf), self.out_size)


class RandomShift(object):

    def __init__(self, min, max, probability=0.5, flags=None, border_mode=None, border_val=None):
        if isinstance(min, tuple) and isinstance(max, tuple):
            if len(min) != len(max) or len(min) != 2:
                raise AssertionError("min/max shift tuple must be 2-elem")
            self.min = min
            self.max = max
        elif isinstance(min, list) and isinstance(max, list):
            if len(min) != len(max) or len(min) != 2:
                raise AssertionError("min/max shift list must be 2-elem")
            self.min = tuple(min)
            self.max = tuple(max)
        elif isinstance(min, (int, float)) and isinstance(max, (int, float)):
            self.min = (min, min)
            self.max = (max, max)
        else:
            raise AssertionError("unexpected min/max combo types")
        if self.max[0]<self.min[0] or self.max[1]<self.min[1]:
            raise AssertionError("bad min/max values")
        if probability < 0 or probability < 1:
            raise AssertionError("bad probability range")
        self.probability = probability
        self.flags = flags
        if self.flags is None:
            self.flags = cv.INTER_LINEAR
        self.border_mode = border_mode
        if self.border_mode is None:
            self.border_mode = cv.BORDER_CONSTANT
        self.border_val = border_val
        if self.border_val is None:
            self.border_val = 0

    def __call__(self, sample):
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        if np.random.uniform(0, 1) > self.probability:
            return sample
        out_size = (sample.shape[1], sample.shape[0])
        x_shift = np.random.uniform(self.min[0],self.max[0])
        y_shift = np.random.uniform(self.min[1],self.max[1])
        transf = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        return cv.warpAffine(sample, transf, dsize=out_size, flags=self.flags, borderMode=self.border_mode, borderValue=self.border_val)

    def invert(self, sample):
        raise AssertionError("stochastic operation cannot be inverted")

    def __repr__(self):
        return self.__class__.__name__ + "(min={0}, max={1}, prob={2})".format(self.min, self.max, self.probability)


class Transpose(object):

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


class NormalizeZeroMeanUnitVar(object):

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


class NormalizeMinMax(object):

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
