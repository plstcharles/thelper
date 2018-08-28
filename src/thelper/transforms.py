"""Transformations module.

This module contains image transformation classes and wrappers for
data augmentation and normalization.
"""
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


def load_transforms(config):
    """Loads a torchvision-compatible transformation pipeline from a dictionary of stages.

    Each entry in the provided dictionary will be considered a stage in the pipeline. The ordering
    of the stages is important, as some transformations might not be compatible if taken out of order.

    The key of each stage in the dictionary will be used to dynamically import a specific type of data
    transformation. Some unique keys have special handling discussed below. The imported transform type
    will be instantiated and given its corresponding dictionary values as parameters.

    If a stage is identified by the 'Augmentor.Pipeline' key, it will be handled as an Augmentor pipeline,
    and its value will be parsed further (a dictionary is expected). Three fields are required in this
    dictionary: 'input_tensor' (bool) which specifies whether the previous stage provides a torch.Tensor
    to the augmentor pipeline; 'output_tensor' (bool) which specifies whether the output of the augmentor
    pipeline should be converted into a torch.Tensor; and 'stages' (dict) which specifies the Augmentor
    pipeline stage names and parameters.

    Finally, if an entry possesses the 'append' key, its value should be a bool, and it will be used to
    specify whether the transformation pipeline should be appended or prepended to other transformations.
    If no 'append' key is provided, the pipeline will be returned with `append_transf=True`.

    Args:
        config: a dictionary containing a list of transformations to apply as a single pipeline.

    Returns:
        A tuple that consists of a pipeline compatible with the torchvision.transforms interfaces, and
        a boolean specifying whether this pipeline should be appended or prefixed to other transforms.
    """
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


class AugmentorWrapper(object):
    """Augmentor pipeline wrapper that allows pickling and multithreading.

    See https://github.com/mdbloice/Augmentor for more information. This wrapper
    was last updated to work with version 0.2.2 --- more recent versions introduced
    yet unfixed (as of 2018/08) issues on some platforms.

    All original transforms are supported here. This wrapper also fixes the list
    output bug for single-image samples when using operations individually.

    Attributes:
        pipeline: the augmentor pipeline instance to apply to images.
    """
    def __init__(self, pipeline):
        """Receives and stores an augmentor pipeline for later use.

        The pipeline itself is instantiated in `thelper.transforms.load_transforms(...)`.
        """
        self.pipeline = pipeline

    def __call__(self, sample):
        """Transforms a single image using the augmentor pipeline.

        Args:
            sample: the image to transform. Should be a numpy array or a PIL image.

        Returns:
            The transformed image, with the same type as the input (numpy array or PIL image).
        """
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

    def __repr__(self):
        """Create a print-friendly representation of inner augmentation stages."""
        return "AugmentorWrapper: " + str([str(op) for op in self.pipeline.operations])


class Compose(torchvision.transforms.Compose):
    """Composes several transforms together (with support for invert ops).

    This interface is fully compatible with torchvision.transforms.Compose.
    """

    def __init__(self, transforms):
        """Forwards the list of transformations to the base class."""
        super().__init__(transforms)

    def invert(self, sample):
        """Tries to invert the transformations applied to a sample.

        Will throw if one of the transformations cannot be inverted.
        """
        for t in reversed(self.transforms):
            if not hasattr(t, "invert"):
                raise AssertionError("missing invert op for transform = %s" % repr(t))
            sample = t.invert(sample)
        return sample


class CenterCrop(object):
    """Returns a center crop from a given image via OpenCV and numpy.

    This operation is deterministic. The code relies on OpenCV, meaning the border arguments
    must be compatible with cv2.copyMakeBorder.

    Attributes:
        size: the size of the target crop (tuple of width, height).
        relative: specifies whether the target crop size is relative to the image size or not.
        bordertype: argument forwarded to cv2.copyMakeBorder.
        borderval: argument forwarded to cv2.copyMakeBorder.
    """

    def __init__(self, size, bordertype=cv.BORDER_CONSTANT, borderval=0):
        """Validates and initializes center crop parameters.

        Args:
            size: size of the target crop, provided as tuple or list. If integer values are used,
                the size is assumed to be absolute. If floating point values are used, the size is
                assumed to be relative, and will be determined dynamically for each sample. If a
                tuple is used, it is assumed to be (width, height).
            bordertype: border copy type to use when the image is too small for the required crop size.
                See cv2.copyMakeBorder for more information.
            borderval: border value to use when the image is too small for the required crop size. See
                cv2.copyMakeBorder for more information.
        """
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
        """Extracts and returns a central crop from the provided image.

        Args:
            sample: the image to generate the crop from; should be a 2d or 3d numpy array.

        Returns:
            The center crop.
        """
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
        """Specifies that this operation cannot be inverted, as data loss is incurred during image transformation."""
        raise AssertionError("cannot be inverted")

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__name__ + "(size={0}, bordertype={1}, bordervalue={2})".format(self.size, self.bordertype, self.borderval)


class Resize(object):
    """Resizes a given image using OpenCV and numpy.

    This operation is deterministic. The code relies on OpenCV, meaning the interpolation arguments
    must be compatible with cv2.resize.

    Attributes:
        interp: interpolation type to use (forwarded to cv2.resize)
        buffer: specifies whether a destination buffer should be used to avoid allocations
        dsize: target image size (tuple of width, height).
        dst: buffer used to avoid reallocations if self.buffer == True
        fx: argument forwarded to cv2.resize.
        fy: argument forwarded to cv2.resize.
    """

    def __init__(self, dsize, fx=0, fy=0, interp=cv.INTER_LINEAR, buffer=False):
        """Validates and initializes resize parameters.

        Args:
            dsize: size of the target image, forwarded to cv2.resize.
            fx: x-scaling factor, forwarded to cv2.resize.
            fy: y-scaling factor, forwarded to cv2.resize.
            interp: resize interpolation type, forwarded to cv2.resize.
            buffer: specifies whether a destination buffer should be used to avoid allocations
        """
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
        """Returns a resized copy of the provided image.

        Args:
            sample: the image to resize; should be a 2d or 3d numpy array.

        Returns:
            The resized image. May be allocated on the spot, or be a pointer to a local buffer.
        """
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
        """Specifies that this operation cannot be inverted, as data loss is incurred during image transformation."""
        # todo, could implement if original size is fixed & known
        raise AssertionError("missing implementation")

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__name__ + "(dsize={0}, fx={1}, fy={2}, interp={3})".format(self.dsize, self.fx, self.fy, self.interp)


class Affine(object):
    """Warps a given image using an affine matrix via OpenCV and numpy.

    This operation is deterministic. The code relies on OpenCV, meaning the border arguments
    must be compatible with cv2.warpAffine.

    Attributes:
        transf: the 2x3 transformation matrix passed to cv2.warpAffine.
        out_size: target image size (tuple of width, height). If None, same as original.
        flags: extra warp flags forwarded to cv2.warpAffine.
        border_mode: border extrapolation mode forwarded to cv2.warpAffine.
        border_val: border constant extrapolation value forwarded to cv2.warpAffine.
    """

    def __init__(self, transf, out_size=None, flags=None, border_mode=None, border_val=None):
        """Validates and initializes affine warp parameters.

        Args:
            transf: the 2x3 transformation matrix passed to cv2.warpAffine.
            out_size: target image size (tuple of width, height). If None, same as original.
            flags: extra warp flags forwarded to cv2.warpAffine.
            border_mode: border extrapolation mode forwarded to cv2.warpAffine.
            border_val: border constant extrapolation value forwarded to cv2.warpAffine.
        """
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
        """Warps a given image using an affine matrix.

        Args:
            sample: the image to warp; should be a 2d or 3d numpy array.

        Returns:
            The warped image.
        """
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        out_size = self.out_size
        if out_size is None:
            out_size = (sample.shape[1], sample.shape[0])
        return cv.warpAffine(sample, self.transf, dsize=out_size, flags=self.flags, borderMode=self.border_mode, borderValue=self.border_val)

    def invert(self, sample):
        """Inverts the warp transformation, but only is the output image has not been cropped before."""
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        out_size = self.out_size
        if out_size is None:
            out_size = (sample.shape[1], sample.shape[0])
        else:
            raise AssertionError("unknown original image size, cannot invert affine transform")
        return cv.warpAffine(sample, self.transf, dsize=out_size, flags=self.flags ^ cv.WARP_INVERSE_MAP, borderMode=self.border_mode, borderValue=self.border_val)

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__name__ + "(transf={0}, out_size={1})".format(np.array2string(self.transf), self.out_size)


class RandomShift(object):
    """Randomly translates an image in a provided range via OpenCV and numpy.

    This operation is stochastic, and thus cannot be inverted. Each time the operation is called,
    a random check will determine whether a transformation is applied or not. The code relies on
    OpenCV, meaning the border arguments must be compatible with cv2.warpAffine.

    Attributes:
        min: the minimum pixel shift that can be applied stochastically.
        max: the maximum pixel shift that can be applied stochastically.
        probability: the probability that the transformation will be applied when called.
        flags: extra warp flags forwarded to cv2.warpAffine.
        border_mode: border extrapolation mode forwarded to cv2.warpAffine.
        border_val: border constant extrapolation value forwarded to cv2.warpAffine.
    """

    def __init__(self, min, max, probability=1.0, flags=None, border_mode=None, border_val=None):
        """Validates and initializes shift parameters.

        Args:
            min: the minimum pixel shift that can be applied stochastically.
            max: the maximum pixel shift that can be applied stochastically.
            probability: the probability that the transformation will be applied when called.
            flags: extra warp flags forwarded to cv2.warpAffine.
            border_mode: border extrapolation mode forwarded to cv2.warpAffine.
            border_val: border constant extrapolation value forwarded to cv2.warpAffine.
        """
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
        if self.max[0] < self.min[0] or self.max[1] < self.min[1]:
            raise AssertionError("bad min/max values")
        if probability < 0 or probability > 1:
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
        """Translates a given image using a predetermined min/max range.

        Args:
            sample: the image to translate; should be a 2d or 3d numpy array.

        Returns:
            The translated image.
        """
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        if np.random.uniform(0, 1) > self.probability:
            return sample
        out_size = (sample.shape[1], sample.shape[0])
        x_shift = np.random.uniform(self.min[0], self.max[0])
        y_shift = np.random.uniform(self.min[1], self.max[1])
        transf = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        return cv.warpAffine(sample, transf, dsize=out_size, flags=self.flags, borderMode=self.border_mode, borderValue=self.border_val)

    def invert(self, sample):
        """Specifies that this operation cannot be inverted, as it is stochastic, and data loss occurs during transformation."""
        raise AssertionError("operation cannot be inverted")

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__name__ + "(min={0}, max={1}, prob={2})".format(self.min, self.max, self.probability)


class Transpose(object):
    """Transposes an image via numpy.

    This operation is deterministic.

    Attributes:
        axes: the axes on which to apply the transpose; forwarded to numpy.transpose(...).
        axes_inv: used to invert the tranpose; also forwarded to numpy.transpose(...).
    """

    def __init__(self, axes):
        """Validates and initializes tranpose parameters.

        Args:
            axes: the axes on which to apply the transpose; forwarded to numpy.transpose(...).
        """
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
        """Transposes a given image.

        Args:
            sample: the image to transpose; should be a numpy array.

        Returns:
            The transposed image.
        """
        return np.transpose(sample, self.axes)

    def invert(self, sample):
        """Invert-transposes a given image.

        Args:
            sample: the image to invert-transpose; should be a numpy array.

        Returns:
            The invert-transposed image.
        """
        return np.transpose(sample, self.axes_inv)

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__name__ + "(axes={0})".format(self.axes)


class NormalizeZeroMeanUnitVar(object):
    """Normalizes a given image using a set of mean and standard deviation parameters.

    The samples will be transformed such that `s = (s - mean) / std`.

    This can be used for whitening; see https://en.wikipedia.org/wiki/Whitening_transformation
    for more information. Note that this operation is also not restricted to images.

    Attributes:
        mean: an array of mean values to subtract from data samples.
        std: an array of standard deviation values to divide with.
        out_type: the output data type to cast the normalization result to.
    """

    def __init__(self, mean, std, out_type=np.float32):
        """Validates and initializes normalization parameters.

        Args:
            mean: an array of mean values to subtract from data samples.
            std: an array of standard deviation values to divide with.
            out_type: the output data type to cast the normalization result to.
        """
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
        """Normalizes a given sample.

        Args:
            sample: the sample to normalize. If given as a PIL image, it
            will be converted to a numpy array first.

        Returns:
            The warped sample, in a numpy array of type 'self.out_type'.
        """
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return ((sample - self.mean) / self.std).astype(self.out_type)

    def invert(self, sample):
        """Inverts the normalization."""
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return (sample * self.std + self.mean).astype(self.out_type)

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__name__ + "(mean={0}, std={1}, out_type={2})".format(self.mean, self.std, self.out_type)


class NormalizeMinMax(object):
    """Normalizes a given image using a set of minimum and maximum values.

    The samples will be transformed such that `s = (s - min) / (max - min)`.

    Note that this operation is also not restricted to images.

    Attributes:
        min: an array of minimum values to subtract with.
        max: an array of maximum values to divide with.
        out_type: the output data type to cast the normalization result to.
    """

    def __init__(self, min, max, out_type=np.float32):
        """Validates and initializes normalization parameters.

        Args:
            min: an array of minimum values to subtract with.
            max: an array of maximum values to divide with.
            out_type: the output data type to cast the normalization result to.
        """
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
        """Normalizes a given sample.

        Args:
            sample: the sample to normalize. If given as a PIL image, it
            will be converted to a numpy array first.

        Returns:
            The warped sample, in a numpy array of type 'self.out_type'.
        """
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return ((sample - self.min) / self.diff).astype(self.out_type)

    def invert(self, sample):
        """Inverts the normalization."""
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return (sample * self.diff + self.min).astype(self.out_type)

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__name__ + "(min={0}, max={1}, out_type={2})".format(self.min, self.max, self.out_type)
