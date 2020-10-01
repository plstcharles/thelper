"""Transformation operations module.

All transforms should aim to be compatible with both numpy arrays and
PyTorch tensors. By default, images are processed using ``__call__``,
meaning that for a given transformation ``t``, we apply it via::

    image_transformed = t(image)

All important parameters for an operation should also be passed in the
constructor and exposed in the operation's ``__repr__`` function so that
external parsers can discover exactly how to reproduce their behavior. For
now, these representations are used for debugging more than anything else.
"""

import copy
import itertools
import logging
import math

import cv2 as cv
import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional
import torchvision.utils

import thelper.utils

logger = logging.getLogger(__name__)


class NoTransform:
    """Used to flag some ops that should not be externally wrapped for sample/key handling."""

    def __call__(self, sample):
        """Identity transform."""
        return sample

    def invert(self, sample):
        """Identity transform."""
        return sample

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + "()"


class ToNumpy:
    """Converts and returns an image in numpy format from a ``torch.Tensor`` or ``PIL.Image`` format.

    This operation is deterministic. The returned image will always be encoded as HxWxC, where
    if the input has three channels, the ordering might be optionally changed.

    Attributes:
        reorder_bgr: specifies whether the channels should be reordered in OpenCV format.
    """

    def __init__(self, reorder_bgr=False):
        """Initializes transformation parameters."""
        self.reorder_bgr = reorder_bgr

    def __call__(self, sample):
        """Converts and returns an image in numpy format.

        Args:
            sample: the image to convert; should be a tensor, numpy array, or PIL image.

        Returns:
            The numpy-converted image.
        """
        assert isinstance(sample, (np.ndarray, torch.Tensor, PIL.Image.Image)), \
            "unknown image type, cannot process sample"
        if isinstance(sample, np.ndarray):
            pass  # no transform needed, channel reordering done at end
        elif isinstance(sample, torch.Tensor):
            sample = sample.cpu().numpy()
        elif isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return sample[..., ::-1] if self.reorder_bgr else sample

    def invert(self, sample):
        """Specifies that this operation cannot be inverted, as the original data type is unknown."""
        raise RuntimeError("cannot be inverted")

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(reorder_bgr={self.reorder_bgr})"


class SelectChannels:
    """Returns selected channels by indices from an array in numpy, ``torch.Tensor`` or ``PIL.Image`` format.

    This operation does not attempt to interpret the meaning of the channel's content.
    It only moves them around by indices. It is up to the user to make sure indices make sense for desired result.
    Input is expected to be encoded as HxWxC and will be returned as numpy array of same format.

    Behavior according to provided ``channels``:

    - single index (``int``):
      only that channel is extracted to form a single channel array as result
    - multiple indices (``list``, ``tuple``, ``set``):
      all specified unique channel indices are extracted and placed in the specified order as result
    - indices map (``dict`` with ``int`` values):
      channels at key index are moved to value index
      values

    Examples::
        out = SelectChannels(0)(img)                    # only the first channel is kept, out is HxWx1
        out = SelectChannels([2,1,0])(img3C)            # out image will have channels in reversed order
        out = SelectChannels([0,1,3])(img4C)            # out image will drop the channel at index #2
        out = SelectChannels({3:0, 0:1, 1:2})(img4C)    # out image is HxWx3 with remapped channels using
                                                        # <from:to> definitions and drops channel #2

        # all of the following are equivalent, some implicit and other explicit with ``None``
        out = SelectChannels([0, 1, 3])(img4C)
        out = SelectChannels({0:0, 1:1, 3:3})(img4C)
        out = SelectChannels({0:0, 1:1, 2:None 3:3})(img4C)

    Attributes:
        channels: indices or map of the channels to select from the original array.
    Returns:
        numpy array of selected channels
    """

    def __init__(self, channels):
        """Initializes transformation parameters."""
        assert isinstance(channels, (int, tuple, list, set, dict)), \
            "unexpected channels (must be an int, an iterable of integers or a dictionary mapping of integers)"
        if isinstance(channels, int):
            assert channels >= 0, "channel index must be a positive integer"
            self.channels = {channels: 0}
        elif isinstance(channels, (tuple, list, set, dict)):
            assert len(channels) > 0, "channels must provide at least one index"
            assert len(set(channels)) == len(channels), "channel indices must be unique"
            assert all(isinstance(c, int) and c >= 0 for c in channels), "all channel indices must be positive integers"
            if not isinstance(channels, dict):
                self.channels = {c: i for i, c in enumerate(channels)}
            else:
                to_channels = [c for c in channels.values() if c is not None]
                assert len(set(to_channels)) == len(to_channels), "mapping destination channel indices must be unique"
                assert all(isinstance(c, int) and c >= 0 for c in to_channels), \
                    "mapping destination channel indices must be positive integers or None"
                to_channels = list(sorted(to_channels))
                assert list(range(len(to_channels))) == to_channels, \
                    f"mapping destination channel indices must be 0-based continuous list, {to_channels} skips indices"
                self.channels = {k: v for k, v in channels.items() if v is not None}

    def __call__(self, sample):
        """Converts and returns an array in numpy format with selected channels.

        Args:
            sample: the array to convert; should be a tensor, numpy array, or PIL image.

        Returns:
            The numpy-converted array with selected channels.
        """
        sample = ToNumpy()(sample)
        n_from_channels = sample.shape[2]
        n_to_channels = len(self.channels)
        assert all(c < n_from_channels for c in self.channels) and n_to_channels <= n_from_channels, \
            f"source channel indices ({list(self.channels)}) " \
            f"cannot be greater than the number of available channels ({n_from_channels})"
        inv_map = {v: k for k, v in self.channels.items()}  # guaranteed 0..N keys with init checks
        sample = np.dstack([sample[:, :, inv_map[i]] for i in range(n_to_channels)])
        if n_to_channels == 1:
            return sample[:, :, 0]  # force cast to have expected format HxWx1
        return sample

    def invert(self, sample):
        """
        Specifies that this operation cannot be inverted. Original data type is unknown and channels can be dropped.
        """
        raise RuntimeError("cannot be inverted")

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}(channels={self.channels})"


class CenterCrop:
    """Returns a center crop from a given image via OpenCV and numpy.

    This operation is deterministic. The code relies on OpenCV, meaning the border arguments
    must be compatible with ``cv2.copyMakeBorder``.

    Attributes:
        size: the size of the target crop (tuple of width, height).
        relative: specifies whether the target crop size is relative to the image size or not.
        bordertype: argument forwarded to ``cv2.copyMakeBorder``.
        borderval: argument forwarded to ``cv2.copyMakeBorder``.
    """

    def __init__(self, size, bordertype=cv.BORDER_CONSTANT, borderval=0):
        """Validates and initializes center crop parameters.

        Args:
            size: size of the target crop, provided as tuple or list. If integer values are used,
                the size is assumed to be absolute. If floating point values are used, the size is
                assumed to be relative, and will be determined dynamically for each sample. If a
                tuple is used, it is assumed to be (width, height).
            bordertype: border copy type to use when the image is too small for the required crop size.
                See ``cv2.copyMakeBorder`` for more information.
            borderval: border value to use when the image is too small for the required crop size. See
                ``cv2.copyMakeBorder`` for more information.
        """
        assert isinstance(size, (tuple, list, int, float)), \
            "unexpected center crop dim input type (need tuple or int/float)"
        if isinstance(size, (tuple, list)):
            assert len(size) == 2, "expected center crop dim input as 2-item tuple (width,height)"
            self.size = tuple(size)
        elif isinstance(size, (int, float)):
            self.size = (size, size)
        assert self.size[0] > 0 and self.size[1] > 0, "crop dimensions must be strictly positive"
        assert isinstance(self.size[0], (float, int)), \
            "unexpected center crop dim input type (need tuple or int/float)"
        self.relative = isinstance(self.size[0], float)
        self.bordertype = thelper.utils.import_class(bordertype) if isinstance(bordertype, str) else bordertype
        self.borderval = borderval

    def __call__(self, sample):
        """Extracts and returns a central crop from the provided image.

        Args:
            sample: the image to generate the crop from; should be a 2d or 3d numpy array.

        Returns:
            The center crop.
        """
        assert isinstance(sample, np.ndarray), f"sample type should be np.ndarray (got {type(sample)})"
        assert 2 <= sample.ndim <= 3, "bad input dimensions; must be 2-d, or 3-d (with channels)"
        crop_height = int(round(self.size[1] * sample.shape[0])) if self.relative else self.size[1]
        crop_width = int(round(self.size[0] * sample.shape[1])) if self.relative else self.size[0]
        tl = [sample.shape[1] // 2 - crop_width // 2, sample.shape[0] // 2 - crop_height // 2]
        br = [tl[0] + crop_width, tl[1] + crop_height]
        return thelper.draw.safe_crop(sample, tl, br, self.bordertype, self.borderval)

    def invert(self, sample):
        """Specifies that this operation cannot be inverted, as data loss is incurred during image transformation."""
        raise RuntimeError("cannot be inverted")

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(size={self.size}, bordertype={self.bordertype}, borderval={self.borderval})"


class RandomResizedCrop:
    """Returns a resized crop of a randomly selected image region.

    This operation is stochastic, and thus cannot be inverted. Each time the operation is called,
    a random check will determine whether a transformation is applied or not. The code relies on
    OpenCV, meaning the interpolation arguments must be compatible with ``cv2.resize``.

    Attributes:
        output_size: size of the output crop, provided as a single element (``edge_size``) or as a
            two-element tuple or list (``[width, height]``). If integer values are used, the size is
            assumed to be absolute. If floating point values are used (i.e. in [0,1]), the output
            size is assumed to be relative to the original image size, and will be determined at
            execution time for each sample. If set to ``None``, the crop will not be resized.
        input_size: range of the input region sizes, provided as a pair of elements
            (``[min_edge_size, max_edge_size]``) or as a pair of tuples or lists
            (``[[min_width, min_height], [max_width, max_height]]``). If the pair-of-pairs format is
            used, the ``ratio`` argument cannot be used. If integer values are used, the ranges are
            assumed to be absolute. If floating point values are used (i.e. in [0,1]), the ranges
            are assumed to be relative to the original image size, and will be determined at
            execution time for each sample.
        ratio: range of minimum/maximum input region aspect ratios to use. This argument cannot be
            used if the pair-of-pairs format is used for the ``input_size`` argument.
        probability: the probability that the transformation will be applied when called; if not
            applied, the returned image will be the original.
        random_attempts: the number of random sampling attempts to try before reverting to center
            or most-probably-valid crop generation.
        min_roi_iou: minimum roi intersection over union (IoU) required for accepting a tile (in [0,1]).
        flags: interpolation flag forwarded to ``cv2.resize``.
    """

    def __init__(self, output_size, input_size=(0.08, 1.0), ratio=(0.75, 1.33), probability=1.0,
                 random_attempts=10, min_roi_iou=1.0, flags=cv.INTER_LINEAR):
        """Validates and initializes center crop parameters.

        Args:
            output_size: size of the output crop, provided as a single element (``edge_size``) or as a
                two-element tuple or list (``[width, height]``). If integer values are used, the size is
                assumed to be absolute. If floating point values are used (i.e. in [0,1]), the output
                size is assumed to be relative to the original image size, and will be determined at
                execution time for each sample. If set to ``None``, the crop will not be resized.
            input_size: range of the input region sizes, provided as a pair of elements
                (``[min_edge_size, max_edge_size]``) or as a pair of tuples or lists
                (``[[min_width, min_height], [max_width, max_height]]``). If the pair-of-pairs format is
                used, the ``ratio`` argument cannot be used. If integer values are used, the ranges are
                assumed to be absolute. If floating point values are used (i.e. in [0,1]), the ranges
                are assumed to be relative to the original image size, and will be determined at
                execution time for each sample.
            ratio: range of minimum/maximum input region aspect ratios to use. This argument cannot be
                used if the pair-of-pairs format is used for the ``input_size`` argument.
            probability: the probability that the transformation will be applied when called; if not
                applied, the returned image will be the original.
            random_attempts: the number of random sampling attempts to try before reverting to center
                or most-probably-valid crop generation.
            min_roi_iou: minimum roi intersection over union (IoU) required for producing a tile.
            flags: interpolation flag forwarded to ``cv2.resize``.
        """
        if output_size is None or isinstance(output_size, (tuple, list)):
            if output_size is not None:
                assert len(output_size) == 2, "expected output size to be two-element list or tuple, or single scalar"
                assert all([isinstance(s, int) for s in output_size]) or all([isinstance(s, float) for s in output_size]), \
                    "expected output size pair elements to be the same type (int or float)"
            self.output_size = output_size
        elif isinstance(output_size, (int, float)):
            self.output_size = (output_size, output_size)
        else:
            raise TypeError("unexpected output size type (need tuple/list/int/float)")
        if self.output_size is not None:
            for s in self.output_size:
                assert ((isinstance(s, float) and 0 < s <= 1) or (isinstance(s, int) and s > 0)), \
                    f"invalid output size value ({str(s)})"
        assert isinstance(input_size, (tuple, list)) and len(input_size) == 2, \
            "expected input size to be provided as a pair of elements or a pair of tuples/lists"
        if all([isinstance(s, int) for s in input_size]) or all([isinstance(s, float) for s in input_size]):
            if ratio is not None and isinstance(ratio, (tuple, list)):
                assert len(ratio) == 2, "invalid ratio tuple/list length (expected two elements)"
                assert all([isinstance(r, float) for r in ratio]), "expected ratio pair elements to be float values"
                self.ratio = (min(ratio), max(ratio))
            elif ratio is not None and isinstance(ratio, float):
                self.ratio = (ratio, ratio)
            else:
                raise TypeError("invalid aspect ratio, expected 2-element tuple/list or single float")
            self.input_size = ((min(input_size), min(input_size)), (max(input_size), max(input_size)))
        elif all([isinstance(s, tuple) and len(s) == 2 for s in input_size]) or \
                all([isinstance(s, list) and len(s) == 2 for s in input_size]):
            assert ratio is None or (isinstance(ratio, (tuple, list)) and len(ratio) == 0), \
                "cannot specify input sizes in two-element tuples/lists and also provide aspect ratios"
            for t in input_size:
                for s in t:
                    assert isinstance(s, (int, float)) and not isinstance(type(s), type(input_size[0][0])), \
                        "input sizes should all be same type, either int or float"
            self.input_size = ((min(input_size[0][0], input_size[1][0]), min(input_size[0][1], input_size[1][1])),
                               (max(input_size[0][0], input_size[1][0]), max(input_size[0][1], input_size[1][1])))
            self.ratio = None  # ignored since input_size contains all necessary info
        else:
            raise TypeError("expected input size to be two-elem list/tuple of int/float or two-element list/tuple of int/float")
        for t in self.input_size:
            assert ((isinstance(t, float) and 0 < s <= 1) or (isinstance(t, int) and s > 0)) or \
                all([((isinstance(s, float) and 0 < s <= 1) or (isinstance(s, int) and s > 0)) for s in t]), \
                "invalid input size value"
        assert 0 <= probability <= 1, "invalid probability value (should be in [0,1]"
        self.probability = probability
        assert random_attempts > 0, "invalid random_attempts value (should be > 0)"
        self.random_attempts = random_attempts
        assert isinstance(min_roi_iou, float) and 0 <= min_roi_iou <= 1, \
            "invalid minimum roi IoU score (should be float in [0,1])"
        self.min_roi_iou = min_roi_iou
        self.flags = thelper.utils.import_class(flags) if isinstance(flags, str) else flags
        self.warned_no_crop_found_with_roi = False

    def __call__(self, image, roi=None, mask=None, bboxes=None):
        """Extracts and returns a random (resized) crop from the provided image.

        Args:
            image: the image to generate the crop from. If given as a 2-element list, it is assumed to
                contain both the image and the roi (passed through a composer).
            roi: the roi to check tile intersections with (may be ``None``).
            mask: a mask to crop simultaneously with the input image (may be ``None``).
            bboxes: a list or array of bounding boxes to crop with the input image (may be ``None``).

        Returns:
            The randomly selected and resized crop. If mask and/or bboxes is given, the output will be
            a dictionary containing the results under the ``image``, ``mask``, and ``bboxes`` keys.
        """
        if isinstance(image, list) and len(image) == 2:
            assert roi is None, "roi provided twice"
            # we assume that the roi was given as the 2nd element of the list
            image, roi = image[0], image[1]
        assert isinstance(image, (PIL.Image.Image, np.ndarray)), \
            "image type should be np.ndarray or PIL image"
        if isinstance(image, PIL.Image.Image):
            image = np.asarray(image)
        if self.probability < 1 and np.random.uniform(0, 1) > self.probability:
            return image
        if bboxes is not None:
            raise NotImplementedError
        image_height, image_width = image.shape[0], image.shape[1]
        target_height, target_width = None, None
        target_row, target_col = None, None
        for attempt in range(self.random_attempts):
            if self.ratio is None:
                target_width = np.random.uniform(self.input_size[0][0], self.input_size[1][0])
                target_height = np.random.uniform(self.input_size[0][1], self.input_size[1][1])
                if isinstance(self.input_size[0][0], float):
                    target_width *= image_width
                    target_height *= image_height
                target_width = int(round(target_width))
                target_height = int(round(target_height))
            elif isinstance(self.input_size[0][0], (int, float)):
                if isinstance(self.input_size[0][0], float):
                    area = image_height * image_width
                    target_area = np.random.uniform(self.input_size[0][0], self.input_size[1][0]) * area
                else:
                    target_area = np.random.uniform(self.input_size[0][0], self.input_size[1][0]) ** 2
                aspect_ratio = np.random.uniform(*self.ratio)
                target_width = int(round(math.sqrt(target_area * aspect_ratio)))
                target_height = int(round(math.sqrt(target_area / aspect_ratio)))
                if np.random.random() < 0.5:
                    target_width, target_height = target_height, target_width
            else:
                raise RuntimeError("unhandled crop strategy")
            if target_width <= image_width and target_height <= image_height:
                target_col = np.random.randint(min(0, image_width - target_width), max(0, image_width - target_width) + 1)
                target_row = np.random.randint(min(0, image_height - target_height), max(0, image_height - target_height) + 1)
                if roi is None:
                    break
                roi = thelper.draw.safe_crop(roi, (target_col, target_row), (target_col + target_width, target_row + target_height))
                if np.count_nonzero(roi) >= target_width * target_height * self.min_roi_iou:
                    break
        if target_row is None or target_col is None:
            # fallback, use centered crop
            target_width = target_height = min(image.shape[0], image.shape[1])
            target_col = (image.shape[1] - target_width) // 2
            target_row = (image.shape[0] - target_height) // 2
            if roi is not None and not self.warned_no_crop_found_with_roi:
                logger.warning("random resized crop failing to find proper ROI matches after max attempt count")
                self.warned_no_crop_found_with_roi = True
        crop = thelper.draw.safe_crop(image, (target_col, target_row), (target_col + target_width, target_row + target_height))
        if self.output_size is None:
            return crop
        elif isinstance(self.output_size[0], float):
            output_width = int(round(self.output_size[0] * image.shape[1]))
            output_height = int(round(self.output_size[1] * image.shape[0]))
            return cv.resize(crop, (output_width, output_height), interpolation=self.flags)
        elif isinstance(self.output_size[0], int):
            return cv.resize(crop, (self.output_size[0], self.output_size[1]), interpolation=self.flags)
        else:
            raise RuntimeError("unhandled crop strategy")

    def invert(self, image):
        """Specifies that this operation cannot be inverted, as data loss is incurred during image transformation."""
        raise RuntimeError("cannot be inverted")

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(output_size={self.output_size}, input_size={self.input_size}, ratio={self.ratio}, " + \
            f"probability={self.probability}, random_attempts={self.random_attempts}, " + \
            f"min_roi_iou={self.min_roi_iou}, flags={self.flags})"

    def set_seed(self, seed):
        """Sets the internal seed to use for stochastic ops."""
        np.random.seed(seed)


class Resize:
    """Resizes a given image using OpenCV and numpy.

    This operation is deterministic. The code relies on OpenCV, meaning the interpolation arguments
    must be compatible with ``cv2.resize``.

    Attributes:
        interp: interpolation type to use (forwarded to ``cv2.resize``)
        buffer: specifies whether a destination buffer should be used to avoid allocations
        dsize: target image size (tuple of width, height).
        dst: buffer used to avoid reallocations if ``self.buffer == True``
        fx: argument forwarded to ``cv2.resize``.
        fy: argument forwarded to ``cv2.resize``.
    """

    def __init__(self, dsize, fx=0, fy=0, interp=cv.INTER_LINEAR, buffer=False):
        """Validates and initializes resize parameters.

        Args:
            dsize: size of the target image, forwarded to ``cv2.resize``.
            fx: x-scaling factor, forwarded to ``cv2.resize``.
            fy: y-scaling factor, forwarded to ``cv2.resize``.
            interp: resize interpolation type, forwarded to ``cv2.resize``.
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
        assert 0 <= fx and 0 <= fy, "scale factors should be null (ignored) or positive"
        assert 0 <= dsize[0] and 0 <= dsize[1], "destination image size should be null (ignored) or positive"
        assert fx != 0 or fy != 0 or dsize[0] != 0 or dsize[1] != 0, \
            "need to specify either destination size or scale factor(s)"

    def __call__(self, sample):
        """Returns a resized copy of the provided image.

        Args:
            sample: the image to resize; should be a 2d or 3d numpy array.

        Returns:
            The resized image. May be allocated on the spot, or be a pointer to a local buffer.
        """
        assert isinstance(sample, (PIL.Image.Image, np.ndarray)), \
            f"sample type should be np.ndarray or PIL image (got {type(sample)})"
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        assert 2 <= sample.ndim <= 3, "bad input dimensions; must be 2-d, or 3-d (with channels)"
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
                slices_dst[idx] = cv.resize(slices[idx], self.dsize, dst=slices_dst[idx],
                                            fx=self.fx, fy=self.fy, interpolation=self.interp)
            if self.buffer:
                self.dst = slices_dst
            return np.stack(slices_dst, 2)

    def invert(self, sample):
        """Specifies that this operation cannot be inverted, as data loss is incurred during image transformation."""
        # todo, could implement if original size is fixed & known
        raise NotImplementedError

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(dsize={self.dsize}, fx={self.fx}, fy={self.fy}, interp={self.interp}, buffer={self.buffer})"


class Affine:
    """Warps a given image using an affine matrix via OpenCV and numpy.

    This operation is deterministic. The code relies on OpenCV, meaning the border arguments
    must be compatible with ``cv2.warpAffine``.

    Attributes:
        transf: the 2x3 transformation matrix passed to ``cv2.warpAffine``.
        out_size: target image size (tuple of width, height). If None, same as original.
        flags: extra warp flags forwarded to ``cv2.warpAffine``.
        border_mode: border extrapolation mode forwarded to ``cv2.warpAffine``.
        border_val: border constant extrapolation value forwarded to ``cv2.warpAffine``.
    """

    def __init__(self, transf, out_size=None, flags=None, border_mode=None, border_val=None):
        """Validates and initializes affine warp parameters.

        Args:
            transf: the 2x3 transformation matrix passed to ``cv2.warpAffine``.
            out_size: target image size (tuple of width, height). If None, same as original.
            flags: extra warp flags forwarded to ``cv2.warpAffine``.
            border_mode: border extrapolation mode forwarded to ``cv2.warpAffine``.
            border_val: border constant extrapolation value forwarded to ``cv2.warpAffine``.
        """
        if isinstance(transf, np.ndarray):
            assert transf.size == 6, "transformation matrix must be 2x3"
            self.transf = transf.reshape((2, 3)).astype(np.float32)
        elif isinstance(transf, list):
            assert len(transf) == 6, "transformation matrix must be 6 elements (2x3)"
            self.transf = np.asarray(transf).reshape((2, 3)).astype(np.float32)
        else:
            raise TypeError("unexpected transformation matrix type")
        self.out_size = None
        if out_size is not None:
            if isinstance(out_size, list):
                assert len(out_size) == 2, "output image size should be 2-elem list or tuple"
                self.out_size = tuple(out_size)
            elif isinstance(out_size, tuple):
                assert len(out_size) == 2, "output image size should be 2-elem list or tuple"
                self.out_size = out_size
            else:
                raise TypeError("unexpected output size type")
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
        assert isinstance(sample, (PIL.Image.Image, np.ndarray)), \
            f"sample type should be np.ndarray or PIL image (got {type(sample)})"
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        out_size = self.out_size
        if out_size is None:
            out_size = (sample.shape[1], sample.shape[0])
        return cv.warpAffine(sample, self.transf, dsize=out_size, flags=self.flags,
                             borderMode=self.border_mode, borderValue=self.border_val)

    def invert(self, sample):
        """Inverts the warp transformation, but only is the output image has not been cropped before."""
        assert isinstance(sample, (PIL.Image.Image, np.ndarray)), \
            f"sample type should be np.ndarray or PIL image (got {type(sample)})"
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        out_size = self.out_size
        if out_size is None:
            out_size = (sample.shape[1], sample.shape[0])
        else:
            raise RuntimeError("unknown original image size, cannot invert affine transform")
        return cv.warpAffine(sample, self.transf, dsize=out_size, flags=self.flags ^ cv.WARP_INVERSE_MAP,
                             borderMode=self.border_mode, borderValue=self.border_val)

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(transf={self.transf}, out_size={self.out_size}, flags={self.flags}, " + \
            f"border_mode={self.border_mode}, border_val={self.border_val})"


class RandomShift:
    """Randomly translates an image in a provided range via OpenCV and numpy.

    This operation is stochastic, and thus cannot be inverted. Each time the operation is called,
    a random check will determine whether a transformation is applied or not. The code relies on
    OpenCV, meaning the border arguments must be compatible with ``cv2.warpAffine``.

    Attributes:
        min: the minimum pixel shift that can be applied stochastically.
        max: the maximum pixel shift that can be applied stochastically.
        probability: the probability that the transformation will be applied when called.
        flags: extra warp flags forwarded to ``cv2.warpAffine``.
        border_mode: border extrapolation mode forwarded to ``cv2.warpAffine``.
        border_val: border constant extrapolation value forwarded to ``cv2.warpAffine``.
    """

    def __init__(self, min, max, probability=1.0, flags=None, border_mode=None, border_val=None):
        """Validates and initializes shift parameters.

        Args:
            min: the minimum pixel shift that can be applied stochastically.
            max: the maximum pixel shift that can be applied stochastically.
            probability: the probability that the transformation will be applied when called.
            flags: extra warp flags forwarded to ``cv2.warpAffine``.
            border_mode: border extrapolation mode forwarded to ``cv2.warpAffine``.
            border_val: border constant extrapolation value forwarded to ``cv2.warpAffine``.
        """
        if isinstance(min, tuple) and isinstance(max, tuple):
            assert len(min) == len(max) and len(min) == 2, "min/max shift tuple must be 2-elem"
            self.min = min
            self.max = max
        elif isinstance(min, list) and isinstance(max, list):
            assert len(min) == len(max) and len(min) == 2, "min/max shift list must be 2-elem"
            self.min = tuple(min)
            self.max = tuple(max)
        elif isinstance(min, (int, float)) and isinstance(max, (int, float)):
            self.min = (min, min)
            self.max = (max, max)
        else:
            raise TypeError("unexpected min/max combo types")
        assert self.max[0] >= self.min[0] or self.max[1] >= self.min[1], "bad min/max values"
        assert 0 <= probability <= 1, "bad probability range"
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
        assert isinstance(sample, (PIL.Image.Image, np.ndarray)), \
            f"sample type should be np.ndarray or PIL image (got {type(sample)})"
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        if self.probability < 1 and np.random.uniform(0, 1) > self.probability:
            return sample
        out_size = (sample.shape[1], sample.shape[0])
        x_shift = np.random.uniform(self.min[0], self.max[0])
        y_shift = np.random.uniform(self.min[1], self.max[1])
        transf = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        return cv.warpAffine(sample, transf, dsize=out_size, flags=self.flags, borderMode=self.border_mode, borderValue=self.border_val)

    def invert(self, sample):
        """Specifies that this operation cannot be inverted, as it is stochastic, and data loss occurs during transformation."""
        raise RuntimeError("operation cannot be inverted")

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(min={self.min}, max={self.max}, probability={self.probability}, flags={self.flags}, " + \
            f"border_mode={self.border_mode}, border_val={self.border_val})"

    def set_seed(self, seed):
        """Sets the internal seed to use for stochastic ops."""
        np.random.seed(seed)


class ToGray:
    """Converts a multi-channel image to grayscale.

    This operation is deterministic, but not reversible. It can be applied to images with
    more than three channels (RGB) --- in that case, it will compute their per-pixel mean
    value. Note that in any case, the last dimension (that corresponds to the channels) will
    remain and be of size 1.
    """

    def __init__(self):
        """Does nothing, there's no attribute to store for this operation."""
        pass

    def __call__(self, sample):
        assert isinstance(sample, (PIL.Image.Image, np.ndarray)), \
            f"sample type should be np.ndarray or PIL image (got {type(sample)})"
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        assert 2 <= sample.ndim <= 3, "array should have at least two dimensions + channels"
        if sample.ndim == 2:
            return np.expand_dims(sample, 2)
        elif sample.shape[2] == 1:
            return sample  # already grayscale, return immediately
        elif sample.shape[2] == 3:
            return np.expand_dims(cv.cvtColor(sample, cv.COLOR_BGR2GRAY), 2)
        return np.mean(sample, axis=2, keepdims=True)

    def invert(self, sample):
        """Specifies that this operation cannot be inverted, as data loss occurs during transformation."""
        raise RuntimeError("operation cannot be inverted")

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + "()"


class ToColor:
    """Converts a single-channel image to color (RGB).

    This operation is deterministic and reversible. It CANNOT be applied to images with
    more than one channel (HxWx1). The byte ordering (BGR or RGB) does not matter.
    """

    def __init__(self):
        """Does nothing, there's no attribute to store for this operation."""
        pass

    def __call__(self, sample):
        assert isinstance(sample, (PIL.Image.Image, np.ndarray)), \
            f"sample type should be np.ndarray or PIL image (got {type(sample)})"
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        assert 2 <= sample.ndim <= 3, "array should have at least two dimensions + channels"
        if sample.ndim == 2:
            return cv.cvtColor(sample, cv.COLOR_GRAY2BGR)
        elif sample.shape[2] == 1:
            return cv.cvtColor(sample[..., 0], cv.COLOR_GRAY2BGR)
        else:
            raise AssertionError("unexpected channel count in input sample")

    def invert(self, sample):
        """Inverts the operation by calling the 'ToGray' operation.

        Note that this operation is probably lossy due to OpenCV's grayscale conversion code
        which uses "0.21 R + 0.72 G + 0.07 B" to compute the luminosity of a pixel.
        """
        return ToGray()(sample)

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + "()"


class Transpose:
    """Transposes an image via numpy.

    This operation is deterministic.

    Attributes:
        axes: the axes on which to apply the transpose; forwarded to ``numpy.transpose``.
        axes_inv: used to invert the tranpose; also forwarded to ``numpy.transpose``.
    """

    def __init__(self, axes):
        """Validates and initializes tranpose parameters.

        Args:
            axes: the axes on which to apply the transpose; forwarded to ``numpy.transpose``.
        """
        axes = np.asarray(axes)
        assert axes.ndim == 1, "tranpose param should be 1-d"
        assert not np.any(axes >= axes.size), "oob dim in axes"
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
        assert isinstance(sample, np.ndarray), f"sample type should be np.ndarray (got {type(sample)})"
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
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + f"(axes={self.axes})"


class Unsqueeze:
    """Expands a dimension in the input array via numpy/PyTorch.

    This operation is deterministic.

    Attributes:
        axis: the axis on which to apply the expansion.
    """

    def __init__(self, axis):
        """Validates and initializes tranpose parameters.

        Args:
            axis: the axis on which to apply the expansion.
        """
        self.axis = axis

    def __call__(self, sample):
        """Expands a dimension in the input array via numpy/PyTorch.

        Args:
            sample: the array to expand.

        Returns:
            The array with an extra dimension.
        """
        if isinstance(sample, np.ndarray):
            return np.expand_dims(sample, self.axis)
        elif isinstance(sample, torch.Tensor):
            return torch.unsqueeze(sample, self.axis)
        else:
            raise TypeError(f"unexpected input type ('{type(sample)}')")

    def invert(self, sample):
        """Squeezes a dimension in the input array via numpy/PyTorch.

        Args:
            sample: the array to squeeze.

        Returns:
            The array with one less dimension.
        """
        if isinstance(sample, np.ndarray):
            return np.squeeze(sample, self.axis)
        elif isinstance(sample, torch.Tensor):
            return torch.squeeze(sample, self.axis)
        else:
            raise TypeError(f"unexpected input type ('{type(sample)}')")

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + f"(axis={self.axis})"


class Duplicator(NoTransform):
    """Duplicates and returns a list of copies of the input sample.

    This operation is used in data augmentation pipelines that rely on probabilistic or preset transformations.
    It can produce a fixed number of simple copies or deep copies of the input samples as required.

    .. warning::
        Since the duplicates will be given directly to the data loader as part of the same minibatch, using too
        many copies can adversely affect gradient descent for that minibatch. To simply increase the total size
        of the training set while still allowing a proper shuffling of samples and/or to keep the minibatch size
        intact, we instead recommend setting the ``train_scale`` configuration value in the data loader. See
        :func:`thelper.data.utils.create_loaders` for more information.

    Attributes:
        count: number of copies to generate.
        deepcopy: specifies whether to deep-copy samples or not.
    """

    def __init__(self, count, deepcopy=False):
        """Validates and initializes duplication parameters.

        Args:
            count: number of copies to generate.
            deepcopy: specifies whether to deep-copy samples or not.
        """
        assert count > 0, "invalid copy count"
        self.count = count
        self.deepcopy = deepcopy

    def __call__(self, sample):
        """Generates and returns duplicates of the sample/object.

        If a dictionary is provided, its values will be expanded into lists that
        will contain all duplicates. Otherwise, the duplicates will be returned
        directly as a list.

        Args:
            sample: the sample/object to duplicate.

        Returns:
            A list of duplicated samples, or a dictionary of duplicate lists.
        """
        copyfct = copy.deepcopy if self.deepcopy else copy.copy
        if isinstance(sample, dict):
            return {k: [copyfct(v) for _ in range(self.count)] for k, v in sample.items()}
        else:
            return [copyfct(sample) for _ in range(self.count)]

    def invert(self, sample):
        """Returns the first instance of the list of duplicates."""
        if isinstance(sample, dict):
            assert all([isinstance(v, list) for v in sample.values()]), "invalid sample type (should be list)"
            assert not any([len(v) != self.count for v in sample.values()]), "invalid sample list length"
            return {k: v[0] for k, v in sample.items()}
        else:
            assert isinstance(sample, list), "invalid sample type (should be list)"
            assert len(sample) == self.count, "invalid sample list length"
            return sample[0]

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(count={self.count}, deepcopy={self.deepcopy})"


class Tile:
    """Returns a list of tiles cut out from a given image.

    This operation can perform tiling given an optional mask with a target intersection over union (IoU) score,
    and with an optional overlap between tiles. The tiling is deterministic and can thus be inverted, but only
    if a mask is not used, as some image regions may be lost otherwise.

    If a mask is used, the first tile position is tested exhaustively by iterating over all input coordinates
    starting from the top-left corner of the image. Otherwise, the first tile position is set as (0,0). Then,
    all other tiles are found by offsetting frm these coordinates, and testing for IoU with the mask (if needed).

    Attributes:
        tile_size: size of the output tiles, provided as a single element (``edge_size``) or as a
            two-element tuple or list (``[width, height]``). If integer values are used, the size is
            assumed to be absolute. If floating point values are used (i.e. in [0,1]), the output
            size is assumed to be relative to the original image size, and will be determined at
            execution time for each image.
        tile_overlap: overlap allowed between two neighboring tiles; should be a ratio in [0,1].
        min_mask_iou: minimum mask intersection over union (IoU) required for accepting a tile (in [0,1]).
        offset_overlap: specifies whether the overlap tiling should be offset outside the image or not.
        bordertype: border copy type to use when the image is too small for the required crop size.
            See ``cv2.copyMakeBorder`` for more information.
        borderval: border value to use when the image is too small for the required crop size. See
            ``cv2.copyMakeBorder`` for more information.
    """

    def __init__(self, tile_size, tile_overlap=0.0, min_mask_iou=1.0, offset_overlap=False, bordertype=cv.BORDER_CONSTANT, borderval=0):
        """Validates and initializes tiling parameters.

        Args:
            tile_size: size of the output tiles, provided as a single element (``edge_size``) or as a
                two-element tuple or list (``[width, height]``). If integer values are used, the size is
                assumed to be absolute. If floating point values are used (i.e. in [0,1]), the output
                size is assumed to be relative to the original image size, and will be determined at
                execution time for each image.
            tile_overlap: overlap ratio between two consecutive (neighboring) tiles; should be in [0,1].
            min_mask_iou: minimum mask intersection over union (IoU) required for producing a tile.
            offset_overlap: specifies whether the overlap tiling should be offset outside the image or not.
            bordertype: border copy type to use when the image is too small for the required crop size.
                See ``cv2.copyMakeBorder`` for more information.
            borderval: border value to use when the image is too small for the required crop size. See
                ``cv2.copyMakeBorder`` for more information.
        """
        if isinstance(tile_size, (tuple, list)):
            assert len(tile_size) == 2, "expected tile size to be two-element list or tuple, or single scalar"
            assert all([isinstance(s, int) for s in tile_size]) or all([isinstance(s, float) for s in tile_size]), \
                "expected tile size pair elements to be the same type (int or float)"
            self.tile_size = tile_size
        elif isinstance(tile_size, (int, float)):
            self.tile_size = (tile_size, tile_size)
        else:
            raise TypeError("unexpected tile size type (need tuple/list/int/float)")
        assert isinstance(tile_overlap, float) and 0 <= tile_overlap < 1, \
            "invalid tile overlap (should be float in [0,1[)"
        self.tile_overlap = tile_overlap
        assert isinstance(min_mask_iou, float) and 0 <= min_mask_iou <= 1, \
            "invalid minimum mask IoU score (should be float in [0,1])"
        self.min_mask_iou = min_mask_iou
        self.offset_overlap = offset_overlap
        self.bordertype = thelper.utils.import_class(bordertype) if isinstance(bordertype, str) else bordertype
        self.borderval = borderval

    def __call__(self, image, mask=None):
        """Extracts and returns a list of tiles cut out from the given image.

        Args:
            image: the image to cut into tiles. If given as a 2-element list, it is assumed to contain both
                the image and the mask (passed through a composer).
            mask: the mask to check tile intersections with (may be ``None``).

        Returns:
            A list of tiles (numpy-compatible images).
        """
        if isinstance(image, list) and len(image) == 2:
            assert mask is None, "mask provided twice"
            # we assume that the mask was given as the 2nd element of the list
            image, mask = image[0], image[1]
        tile_rects, tile_images = self._get_tile_rects(image, mask), []
        for rect in tile_rects:
            tile_images.append(thelper.draw.safe_crop(image, (rect[0], rect[1]),
                                                      (rect[0] + rect[2], rect[1] + rect[3]),
                                                      self.bordertype, self.borderval))
        return tile_images

    def count_tiles(self, image, mask=None):
        """Returns the number of tiles that would be cut out from the given image.

        Args:
            image: the image to cut into tiles. If given as a 2-element list, it is assumed to contain both
                the image and the mask (passed through a composer).
            mask: the mask to check tile intersections with (may be ``None``).

        Returns:
            The number of tiles that would be cut with :func:`thelper.transforms.operations.Tile.__call__`.
        """
        if isinstance(image, list) and len(image) == 2:
            assert mask is None, "mask provided twice"
            # we assume that the mask was given as the 2nd element of the list
            image, mask = image[0], image[1]
        return len(self._get_tile_rects(image, mask))

    def _get_tile_rects(self, image, mask=None):
        assert isinstance(image, (PIL.Image.Image, np.ndarray)), \
            "image type should be np.ndarray or PIL image"
        if isinstance(image, PIL.Image.Image):
            image = np.asarray(image)
        if mask is not None:
            assert isinstance(mask, (PIL.Image.Image, np.ndarray)), \
                "mask type should be np.ndarray or PIL image"
            if isinstance(mask, PIL.Image.Image):
                mask = np.asarray(mask)
        tile_rects = []
        height, width = image.shape[0], image.shape[1]
        if isinstance(self.tile_size[0], float):
            tile_size = (int(round(self.tile_size[0] * width)), int(round(self.tile_size[1] * height)))
        else:
            tile_size = self.tile_size
        overlap = (int(round(tile_size[0] * self.tile_overlap)), int(round(tile_size[1] * self.tile_overlap)))
        overlap_offset = (-overlap[0] // 2, -overlap[1] // 2) if self.offset_overlap else (0, 0)
        step_size = (max(tile_size[0] - (overlap[0] // 2) * 2, 1), max(tile_size[1] - (overlap[1] // 2) * 2, 1))
        req_mask_area = tile_size[0] * tile_size[1] * self.min_mask_iou
        if mask is not None:
            assert height == mask.shape[0] and width == mask.shape[1], "image and mask dimensions mismatch"
            assert mask.ndim == 2, "mask should be 2d binary (uchar) array"
            offset_coord = None
            row_range = range(overlap_offset[1], height - overlap_offset[1] - tile_size[1] + 1)
            col_range = range(overlap_offset[0], width - overlap_offset[0] - tile_size[0] + 1)
            for row, col in itertools.product(row_range, col_range):
                crop = thelper.draw.safe_crop(mask, (col, row), (col + tile_size[0], row + tile_size[1]))
                if np.count_nonzero(crop) >= req_mask_area:
                    offset_coord = (overlap_offset[0] + ((col - overlap_offset[0]) % step_size[0]),
                                    overlap_offset[1] + ((row - overlap_offset[1]) % step_size[1]))
                    break
            if offset_coord is None:
                return tile_rects
        else:
            offset_coord = overlap_offset
        row = offset_coord[1]
        while row + tile_size[1] <= height - overlap_offset[1]:
            col = offset_coord[0]
            while col + tile_size[0] <= width - overlap_offset[0]:
                if mask is not None:
                    crop = thelper.draw.safe_crop(mask, (col, row), (col + tile_size[0], row + tile_size[1]))
                    if np.count_nonzero(crop) >= req_mask_area:
                        tile_rects.append((col, row, tile_size[0], tile_size[1]))  # rect = (x, y, w, h)
                else:
                    tile_rects.append((col, row, tile_size[0], tile_size[1]))  # rect = (x, y, w, h)
                col += step_size[0]
            row += step_size[1]
        return tile_rects

    def invert(self, image, mask=None):
        """Returns the reconstituted image from a list of tiles, or throws if a mask was used."""
        assert mask is None, "cannot invert operation, mask might have forced the loss of image content"
        raise NotImplementedError  # TODO

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(tile_size={self.tile_size}, tile_overlap={self.tile_overlap}, min_mask_iou={self.min_mask_iou}, " + \
            f"offset_overlap={self.offset_overlap}, bordertype={self.bordertype}, borderval={self.borderval})"


class NormalizeZeroMeanUnitVar:
    """Normalizes a given image using a set of mean and standard deviation parameters.

    The samples will be transformed such that ``s = (s - mean) / std``.

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
        assert self.mean.ndim == 1 and self.std.ndim == 1, "normalization params should be 1-d"
        assert self.mean.size == self.std.size, "normalization params size mismatch"
        assert not any([d == 0 for d in self.std]), "normalization std must be non-null"

    def __call__(self, sample):
        """Normalizes a given sample.

        Args:
            sample: the sample to normalize. If given as a PIL image, it
            will be converted to a numpy array first.

        Returns:
            The warped sample, in a numpy array of type ``self.out_type``.
        """
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        if isinstance(sample, np.ndarray):
            return ((sample - self.mean) / self.std).astype(self.out_type)
        elif isinstance(sample, torch.Tensor):
            out = torchvision.transforms.functional.normalize(sample,
                                                              torch.from_numpy(self.mean),
                                                              torch.from_numpy(self.std))
            assert self.out_type != np.float32, "missing impl for non-float torch normalize output"
            return out.float()
        else:
            raise TypeError(f"sample type should be np.ndarray or torch.Tensor (got {type(sample)})")

    def invert(self, sample):
        """Inverts the normalization."""
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return (sample * self.std + self.mean).astype(self.out_type)

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(mean={self.mean}, std={self.std}, out_type={self.out_type})"


class NormalizeMinMax:
    """Normalizes a given image using a set of minimum and maximum values.

    The samples will be transformed such that ``s = (s - min) / (max - min)``.

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
        assert self.min.ndim == 1 and self.max.ndim == 1, \
            "normalization params should be a 1-d array (one value per channel)"
        assert self.min.size == self.max.size, "normalization params size mismatch"
        self.diff = self.max - self.min
        assert not any([d == 0 for d in self.diff]), "normalization diff must be non-null"

    def __call__(self, sample):
        """Normalizes a given sample.

        Args:
            sample: the sample to normalize. If given as a PIL image, it
            will be converted to a numpy array first.

        Returns:
            The warped sample, in a numpy array of type ``self.out_type``.
        """
        assert isinstance(sample, (PIL.Image.Image, np.ndarray)), \
            f"sample type should be np.ndarray or PIL image (got {type(sample)})"
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return ((sample - self.min) / self.diff).astype(self.out_type)

    def invert(self, sample):
        """Inverts the normalization."""
        assert isinstance(sample, (PIL.Image.Image, np.ndarray)), \
            f"sample type should be np.ndarray or PIL image (got {type(sample)})"
        if isinstance(sample, PIL.Image.Image):
            sample = np.asarray(sample)
        return (sample * self.diff + self.min).astype(self.out_type)

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(min={self.min}, max={self.max}, out_type={self.out_type})"
