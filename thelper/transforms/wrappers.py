"""Transformations wrappers module.

The wrapper classes herein are used to either support inline operations on odd sample types (e.g. lists
of images) or for external libraries (e.g. Augmentor).
"""

import logging
import random

import numpy as np
import PIL.Image
import torch

import thelper.utils

logger = logging.getLogger(__name__)


class AugmentorWrapper(object):
    """Augmentor pipeline wrapper that allows pickling and multithreading.

    See https://github.com/mdbloice/Augmentor for more information. This wrapper was last updated to work
    with version 0.2.2 --- more recent versions introduced yet unfixed (as of 2018/08) issues on some platforms.

    All original transforms are supported here. This wrapper also fixes the list output bug for single-image
    samples when using operations individually.

    Attributes:
        pipeline: the augmentor pipeline instance to apply to images.
        linked_fate: specifies whether input list samples should all have the same fate or not.

    .. seealso::
        | :func:`thelper.transforms.utils.load_transforms`
    """

    def __init__(self, pipeline, linked_fate=True):
        """Receives and stores an augmentor pipeline for later use.

        The pipeline itself is instantiated in :func:`thelper.transforms.utils.load_transforms`.
        """
        self.pipeline = pipeline
        self.linked_fate = linked_fate

    def __call__(self, samples, force_linked_fate=False, op_seed=None, in_cvts=None):
        """Transforms a single image (or a list of images) using the augmentor pipeline.

        Args:
            samples: the image(s) to transform (can also contain embedded lists/tuples of images).
            force_linked_fate: override flag for recursive use allowing forced linking of arrays.
            op_seed: seed to set before calling the wrapped operation.
            in_cvts: holds the input conversion flag array (for recursive usage).

        Returns:
            The transformed image(s), with the same list/tuple formatting as the input.
        """
        out_cvts = in_cvts is not None
        out_list = isinstance(samples, (list, tuple))
        if samples is None or (out_list and not samples):
            return ([], []) if out_cvts else []
        elif not out_list:
            samples = [samples]
        skip_unpack = in_cvts is not None and isinstance(in_cvts, bool) and in_cvts
        if self.linked_fate or force_linked_fate:  # process all samples/arrays with the same operations below
            if not skip_unpack:
                samples, cvts = ImageTransformWrapper._unpack(samples, convert_pil=True)
                if not isinstance(samples, (list, tuple)):
                    samples = [samples]
                    cvts = [cvts]
            else:
                cvts = in_cvts
            if op_seed is None:
                op_seed = np.random.randint(np.iinfo(np.int32).max)
            np.random.seed(op_seed)
            prev_state = np.random.get_state()
            for idx, _ in enumerate(samples):
                if not isinstance(samples[idx], PIL.Image.Image):
                    samples[idx], cvts[idx] = self(samples[idx], force_linked_fate=True, op_seed=op_seed, in_cvts=cvts[idx])
                else:
                    np.random.set_state(prev_state)
                    random.seed(np.random.randint(np.iinfo(np.int32).max))
                    for operation in self.pipeline.operations:
                        r = round(np.random.uniform(0, 1), 1)
                        if r <= operation.probability:
                            samples[idx] = operation.perform_operation([samples[idx]])[0]
        else:  # each element of the top array will be processed independently below (current seeds are kept)
            cvts = [False] * len(samples)
            for idx, _ in enumerate(samples):
                samples[idx], cvts[idx] = ImageTransformWrapper._unpack(samples[idx], convert_pil=True)
                if not isinstance(samples[idx], PIL.Image.Image):
                    samples[idx], cvts[idx] = self(samples[idx], force_linked_fate=True, op_seed=op_seed, in_cvts=cvts[idx])
                else:
                    random.seed(np.random.randint(np.iinfo(np.int32).max))
                    for operation in self.pipeline.operations:
                        r = round(np.random.uniform(0, 1), 1)
                        if r <= operation.probability:
                            samples[idx] = operation.perform_operation([samples[idx]])[0]
        samples, cvts = ImageTransformWrapper._pack(samples, cvts, convert_pil=True)
        if len(samples) != len(cvts):
            raise AssertionError("messed up packing/unpacking logic")
        if (skip_unpack or not out_list) and len(samples) == 1:
            samples = samples[0]
            cvts = cvts[0]
        return (samples, cvts) if out_cvts else samples

    def __repr__(self):
        """Create a print-friendly representation of inner augmentation stages."""
        return self.__class__.__name__ + "(" + ", ".join([str(t) for t in self.pipeline.operations]) + ")"

    def set_seed(self, seed):
        """Sets the internal seed to use for stochastic ops."""
        np.random.seed(seed)


class ImageTransformWrapper(object):
    """Image tranform wrapper that allows operations on lists/tuples.

    Can be used to wrap the operations in ``thelper.transforms`` or in ``torchvision.transforms``
    that only accept images as their input. Will optionally force-convert the samples to PIL images.

    Can also be used to transform a list/tuple of images uniformly based on a shared dice roll, or
    to ensure that each image is transformed independently.

    .. warning::
        Stochastic transforms (e.g. ``torchvision.transforms.RandomCrop``) will always treat each
        image in a list differently. If the same operations are to be applied to all images, you
        should consider using a series non-stochastic operations wrapped inside an instance of
        ``torchvision.transforms.RandomApply``, or simply provide the probability of applying the
        transforms to this wrapper's constructor.

    Attributes:
        operation: the wrapped operation (callable object or class name string to import).
        params: the parameters that are passed to the operation when init'd or called.
        probability: the probability that the wrapped operation will be applied.
        convert_pil: specifies whether images should be converted into PIL format or not.
        linked_fate: specifies whether images given in a list/tuple should have the same fate or not.
    """

    def __init__(self, operation, params=None, probability=1, convert_pil=False, linked_fate=True):
        """Receives and stores a torchvision transform operation for later use.

        If the operation is given as a string, it is assumed to be a class name and it will
        be imported. The parameters (if any) will then be given to the constructor of that
        class. Otherwise, the operation is assumed to be a callable object, and its parameters
        (if any) will be provided at call-time.

        Args:
            operation: the wrapped operation (callable object or class name string to import).
            params: the parameters that are passed to the operation when init'd or called.
            probability: the probability that the wrapped operation will be applied.
            convert_pil: specifies whether images should be forced into PIL format or not.
            linked_fate: specifies whether images given in a list/tuple should have the same fate or not.
        """
        if params is not None and not isinstance(params, dict):
            raise AssertionError("expected params to be passed in as a dictionary")
        if isinstance(operation, str):
            operation_type = thelper.utils.import_class(operation)
            self.operation = operation_type(**params) if params is not None else operation_type()
            self.params = {}
        else:
            self.operation = operation
            self.params = params if params is not None else {}
        if probability < 0 or probability > 1:
            raise AssertionError("invalid probability value (range is [0,1]")
        self.probability = probability
        self.convert_pil = convert_pil
        self.linked_fate = linked_fate

    @staticmethod
    def _unpack(sample, force_flatten=False, convert_pil=False):
        if isinstance(sample, (list, tuple)):
            if len(sample) > 1:
                if not force_flatten:
                    return sample, [False] * len(sample)
                flat_samples = []
                cvts = []
                for s in sample:
                    out, cvt = ImageTransformWrapper._unpack(s, force_flatten=force_flatten)
                    if isinstance(cvt, (list, tuple)):
                        if not isinstance(out, (list, tuple)):
                            raise AssertionError("unexpected out/cvt types")
                        flat_samples += list(out)
                        cvts += list(cvt)
                    else:
                        flat_samples.append(out)
                        cvts.append(cvt)
                return flat_samples, cvts
            else:
                sample = sample[0]
        if convert_pil:
            if isinstance(sample, torch.Tensor):
                sample = sample.numpy()
            if isinstance(sample, np.ndarray) and sample.ndim > 2 and sample.shape[-1] > 1 and (sample.dtype != np.uint8):
                # PIL images cannot handle multi-channel non-byte arrays; we handle these manually
                flat_samples = []
                for c in range(sample.shape[-1]):
                    flat_samples.append(PIL.Image.fromarray(sample[..., c]))
                return flat_samples, True  # this is the only case where an array can be paired with a single cvt flag
            else:
                out = PIL.Image.fromarray(np.squeeze(sample))
                return out, True
        return sample, False

    @staticmethod
    def _pack(samples, cvts, convert_pil=False):
        if not isinstance(samples, (list, tuple)) or not isinstance(cvts, (list, tuple)) or len(samples) != len(cvts):
            if not convert_pil or not isinstance(cvts, bool) or not cvts:
                raise AssertionError("unexpected cvts len w/ pil conversion (bad logic somewhere)")
            if not all([isinstance(s, PIL.Image.Image) for s in samples]):
                raise AssertionError("unexpected packed list sample types")
            samples = [np.asarray(s) for s in samples]
            if not all([s.ndim == 2 for s in samples]):
                raise AssertionError("unexpected packed list sample depths")
            samples = [np.expand_dims(s, axis=2) for s in samples]
            return [np.concatenate(samples, axis=2)], [False]
        for idx, cvt in enumerate(cvts):
            if not isinstance(cvt, (list, tuple)):
                if not isinstance(cvt, bool):
                    raise AssertionError("unexpected cvt type")
                if cvt:
                    if isinstance(samples[idx], (list, tuple)):
                        raise AssertionError("unexpected packed sample type")
                    samples[idx] = np.asarray(samples[idx])
                    cvts[idx] = False
        return samples, cvts

    def __call__(self, samples, force_linked_fate=False, op_seed=None, in_cvts=None):
        """Transforms a single image (or embedded lists/tuples of images) using a wrapped operation.

        Args:
            samples: the image(s) to transform (can also contain embedded lists/tuples of images).
            force_linked_fate: override flag for recursive use allowing forced linking of arrays.
            op_seed: seed to set before calling the wrapped operation.
            in_cvts: holds the input conversion flag array (for recursive usage).

        Returns:
            The transformed image(s), with the same list/tuple formatting as the input.
        """
        out_cvts = in_cvts is not None
        out_list = isinstance(samples, (list, tuple))
        if samples is None or (out_list and not samples):
            return ([], []) if out_cvts else []
        elif not out_list:
            samples = [samples]
        skip_unpack = in_cvts is not None and isinstance(in_cvts, bool) and in_cvts
        if self.linked_fate or force_linked_fate:  # process all samples/arrays with the same operations below
            if not skip_unpack:
                samples, cvts = self._unpack(samples, convert_pil=self.convert_pil)
                if not isinstance(samples, (list, tuple)):
                    samples = [samples]
                    cvts = [cvts]
            else:
                cvts = in_cvts
            if self.probability >= 1 or round(np.random.uniform(0, 1), 1) <= self.probability:
                if op_seed is None:
                    op_seed = np.random.randint(np.iinfo(np.int32).max)
                for idx, _ in enumerate(samples):
                    if isinstance(samples[idx], (list, tuple)):
                        samples[idx], cvts[idx] = self(samples[idx], force_linked_fate=True, op_seed=op_seed, in_cvts=cvts[idx])
                    else:
                        if hasattr(self.operation, "set_seed") and callable(self.operation.set_seed):
                            self.operation.set_seed(op_seed)
                        # watch out: if operation is stochastic and we cannot seed above, then there is no
                        # guarantee that the samples will truly have a 'linked fate' (this might cause issues!)
                        samples[idx] = self.operation(samples[idx], **self.params)
        else:  # each element of the top array will be processed independently below (current seeds are kept)
            cvts = [False] * len(samples)
            for idx, _ in enumerate(samples):
                samples[idx], cvts[idx] = self._unpack(samples[idx], convert_pil=self.convert_pil)
                if self.probability >= 1 or round(np.random.uniform(0, 1), 1) <= self.probability:
                    if isinstance(samples[idx], (list, tuple)):
                        # we will now force fate linkage for all subelements of this array
                        samples[idx], cvts[idx] = self(samples[idx], force_linked_fate=True, op_seed=op_seed, in_cvts=cvts[idx])
                    else:
                        samples[idx] = self.operation(samples[idx], **self.params)
        samples, cvts = ImageTransformWrapper._pack(samples, cvts, convert_pil=self.convert_pil)
        if len(samples) != len(cvts):
            raise AssertionError("messed up packing/unpacking logic")
        if (skip_unpack or not out_list) and len(samples) == 1:
            samples = samples[0]
            cvts = cvts[0]
        return (samples, cvts) if out_cvts else samples

    def __repr__(self):
        """Create a print-friendly representation of inner augmentation stages."""
        return self.__class__.__name__ + "(probability={0}, operation={1})".format(self.probability, str(self.operation))

    def set_seed(self, seed):
        """Sets the internal seed to use for stochastic ops."""
        np.random.seed(seed)
