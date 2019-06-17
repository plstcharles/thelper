"""Transformations wrappers module.

The wrapper classes herein are used to either support inline operations on odd sample types (e.g. lists
of images) or for external libraries (e.g. Augmentor).
"""

import functools
import logging
import random

import numpy as np
import PIL.Image
import torch

import thelper.data
import thelper.utils

logger = logging.getLogger(__name__)


class AlbumentationsWrapper:
    """Albumentations pipeline wrapper that allows dictionary unpacking.

    See https://github.com/albu/albumentations for more information.

    Attributes:
        pipeline: the augmentor pipeline instance to apply to images.
        image_key: the key to fetch images from (when dictionaries are passed in).
        bboxes_key: the key to fetch bounding boxes from (when dictionaries are passed in).
        mask_key: the key to fetch masks from (when dictionaries are passed in).
        keypoints_key: the key to fetch keypoints from (when dictionaries are passed in).
        cvt_kpts_to_bboxes: specifies whether keypoints should be converted to bboxes for compatbility.
        linked_fate: specifies whether input list samples should all have the same fate or not.

    .. seealso::
        | :func:`thelper.transforms.utils.load_transforms`
    """

    def __init__(self, transforms, to_tensor=None, bbox_params=None, add_targets=None, image_key="image",
                 bboxes_key="bboxes", mask_key="mask", keypoints_key="keypoints", probability=1.0,
                 cvt_kpts_to_bboxes=False, linked_fate=False):
        """Receives and stores an augmentor pipeline for later use.

        The pipeline itself is instantiated in :func:`thelper.transforms.utils.load_transforms`.
        """
        if bbox_params is None or not bbox_params:
            bbox_params = {"format": "coco"}  # i.e. opencv format (X,Y,W,H)
        if add_targets is None:
            add_targets = {}
        if isinstance(image_key, (list, tuple)):
            assert len(image_key) <= 1, "current implementation cannot handle more than one input image key per packet"
            image_key = image_key[0]
        self.image_key = image_key
        assert not (isinstance(bboxes_key, (list, tuple)) or
                    isinstance(keypoints_key, (list, tuple)) or
                    isinstance(mask_key, (list, tuple))), \
            "bboxes/keypoints/masks keys should never be passed as lists"
        self.bboxes_key = bboxes_key
        self.mask_key = mask_key
        self.keypoints_key = keypoints_key
        self.cvt_kpts_to_bboxes = cvt_kpts_to_bboxes
        assert not (cvt_kpts_to_bboxes and "format" not in bbox_params or bbox_params["format"] != "coco"), \
            "if converting kpts to bboxes, must use coco format"
        self.bbox_params = bbox_params
        self.linked_fate = linked_fate
        import albumentations
        self.transforms = transforms
        self.to_tensor = to_tensor
        self.add_targets = add_targets
        self.pipeline = albumentations.Compose(transforms, to_tensor=to_tensor, bbox_params=self.bbox_params,
                                               additional_targets=add_targets, p=probability)

    def __call__(self, sample, force_linked_fate=False, op_seed=None):
        """Transforms a (dict) sample, a single image, or a list of images using the augmentor pipeline.

        Args:
            sample: the sample or image(s) to transform (can also contain embedded lists/tuples of images).
            force_linked_fate: override flag for recursive use allowing forced linking of arrays.
            op_seed: seed to set before calling the wrapped operation.

        Returns:
            The transformed image(s), with the same list/tuple formatting as the input.
        """
        # todo: add list unwrapping/interlacing support like in other wrappers?
        params = {}
        unpack_bboxes, decode_bboxes = False, False
        if isinstance(sample, dict):
            assert self.image_key in sample, \
                f"image is missing from sample (key={self.image_key}) but it is mandatory"
            image = sample[self.image_key]
            if isinstance(image, (list, tuple)):
                raise NotImplementedError
                # impl should use linked_fate and force_linked_fate
            params["image"] = sample[self.image_key]
            if self.keypoints_key in sample and sample[self.keypoints_key] is not None:
                keypoints = sample[self.keypoints_key]
                if self.cvt_kpts_to_bboxes:
                    assert self.bboxes_key not in sample, \
                        "trying to override bboxes w/ keypoints while bboxes already exist"
                    # fake x,y,w,h,c format (w/ labels)
                    msize = params["image"].shape
                    params["bboxes"] = [[min(max(kp[0], 0), msize[1] - 1),
                                         min(max(kp[1], 0), msize[0] - 1), 1, 1, 0] for kp in keypoints]
                else:
                    params["keypoints"] = keypoints
            if self.bboxes_key in sample and sample[self.bboxes_key] is not None:
                bboxes = sample[self.bboxes_key]
                if isinstance(bboxes, thelper.data.BoundingBox):
                    bboxes = [bboxes]
                    unpack_bboxes = True
                if isinstance(bboxes, list) and all([isinstance(bbox, thelper.data.BoundingBox) for bbox in bboxes]):
                    assert self.bbox_params["format"] in ["coco", "pascal_voc"], "unsupported/unknown bbox format"
                    bboxes = [bbox.encode(format=self.bbox_params["format"]) for bbox in bboxes]
                    decode_bboxes = True
                params["bboxes"] = bboxes
            else:
                params["bboxes"] = []
            if self.mask_key in sample and sample[self.mask_key] is not None:
                params["mask"] = sample[self.mask_key]
            output = self.pipeline(**params)
            sample[self.image_key] = output["image"]
            if "keypoints" in output:
                sample[self.keypoints_key] = output["keypoints"]
            if "bboxes" in output:
                if self.cvt_kpts_to_bboxes:
                    sample[self.keypoints_key] = [[kp[0], kp[1]] for kp in output["bboxes"]]
                else:
                    bboxes = output["bboxes"]
                    if decode_bboxes:
                        bboxes = [thelper.data.BoundingBox.decode(bbox, self.bbox_params["format"]) for bbox in bboxes]
                    sample[self.bboxes_key] = bboxes[0] if unpack_bboxes else bboxes
            if "mask" in output:
                sample[self.mask_key] = output["mask"]
            return sample
        elif isinstance(sample, (list, tuple)):
            raise NotImplementedError
            # impl should use linked_fate and force_linked_fate
        else:
            assert sample is None or isinstance(sample, np.ndarray)
            if sample is None:
                return None
            params["image"] = sample
        output = self.pipeline(**params)
        return output["image"]

    def __repr__(self):
        """Create a print-friendly representation of inner augmentation stages."""
        # for debug purposes only, transforms probably cannot be expressed as a string
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(transforms={repr(self.transforms)}, to_tensor={repr(self.to_tensor)}, " + \
            f"bbox_params={repr(self.bbox_params)}, add_targets={repr(self.add_targets)}, " + \
            f"image_key={repr(self.image_key)}, bboxes_key={repr(self.bboxes_key)}, " + \
            f"mask_key={repr(self.mask_key)}, keypoints_key={repr(self.keypoints_key)}, " + \
            f"probability={repr(self.probability)}, cvt_kpts_to_bboxes={repr(self.cvt_kpts_to_bboxes)}, " + \
            f"linked_fate={repr(self.linked_fate)})"

    # noinspection PyMethodMayBeStatic
    def set_seed(self, seed):
        """Sets the internal seed to use for stochastic ops."""
        if self.pipeline.transforms is not None:
            random.random(seed)
            np.random.seed(seed)
            for t in self.pipeline.transforms:
                if hasattr(t, "set_seed") and callable(t.set_seed):
                    t.set_seed(seed)

    def set_epoch(self, epoch=0):
        """Sets the current epoch number in order to change the behavior of some suboperations."""
        assert isinstance(epoch, int) and epoch >= 0, "invalid epoch value"
        if self.pipeline.transforms is not None:
            for t in self.pipeline.transforms:
                if hasattr(t, "set_epoch") and callable(t.set_epoch):
                    t.set_epoch(epoch)


class AugmentorWrapper:
    """Augmentor pipeline wrapper that allows pickling and multi-threading.

    See https://github.com/mdbloice/Augmentor for more information. This wrapper was last updated to work
    with version 0.2.2 --- more recent versions introduced yet unfixed (as of 2018/08) issues on some platforms.

    All original transforms are supported here. This wrapper also fixes the list output bug for single-image
    samples when using operations individually.

    Attributes:
        pipeline: the augmentor pipeline instance to apply to images.
        target_keys: the sample keys to apply the pipeline to (when dictionaries are passed in).
        linked_fate: specifies whether input list samples should all have the same fate or not.

    .. seealso::
        | :func:`thelper.transforms.utils.load_transforms`
    """

    def __init__(self, pipeline, target_keys=None, linked_fate=True):
        """Receives and stores an augmentor pipeline for later use.

        The pipeline itself is instantiated in :func:`thelper.transforms.utils.load_transforms`.
        """
        self.pipeline = pipeline
        self.target_keys = target_keys
        self.linked_fate = linked_fate

    def __call__(self, sample, force_linked_fate=False, op_seed=None, in_cvts=None):
        """Transforms a (dict) sample, a single image, or a list of images using the augmentor pipeline.

        Args:
            sample: the sample or image(s) to transform (can also contain embedded lists/tuples of images).
            force_linked_fate: override flag for recursive use allowing forced linking of arrays.
            op_seed: seed to set before calling the wrapped operation.
            in_cvts: holds the input conversion flag array (for recursive usage).

        Returns:
            The transformed image(s), with the same list/tuple formatting as the input.
        """
        if isinstance(sample, dict):
            # recursive call for unpacking sample content w/ target keys
            assert in_cvts is None, "top-level call should never provide in_cvts"
            # capture non-scalar objects (according to numpy) if no keys are provided
            key_vals = [(k, v) for k, v in sample.items() if (
                (self.target_keys is None and not np.isscalar(v)) or
                (self.target_keys is not None and k in self.target_keys))]
            keys, vals = map(list, zip(*key_vals))
            lengths = [len(v) if isinstance(v, (list, tuple)) else -1 for v in vals]
            if len(lengths) > 0 and all(n == lengths[0] for n in lengths) and lengths[0] > 0:
                # interlace input lists for internal linked fate (if needed; otherwise, it won't change anything)
                vals = [[v[idx] if isinstance(v, (list, tuple)) else
                         v[idx, ...] for v in vals] for idx in range(lengths[0])]
                vals = self(vals, force_linked_fate=force_linked_fate, op_seed=op_seed, in_cvts=in_cvts)
                assert isinstance(vals, list) and len(vals) == lengths[0], "messed up something internally"
                out_vals = [[v] for v in vals[0]] if isinstance(vals[0], list) else [[vals[0]]]
                for idx1 in range(1, lengths[0]):
                    for idx2 in range(len(out_vals)):
                        out_vals[idx2].append(vals[idx1][idx2] if isinstance(vals[idx1], list) else vals[idx1])
                vals = out_vals
            else:
                vals = self(vals, force_linked_fate=force_linked_fate, op_seed=op_seed, in_cvts=in_cvts)
            sample = {k: vals[keys.index(k)] if k in keys else sample[k] for k in sample}
            return sample
        out_cvts = in_cvts is not None
        out_list = isinstance(sample, (list, tuple))
        if sample is None or (out_list and not sample):
            return ([], []) if out_cvts else []
        elif not out_list:
            sample = [sample]
        assert not any([isinstance(v, dict) for v in sample]), \
            "augmentor wrapper cannot handle sample-in-sample (or dict-in-list) inputs"
        skip_unpack = in_cvts is not None and isinstance(in_cvts, bool) and in_cvts
        if self.linked_fate or force_linked_fate:  # process all content with the same operations below
            if not skip_unpack:
                # noinspection PyProtectedMember
                sample, cvts = TransformWrapper._unpack(sample, convert_pil=True)
                if not isinstance(sample, (list, tuple)):
                    sample = [sample]
                    cvts = [cvts]
            else:
                cvts = in_cvts
            if op_seed is None:
                op_seed = np.random.randint(np.iinfo(np.int32).max)
            np.random.seed(op_seed)
            prev_state = np.random.get_state()
            for idx, _ in enumerate(sample):
                if not isinstance(sample[idx], PIL.Image.Image):
                    sample[idx], cvts[idx] = self(sample[idx], force_linked_fate=True,
                                                  op_seed=op_seed, in_cvts=cvts[idx])
                else:
                    np.random.set_state(prev_state)
                    random.seed(np.random.randint(np.iinfo(np.int32).max))
                    for operation in self.pipeline.operations:
                        r = round(np.random.uniform(0, 1), 1)
                        if r <= operation.probability:
                            if sample[idx] is not None:
                                sample[idx] = operation.perform_operation([sample[idx]])[0]
        else:  # each element of the top array will be processed independently below (current seeds are kept)
            cvts = [False] * len(sample)
            for idx, _ in enumerate(sample):
                # noinspection PyProtectedMember
                sample[idx], cvts[idx] = TransformWrapper._unpack(sample[idx], convert_pil=True)
                if not isinstance(sample[idx], PIL.Image.Image):
                    sample[idx], cvts[idx] = self(sample[idx], force_linked_fate=True,
                                                  op_seed=op_seed, in_cvts=cvts[idx])
                else:
                    random.seed(np.random.randint(np.iinfo(np.int32).max))
                    for operation in self.pipeline.operations:
                        r = round(np.random.uniform(0, 1), 1)
                        if r <= operation.probability:
                            if sample[idx] is not None:
                                sample[idx] = operation.perform_operation([sample[idx]])[0]
        # noinspection PyProtectedMember
        sample, cvts = TransformWrapper._pack(sample, cvts, convert_pil=True)
        assert len(sample) == len(cvts), "messed up packing/unpacking logic"
        if (skip_unpack or not out_list) and len(sample) == 1:
            sample = sample[0]
            cvts = cvts[0]
        return (sample, cvts) if out_cvts else sample

    def __repr__(self):
        """Create a print-friendly representation of inner augmentation stages."""
        # for debug purposes only, pipeline probably cannot be expressed as a string
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(pipeline={repr(self.pipeline)}, target_keys={repr(self.target_keys)}, linked_fate={repr(self.linked_fate)})"

    # noinspection PyMethodMayBeStatic
    def set_seed(self, seed):
        """Sets the internal seed to use for stochastic ops."""
        np.random.seed(seed)

    def set_epoch(self, epoch=0):
        """Sets the current epoch number in order to change the behavior of some suboperations."""
        assert isinstance(epoch, int) and epoch >= 0, "invalid epoch value"
        if self.pipeline.operations is not None:
            for op in self.pipeline.operations:
                if hasattr(op, "set_epoch") and callable(op.set_epoch):
                    op.set_epoch(epoch)


class TransformWrapper:
    """Transform wrapper that allows operations on samples, lists, tuples, and single elements.

    Can be used to wrap the operations in ``thelper.transforms`` or in ``torchvision.transforms``
    that only accept array-like objects as input. Will optionally force-convert content to PIL images.

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
        target_keys: the sample keys to apply the transform to (when dictionaries are passed in).
        linked_fate: specifies whether images given in a list/tuple should have the same fate or not.
    """

    def __init__(self, operation, params=None, probability=1, convert_pil=False, target_keys=None, linked_fate=True):
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
            target_keys: the sample keys to apply the pipeline to (when dictionaries are passed in).
            linked_fate: specifies whether images given in a list/tuple should have the same fate or not.
        """
        assert params is None or isinstance(params, dict), "expected params to be passed in as a dictionary"
        assert 0 <= probability <= 1, "invalid probability value (range is [0,1]"
        self.params = {} if params is None else params
        self.operation = operation
        if isinstance(self.operation, str):
            operation_type = thelper.utils.import_class(operation)
            self.opcall = operation_type(**self.params)
        else:
            self.opcall = functools.partial(operation, **self.params)
        self.probability = probability
        self.convert_pil = convert_pil
        self.target_keys = target_keys
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
                    out, cvt = TransformWrapper._unpack(s, force_flatten=force_flatten)
                    if isinstance(cvt, (list, tuple)):
                        assert isinstance(out, (list, tuple)), "unexpected out/cvt types"
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
            if isinstance(sample, np.ndarray) and sample.ndim > 2 and \
                    sample.shape[-1] > 1 and (sample.dtype != np.uint8):
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
            assert convert_pil and isinstance(cvts, bool) and cvts, \
                "unexpected cvts len w/ pil conversion (bad logic somewhere)"
            assert all([isinstance(s, PIL.Image.Image) for s in samples]), "unexpected packed list sample types"
            samples = [np.asarray(s) for s in samples]
            assert all([s.ndim == 2 for s in samples]), "unexpected packed list sample depths"
            samples = [np.expand_dims(s, axis=2) for s in samples]
            return [np.concatenate(samples, axis=2)], [False]
        for idx, cvt in enumerate(cvts):
            if not isinstance(cvt, (list, tuple)):
                assert isinstance(cvt, bool), "unexpected cvt type"
                if cvt:
                    assert not isinstance(samples[idx], (list, tuple)), "unexpected packed sample type"
                    samples[idx] = np.asarray(samples[idx])
                    cvts[idx] = False
        return samples, cvts

    def __call__(self, sample, force_linked_fate=False, op_seed=None, in_cvts=None):
        """Transforms a (dict) sample, a single image, or a list of images using a wrapped operation.

        Args:
            sample: the sample or image(s) to transform (can also contain embedded lists/tuples of images).
            force_linked_fate: override flag for recursive use allowing forced linking of arrays.
            op_seed: seed to set before calling the wrapped operation.
            in_cvts: holds the input conversion flag array (for recursive usage).

        Returns:
            The transformed image(s), with the same list/tuple formatting as the input.
        """
        if isinstance(sample, dict):
            # recursive call for unpacking sample content w/ target keys
            assert in_cvts is None, "top-level call should never provide in_cvts"
            # capture non-scalar objects (according to numpy) if no keys are provided
            key_vals = [(k, v) for k, v in sample.items() if (
                (self.target_keys is None and not thelper.utils.is_scalar(v)) or
                (self.target_keys is not None and k in self.target_keys))]
            keys, vals = map(list, zip(*key_vals))
            lengths = [len(v) if isinstance(v, (list, tuple)) else -1 for v in vals]
            if len(lengths) > 0 and all(n == lengths[0] for n in lengths) and lengths[0] > 0:
                # interlace input lists for internal linked fate (if needed; otherwise, it won't change anything)
                vals = [[v[idx] if isinstance(v, (list, tuple)) else
                         v[idx, ...] for v in vals] for idx in range(lengths[0])]
                vals = self(vals, force_linked_fate=force_linked_fate, op_seed=op_seed, in_cvts=in_cvts)
                assert isinstance(vals, list) and len(vals) == lengths[0], "messed up something internally"
                out_vals = [[v] for v in vals[0]] if isinstance(vals[0], list) else [[vals[0]]]
                for idx1 in range(1, lengths[0]):
                    for idx2 in range(len(out_vals)):
                        out_vals[idx2].append(vals[idx1][idx2] if isinstance(vals[idx1], list) else vals[idx1])
                vals = out_vals
            else:
                vals = self(vals, force_linked_fate=force_linked_fate, op_seed=op_seed, in_cvts=in_cvts)
            sample = {k: vals[keys.index(k)] if k in keys else sample[k] for k in sample}
            return sample
        out_cvts = in_cvts is not None
        out_list = isinstance(sample, (list, tuple))
        if sample is None or (out_list and not sample):
            return ([], []) if out_cvts else []
        elif not out_list:
            sample = [sample]
        assert not any([isinstance(v, dict) for v in sample]), \
            "sample transform wrapper cannot handle sample-in-sample (or dict-in-list) inputs"
        skip_unpack = in_cvts is not None and isinstance(in_cvts, bool) and in_cvts
        if self.linked_fate or force_linked_fate:  # process all content with the same operations below
            if not skip_unpack:
                sample, cvts = self._unpack(sample, convert_pil=self.convert_pil)
                if not isinstance(sample, (list, tuple)):
                    sample = [sample]
                    cvts = [cvts]
            else:
                cvts = in_cvts
            if self.probability >= 1 or round(np.random.uniform(0, 1), 1) <= self.probability:
                if op_seed is None:
                    op_seed = np.random.randint(np.iinfo(np.int32).max)
                for idx, _ in enumerate(sample):
                    if isinstance(sample[idx], (list, tuple)):
                        sample[idx], cvts[idx] = self(sample[idx], force_linked_fate=True,
                                                      op_seed=op_seed, in_cvts=cvts[idx])
                    else:
                        if hasattr(self.opcall, "set_seed") and callable(self.opcall.set_seed):
                            self.opcall.set_seed(op_seed)
                        # watch out: if operation is stochastic and we cannot seed above, then there is no
                        # guarantee that the content will truly have a 'linked fate' (this might cause issues!)
                        if sample[idx] is not None:
                            sample[idx] = self.opcall(sample[idx], **self.params)
        else:  # each element of the top array will be processed independently below (current seeds are kept)
            cvts = [False] * len(sample)
            for idx, _ in enumerate(sample):
                sample[idx], cvts[idx] = self._unpack(sample[idx], convert_pil=self.convert_pil)
                if self.probability >= 1 or round(np.random.uniform(0, 1), 1) <= self.probability:
                    if isinstance(sample[idx], (list, tuple)):
                        # we will now force fate linkage for all sub-elements of this array
                        sample[idx], cvts[idx] = self(sample[idx], force_linked_fate=True,
                                                      op_seed=op_seed, in_cvts=cvts[idx])
                    else:
                        if sample[idx] is not None:
                            sample[idx] = self.opcall(sample[idx], **self.params)
        sample, cvts = TransformWrapper._pack(sample, cvts, convert_pil=self.convert_pil)
        assert len(sample) == len(cvts), "messed up packing/unpacking logic"
        if (skip_unpack or not out_list) and len(sample) == 1:
            sample = sample[0]
            cvts = cvts[0]
        return (sample, cvts) if out_cvts else sample

    def __repr__(self):
        """Create a print-friendly representation of inner augmentation stages."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(operation={repr(self.operation)}, params={repr(self.params)}, probability={repr(self.probability)}, " + \
            f"convert_pil={repr(self.convert_pil)}, target_keys={repr(self.target_keys)}, linked_fate={repr(self.linked_fate)})"

    # noinspection PyMethodMayBeStatic
    def set_seed(self, seed):
        """Sets the internal seed to use for stochastic ops."""
        np.random.seed(seed)

    def set_epoch(self, epoch=0):
        """Sets the current epoch number in order to change the behavior of some suboperations."""
        assert isinstance(epoch, int) and epoch >= 0, "invalid epoch value"
        if hasattr(self.opcall, "set_epoch") and callable(self.opcall.set_epoch):
            self.opcall.set_epoch(epoch)
