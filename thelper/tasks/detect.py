"""Detection task interface module.

This module contains classes that define object detection utilities and task interfaces.
"""
import copy
import json
import logging
import os

import numpy as np
import torch
import tqdm

from thelper.tasks.regr import Regression
from thelper.tasks.utils import Task

logger = logging.getLogger(__name__)


class BoundingBox:
    """Interface used to hold instance metadata for object detection tasks.

    Object detection trainers and display utilities in the framework will expect this interface to be
    used when parsing a predicted detection or an annotation. The default contents are based on the
    PASCALVOC metadata structure, and this class can be derived if necessary to contain more metadata.

    Attributes:
        class_id: type identifier for the underlying object instance.
        bbox: four-element tuple holding the (xmin,xmax,ymin,ymax) bounding box parameters.
        difficult: defines whether this instance is considered "difficult" (false by default).
        occluded: defines whether this instance is considered "occluded" (false by default).
        truncated: defines whether this instance is considered "truncated" (false by default).

    .. seealso::
        | :class:`thelper.tasks.utils.Task`
        | :class:`thelper.tasks.detect.Detection`
    """

    def __init__(self, class_id, bbox, difficult=False, occluded=False, truncated=False):
        """Receives and stores low-level input detection metadata for later access."""
        # note: the input bbox is expected to be a 4 element array (xmin,xmax,ymin,ymax)
        self.class_id = class_id  # should be string or int to allow batching in data loaders
        assert isinstance(bbox, (list, tuple, np.ndarray, torch.Tensor)) and len(bbox) == 4, "invalid input bbox type/len"
        assert bbox[0] <= bbox[1] and bbox[2] <= bbox[3], "invalid min/max values for bbox coordinates"
        self.bbox = bbox
        self.difficult = difficult
        self.occluded = occluded
        self.truncated = truncated

    def get_class_id(self):
        """Returns the instance's object class type identifier (should be string/int)."""
        return self.class_id

    def get_bbox(self):
        """Returns the instance's bounding box tuple (xmin,xmax,ymin,ymax)."""
        return self.bbox

    def get_top_left(self):
        """Returns the instance's top left bounding box corner coordinates (x,y)."""
        return self.bbox[0], self.bbox[2]

    def get_bottom_right(self):
        """Returns the instance's top left bounding box corner coordinates (x,y)."""
        return self.bbox[1], self.bbox[3]

    def get_width(self):
        """Returns the width of the bounding box."""
        return self.bbox[1] - self.bbox[0]

    def get_height(self):
        """Returns the height of the bounding box."""
        return self.bbox[3] - self.bbox[2]

    def get_centroid(self, floor=False):
        """Returns the instance's bounding box centroid coordinates (x,y)."""
        if floor:
            return (self.bbox[0] + self.bbox[1]) // 2, (self.bbox[2] + self.bbox[3]) // 2
        return (self.bbox[0] + self.bbox[1]) / 2, (self.bbox[2] + self.bbox[3]) / 2

    def is_difficult(self):
        """Returns whether this instance is considered "difficult" by the dataset (false by default)."""
        return self.difficult

    def is_occluded(self):
        """Returns whether this instance is considered "occluded" by the dataset (false by default)."""
        return self.occluded

    def is_truncated(self):
        """Returns whether this instance is considered "truncated" by the dataset (false by default)."""
        return self.truncated

    def encode(self, format=None):
        """Returns a vectorizable representation of this bounding box in a specified format."""
        if format == "coco":
            return [*self.get_top_left(), self.get_width(), self.get_height(), self.get_class_id(),
                    self.is_difficult(), self.is_occluded(), self.is_truncated()]
        elif format == "pascal_voc":
            return [*self.get_top_left(), *self.get_bottom_right(), self.get_class_id(),
                    self.is_difficult(), self.is_occluded(), self.is_truncated()]
        else:
            assert format is None, "unrecognized/unknown encoding format"
            return [self.get_class_id(), *self.get_bbox(), self.is_difficult(),
                    self.is_occluded(), self.is_truncated()]

    @staticmethod
    def decode(vec, format=None):
        """Returns a BoundingBox object from a vectorized representation in a specified format."""
        assert len(vec) == 8, "unexpected vector length (should contain 8 values)"
        # note: the input bbox is expected to be a 4 element array (xmin,xmax,ymin,ymax)
        if format == "coco":
            return BoundingBox(class_id=vec[4], bbox=[vec[0], vec[0] + vec[2], vec[1], vec[1] + vec[3]],
                               difficult=vec[5], occluded=vec[6], truncated=vec[7])
        elif format == "pascal_voc":
            return BoundingBox(class_id=vec[4], bbox=[vec[0], vec[2], vec[1], vec[3]],
                               difficult=vec[5], occluded=vec[6], truncated=vec[7])
        else:
            assert format is None, "unrecognized/unknown encoding format"
            return BoundingBox(class_id=vec[0], bbox=vec[1:5], difficult=vec[5],
                               occluded=vec[6], truncated=vec[7])

    def __repr__(self):
        """Creates a print-friendly representation of the object detection bbox instance."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + ": " + str({
            "class_id": self.get_class_id(),
            "bbox": self.get_bbox(),
            "difficult": self.is_difficult(),
            "occluded": self.is_occluded(),
            "truncated": self.is_truncated()
        })


class Detection(Regression):
    """Interface for object detection tasks.

    This specialization requests that when given an input image, the trained model should
    provide a list of bounding box (bbox) proposals that correspond to probable objects detected
    in the image.

    This specialized regression interface is currently used to help display functions.

    Attributes:
        class_map: map of class name-value pairs for object types to detect.
        input_shape: a numpy-compatible shape to expect input images to possess.
        target_shape: a numpy-compatible shape to expect the predictions to be in.
        target_min: a 2-dim tensor containing minimum bounding box corner values.
        target_max: a 2-dim tensor containing maximum bounding box corner values.
        input_key: the key used to fetch input tensors from a sample dictionary.
        bboxes_key: the key used to fetch target (groundtruth) bboxes from a sample dictionary.
        meta_keys: the list of extra keys provided by the data parser inside each sample.
        color_map: map of class name-color pairs to use when displaying results.

    .. seealso::
        | :class:`thelper.tasks.utils.Task`
        | :class:`thelper.tasks.regr.Regression`
        | :class:`thelper.train.regr.RegressionTrainer`
    """

    def __init__(self, class_names, input_key, bboxes_key, meta_keys=None, input_shape=None,
                 target_shape=None, target_min=None, target_max=None, color_map=None):
        """Receives and stores the bbox types to detect, the input tensor key, the groundtruth
        bboxes list key, the extra (meta) keys produced by the dataset parser(s), and the color
        map used to color bboxes when displaying results.

        The class names can be provided as a list of strings, as a path to a json file that
        contains such a list, or as a map of predefined name-value pairs to use in gt maps.
        This list/map must contain at least two elements (background and one class). All
        other arguments are used as-is to index dictionaries, and must therefore be key-
        compatible types.
        """
        super().__init__(input_key, bboxes_key, meta_keys, input_shape=input_shape,
                         target_shape=target_shape, target_min=target_min, target_max=target_max)
        if isinstance(class_names, str) and os.path.exists(class_names):
            with open(class_names, "r") as fd:
                class_names = json.load(fd)
        assert isinstance(class_names, (list, dict)), "expected class names to be provided as a list or map"
        if isinstance(class_names, list):
            assert len(class_names) == len(set(class_names)), "class names should not contain duplicates"
            class_map = {class_name: class_idx for class_idx, class_name in enumerate(class_names)}
        else:
            class_map = copy.copy(class_names)
        assert len(class_map) >= 1, "should have at least one class!"
        assert len(class_map) == len(set(class_map)), "class set should not contain duplicates"
        self.class_map = class_map
        self.color_map = None
        if color_map is not None:
            assert isinstance(color_map, dict), "color map should be given as dictionary"
            self.color_map = {}
            for key, val in color_map.items():
                assert key in self.class_map, "unknown color map entry '%s'" % key
                if isinstance(val, (list, tuple)):
                    val = np.ndarray(val)
                assert isinstance(val, np.ndarray) and val.size == 3, "color values should be given as triplets"
                self.color_map[key] = val

    def get_class_names(self):
        """Returns the list of class names to be predicted by the model."""
        return list(self.class_map.keys())

    def get_nb_classes(self):
        """Returns the number of object types to be detected by the model."""
        return len(self.class_map)

    def get_class_idxs_map(self):
        """Returns the object-type-to-index map used for encoding class labels as integers."""
        return self.class_map

    def get_class_sizes(self, samples, bbox_format=None):
        """Given a list of samples, returns a map of element counts for each object type."""
        assert samples is not None and samples, "provided invalid sample list"
        elem_counts = {class_name: 0 for class_name in self.class_map}
        bboxes_key = self.get_gt_key()
        for sample_idx, sample in tqdm.tqdm(enumerate(samples), desc="cumulating bbox counts", total=len(samples)):
            if bboxes_key is None or bboxes_key not in sample:
                continue
            else:
                bboxes = sample[bboxes_key]
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                if isinstance(bboxes, (np.ndarray, list, tuple)):
                    bboxes = [BoundingBox.decode(bbox, format=bbox_format) for bbox in bboxes]
                assert all([isinstance(bbox, BoundingBox) for bbox in bboxes]), "unrecognized sample bbox format"
                assert all([bbox.get_class_id() in self.class_map.values() for bbox in bboxes]), "bboxes contain unknown class ids"
                for class_name in self.class_map:
                    elem_counts[class_name] += len([b for b in bboxes if b.get_class_id() == self.class_map[class_name]])
        return elem_counts

    def get_color_map(self):
        """Returns the color map used to swap label indices for colors when displaying results."""
        return self.color_map

    def check_compat(self, other, exact=False):
        """Returns whether the current task is compatible with the provided one or not.

        This is useful for sanity-checking, and to see if the inputs/outputs of two models
        are compatible. If ``exact = True``, all fields will be checked for exact (perfect)
        compatibility (in this case, matching meta keys and class maps).
        """
        if not super(Detection, self).check_compat(other, exact=exact):
            return False
        if isinstance(other, Detection):
            # if both tasks are related to segmentation: gt keys, class names, and dc must match
            return (self.get_input_key() == other.get_input_key() and
                    (self.get_gt_key() is None or other.get_gt_key() is None or self.get_gt_key() == other.get_gt_key()) and
                    all([cls in self.get_class_names() for cls in other.get_class_names()]) and
                    (not exact or (self.get_class_idxs_map() == other.get_class_idxs_map() and
                                   set(self.get_meta_keys()) == set(other.get_meta_keys()))))
        return False

    def get_compat(self, other):
        """Returns a task instance compatible with the current task and the given one."""
        assert isinstance(other, (Detection, Task)), "cannot combine '%s' with '%s'" % (str(other.__class__), str(self.__class__))
        if isinstance(other, Detection):
            assert self.get_input_key() == other.get_input_key(), "input key mismatch, cannot create compatible task"
            assert self.get_gt_key() is None or other.get_gt_key() is None or self.get_gt_key() == other.get_gt_key(), \
                "gt key mismatch, cannot create compatible task"
            meta_keys = list(set(self.get_meta_keys() + other.get_meta_keys()))
            # cannot use set for class names, order needs to stay intact!
            class_names = self.get_class_names() + [name for name in other.get_class_names() if name not in self.get_class_names()]
            return Detection(class_names, self.get_input_key(), self.get_gt_key(), meta_keys=meta_keys,
                             input_shape=self.get_input_shape(), target_shape=self.get_target_shape(),
                             target_min=self.get_target_min(), target_max=self.get_target_max(),
                             color_map=self.get_color_map())
        elif type(other) == Task:
            assert self.check_compat(other), "cannot create compat task between:\n\tself: %s\n\tother: %s" % (str(self), str(other))
            meta_keys = list(set(self.get_meta_keys() + other.get_meta_keys()))
            return Detection(self.get_class_idxs_map(), self.get_input_key(), self.get_gt_key(), meta_keys=meta_keys,
                             input_shape=self.get_input_shape(), target_shape=self.get_target_shape(),
                             target_min=self.get_target_min(), target_max=self.get_target_max(),
                             color_map=self.get_color_map())

    def __repr__(self):
        """Creates a print-friendly representation of a segmentation task."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + ": " + str({
            "class_names": self.get_class_idxs_map(),
            "input_key": self.get_input_key(),
            "bboxes_key": self.get_gt_key(),
            "meta_keys": self.get_meta_keys(),
            "input_shape": self.get_input_shape(),
            "target_shape": self.get_target_shape(),
            "target_min": self.get_target_min(),
            "target_max": self.get_target_max(),
            "color_map": self.get_color_map(),
        })
