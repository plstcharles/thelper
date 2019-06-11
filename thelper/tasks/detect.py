"""Detection task interface module.

This module contains classes that define object detection utilities and task interfaces.
"""
import copy
import json
import logging
import os
from typing import Optional  # noqa: F401

import numpy as np
import torch
import tqdm

from thelper.tasks.regr import Regression
from thelper.tasks.utils import Task

logger = logging.getLogger(__name__)


class BoundingBox:
    """Interface used to hold instance metadata for object detection tasks.

    Object detection trainers and display utilities in the framework will expect this interface to be
    used when parsing a predicted detection or an annotation. The base contents are based on the
    PASCALVOC metadata structure, and this class can be derived if necessary to contain more metadata.

    Attributes:
        class_id: type identifier for the underlying object instance.
        bbox: four-element tuple holding the (xmin,ymin,xmax,ymax) bounding box parameters.
        difficult: defines whether this instance is considered "difficult" (false by default).
        occluded: defines whether this instance is considered "occluded" (false by default).
        truncated: defines whether this instance is considered "truncated" (false by default).
        iscrowd: defines whether this instance covers a "crowd" of objects or not (false by default).
        area: scalar indicating the total surface of the annotation (will be computed automatically if None).
        confidence: scalar or array of prediction confidence values tied to class types (empty by default).
        image_id: string used to identify the image containing this bounding box (i.e. file path or uuid).
        task: reference to the task object that holds extra metadata regarding the content of the bbox (None by default).

    .. seealso::
        | :class:`thelper.tasks.utils.Task`
        | :class:`thelper.tasks.detect.Detection`
    """

    def __init__(self, class_id, bbox, difficult=False, occluded=False, truncated=False,
                 iscrowd=False, area=None, confidence=None, image_id=None, task=None):
        """Receives and stores low-level input detection metadata for later access."""
        self.class_id = class_id  # should be string or int to allow batching in data loaders
        # note: the input bbox is expected to be a 4 element array (xmin,ymin,xmax,ymax)
        self.bbox = bbox
        self.difficult = difficult
        self.occluded = occluded
        self.truncated = truncated
        self.iscrowd = iscrowd
        self.area = area
        self.confidence = confidence
        self.image_id = image_id  # should be string that identifies the associated image (file path or uuid)
        self.task = task

    @property
    def class_id(self):
        """Returns the object class type identifier."""
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """Sets the object class type identifier  (should be string/int)."""
        assert isinstance(value, (int, str)), "class should be defined as integer (index) or string (name)"
        self._class_id = value

    @property
    def bbox(self):
        """Returns the bounding box tuple (xmin,ymin,xmax,ymax)."""
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        """Sets the bounding box tuple (xmin,ymin,xmax,ymax)."""
        assert isinstance(value, (list, tuple, np.ndarray, torch.Tensor)) and len(value) == 4, "invalid input type/len"
        assert not isinstance(value, (list, tuple)) or all([isinstance(v, (int, float)) for v in value]), \
            "input bbox values must be integer/float"
        assert value[0] <= value[2] and value[1] <= value[3], "invalid min/max values for bbox coordinates"
        self._bbox = value

    @property
    def top_left(self):
        """Returns the top left bounding box corner coordinates (x,y)."""
        return self._bbox[0], self._bbox[1]

    @top_left.setter
    def top_left(self, value):
        """Sets the top left bounding box corner coordinates (x,y)."""
        assert isinstance(value, (list, tuple, np.ndarray, torch.Tensor)) and len(value) == 2, "invalid input type/len"
        self._bbox[0], self._bbox[1] = value[0], value[1]

    @property
    def bottom_right(self):
        """Returns the bottom right bounding box corner coordinates (x,y)."""
        return self._bbox[2], self._bbox[3]

    @bottom_right.setter
    def bottom_right(self, value):
        """Sets the bottom right bounding box corner coordinates (x,y)."""
        assert isinstance(value, (list, tuple, np.ndarray, torch.Tensor)) and len(value) == 2, "invalid input type/len"
        assert value[0] >= self._bbox[0] and value[1] >= self._bbox[1]
        self._bbox[2], self._bbox[3] = value[0], value[1]

    @property
    def width(self):
        """Returns the width of the bounding box."""
        return self._bbox[2] - self._bbox[0]

    @property
    def height(self):
        """Returns the height of the bounding box."""
        return self._bbox[3] - self._bbox[1]

    @property
    def centroid(self, floor=False):
        """Returns the bounding box centroid coordinates (x,y)."""
        if floor:
            return (self._bbox[0] + self._bbox[2]) // 2, (self._bbox[1] + self._bbox[3]) // 2
        return (self._bbox[0] + self._bbox[2]) / 2, (self._bbox[1] + self._bbox[3]) / 2

    @property
    def difficult(self):
        """Returns whether this bounding box is considered "difficult" by the dataset (false by default)."""
        return self._difficult

    @difficult.setter
    def difficult(self, value):
        """Sets whether this bounding box is considered "difficult" by the dataset."""
        assert isinstance(value, (int, bool)), "flag type must be integer or boolean"
        self._difficult = value

    @property
    def occluded(self):
        """Returns whether this bounding box is considered "occluded" by the dataset (false by default)."""
        return self._occluded

    @occluded.setter
    def occluded(self, value):
        """Sets whether this bounding box is considered "occluded" by the dataset."""
        assert isinstance(value, (int, bool)), "flag type must be integer or boolean"
        self._occluded = value

    @property
    def truncated(self):
        """Returns whether this bounding box is considered "truncated" by the dataset (false by default)."""
        return self._truncated

    @truncated.setter
    def truncated(self, value):
        """Sets whether this bounding box is considered "truncated" by the dataset."""
        assert isinstance(value, (int, bool)), "flag type must be integer or boolean"
        self._truncated = value

    @property
    def iscrowd(self):
        """Returns whether this instance covers a "crowd" of objects or not (false by default)."""
        return self._iscrowd

    @iscrowd.setter
    def iscrowd(self, value):
        """Sets whether this instance covers a "crowd" of objects or not."""
        assert isinstance(value, (int, bool)), "flag type must be integer or boolean"
        self._iscrowd = value

    @property
    def area(self):
        """Returns a scalar indicating the total surface of the annotation (may be None if unknown/unspecified)."""
        return self._area

    @area.setter
    def area(self, value):
        if value is None:
            value = self.width * self.height
        else:
            # may be assigned if for example we are using relative coords but want to keep track of the original surface
            assert (isinstance(value, (int, float)) and value > 0), "invalid annotation surface area value"
        self._area = value

    @property
    def confidence(self):
        """Returns the confidence value (or array of confidence values) associated to the predicted class types."""
        return self._confidence

    @confidence.setter
    def confidence(self, value):
        """Sets the confidence value (or array of confidence values) associated to the predicted class types."""
        assert value is None or isinstance(value, (float, list, np.ndarray, torch.Tensor)), "value should be float/list/ndarray/tensor"
        self._confidence = value

    @property
    def image_id(self):
        """Returns the image string identifier."""
        return self._image_id

    @image_id.setter
    def image_id(self, value):
        """Sets the image string identifier."""
        assert value is None or isinstance(value, (str, int)), "image identifier should be a string/int (file path or uuid)"
        self._image_id = value

    @property
    def task(self):
        """Returns the reference to the task object that holds extra metadata regarding the content of the bbox."""
        return self._task

    @task.setter
    def task(self, value):
        """Sets the reference to the task object that holds extra metadata regarding the content of the bbox."""
        assert isinstance(value, Detection), "task should be detection-related"
        assert self.class_id in value.class_indices.values(), f"cannot find class_id '{self.class_id}' in task indices"
        self._task = value

    def encode(self, format=None):
        """Returns a vectorizable representation of this bounding box in a specified format.

        WARNING: Encoding might cause information loss (e.g. task reference is discarded).
        """
        if format == "coco":
            return [*self.top_left, self.width, self.height, self.class_id]
        elif format == "pascal_voc":
            return [*self.top_left, *self.bottom_right, self.class_id,
                    self.difficult, self.occluded, self.truncated]
        else:
            assert format is None, "unrecognized/unknown encoding format"
            vec = [*self.bbox, self.class_id, self.difficult, self.occluded,
                   self.truncated, self.iscrowd, self.area, self.image_id]
            if self.confidence is not None:
                vec += [self.confidence] if isinstance(self.confidence, float) else [*self.confidence]
            return vec

    @staticmethod
    def decode(vec, format=None):
        """Returns a BoundingBox object from a vectorized representation in a specified format."""
        # note: the input bbox is expected to be a 4 element array (xmin,ymin,xmax,ymax)
        if format == "coco":
            assert len(vec) == 5, "unexpected vector length (should contain 5 values)"
            return BoundingBox(class_id=vec[4], bbox=[vec[0], vec[1], vec[0] + vec[2], vec[1] + vec[3]])
        elif format == "pascal_voc":
            assert len(vec) == 8, "unexpected vector length (should contain 8 values)"
            return BoundingBox(class_id=vec[4], bbox=vec[0:4], difficult=vec[5], occluded=vec[6], truncated=vec[7])
        else:
            assert format is None, "unrecognized/unknown encoding format"
            assert len(vec) >= 11, "unexpected vector length (should contain 9 values or more)"
            return BoundingBox(class_id=vec[4], bbox=vec[0:4], difficult=vec[5], occluded=vec[6],
                               truncated=vec[7], iscrowd=vec[8], area=vec[9],
                               confidence=(None if len(vec) == 11 else vec[11:]),
                               image_id=vec[10], task=None)

    def __repr__(self):
        """Creates a print-friendly representation of the object detection bbox instance."""
        # note: we do not export the task reference here (it might be too heavy for logs)
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(class_id={repr(self.class_id)}, bbox={repr(self.bbox)}, difficult={repr(self.difficult)}, " + \
            f"occluded={repr(self.occluded)}, truncated={repr(self.truncated)}, iscrowd={repr(self.iscrowd)}, " + \
            f"area={repr(self.area)}, confidence={repr(self.confidence)}, image_id={repr(self.image_id)})"


class Detection(Regression):
    """Interface for object detection tasks.

    This specialization requests that when given an input image, the trained model should
    provide a list of bounding box (bbox) proposals that correspond to probable objects detected
    in the image.

    This specialized regression interface is currently used to help display functions.

    Attributes:
        class_names: map of class name-value pairs for object types to detect.
        input_key: the key used to fetch input tensors from a sample dictionary.
        bboxes_key: the key used to fetch target (groundtruth) bboxes from a sample dictionary.
        meta_keys: the list of extra keys provided by the data parser inside each sample.
        input_shape: a numpy-compatible shape to expect input images to possess.
        target_shape: a numpy-compatible shape to expect the predictions to be in.
        target_min: a 2-dim tensor containing minimum (x,y) bounding box corner values.
        target_max: a 2-dim tensor containing maximum (x,y) bounding box corner values.
        background: value of the 'background' label (if any) used in the class map.
        color_map: map of class name-color pairs to use when displaying results.

    .. seealso::
        | :class:`thelper.tasks.utils.Task`
        | :class:`thelper.tasks.regr.Regression`
        | :class:`thelper.train.detect.ObjDetectTrainer`
    """

    def __init__(self, class_names, input_key, bboxes_key, meta_keys=None, input_shape=None,
                 target_shape=None, target_min=None, target_max=None, background=None, color_map=None):
        """Receives and stores the bbox types to detect, the input tensor key, the groundtruth
        bboxes list key, the extra (meta) keys produced by the dataset parser(s), and the color
        map used to color bboxes when displaying results.

        The class names can be provided as a list of strings, as a path to a json file that
        contains such a list, or as a map of predefined name-value pairs to use in gt maps.
        This list/map must contain at least two elements (background and one class). All
        other arguments are used as-is to index dictionaries, and must therefore be key-
        compatible types.
        """
        super(Detection, self).__init__(input_key, bboxes_key, meta_keys,
                                        input_shape=input_shape, target_shape=target_shape,
                                        target_min=target_min, target_max=target_max)
        self.class_names = class_names
        self.background = background
        self.color_map = color_map

    @property
    def class_names(self):
        """Returns the list of class names to be predicted."""
        return self._class_names

    @class_names.setter
    def class_names(self, class_names):
        """Sets the list of class names to be predicted."""
        if isinstance(class_names, str) and os.path.exists(class_names):
            with open(class_names, "r") as fd:
                class_names = json.load(fd)
        assert isinstance(class_names, (list, dict)), "expected class names to be provided as a list or map"
        if isinstance(class_names, list):
            if len(class_names) != len(set(class_names)):
                # no longer throwing here, imagenet possesses such a case ('crane#134' and 'crane#517')
                logger.warning("found duplicated name in class list, might be a data entry problem...")
                class_names = [name if class_names.count(name) == 1 else name + "#" + str(idx)
                               for idx, name in enumerate(class_names)]
            class_indices = {class_name: class_idx for class_idx, class_name in enumerate(class_names)}
        else:
            class_indices = copy.deepcopy(class_names)
        assert isinstance(class_indices, dict), "expected class names to be provided as a dictionary"
        assert all([isinstance(name, str) for name in class_indices.keys()]), "all classes must be named with strings"
        assert all([isinstance(idx, int) for idx in class_indices.values()]), "all classes must be indexed with integers"
        assert len(class_indices) >= 1, "should have at least one class!"
        background = None if "background" not in class_indices else class_indices["background"]
        self._class_names = [class_name for class_name in class_indices.keys()]
        self._class_indices = class_indices
        self.background = background

    @property
    def class_indices(self):
        """Returns the class-name-to-index map used for encoding labels as integers."""
        return self._class_indices

    @class_indices.setter
    def class_indices(self, class_indices):
        """Sets the class-name-to-index map used for encoding labels as integers."""
        assert isinstance(class_indices, dict), "class indices must be provided as dictionary"
        self.class_names = class_indices

    @property
    def background(self):
        """Returns the 'background' label value used in loss functions (can be ``None``)."""
        return self._background

    @background.setter
    def background(self, background):
        """Sets the 'background' label value for this segmentation task (can be ``None``)."""
        if background is not None:
            assert isinstance(background, int), "'background' value should be integer (index)"
            assert background not in self.class_indices.values() or self.class_indices["background"] == background, \
                "found 'background' value tied to another class label"
        self._background = background

    @property
    def color_map(self):
        """Returns the color map used to swap label indices for colors when displaying results."""
        return self._color_map

    @color_map.setter
    def color_map(self, color_map):
        """Sets the color map used to swap label indices for colors when displaying results."""
        if color_map is not None:
            assert isinstance(color_map, dict), "color map should be given as dictionary"
            self._color_map = {}
            assert all([isinstance(k, int) for k in color_map]) or all([isinstance(k, str) for k in color_map]), \
                "color map keys should be only class names or only class indices"
            for key, val in color_map.items():
                if isinstance(key, str):
                    if key == "background" and self.background is not None:
                        key = self.background
                    else:
                        assert key in self.class_indices, f"could not find color map key '{key}' in class names"
                        key = self.class_indices[key]
                assert key in self.class_indices.values() or key == self.background, f"unrecognized class index '{key}'"
                if isinstance(val, (list, tuple)):
                    val = np.asarray(val)
                assert isinstance(val, np.ndarray) and val.size == 3, "color values should be given as triplets"
                self._color_map[key] = val
            if self.background is not None and self.background not in self._color_map:
                self._color_map[self.background] = np.asarray([0, 0, 0])  # use black as default 'background' color
        else:
            self._color_map = None

    def get_class_sizes(self, samples, bbox_format=None):
        """Given a list of samples, returns a map of element counts for each object type."""
        assert samples is not None and samples, "provided invalid sample list"
        elem_counts = {class_name: 0 for class_name in self.class_names}
        for sample_idx, sample in tqdm.tqdm(enumerate(samples), desc="cumulating bbox counts", total=len(samples)):
            if self.gt_key is None or self.gt_key not in sample:
                continue
            else:
                bboxes = sample[self.gt_key]
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                if isinstance(bboxes, (np.ndarray, list, tuple)):
                    bboxes = [BoundingBox.decode(bbox, format=bbox_format)
                              if isinstance(bbox, (np.ndarray, list, tuple)) else bbox for bbox in bboxes]
                assert all([isinstance(bbox, BoundingBox) for bbox in bboxes]), "unrecognized sample bbox format"
                assert all([bbox.class_id in self.class_indices.values() for bbox in bboxes]), \
                    "bboxes contain unknown class ids"
                for cname, cval in self.class_indices.items():
                    elem_counts[cname] += len([b for b in bboxes if b.class_id == cval])
        return elem_counts

    def check_compat(self, task, exact=False):
        # type: (Detection, Optional[bool]) -> bool
        """Returns whether the current task is compatible with the provided one or not.

        This is useful for sanity-checking, and to see if the inputs/outputs of two models
        are compatible. If ``exact = True``, all fields will be checked for exact (perfect)
        compatibility (in this case, matching meta keys and class maps).
        """
        if isinstance(task, Detection):
            if not Regression.check_compat(self, task, exact=exact):
                return False
            return self.background == task.background and \
                all([cls in self.class_names for cls in task.class_names]) and \
                (not exact or (self.class_names == task.class_names and
                               self.color_map == task.color_map))
        elif type(task) == Task:
            # if 'task' simply has no gt, compatibility rests on input key only
            return not exact and self.input_key == task.input_key and task.gt_key is None
        return False

    def get_compat(self, task):
        """Returns a task instance compatible with the current task and the given one."""
        assert isinstance(task, Detection) or type(task) == Task, \
            f"cannot create compatible task from types '{type(task)}' and '{type(self)}'"
        if isinstance(task, Detection):
            assert self.input_key == task.input_key, "input key mismatch, cannot create compatible task"
            assert self.gt_key is None or task.gt_key is None or self.gt_key == task.gt_key, \
                "gt key mismatch, cannot create compatible task"
            assert self.background == task.background, "background value mismatch, cannot create compatible task"
            meta_keys = list(set(self.meta_keys + task.meta_keys))
            # cannot use set for class names, order needs to stay intact!
            class_indices = {cname: cval for cname, cval in task.class_indices.items() if cname not in self.class_indices}
            class_indices = {**self.class_indices, **class_indices}
            color_map = {cname: cval for cname, cval in task.color_map.items() if cname not in self.color_map}
            color_map = {**self.color_map, **color_map}
            return Detection(class_names=class_indices, input_key=self.input_key, bboxes_key=self.gt_key, meta_keys=meta_keys,
                             input_shape=self.input_shape if self.input_shape is not None else task.input_shape,
                             target_shape=self.target_shape if self.target_shape is not None else task.target_shape,
                             target_min=self.target_min if self.target_min is not None else task.target_min,
                             target_max=self.target_max if self.target_max is not None else task.target_max,
                             background=self.background, color_map=color_map)
        elif type(task) == Task:
            assert self.check_compat(task), f"cannot create compatible task between:\n\t{str(self)}\n\t{str(task)}"
            meta_keys = list(set(self.meta_keys + task.meta_keys))
            return Detection(class_names=self.class_indices, input_key=self.input_key, bboxes_key=self.gt_key,
                             meta_keys=meta_keys, input_shape=self.input_shape, target_shape=self.target_shape,
                             target_min=self.target_min, target_max=self.target_max, background=self.background,
                             color_map=self.color_map)

    def __repr__(self):
        """Creates a print-friendly representation of a segmentation task."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(class_names={repr(self.class_indices)}, input_key={repr(self.input_key)}, " + \
            f"bboxes_key={repr(self.gt_key)}, meta_keys={repr(self.meta_keys)}, " + \
            f"input_shape={repr(self.input_shape)}, target_shape={repr(self.target_shape)}, " + \
            f"target_min={repr(self.target_min)}, target_max={repr(self.target_max)}, " + \
            f"background={repr(self.background)}, color_map={self.color_map})"
