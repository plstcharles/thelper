"""Detection task interface module.

This module contains classes that define object detection utilities and task interfaces.
"""
import logging
from typing import List, Optional, Tuple, Union  # noqa: F401

import numpy as np
import torch
import tqdm

import thelper.concepts
import thelper.utils
from thelper.ifaces import ClassNamesHandler
from thelper.tasks.regr import Regression
from thelper.tasks.utils import Task

logger = logging.getLogger(__name__)


@thelper.concepts.detection
class BoundingBox:
    """Interface used to hold instance metadata for object detection tasks.

    Object detection trainers and display utilities in the framework will expect this interface to be
    used when parsing a predicted detection or an annotation. The base contents are based on the
    PASCALVOC metadata structure, and this class can be derived if necessary to contain more metadata.

    Attributes:
        class_id: type identifier for the underlying object instance.
        bbox: four-element tuple holding the (xmin,ymin,xmax,ymax) bounding box parameters.
        include_margin: defines whether xmax/ymax is included in the bounding box area or not.
        difficult: defines whether this instance is considered "difficult" (false by default).
        occluded: defines whether this instance is considered "occluded" (false by default).
        truncated: defines whether this instance is considered "truncated" (false by default).
        iscrowd: defines whether this instance covers a "crowd" of objects or not (false by default).
        confidence: scalar or array of prediction confidence values tied to class types (empty by default).
        image_id: string used to identify the image containing this bounding box (i.e. file path or uuid).
        task: reference to the task object that holds extra metadata regarding the content of the bbox (None by default).

    .. seealso::
        | :class:`thelper.tasks.utils.Task`
        | :class:`thelper.tasks.detect.Detection`
    """

    def __init__(self, class_id, bbox, include_margin=True, difficult=False, occluded=False,
                 truncated=False, iscrowd=False, confidence=None, image_id=None, task=None):
        """Receives and stores low-level input detection metadata for later access."""
        self.class_id = class_id  # should be string or int to allow batching in data loaders
        # note: the input bbox is expected to be a 4 element array (xmin,ymin,xmax,ymax)
        self.include_margin = include_margin
        self.bbox = bbox
        self.difficult = difficult
        self.occluded = occluded
        self.truncated = truncated
        self.iscrowd = iscrowd
        self.confidence = confidence
        self.image_id = image_id  # should be string that identifies the associated image (file path or uuid)
        self.task = task

    @property
    def class_id(self):
        # type: () -> Union[int, str]
        """Returns the object class type identifier."""
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        # type: (Union[int, str]) -> None
        """Sets the object class type identifier  (should be string/int)."""
        assert isinstance(value, (int, str)), "class should be defined as integer (index) or string (name)"
        self._class_id = value

    @property
    def bbox(self):
        """Returns the bounding box tuple :math:`(x_min,y_min,x_max,y_max)`."""
        return self._bbox

    @bbox.setter
    def bbox(self, value):
        """Sets the bounding box tuple :math:`(x_min,y_min,x_max,y_max)`."""
        assert isinstance(value, (list, tuple, np.ndarray, torch.Tensor)) and len(value) == 4, "invalid input type/len"
        assert not isinstance(value, (list, tuple)) or all([isinstance(v, (int, float)) for v in value]), \
            "input bbox values must be integer/float"
        assert value[0] <= value[2] and value[1] <= value[3], "invalid min/max values for bbox coordinates"
        assert not self.include_margin or not any([isinstance(v, float) for v in value]), \
            "it makes no sense to include xmax/ymax margin if using floating point coordinates"
        self._bbox = value

    @property
    def left(self):
        """Returns the left bounding box edge origin offset value."""
        return self._bbox[0]

    @left.setter
    def left(self, value):
        """Sets the left bounding box edge origin offset value."""
        self._bbox[0] = value

    @property
    def top(self):
        """Returns the top bounding box edge origin offset value."""
        return self._bbox[1]

    @top.setter
    def top(self, value):
        """Sets the top bounding box edge origin offset value."""
        self._bbox[1] = value

    @property
    def top_left(self):
        """Returns the top left bounding box corner coordinates :math:`(x,y)`."""
        return self._bbox[0], self._bbox[1]

    @top_left.setter
    def top_left(self, value):
        """Sets the top left bounding box corner coordinates :math:`(x,y)`."""
        assert isinstance(value, (list, tuple, np.ndarray, torch.Tensor)) and len(value) == 2, "invalid input type/len"
        self._bbox[0], self._bbox[1] = value[0], value[1]

    @property
    def right(self):
        """Returns the right bounding box edge origin offset value."""
        return self._bbox[2]

    @right.setter
    def right(self, value):
        """Sets the right bounding box edge origin offset value."""
        self._bbox[2] = value

    @property
    def bottom(self):
        """Returns the bottom bounding box edge origin offset value."""
        return self._bbox[3]

    @bottom.setter
    def bottom(self, value):
        """Sets the bottom bounding box edge origin offset value."""
        self._bbox[3] = value

    @property
    def bottom_right(self):
        """Returns the bottom right bounding box corner coordinates :math:`(x,y)`."""
        return self._bbox[2], self._bbox[3]

    @bottom_right.setter
    def bottom_right(self, value):
        """Sets the bottom right bounding box corner coordinates :math:`(x,y)`."""
        assert isinstance(value, (list, tuple, np.ndarray, torch.Tensor)) and len(value) == 2, "invalid input type/len"
        assert value[0] >= self._bbox[0] and value[1] >= self._bbox[1]
        self._bbox[2], self._bbox[3] = value[0], value[1]

    @property
    def width(self):
        """Returns the width of the bounding box."""
        return (self._bbox[2] - self._bbox[0]) + 1 if self.include_margin else 0

    @property
    def height(self):
        """Returns the height of the bounding box."""
        return (self._bbox[3] - self._bbox[1]) + 1 if self.include_margin else 0

    @property
    def centroid(self, floor=False):
        """Returns the bounding box centroid coordinates :math:`(x,y)`."""
        if self.include_margin:
            if floor:
                return (self._bbox[0] + self._bbox[2] + 1) // 2, (self._bbox[1] + self._bbox[3] + 1) // 2
            return (self._bbox[0] + self._bbox[2] + 1) / 2, (self._bbox[1] + self._bbox[3] + 1) / 2
        else:
            if floor:
                return (self._bbox[0] + self._bbox[2]) // 2, (self._bbox[1] + self._bbox[3]) // 2
            return (self._bbox[0] + self._bbox[2]) / 2, (self._bbox[1] + self._bbox[3]) / 2

    @property
    def include_margin(self):
        """Returns whether :math:`x_max` and :math:`y_max` are included in the bounding box area or not"""
        return self._include_margin

    @include_margin.setter
    def include_margin(self, value):
        """Sets whether :math:`x_max` and :math:`y_max` are is included in the bounding box area or not"""
        assert isinstance(value, (int, bool)), "flag type must be integer or boolean"
        self._include_margin = value

    @property
    def difficult(self):
        """Returns whether this bounding box is considered *difficult* by the dataset (false by default)."""
        return self._difficult

    @difficult.setter
    def difficult(self, value):
        """Sets whether this bounding box is considered *difficult* by the dataset."""
        assert isinstance(value, (int, bool)), "flag type must be integer or boolean"
        self._difficult = value

    @property
    def occluded(self):
        """Returns whether this bounding box is considered *occluded* by the dataset (false by default)."""
        return self._occluded

    @occluded.setter
    def occluded(self, value):
        """Sets whether this bounding box is considered *occluded* by the dataset."""
        assert isinstance(value, (int, bool)), "flag type must be integer or boolean"
        self._occluded = value

    @property
    def truncated(self):
        """Returns whether this bounding box is considered *truncated* by the dataset (false by default)."""
        return self._truncated

    @truncated.setter
    def truncated(self, value):
        """Sets whether this bounding box is considered *truncated* by the dataset."""
        assert isinstance(value, (int, bool)), "flag type must be integer or boolean"
        self._truncated = value

    @property
    def iscrowd(self):
        """Returns whether this instance covers a *crowd* of objects or not (false by default)."""
        return self._iscrowd

    @iscrowd.setter
    def iscrowd(self, value):
        """Sets whether this instance covers a *crowd* of objects or not."""
        assert isinstance(value, (int, bool)), "flag type must be integer or boolean"
        self._iscrowd = value

    @property
    def area(self):
        """Returns a scalar indicating the total surface of the annotation (may be None if unknown/unspecified)."""
        return self.width * self.height

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
        if value is not None:
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
            vec = [*self.bbox, self.class_id, self.include_margin, self.difficult, self.occluded,
                   self.truncated, self.iscrowd, self.image_id]
            if self.confidence is not None:
                vec += [self.confidence] if isinstance(self.confidence, float) else [*self.confidence]
            return vec

    @staticmethod
    def decode(vec, format=None):
        """Returns a BoundingBox object from a vectorized representation in a specified format.

        .. note::
            The input bbox is expected to be a 4 element array :math:`(x_min,y_min,x_max,y_max)`.
        """
        if format == "coco":
            assert len(vec) == 5, "unexpected vector length (should contain 5 values)"
            return BoundingBox(class_id=vec[4], bbox=[vec[0], vec[1], vec[0] + vec[2], vec[1] + vec[3]])
        elif format == "pascal_voc":
            assert len(vec) == 8, "unexpected vector length (should contain 8 values)"
            return BoundingBox(class_id=vec[4], bbox=vec[0:4], difficult=vec[5], occluded=vec[6], truncated=vec[7])
        else:
            assert format is None, "unrecognized/unknown encoding format"
            assert len(vec) >= 12, "unexpected vector length (should contain 9 values or more)"
            return BoundingBox(class_id=vec[4], bbox=vec[0:4], include_margin=vec[5], difficult=vec[6],
                               occluded=vec[7], truncated=vec[8], iscrowd=vec[9],
                               confidence=(None if len(vec) == 11 else vec[11:]),
                               image_id=vec[10], task=None)

    def intersects(self, geom):
        """Returns whether the bounding box intersects a geometry (i.e. a 2D point or another bbox)."""
        assert isinstance(geom, (tuple, list, np.ndarray, BoundingBox)), "unexpected input geometry type"
        if isinstance(geom, (tuple, list, np.ndarray)):
            # check intersection with point
            assert len(geom) == 2, "point should be given as list of coordinates (x,y)"
            return self._bbox[0] <= geom[0] <= self.bbox[2] and self._bbox[1] <= geom[1] <= self.bbox[3]
        else:
            return not (self._bbox[0] > geom._bbox[2] or geom._bbox[0] > self._bbox[2] or
                        self._bbox[3] < geom._bbox[1] or geom._bbox[3] < self._bbox[1])

    def totuple(self):
        # type: () -> Tuple[int, int, int, int]
        """Gets a ``tuple`` representation of the underlying bounding box tuple :math:`(x_min,y_min,x_max,y_max)`.

        This ensures that ``Tensor`` objects are converted to native *Python* types."""
        return tuple(self.tolist())

    def tolist(self):
        # type: () -> List[int]
        """Gets a ``list`` representation of the underlying bounding box tuple :math:`(x_min,y_min,x_max,y_max)`.

        This ensures that ``Tensor`` objects are converted to native *Python* types."""
        return self._bbox.tolist() if isinstance(self._bbox, torch.Tensor) else list(self._bbox)

    def json(self):
        # type: () -> thelper.typedefs.JSON
        """Gets a JSON-serializable representation of the bounding box parameters."""
        return {
            "class_id": self.class_id,
            "image_id": self.image_id,
            "bbox": self.tolist(),
            "confidence": self.confidence,
            "include_margin": self.include_margin,
            "difficult": self.difficult,
            "occluded": self.occluded,
            "truncated": self.truncated,
            "is_crowd": self.iscrowd,
        }

    def __repr__(self):
        """Creates a print-friendly representation of the object detection bbox instance."""
        # note: we do not export the task reference here (it might be too heavy for logs)
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(class_id={repr(self.class_id)}, bbox={repr(self.bbox)}, include_margin={repr(self.include_margin)}, " + \
            f"difficult={repr(self.difficult)}, occluded={repr(self.occluded)}, truncated={repr(self.truncated)}, " + \
            f"iscrowd={repr(self.iscrowd)}, confidence={repr(self.confidence)}, image_id={repr(self.image_id)})"


@thelper.concepts.detection
class Detection(Regression, ClassNamesHandler):
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
        ClassNamesHandler.__init__(self, class_names=class_names)
        if background is not None:
            background = None if "background" not in self.class_indices else self.class_indices["background"]
        self.background = background
        self.color_map = color_map

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
            self._color_map = {}

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
                               self.color_map.keys() == task.color_map.keys() and
                               all([np.array_equal(self.color_map[k], task.color_map[k]) for k in self.color_map])))
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
        color_map = {k: v.tolist() for k, v in self.color_map.items()}
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(class_names={repr(self.class_indices)}, input_key={repr(self.input_key)}, " + \
            f"bboxes_key={repr(self.gt_key)}, meta_keys={repr(self.meta_keys)}, " + \
            f"input_shape={repr(self.input_shape)}, target_shape={repr(self.target_shape)}, " + \
            f"target_min={repr(self.target_min)}, target_max={repr(self.target_max)}, " + \
            f"background={repr(self.background)}, color_map={repr(color_map)})"
