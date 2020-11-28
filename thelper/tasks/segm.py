"""Segmentation task interface module.

This module contains a class that defines the objectives of models/trainers for segmentation tasks.
"""
import logging
from typing import Optional  # noqa: F401

import numpy as np
import PIL.Image
import torch
import tqdm

import thelper.concepts
import thelper.utils
from thelper.ifaces import ClassNamesHandler, ColorMapHandler
from thelper.tasks.utils import Task

logger = logging.getLogger(__name__)


@thelper.concepts.segmentation
class Segmentation(Task, ClassNamesHandler, ColorMapHandler):
    """Interface for pixel-level labeling/classification (segmentation) tasks.

    This specialization requests that when given an input tensor, the trained model should
    provide prediction scores for each predefined label (or class), for each element of the
    input tensor. The label names are used here to help categorize samples, and to assure that
    two tasks are only identical when their label counts and ordering match.

    Attributes:
        class_names: map of class name-value pairs to predict for each pixel.
        input_key: the key used to fetch input tensors from a sample dictionary.
        label_map_key: the key used to fetch label (class) maps from a sample dictionary.
        meta_keys: the list of extra keys provided by the data parser inside each sample.
        dontcare: value of the 'dontcare' label (if any) used in the class map.
        color_map: map of class name-color pairs to use when displaying results.

    .. seealso::
        | :class:`thelper.tasks.utils.Task`
        | :class:`thelper.train.segm.ImageSegmTrainer`
    """

    def __init__(self, class_names, input_key, label_map_key,
                 meta_keys=None, dontcare=None, color_map=None):
        """Receives and stores the class (or label) names to predict, the input tensor key,
        the groundtruth label (class) map key, the extra (meta) keys produced by the dataset
        parser(s), the 'dontcare' label value that might be present in gt maps (if any), and
        the color map used to swap label indices for colors when displaying results.

        The class names can be provided as a list of strings, as a path to a json file that
        contains such a list, or as a map of predefined name-value pairs to use in gt maps.
        This list/map must contain at least two elements. All other arguments are used as-is
        to index dictionaries, and must therefore be key-compatible types.
        """
        super(Segmentation, self).__init__(input_key, label_map_key, meta_keys)
        if "dontcare" in class_names:
            logger.warning("found reserved 'dontcare' label in input classes; it will be removed from the internal list")
            if isinstance(class_names, list):
                del class_names[class_names.index("dontcare")]
            elif isinstance(class_names, dict):
                del class_names["dontcare"]
        ClassNamesHandler.__init__(self, class_names=class_names)
        ColorMapHandler.__init__(self, color_map=color_map, dontcare=dontcare)

    def get_class_sizes(self, samples):
        """Given a list of samples, returns a map of element counts for each class label."""
        assert samples is not None and len(samples) > 0, "provided invalid sample list"
        elem_counts = {class_name: 0 for class_name in self.class_names}
        if self.dontcare is not None:
            elem_counts["dontcare"] = 0
        warned_unknown_value_flag = False
        for sample_idx, sample in tqdm.tqdm(enumerate(samples), desc="cumulating label counts", total=len(samples)):
            if self.gt_key is None or self.gt_key not in sample:
                continue
            else:
                labels = sample[self.gt_key]
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                if isinstance(labels, PIL.Image.Image):
                    labels = np.array(labels)
                assert isinstance(labels, np.ndarray), "unsupported label map type ('%s')" % str(type(labels))
                # here, we assume labels are given as some integer type that corresponds to class name indices
                curr_elem_counts = {cname: np.count_nonzero(labels == cval) for cname, cval in self.class_indices.items()}
                dontcare_elem_count = 0 if self.dontcare is None else np.count_nonzero(labels == self.dontcare)
                if (sum(curr_elem_counts.values()) + dontcare_elem_count) != labels.size and not warned_unknown_value_flag:
                    logger.warning("some label maps contain values that are unknown (i.e. with no proper class mapping)")
                    warned_unknown_value_flag = True
                for class_name in self.class_names:
                    elem_counts[class_name] += curr_elem_counts[class_name]
                if dontcare_elem_count > 0:
                    elem_counts["dontcare"] += dontcare_elem_count
        return elem_counts

    def check_compat(self, task, exact=False):
        # type: (Segmentation, Optional[bool]) -> bool
        """Returns whether the current task is compatible with the provided one or not.

        This is useful for sanity-checking, and to see if the inputs/outputs of two models
        are compatible. If ``exact = True``, all fields will be checked for exact (perfect)
        compatibility (in this case, matching meta keys and class maps).
        """
        if isinstance(task, Segmentation):
            # if both tasks are related to segmentation: gt keys, class names, and dc must match
            return self.input_key == task.input_key and self.dontcare == task.dontcare and \
                (self.gt_key is None or task.gt_key is None or self.gt_key == task.gt_key) and \
                all([cls in self.class_names for cls in task.class_names]) and \
                (not exact or (self.class_names == task.class_names and
                               set(self.meta_keys) == set(task.meta_keys) and
                               self.color_map.keys() == task.color_map.keys() and
                               all([np.array_equal(self.color_map[k], task.color_map[k]) for k in self.color_map]) and
                               self.gt_key == task.gt_key))
        elif type(task) == Task:
            # if 'task' simply has no gt, compatibility rests on input key only
            return not exact and self.input_key == task.input_key and task.gt_key is None
        return False

    def get_compat(self, task):
        """Returns a task instance compatible with the current task and the given one."""
        assert isinstance(task, Segmentation) or type(task) == Task, \
            f"cannot create compatible task from types '{type(task)}' and '{type(self)}'"
        if isinstance(task, Segmentation):
            assert self.input_key == task.input_key, "input key mismatch, cannot create compatible task"
            assert self.gt_key is None or task.gt_key is None or self.gt_key == task.gt_key, \
                "gt key mismatch, cannot create compatible task"
            #TODO Check the validity of this assert
            #assert self.dontcare == task.dontcare, "dontcare value mismatch, cannot create compatible task"
            meta_keys = list(set(self.meta_keys + task.meta_keys))
            # cannot use set for class names, order needs to stay intact!
            class_indices = {cname: cval for cname, cval in task.class_indices.items() if cname not in self.class_indices}
            class_indices = {**self.class_indices, **class_indices}
            color_map = {cname: cval for cname, cval in task.color_map.items() if cname not in self.color_map}
            color_map = {**self.color_map, **color_map}
            return Segmentation(class_names=class_indices, input_key=self.input_key, label_map_key=self.gt_key,
                                meta_keys=meta_keys, dontcare=self.dontcare, color_map=color_map)
        elif type(task) == Task:
            assert self.check_compat(task), f"cannot create compatible task between:\n\t{str(self)}\n\t{str(task)}"
            meta_keys = list(set(self.meta_keys + task.meta_keys))
            return Segmentation(class_names=self.class_indices, input_key=self.input_key, label_map_key=self.gt_key,
                                meta_keys=meta_keys, dontcare=self.dontcare, color_map=self.color_map)

    def __repr__(self):
        """Creates a print-friendly representation of a segmentation task."""
        color_map = {k: v.tolist() for k, v in self.color_map.items()}
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(class_names={repr(self.class_indices)}, input_key={repr(self.input_key)}, " + \
            f"label_map_key={repr(self.gt_key)}, meta_keys={repr(self.meta_keys)}, " + \
            f"dontcare={repr(self.dontcare)}, color_map={repr(color_map)})"
