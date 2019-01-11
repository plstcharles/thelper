"""Segmentation task interface module.

This module contains a class that defines the objectives of models/trainers for segmentation tasks.
"""
import copy
import json
import logging
import os

import numpy as np
import PIL.Image
import torch
import tqdm

from thelper.tasks.utils import Task

logger = logging.getLogger(__name__)


class Segmentation(Task):
    """Interface for pixel-level labeling/classification (segmentation) tasks.

    This specialization requests that when given an input tensor, the trained model should
    provide prediction scores for each predefined label (or class), for each element of the
    input tensor. The label names are used here to help categorize samples, and to assure that
    two tasks are only identical when their label counts and ordering match.

    Attributes:
        class_map: map of class name-value pairs to predict for each pixel.
        dontcare: name of the 'dontcare' label (if any) used in the class map.
        input_key: the key used to fetch input tensors from a sample dictionary.
        label_map_key: the key used to fetch label (class) maps from a sample dictionary.
        meta_keys: the list of extra keys provided by the data parser inside each sample.
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
        super().__init__(input_key, label_map_key, meta_keys)
        if isinstance(class_names, str) and os.path.exists(class_names):
            with open(class_names, "r") as fd:
                class_names = json.load(fd)
        if isinstance(class_names, list):
            class_map = {class_name: class_idx for class_idx, class_name in enumerate(class_names)}
        elif isinstance(class_names, dict):
            class_map = copy.copy(class_names)
        else:
            raise AssertionError("expected class names to be provided as a list or a map of pixel values")
        if "dontcare" in class_map and dontcare is None:
            raise AssertionError("'dontcare' class name is reserved")
        if dontcare is not None:
            if not isinstance(dontcare, (int, float)):
                raise AssertionError("'dontcare' value should be int or float")
            if "dontcare" in class_map:
                if dontcare != class_map["dontcare"]:
                    raise AssertionError("'dontcare' value mismatch with pre-existing class map")
                del class_map["dontcare"]
            else:
                if any([dontcare == val for val in class_map.values()]):
                    raise AssertionError("'dontcare' value matches a pre-existing class name that is not 'dontcare'")
        if len(class_map) < 1:
            raise AssertionError("should have at least one class!")
        if len(class_map) != len(set(class_map)):
            raise AssertionError("class set should not contain duplicates")
        self.class_map = class_map
        self.dontcare = dontcare
        self.color_map = None
        if color_map is not None:
            if not isinstance(color_map, dict):
                raise AssertionError("color map should be given as dictionary")
            self.color_map = {}
            for key, val in color_map.items():
                if key not in self.class_map and key != "dontcare":
                    raise AssertionError("unknown color map entry '%s'" % key)
                if isinstance(val, (list, tuple)):
                    val = np.ndarray(val)
                if not isinstance(val, np.ndarray) or val.size != 3:
                    raise AssertionError("color values should be given as triplets")
                self.color_map[key] = val

    def get_class_names(self):
        """Returns the list of class names to be predicted by the model."""
        return list(self.class_map.keys())

    def get_nb_classes(self):
        """Returns the number of classes (or labels) to be predicted by the model."""
        return len(self.class_map)

    def get_class_idxs_map(self):
        """Returns the class-label-to-index map used for encoding class labels as integers."""
        return self.class_map

    def get_class_sizes(self, samples):
        """Given a list of samples, returns a map of element counts for each class label."""
        if samples is None or not samples:
            raise AssertionError("provided invalid sample list")
        elem_counts = {class_name: 0 for class_name in self.class_map}
        label_map_key = self.get_gt_key()
        warned_unknown_value_flag = False
        for sample_idx, sample in tqdm.tqdm(enumerate(samples), desc="cumulating label counts", total=len(samples)):
            if label_map_key is None or label_map_key not in sample:
                continue
            else:
                labels = sample[label_map_key]
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                if isinstance(labels, PIL.Image.Image):
                    labels = np.array(labels)
                if not isinstance(labels, np.ndarray):
                    raise AssertionError("unsupported label map type ('%s')" % str(type(labels)))
                # here, we assume labels are given as some integer type that corresponds to class name indices
                curr_elem_counts = {class_name: np.count_nonzero(labels == class_val) for class_name, class_val in self.class_map.items()}
                dontcare_elem_count = 0 if self.dontcare is None else np.count_nonzero(labels == self.dontcare)
                if (sum(curr_elem_counts.values()) + dontcare_elem_count) != labels.size and not warned_unknown_value_flag:
                    logger.warning("some label maps contain values that are unknown (i.e. with no proper class mapping)")
                    warned_unknown_value_flag = True
                for class_name in self.class_map:
                    elem_counts[class_name] += curr_elem_counts[class_name]
                if dontcare_elem_count > 0:
                    if "dontcare" not in elem_counts:
                        elem_counts["dontcare"] = 0
                    elem_counts["dontcare"] += dontcare_elem_count
        return elem_counts

    def get_dontcare_val(self):
        """Returns the 'dontcare' label value for this segmentation task (can be ``None``)."""
        return self.dontcare

    def get_color_map(self):
        """Returns the color map used to swap label indices for colors when displaying results."""
        return self.color_map

    def check_compat(self, other, exact=False):
        """Returns whether the current task is compatible with the provided one or not.

        This is useful for sanity-checking, and to see if the inputs/outputs of two models
        are compatible. If ``exact = True``, all fields will be checked for exact (perfect)
        compatibility (in this case, matching meta keys and class maps).
        """
        if isinstance(other, Segmentation):
            # if both tasks are related to segmentation: gt keys, class names, and dc must match
            return (self.get_input_key() == other.get_input_key() and
                    self.get_dontcare_val() == other.get_dontcare_val() and
                    (self.get_gt_key() is None or other.get_gt_key() is None or self.get_gt_key() == other.get_gt_key()) and
                    all([cls in self.get_class_names() for cls in other.get_class_names()]) and
                    (not exact or (self.get_class_idxs_map() == other.get_class_idxs_map() and
                                   self.get_meta_keys() == other.get_meta_keys())))
        elif type(other) == Task:
            # if 'other' simply has no gt, compatibility rests on input key only
            return not exact and self.get_input_key() == other.get_input_key() and other.get_gt_key() is None
        return False

    def get_compat(self, other):
        """Returns a task instance compatible with the current task and the given one."""
        if isinstance(other, Segmentation):
            if self.get_input_key() != other.get_input_key():
                raise AssertionError("input key mismatch, cannot create compatible task")
            if self.get_dontcare_val() != other.get_dontcare_val():
                raise AssertionError("dontcare value mismatch, cannot create compatible task")
            if self.get_gt_key() is not None and other.get_gt_key() is not None and self.get_gt_key() != other.get_gt_key():
                raise AssertionError("gt key mismatch, cannot create compatible task")
            meta_keys = list(set(self.get_meta_keys() + other.get_meta_keys()))
            # cannot use set for class names, order needs to stay intact!
            class_names = self.get_class_names() + [name for name in other.get_class_names() if name not in self.get_class_names()]
            return Segmentation(class_names, self.get_input_key(), self.get_gt_key(),
                                meta_keys=meta_keys, dontcare=self.get_dontcare_val())
        elif type(other) == Task:
            if not self.check_compat(other):
                raise AssertionError("cannot create compatible task instance between:\n"
                                     "\tself: %s\n\tother: %s" % (str(self), str(other)))
            meta_keys = list(set(self.get_meta_keys() + other.get_meta_keys()))
            return Segmentation(self.get_class_idxs_map(), self.get_input_key(), self.get_gt_key(),
                                meta_keys=meta_keys, dontcare=self.get_dontcare_val())
        else:
            raise AssertionError("cannot combine task type '%s' with '%s'" % (str(other.__class__), str(self.__class__)))

    def __repr__(self):
        """Creates a print-friendly representation of a segmentation task."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + ": " + str({
            "class_names": self.get_class_idxs_map(),
            "input_key": self.get_input_key(),
            "label_map_key": self.get_gt_key(),
            "meta_keys": self.get_meta_keys(),
            "dontcare": self.get_dontcare_val()
        })
