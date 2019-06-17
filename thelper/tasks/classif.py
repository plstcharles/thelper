"""Classification task interface module.

This module contains a class that defines the objectives of models/trainers for classification tasks.
"""
import copy
import json
import logging
import os
from typing import Optional  # noqa: F401

import numpy as np
import torch

import thelper.utils
from thelper.tasks.utils import Task

logger = logging.getLogger(__name__)


class Classification(Task):
    """Interface for input labeling/classification tasks.

    This specialization requests that when given an input tensor, the trained model should
    provide prediction scores for each predefined label (or class). The label names are used
    here to help categorize samples, and to assure that two tasks are only identical when their
    label counts and ordering match.

    Attributes:
        class_names: list of label (class) names to predict (each name should be a string).
        input_key: the key used to fetch input tensors from a sample dictionary.
        label_key: the key used to fetch label (class) names/indices from a sample dictionary.
        meta_keys: the list of extra keys provided by the data parser inside each sample.

    .. seealso::
        | :class:`thelper.tasks.utils.Task`
        | :class:`thelper.train.classif.ImageClassifTrainer`
    """

    def __init__(self, class_names, input_key, label_key, meta_keys=None):
        """Receives and stores the class (or label) names to predict, the input tensor key, the
        groundtruth label (class) key, and the extra (meta) keys produced by the dataset parser(s).

        The class names can be provided as a list of strings, or as a path to a json file that
        contains such a list. The list must contain at least two items. All other arguments are
        used as-is to index dictionaries, and must therefore be key-compatible types.
        """
        super(Classification, self).__init__(input_key, label_key, meta_keys)
        self.class_names = class_names

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
        if isinstance(class_names, dict):
            assert all([idx in class_names or str(idx) in class_names for idx in range(len(class_names))]), \
                "missing class indices (all integers must be consecutive)"
            class_names = [thelper.utils.get_key([idx, str(idx)], class_names) for idx in range(len(class_names))]
        assert isinstance(class_names, list), "expected class names to be provided as an array"
        assert all([isinstance(name, str) for name in class_names]), "all classes must be named with strings"
        assert len(class_names) >= 1, "should have at least one class!"
        if len(class_names) != len(set(class_names)):
            # no longer throwing here, imagenet possesses such a case ('crane#134' and 'crane#517')
            logger.warning("found duplicated name in class list, might be a data entry problem...")
            class_names = [name if class_names.count(name) == 1 else name + "#" + str(idx)
                           for idx, name in enumerate(class_names)]
        self._class_names = copy.deepcopy(class_names)
        self._class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

    @property
    def class_indices(self):
        """Returns the class-name-to-index map used for encoding labels as integers."""
        return self._class_indices

    @class_indices.setter
    def class_indices(self, class_indices):
        """Sets the class-name-to-index map used for encoding labels as integers."""
        assert isinstance(class_indices, dict), "class indices must be provided as dictionary"
        self.class_names = class_indices

    def get_class_sizes(self, samples):
        """Given a list of samples, returns a map of sample counts for each class label."""
        class_idxs = self.get_class_sample_map(samples)
        return {class_name: len(class_idxs[class_name]) for class_name in class_idxs}

    def get_class_sample_map(self, samples, unset_key=None):
        """Splits a list of samples based on their labels into a map of sample lists.

        This function is useful if we need to split a dataset based on its label categories in
        order to sort it, augment it, or re-balance it. The samples do not need to be fully loaded
        for this to work, as only their label (gt) value will be queried. If a sample is missing
        its label, it will be ignored and left out of the generated dictionary unless a value is
        given for ``unset_key``.

        Args:
            samples: the samples to split, where each sample is provided as a dictionary.
            unset_key: a key under which all unlabeled samples should be kept (``None`` = ignore).

        Returns:
            A dictionary that maps each class label to its corresponding list of samples.
        """
        assert samples is not None and isinstance(samples, list), "invalid sample list "
        sample_idxs = {class_name: [] for class_name in self.class_names}
        import collections
        if unset_key is not None:
            assert isinstance(unset_key, collections.Hashable), "unset class name key should be hashable"
            assert unset_key not in sample_idxs, "unset class name key cannot already be in class names list"
            sample_idxs[unset_key] = []
        for sample_idx, sample in enumerate(samples):
            if self.gt_key is None or self.gt_key not in sample:
                if unset_key is not None:
                    class_name = unset_key
                else:
                    continue
            else:
                class_name = sample[self.gt_key]
                assert isinstance(class_name, (str, int, np.ndarray, torch.Tensor)) and thelper.utils.is_scalar(class_name), \
                    "unexpected sample label type (need scalar, string or int)"
                if isinstance(class_name, str):
                    assert class_name in self.class_names, f"label '{class_name}' not found in original task class names list"
                else:
                    if isinstance(class_name, torch.Tensor):
                        class_name = class_name.item()
                    # dataset must already be using indices, we will forgive this...
                    # (this is pretty much always the case for torchvision datasets)
                    assert 0 <= class_name < len(self.class_names), "class name given as out-of-range index"
                    class_name = self.class_names[class_name]
            sample_idxs[class_name].append(sample_idx)
        return sample_idxs

    def check_compat(self, task, exact=False):
        # type: (Classification, Optional[bool]) -> bool
        """Returns whether the current task is compatible with the provided one or not.

        This is useful for sanity-checking, and to see if the inputs/outputs of two models
        are compatible. If ``exact = True``, all fields will be checked for exact (perfect)
        compatibility (in this case, matching meta keys and class name order).
        """
        if isinstance(task, Classification):
            # if both tasks are related to classification, gt keys and class names must match
            return self.input_key == task.input_key and \
                (self.gt_key is None or task.gt_key is None or self.gt_key == task.gt_key) and \
                all([cls in self.class_names for cls in task.class_names]) and \
                (not exact or (self.class_names == task.class_names and
                               set(self.meta_keys) == set(task.meta_keys) and
                               self.gt_key == task.gt_key))
        elif type(task) == Task:
            # if 'task' simply has no gt, compatibility rests on input key only
            return not exact and self.input_key == task.input_key and task.gt_key is None
        return False

    def get_compat(self, task):
        """Returns a task instance compatible with the current task and the given one."""
        assert isinstance(task, Classification) or type(task) == Task, \
            f"cannot create compatible task from types '{type(task)}' and '{type(self)}'"
        if isinstance(task, Classification):
            assert self.input_key == task.input_key, "input key mismatch, cannot create compatible task"
            assert self.gt_key is None or task.gt_key is None or self.gt_key == task.gt_key, \
                "gt key mismatch, cannot create compatible task"
            meta_keys = list(set(self.meta_keys + task.meta_keys))
            # cannot use set for class names, order needs to stay intact!
            class_names = self.class_names + [name for name in task.class_names if name not in self.class_names]
            return Classification(class_names=class_names, input_key=self.input_key,
                                  label_key=self.gt_key, meta_keys=meta_keys)
        elif type(task) == Task:
            assert self.check_compat(task), f"cannot create compatible task between:\n\t{str(self)}\n\t{str(task)}"
            meta_keys = list(set(self.meta_keys + task.meta_keys))
            return Classification(class_names=self.class_names, input_key=self.input_key,
                                  label_key=self.gt_key, meta_keys=meta_keys)

    def __repr__(self):
        """Creates a print-friendly representation of a classification task."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(class_names={repr(self.class_names)}, input_key={repr(self.input_key)}, " + \
            f"label_key={repr(self.gt_key)}, meta_keys={repr(self.meta_keys)})"
