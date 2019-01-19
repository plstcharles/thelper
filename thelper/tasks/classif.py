"""Classification task interface module.

This module contains a class that defines the objectives of models/trainers for classification tasks.
"""
import json
import logging
import os
from typing import Optional  # noqa: F401

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
        if isinstance(class_names, str) and os.path.exists(class_names):
            with open(class_names, "r") as fd:
                self.class_names = json.load(fd)
        if not isinstance(self.class_names, list):
            raise AssertionError("expected class names to be provided as a list")
        if len(self.class_names) < 1:
            raise AssertionError("should have at least one class!")
        if len(self.class_names) != len(set(self.class_names)):
            raise AssertionError("class names should not contain duplicates")

    def get_class_names(self):
        """Returns the list of class names to be predicted by the model."""
        return self.class_names

    def get_nb_classes(self):
        """Returns the number of classes (or labels) to be predicted by the model."""
        return len(self.class_names)

    def get_class_idxs_map(self):
        """Returns the class-label-to-index map used for encoding class labels as integers."""
        return {class_name: idx for idx, class_name in enumerate(self.class_names)}

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
        if samples is None or not samples:
            raise AssertionError("provided invalid sample list")
        sample_idxs = {class_name: [] for class_name in self.class_names}
        if unset_key is not None and not isinstance(unset_key, str):
            raise AssertionError("unset class name key should be string, just like other class names")
        elif unset_key in sample_idxs:
            raise AssertionError("unset class name key already in class names list")
        else:
            sample_idxs[unset_key] = []
        label_key = self.get_gt_key()
        for sample_idx, sample in enumerate(samples):
            if label_key is None or label_key not in sample:
                if unset_key is not None:
                    class_name = unset_key
                else:
                    continue
            else:
                class_name = sample[label_key]
                if isinstance(class_name, str):
                    if class_name not in self.class_names:
                        raise AssertionError("label '%s' not found in class names provided earlier" % class_name)
                elif isinstance(class_name, int):
                    # dataset must already be using indices, we will forgive this...
                    # (this is pretty much always the case for torchvision datasets)
                    if class_name < 0 or class_name >= len(self.class_names):
                        raise AssertionError("class name given as out-of-range index (%d) for class list" % class_name)
                    class_name = self.class_names[class_name]
                else:
                    raise AssertionError("unexpected sample label type (need string!)")
            sample_idxs[class_name].append(sample_idx)
        return sample_idxs

    def check_compat(self, other, exact=False):
        # type: (Classification, Optional[bool]) -> bool
        """Returns whether the current task is compatible with the provided one or not.

        This is useful for sanity-checking, and to see if the inputs/outputs of two models
        are compatible. If ``exact = True``, all fields will be checked for exact (perfect)
        compatibility (in this case, matching meta keys and class name order).
        """
        if isinstance(other, Classification):
            # if both tasks are related to classification, gt keys and class names must match
            return (self.get_input_key() == other.get_input_key() and
                    (self.get_gt_key() is None or other.get_gt_key() is None or
                     self.get_gt_key() == other.get_gt_key()) and
                    all([cls in self.get_class_names() for cls in other.get_class_names()]) and
                    (not exact or (self.get_class_names() == other.get_class_names() and
                                   self.get_meta_keys() == other.get_meta_keys())))
        elif type(other) == Task:
            # if 'other' simply has no gt, compatibility rests on input key only
            return not exact and self.get_input_key() == other.get_input_key() and other.get_gt_key() is None
        return False

    def get_compat(self, other):
        """Returns a task instance compatible with the current task and the given one."""
        if isinstance(other, Classification):
            if self.get_input_key() != other.get_input_key():
                raise AssertionError("input key mismatch, cannot create compatible task")
            if self.get_gt_key() is not None \
                    and other.get_gt_key() is not None \
                    and self.get_gt_key() != other.get_gt_key():
                raise AssertionError("gt key mismatch, cannot create compatible task")
            meta_keys = list(set(self.get_meta_keys() + other.get_meta_keys()))
            # cannot use set for class names, order needs to stay intact!
            other_names = [name for name in other.get_class_names() if name not in self.get_class_names()]
            class_names = self.get_class_names() + other_names
            return Classification(class_names, self.get_input_key(), self.get_gt_key(), meta_keys=meta_keys)
        elif type(other) == Task:
            if not self.check_compat(other):
                raise AssertionError("cannot create compatible task instance between:\n"
                                     "\tself: %s\n\tother: %s" % (str(self), str(other)))
            meta_keys = list(set(self.get_meta_keys() + other.get_meta_keys()))
            return Classification(self.get_class_names(), self.get_input_key(), self.get_gt_key(), meta_keys=meta_keys)
        else:
            raise AssertionError("cannot combine task type '%s' with '%s'"
                                 % (str(other.__class__), str(self.__class__)))

    def __repr__(self):
        """Creates a print-friendly representation of a classification task."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + ": " + str({
            "class_names": self.get_class_names(),
            "input_key": self.get_input_key(),
            "label_key": self.get_gt_key(),
            "meta_keys": self.get_meta_keys(),
        })
