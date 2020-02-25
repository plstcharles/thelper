"""Classification task interface module.

This module contains a class that defines the objectives of models/trainers for classification tasks.
"""
import logging
import typing

import numpy as np
import torch

import thelper.concepts
import thelper.utils
from thelper.ifaces import ClassNamesHandler
from thelper.tasks.utils import Task

logger = logging.getLogger(__name__)


@thelper.concepts.classification
class Classification(Task, ClassNamesHandler):
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

    def __init__(
        self,
        class_names: typing.Iterable[typing.AnyStr],
        input_key: typing.Hashable,
        label_key: typing.Hashable,
        meta_keys: typing.Optional[typing.Iterable[typing.Hashable]] = None,
        multi_label: bool = False,
    ):
        """Receives and stores the class (or label) names to predict, the input tensor key, the
        groundtruth label (class) key, and the extra (meta) keys produced by the dataset parser(s).

        The class names can be provided as a list of strings, or as a path to a json file that
        contains such a list. The list must contain at least two items. All other arguments are
        used as-is to index dictionaries, and must therefore be key-compatible types.

        If the `multi_label` is activated, samples with non-scalar class labels will be allowed
        in the `get_class_sizes` and `get_class_sample_map` functions.
        """
        super(Classification, self).__init__(input_key, label_key, meta_keys)
        ClassNamesHandler.__init__(self, class_names=class_names)
        self.multi_label = multi_label

    def get_class_sizes(self, samples: typing.Iterable) -> typing.Dict[typing.AnyStr, int]:
        """Given a list of samples, returns a map of sample counts for each class label."""
        class_idxs = self.get_class_sample_map(samples)
        return {class_name: len(class_idxs[class_name]) for class_name in class_idxs}

    def get_class_sample_map(
        self,
        samples: typing.Iterable,
        unset_key: typing.Optional[typing.Hashable] = None,
    ) -> typing.Dict[typing.AnyStr, typing.List[int]]:
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
        sample_idxs = {class_name: [] for class_name in self.class_names}
        import collections
        if unset_key is not None:
            assert isinstance(unset_key, collections.abc.Hashable), "unset class name key should be hashable"
            assert unset_key not in sample_idxs, "unset class name key cannot already be in class names list"
            sample_idxs[unset_key] = []
        for sample_idx, sample in enumerate(samples):
            gt_attrib = None
            if self.gt_key is not None:
                if isinstance(sample, dict) and self.gt_key in sample:
                    gt_attrib = sample[self.gt_key]
                elif isinstance(self.gt_key, str) and hasattr(sample, self.gt_key):
                    gt_attrib = getattr(sample, self.gt_key)
            if gt_attrib is None:
                if unset_key is not None:
                    sample_idxs[unset_key].append(sample_idx)
                else:
                    continue
            else:
                if self.multi_label:
                    assert isinstance(gt_attrib, typing.Iterable), \
                        "unexpected multi-label classification sample gt type (need iterable, string or int)"
                    if not isinstance(gt_attrib, (typing.List, typing.Tuple, np.ndarray, torch.Tensor)):
                        gt_attrib = [gt for gt in gt_attrib]
                    assert all([isinstance(gt, str) for gt in gt_attrib]) or \
                        all([isinstance(gt, (np.integer, int, bool)) for gt in gt_attrib]), \
                        "multi-label groundtruth array should contain class names or class-wise binary flags"
                    if all([isinstance(gt, str) for gt in gt_attrib]):
                        for gt in gt_attrib:
                            assert gt in self.class_names, f"label '{gt}' not found in task class names"
                            sample_idxs[gt].append(sample_idx)
                    else:
                        assert len(gt_attrib) == len(self.class_names), \
                            "unexpected multi-label one-hot vector shape\n" \
                            f"(should be {len(self.class_names)}-element long, was {len(gt_attrib)})"
                        for class_name, class_flag in zip(self.class_names, gt_attrib):
                            if class_flag:
                                sample_idxs[class_name].append(sample_idx)
                else:
                    assert isinstance(gt_attrib, (str, int, np.ndarray, torch.Tensor)) and \
                        thelper.utils.is_scalar(gt_attrib), \
                        "unexpected classification sample gt type (need scalar, string or int)"
                    if isinstance(gt_attrib, str):
                        assert gt_attrib in self.class_names, f"label '{gt_attrib}' not found in task class names"
                    else:
                        if isinstance(gt_attrib, torch.Tensor):
                            gt_attrib = gt_attrib.item()
                        # dataset must already be using indices, we will forgive this...
                        # (this is pretty much always the case for torchvision datasets)
                        assert 0 <= gt_attrib < len(self.class_names), "class name given as out-of-range index"
                        gt_attrib = self.class_names[gt_attrib]
                    sample_idxs[gt_attrib].append(sample_idx)
        # remember: when using multi-label mode, the sample indices might be duplicated across class groups
        return sample_idxs

    def check_compat(self, task: Task, exact: bool = False) -> bool:
        """Returns whether the current task is compatible with the provided one or not.

        This is useful for sanity-checking, and to see if the inputs/outputs of two models
        are compatible. If ``exact = True``, all fields will be checked for exact (perfect)
        compatibility (in this case, matching meta keys and class name order).
        """
        if isinstance(task, Classification):
            # if both tasks are related to classification, gt keys and class names must match
            return self.multi_label == task.multi_label and self.input_key == task.input_key and \
                (self.gt_key is None or task.gt_key is None or self.gt_key == task.gt_key) and \
                all([cls in self.class_names for cls in task.class_names]) and \
                (not exact or (self.class_names == task.class_names and
                               set(self.meta_keys) == set(task.meta_keys) and
                               self.gt_key == task.gt_key))
        elif type(task) == Task:
            # if 'task' simply has no gt, compatibility rests on input key only
            return not exact and self.input_key == task.input_key and task.gt_key is None
        return False

    def get_compat(self, task: Task) -> Task:
        """Returns a task instance compatible with the current task and the given one."""
        assert isinstance(task, Classification) or type(task) == Task, \
            f"cannot create compatible task from types '{type(task)}' and '{type(self)}'"
        if isinstance(task, Classification):
            assert self.multi_label == task.multi_label, "cannot mix multi-label and single-label classif tasks"
            assert self.input_key == task.input_key, "input key mismatch, cannot create compatible task"
            assert self.gt_key is None or task.gt_key is None or self.gt_key == task.gt_key, \
                "gt key mismatch, cannot create compatible task"
            meta_keys = list(set(self.meta_keys + task.meta_keys))
            # cannot use set for class names, order needs to stay intact!
            class_names = self.class_names + [name for name in task.class_names if name not in self.class_names]
            return Classification(class_names=class_names, input_key=self.input_key,
                                  label_key=self.gt_key, meta_keys=meta_keys, multi_label=self.multi_label)
        elif type(task) == Task:
            assert self.check_compat(task), f"cannot create compatible task between:\n\t{str(self)}\n\t{str(task)}"
            meta_keys = list(set(self.meta_keys + task.meta_keys))
            return Classification(class_names=self.class_names, input_key=self.input_key,
                                  label_key=self.gt_key, meta_keys=meta_keys, multi_label=self.multi_label)

    def __repr__(self) -> str:
        """Creates a print-friendly representation of a classification task."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(class_names={repr(self.class_names)}, input_key={repr(self.input_key)}, " + \
            f"label_key={repr(self.gt_key)}, meta_keys={repr(self.meta_keys)}, " + \
            f"multi_label={repr(self.multi_label)})"
