"""Tasks interface module.

This module contains task interfaces that define the objectives of models/trainers. These
tasks are deduced from a configuration file or obtained from a dataset interface. They
essentially contain information about model input/output tensor formats and keys. All
models/trainers instantiated by this framework are specialized using a task object.
"""
import json
import logging
import os

import thelper.utils

logger = logging.getLogger(__name__)


def load_task(config):
    """Parses a configuration dictionary or repr string and returns an instantiated task from it.

    If a string is provided, it will first be parsed to get the class, and then the object will be
    instantiated by forwarding the parameters contained in the string to the constructor of that
    class. Note that it is important for this function to work that the constructor argument names
    match the names of parameters printed in the task's ``__repr__`` function.

    If a dict is provided, it should contain a 'type' and a 'params' field with the values required
    for direct instantiation.

    .. seealso::
        :class:`thelper.tasks.Task`
        :func:`thelper.tasks.Task.__repr__`
    """
    if config is None or not isinstance(config, (str, dict)):
        raise AssertionError("unexpected config type (should be str or dict)")
    if isinstance(config, dict):
        if "type" not in config or not isinstance(config["type"], str):
            raise AssertionError("invalid field 'type' in task config")
        task_type = thelper.utils.import_class(config["type"])
        if "params" not in config or not isinstance(config["params"], dict):
            raise AssertionError("invalid field 'params' in task config")
        task_params = config["params"]
        task = task_type(**task_params)
        if not isinstance(task, thelper.tasks.Task):
            raise AssertionError("the task must be derived from 'thelper.tasks.Task'")
        return task
    elif isinstance(config, str):
        task_type = thelper.utils.import_class(repr.split(": ")[0])
        task_params = eval(": ".join(repr.split(": ")[1:]))
        task = task_type(**task_params)
        if not isinstance(task, thelper.tasks.Task):
            raise AssertionError("the task must be derived from 'thelper.tasks.Task'")
        return task


def get_global_task(tasks):
    """Returns a global task object that is compatible with a list of subtasks.

    When different datasets must be combined in a session, the tasks they define must also be
    merged. This functions allows us to do so as long as the tasks all share a common objective.
    If creating a globally-compatible task is impossible, this function will raise an exception.
    Otherwise, the returned task object can be used to replace the subtasks of all used datasets.

    .. seealso::
        :class:`thelper.tasks.Task`
        :func:`thelper.data.load_datasets`
    """
    if tasks is None:
        return None
    if not isinstance(tasks, list):
        raise AssertionError("tasks should be provided as list")
    ref_task = None
    for task in tasks:
        if task is None:
            # skip all undefined tasks
            continue
        if not isinstance(task, Task):
            raise AssertionError("all tasks should derive from thelper.tasks.Task")
        if ref_task is None:
            # no reference task set; take the first instance and continue to next
            ref_task = task
            continue
        if type(ref_task) != Task:
            # reference task already specialized, we can ask it for compatible instances
            ref_task = ref_task.get_compat(task)
        else:
            # otherwise, keep asking the new one to stay compatible with the base ref
            ref_task = task.get_compat(ref_task)
    return ref_task


class Task(object):
    """Basic task interface that defines a training objective and that holds sample i/o keys.

    Since the framework's data loaders expect samples to be passed in as dictionaries, keys
    are required to obtain the input that should be forwarded to a model, and to obtain the
    groundtruth required for the evaluation of model predictions. Other keys might also be
    kept by this interface for reference (these are considered meta keys).

    Note that while this interface can be instantiated directly, trainers and models might
    not be provided enough information about their goal to be corectly instantiated. Thus,
    specialized task objects derived from this base class should be used if possible.

    Attributes:
        input_key: the key used to fetch input tensors from a sample dictionary.
        gt_key: the key used to fetch gt tensors from a sample dictionary.
        meta_keys: the list of extra keys provided by the data parser inside each sample.

    .. seealso::
        :class:`thelper.tasks.Classification`
    """

    def __init__(self, input_key, gt_key=None, meta_keys=None):
        """Receives and stores the input tensor key, the groundtruth tensor key, and the extra
        (meta) keys produced by the dataset parser(s).
        """
        if input_key is None:
            raise AssertionError("task input key cannot be None (input tensors must always be available)")
        self.input_key = input_key
        self.gt_key = gt_key
        self.meta_keys = []
        if meta_keys is not None:
            if not isinstance(meta_keys, list):
                raise AssertionError("meta keys should be provided as a list")
            self.meta_keys = meta_keys

    def get_input_key(self):
        """Returns the key used to fetch input data tensors from a sample dictionary.

        The key can be of any type, as long as it can be used to index a dictionary. Print-
        friendly types (e.g. string) are recommended for debugging. This key can never be
        ``None``, as input tensors should always be available in loaded samples.
        """
        return self.input_key

    def get_gt_key(self):
        """Returns the key used to fetch groundtruth data tensors from a sample dictionary.

        The key can be of any type, as long as it can be used to index a dictionary. Print-
        friendly types (e.g. string) are recommended for debugging. If groundtruth is not
        available through the dataset parsers, this function should teturn ``None``.
        """
        return self.gt_key

    def get_meta_keys(self):
        """Returns a list of keys used to carry metadata and auxiliary info in samples.

        The keys can be of any type, as long as they can be used to index a dictionary.
        Print-friendly types (e.g. string) are recommended for debugging. This list can
        be empty if the dataset/model does not provide/require any extra inputs.
        """
        return self.meta_keys

    def check_compat(self, other):
        """Returns whether the current task is compatible with the provided one or not.

        This is useful for sanity-checking, and to see if the inputs/outputs of two models
        are compatible. It should ideally be overridden in derived classes to specialize
        the compatibility verification.
        """
        if type(other) == Task:
            return (self.get_input_key() == other.get_input_key() and
                    self.get_gt_key() == other.get_gt_key())
        return False

    def get_compat(self, other):
        """Returns a task instance compatible with the current task and the given one."""
        if type(other) == Task:
            if not self.check_compat(other):
                raise AssertionError("cannot create compatible task instance between:\n"
                                     "\tself: %s\n\tother: %s" % (str(self), str(other)))
            meta_keys = list(set(self.get_meta_keys() + other.get_meta_keys()))
            return Task(self.get_input_key(), self.get_gt_key(), meta_keys)
        else:
            raise AssertionError("cannot combine task type '%s' with '%s'" % (str(other.__class__), str(self.__class__)))

    def __repr__(self):
        """Creates a print-friendly representation of an abstract task.

        Note that this representation might also be used to check the compatibility of tasks
        without importing the whole framework. Therefore, it should contain all the necessary
        information about the task. The name of the parameters herein should also match the
        argument names given to the constructor in case we need to recreate a task object from
        this string.
        """
        return self.__class__.__qualname__ + ": " + str({
            "input_key": self.get_input_key(),
            "gt_key": self.get_gt_key(),
            "meta_keys": self.get_meta_keys()
        })


class Classification(Task):
    """Interface for input-to-label classification tasks.

    This specialization requests that the model provides prediction scores for each predefined
    label (or class) given an input tensor. The label names are used here to help categorize
    samples, and to assure that two tasks are only identical when their label counts and
    ordering match.

    Attributes:
        class_names: list of label (class) names to predict (each name should be a string).
        input_key: the key used to fetch input tensors from a sample dictionary.
        label_key: the key used to fetch label (class) names/indices from a sample dictionary.
        meta_keys: the list of extra keys provided by the data parser inside each sample.

    .. seealso::
        :class:`thelper.tasks.Task`
        :class:`thelper.train.ImageClassifTrainer`
    """

    def __init__(self, class_names, input_key, label_key, meta_keys=None):
        """Receives and stores the class (or label) names to predict, the input tensor key, the
        groundtruth label (class) key, and the extra (meta) keys produced by the dataset parser(s).

        The class names can be provided as a list of strings, or as a path to a json file that
        contains such a list. The list must contain at least two items. All other arguments are
        used as-is to index dictionaries, and must therefore be key-compatible types.
        """
        super().__init__(input_key, label_key, meta_keys)
        self.class_names = class_names
        if isinstance(class_names, str) and os.path.exists(class_names):
            with open(class_names, "r") as fd:
                self.class_names = json.load(fd)
        if not isinstance(self.class_names, list):
            raise AssertionError("expected class names to be provided as a list")
        if len(self.class_names) < 1:
            raise AssertionError("should have at least one class!")
        if len(self.class_names) != len(set(self.class_names)):
            raise AssertionError("list should not contain duplicates")

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
        order to sort it, augment it, or rebalance it. The samples do not need to be fully loaded
        for this to work, as only their label (gt) value will be queried. If a sample is missing
        its label, it will be ignored and left out of the generated dictionary unless a value is
        given for ``unset_key``.

        Args:
            samples: a list of samples to split, where each sample is a dictionary.
            unset_key: a key under which all unlabeled samples should be kept (``None`` = ignore).

        Returns:
            A dictionary that maps each class label to its corresponding list of samples.
        """
        if samples is None or not samples:
            raise AssertionError("provided invalid sample list")
        elif not isinstance(samples, list) or not isinstance(samples[0], dict):
            raise AssertionError("dataset samples should be given as list of dictionaries")
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

    def check_compat(self, other):
        """Returns whether the current task is compatible with the provided one or not.

        In this case, an extra check regarding class names is added when all other fields match.
        """
        if isinstance(other, Classification):
            # if both tasks are related to classification, gt keys and class names must match
            # note: one class name array can be bigger than the other, as long as the overlap is the same
            return (self.get_input_key() == other.get_input_key() and (
                # REALLY DIRTY HACK FOR CHECKPOINT BACKWARD COMPAT HERE, TO BE REMOVED ASAP @@@@@@
                (hasattr(self, "label_key") and isinstance(self.label_key, str) and
                 (other.get_gt_key() is None or self.label_key == other.get_gt_key())) or
                # line below is ok, should be default check later on
                (self.get_gt_key() is None or other.get_gt_key() is None or self.get_gt_key() == other.get_gt_key())
            ) and all([cls1 == cls2 for cls1, cls2 in zip(self.get_class_names(), other.get_class_names())]))
        elif type(other) == Task:
            # if 'other' simply has no gt, compatibility rests on input key only
            return self.get_input_key() == other.get_input_key() and other.get_gt_key() is None
        return False

    def get_compat(self, other):
        """Returns a task instance compatible with the current task and the given one."""
        if isinstance(other, Classification):
            if self.get_input_key() != other.get_input_key():
                raise AssertionError("input key mismatch, cannot create compatible task")
            if self.get_gt_key() is not None and other.get_gt_key() is not None and self.get_gt_key() != other.get_gt_key():
                raise AssertionError("gt key mismatch, cannot create compatible task")
            meta_keys = list(set(self.get_meta_keys() + other.get_meta_keys()))
            # cannot use set for class names, order needs to stay intact!
            class_names = self.get_class_names() + [name for name in other.get_class_names() if name not in self.get_class_names()]
            return Classification(class_names, self.get_input_key(), self.get_gt_key(), meta_keys)
        elif type(other) == Task:
            if not self.check_compat(other):
                raise AssertionError("cannot create compatible task instance between:\n"
                                     "\tself: %s\n\tother: %s" % (str(self), str(other)))
            meta_keys = list(set(self.get_meta_keys() + other.get_meta_keys()))
            return Classification(self.get_class_names(), self.get_input_key(), self.get_gt_key(), meta_keys)
        else:
            raise AssertionError("cannot combine task type '%s' with '%s'" % (str(other.__class__), str(self.__class__)))

    def __repr__(self):
        """Creates a print-friendly representation of an abstract task.

        Note that this representation might also be used to check the compatibility of tasks
        without importing the whole framework. Therefore, it should contain all the necessary
        information about the task. The name of the parameters herein should also match the
        argument names given to the constructor in case we need to recreate a task object from
        this string.
        """
        return self.__class__.__qualname__ + ": " + str({
            "class_names": self.get_class_names(),
            "input_key": self.get_input_key(),
            "label_key": self.get_gt_key(),
            "meta_keys": self.get_meta_keys(),
        })

# todo: add new task types (objdetecton, segmentation, regression, superres, ...)
