"""Tasks interface module.

This module contains task interfaces that define the (training) goal of a model. These
tasks are deduced from a configuration file, or obtained from a dataset interface. They
essentially contain information about model input/output tensor formats and keys. All
models instantiated by this framework are attached to a task.

For now, only 'classification' (i.e. generic image recognition) is implemented. New tasks
such as object detection and image segmentation will later be added here.
"""
import logging
import json
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Task(ABC):
    """Abstract task interface that holds sample keys.

    Since the framework's data loaders expect samples to be passed in as dictionaries, keys
    are required to obtain the input that should be forwarded to a model, and to obtain the
    groundtruth required for the evaluation of model predictions. Other keys might also be
    kept by this abstract interface for reference (these are considered meta keys). Getter
    functions thus have to be implemented in the derived class to provide all these keys.
    """

    @abstractmethod
    def get_input_key(self):
        """Returns the key used to fetch input data tensors from loaded samples.

        The key can be of any type, as long as it can be used to index a dictionary. Print-
        friendly types (e.g. string) are recommended for debugging.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gt_key(self):
        """Returns the key used to fetch groundtruth data tensors from loaded samples.

        The key can be of any type, as long as it can be used to index a dictionary. Print-
        friendly types (e.g. string) are recommended for debugging.
        """
        raise NotImplementedError

    @abstractmethod
    def get_meta_keys(self):
        """Returns a list of keys used to carry metadata and auxiliary info in samples.

        The keys can be of any type, as long as they can be used to index a dictionary.
        Print-friendly types (e.g. string) are recommended for debugging. This list can
        be empty if the dataset/model does not provide/require any extra inputs.
        """
        raise NotImplementedError

    def __eq__(self, other):
        """Checks whether two tasks are compatible or not.

        This is useful for sanity-checking, and to see if the inputs/outputs of two models
        are compatible. It should ideally be overridden in derived classes to add checks.
        """
        if isinstance(other, self.__class__):
            return (self.get_input_key() == other.get_input_key() and
                    self.get_gt_key() == other.get_gt_key() and
                    self.get_meta_keys() == other.get_meta_keys())
        return False

    def __ne__(self, other):
        """Checks whether two tasks are compatible or not. See __eq__ for more info."""
        return not (self == other)

    def __repr__(self):
        """Creates a print-friendly representation of an abstract task."""
        return self.__class__.__name__ + ": " + str({
            "input": self.get_input_key(),
            "gt": self.get_gt_key(),
            "meta": self.get_meta_keys()
        })


class Classification(Task):
    """Classification interface for input-to-label translation.

    This specialization requests that for a given input, the model should provide prediction
    scores for each predefined label (or class). The label names are not used by the model,
    but the number of labels will affect the complexity of its final layer. The names are used
    here to help categorize samples, and to assure that two tasks are only identical when
    their label counts and ordering match, which helps during sanity-checks.
    """

    def __init__(self, class_names, input_key, label_key, meta_keys=None):
        self.class_names = class_names
        if isinstance(class_names, str) and os.path.exists(class_names):
            with open(class_names, "r") as fd:
                self.class_names = json.load(fd)
        if not isinstance(self.class_names, list):
            raise AssertionError("expected class names to be provided as a list")
        if len(self.class_names) < 1:
            raise AssertionError("should have at least one class!")
        self.input_key = input_key
        self.label_key = label_key
        self.meta_keys = []
        if meta_keys is not None:
            if not isinstance(meta_keys, list):
                raise AssertionError("meta keys should be provided as a list")
            self.meta_keys = meta_keys

    def get_input_key(self):
        return self.input_key

    def get_gt_key(self):
        return self.label_key

    def get_meta_keys(self):
        return self.meta_keys

    def get_nb_classes(self):
        return len(self.class_names)

    def get_class_idxs_map(self):
        return {class_name: idx for idx, class_name in enumerate(self.class_names)}

    def get_class_sizes(self, samples):
        class_idxs = self.get_class_sample_map(samples)
        return {class_name: len(class_idxs[class_name]) for class_name in class_idxs}

    def get_class_sample_map(self, samples):
        if samples is None or not samples:
            raise AssertionError("provided invalid sample list")
        elif not isinstance(samples, list) or not isinstance(samples[0], dict):
            raise AssertionError("dataset samples should be given as list of dictionaries")
        sample_idxs = {class_name: [] for class_name in self.class_names}
        label_keys = self.label_key if isinstance(self.label_key, list) else [self.label_key]
        for sample_idx, sample in enumerate(samples):
            class_name = None
            for key in label_keys:
                if key in sample:
                    class_name = sample[key]
                    break  # by default, stop after finding first match
            if class_name is None:
                raise AssertionError("could not find label key match in sample dict")
            if isinstance(class_name, str):
                if class_name not in self.class_names:
                    raise AssertionError("label '%s' not found in class names provided earlier" % label)
            else:
                raise AssertionError("unexpected sample label type (need string!)")
            sample_idxs[class_name].append(sample_idx)
        return sample_idxs

    def get_class_names(self):
        return self.class_names

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.get_input_key() == other.get_input_key() and
                    self.get_gt_key() == other.get_gt_key() and
                    self.get_meta_keys() == other.get_meta_keys() and
                    self.get_class_names() == other.get_class_names())
        return False

    def __repr__(self):
        return self.__class__.__name__ + ": " + str({
            "input": self.get_input_key(),
            "gt": self.get_gt_key(),
            "meta": self.get_meta_keys(),
            "classes": self.get_class_names()
        })
