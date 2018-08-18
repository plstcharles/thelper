import logging
import json
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Task(ABC):

    def __init__(self, meta_keys=None):
        self.meta_keys = []
        if meta_keys is not None:
            if not isinstance(meta_keys, list):
                raise AssertionError("meta keys should be provided as a list")
            self.meta_keys = meta_keys

    @abstractmethod
    def get_input_key(self):
        raise NotImplementedError

    @abstractmethod
    def get_gt_key(self):
        raise NotImplementedError

    def get_meta_keys(self):
        return self.meta_keys

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return (self.get_input_key() == other.get_input_key() and
                    self.get_gt_key() == other.get_gt_key() and
                    self.get_meta_keys() == other.get_meta_keys())
        return False

    def __ne__(self, other):
        return not (self == other)


class Classification(Task):

    def __init__(self, class_names, input_key, label_key, meta_keys=None):
        super().__init__(meta_keys)
        self.class_names = class_names
        if isinstance(class_names, str) and os.path.exists(class_names):
            with open(class_names, "r") as fd:
                self.class_names = json.load(fd)
        if not isinstance(self.class_names, list):
            raise AssertionError("expected class names to be provided as a list")
        if len(self.class_names) < 1:
            raise AssertionError("should have at least one class!")
        if not isinstance(self.class_names[0], str):
            raise AssertionError("class names should be provided as strings")
        self.input_key = input_key
        self.label_key = label_key

    def get_input_key(self):
        return self.input_key

    def get_gt_key(self):
        return self.label_key

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
