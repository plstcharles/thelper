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

    def __init__(self, class_map, input_key, label_key=None, meta_keys=None):
        super().__init__(meta_keys)
        self.class_map = class_map
        if isinstance(class_map, str) and os.path.exists(class_map):
            with open(class_map, "r") as fd:
                self.class_map = json.load(fd)
        if not isinstance(self.class_map, dict):
            raise AssertionError("expected class map to be dict (idx->name)")
        if len(self.class_map) < 2:
            raise AssertionError("should have at least two classes!")
        self.binary = len(self.class_map) == 2
        self.input_key = input_key
        self.label_key = label_key

    def get_input_key(self):
        return self.input_key

    def get_gt_key(self):
        return self.label_key

    def get_nb_classes(self):
        return len(self.class_map)

    def get_class_map(self):
        return self.class_map
