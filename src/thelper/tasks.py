import logging
import json
import os
from abc import ABC,abstractmethod

logger = logging.getLogger(__name__)


class Task(ABC):
    @abstractmethod
    def get_input_key(self):
        raise NotImplementedError

    @abstractmethod
    def get_gt_key(self):
        raise NotImplementedError


class Classification(Task):
    def __init__(self,class_map,input_key,label_key=None):
        self.class_map = class_map
        if isinstance(class_map,str) and os.path.exists(class_map):
            with open(class_map,"r") as fd:
                self.class_map = json.load(fd)
        if not isinstance(self.class_map,dict):
            raise AssertionError("expected class map to be dict (idx->name)")
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

    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return self.__dict__==other.__dict__
        return False

    def __ne__(self,other):
        if isinstance(other,self.__class__):
            return self.__dict__!=other.__dict__
        return True
