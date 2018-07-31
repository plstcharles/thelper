import logging
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
    def __init__(self,nb_classes,input_key,label_key=None):
        self.nb_classes = nb_classes
        self.input_key = input_key
        self.label_key = label_key

    def get_input_key(self):
        return self.input_key

    def get_gt_key(self):
        return self.label_key

    def get_nb_classes(self):
        return self.nb_classes

    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return self.__dict__==other.__dict__
        return False

    def __ne__(self,other):
        if isinstance(other,self.__class__):
            return self.__dict__!=other.__dict__
        return True
