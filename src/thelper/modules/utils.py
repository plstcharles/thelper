import logging
from abc import ABC,abstractmethod

import numpy as np
import torch
import torch.nn

import thelper
import thelper.utils
import thelper.tasks
import thelper.modules

logger = logging.getLogger(__name__)


class Module(torch.nn.Module,ABC):
    def __init__(self,task,name=None):
        super().__init__()
        self.logger = thelper.utils.get_class_logger()
        self.task = task
        self.name = name

    @abstractmethod
    def forward(self,*input):
        raise NotImplementedError

    def summary(self):
        params = filter(lambda p:p.requires_grad,self.parameters())
        count = sum([np.prod(p.size()) for p in params])
        self.logger.info("module '%s' parameter count: %d"%(self.name,count))
        self.logger.info(self)


class ExternalModule(Module):
    def __init__(self,model_type,task,name=None,config=None):
        super().__init__(task,name=name)
        self.logger.info("instantiating external module '%s'..."%name)
        self.model_type = model_type
        self.task = task
        self.model = model_type(**config)
        if not hasattr(self.model,"forward"):
            raise AssertionError("external module must implement 'forward' method")

    def forward(self,*input):
        return self.model(*input)


class ExternalClassifModule(ExternalModule):
    def __init__(self,model_type,task,name=None,module_config=None):
        super().__init__(model_type,task,name=name,config=module_config)
        if type(task)!=thelper.tasks.Classification:
            raise AssertionError("task passed to ExternalClassifModule should be 'thelper.tasks.Classification'")
        self.nb_classes = self.task.get_nb_classes()
        if hasattr(self.model,"fc"):
            self.logger.info("reconnecting fc layer for outputting %d classes..."%self.nb_classes)
            nb_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(nb_features,self.nb_classes)
        else:
            raise AssertionError("could not reconnect fully connected layer for new classes")
