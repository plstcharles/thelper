import inspect
import logging
from abc import ABC,abstractmethod

import numpy as np
import torch.nn

import thelper.utils
import thelper.tasks

logger = logging.getLogger(__name__)


def load_model(config,task):
    if not issubclass(type(task),thelper.tasks.Task):
        raise AssertionError("bad task type passed to load_model")
    if "type" not in config or not config["type"]:
        raise AssertionError("model config missing 'type' field")
    model_type = thelper.utils.import_class(config["type"])
    if "params" not in config:
        raise AssertionError("model config missing 'params' field")
    params = thelper.utils.keyvals2dict(config["params"])
    name = config["name"] if "name" in config else None
    model = None
    if issubclass(model_type,thelper.modules.Module):
        model = model_type(name=name,config=params)
    else:
        if type(task)==thelper.tasks.Classification:
            model = thelper.modules.ExternalClassifModule(model_type,task,name=name,module_config=params)
        else:
            model = thelper.modules.ExternalModule(model_type,task,name=name,config=params)
    return model


class Module(torch.nn.Module,ABC):
    def __init__(self,task,name=None,config=None):
        super().__init__()
        self.logger = thelper.utils.get_class_logger()
        self.task = task
        self.name = name
        self.config = config

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
        super().__init__(task,name=name,config=config)
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
