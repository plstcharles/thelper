import logging
import inspect
from abc import abstractmethod

import numpy as np
import torch.nn

import thelper.utils

logger = logging.getLogger(__name__)


def load_model(config):
    if "type" not in config or not config["type"]:
        raise AssertionError("model config missing 'type' field")
    type = thelper.utils.import_class(config["type"])
    if "params" not in config:
        raise AssertionError("model config missing 'params' field")
    params = thelper.utils.keyvals2dict(config["params"])
    name = config["name"] if "name" in config else None
    model = None
    if inspect.isclass(type) and issubclass(type,thelper.modules.Module):
        model = type(name=name,config=params)
    else:
        if "classif_params" in config:
            classif_params = thelper.utils.keyvals2dict(config["classif_params"])
            model = thelper.modules.ExternalClassifModule(type,classif_params,name=name,module_config=params)
        else:
            model = thelper.modules.ExternalModule(type,name=name,config=params)
    return model


class Module(torch.nn.Module):
    def __init__(self,name=None,config=None):
        super().__init__()
        self.logger = thelper.utils.get_class_logger()
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
    def __init__(self,type,name=None,config=None):
        super().__init__(name=name,config=config)
        self.logger.info("instantiating external module '%s'..."%name)
        self.type = type
        self.model = type(**config)
        if not hasattr(self.model,"forward"):
            raise AssertionError("external module must implement 'forward' method")

    def forward(self,*input):
        return self.model(*input)


class ExternalClassifModule(ExternalModule):
    def __init__(self,type,classif_config,name=None,module_config=None):
        super().__init__(type,name=name,config=module_config)
        if "nb_classes" not in classif_config or not classif_config["nb_classes"]:
            raise AssertionError("missing 'nb_classes' field in classif module config")
        self.nb_classes = classif_config["nb_classes"]
        if hasattr(self.model,"fc"):
            nb_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(nb_features,self.nb_classes)
        else:
            raise AssertionError("could not reconnect fully connected layer for new classes")
