import logging
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
import torch.nn

import thelper
import thelper.modules
import thelper.tasks
import thelper.utils

logger = logging.getLogger(__name__)


class Module(torch.nn.Module, ABC):

    def __init__(self, task, name=None):
        super().__init__()
        if task is None or not isinstance(task, thelper.tasks.Task):
            raise AssertionError("task must derive from thelper.tasks.Task")
        self.task = task
        self.name = name

    def _get_derived_name(self):
        dname = str(self.__class__.__qualname__)
        if self.name:
            dname += "." + self.name
        return dname

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    def summary(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        count = sum([np.prod(p.size()) for p in params])
        logger.info("module '%s' parameter count: %d" % (self.name, count))
        logger.info(self)


class ExternalModule(Module):

    def __init__(self, model_type, task, name=None, config=None):
        super().__init__(task, name=name)
        logger.info("instantiating external module '%s'..." % name)
        self.model_type = model_type
        self.task = task
        self.model = model_type(**config)
        if not hasattr(self.model, "forward"):
            raise AssertionError("external module must implement 'forward' method")

    def _get_derived_name(self):
        dname = thelper.utils.get_caller_name(0).rsplit(".", 1)[0]
        if self.name:
            dname += "." + self.name
        return dname

    def forward(self, *input):
        return self.model(*input)


class ExternalClassifModule(ExternalModule):

    def __init__(self, model_type, task, name=None, module_config=None):
        super().__init__(model_type, task, name=name, config=module_config)
        if type(task) != thelper.tasks.Classification:
            raise AssertionError("task passed to ExternalClassifModule should be 'thelper.tasks.Classification'")
        self.nb_classes = self.task.get_nb_classes()
        if hasattr(self.model, "fc"):
            logger.info("reconnecting fc layer for outputting %d classes..." % self.nb_classes)
            nb_features = self.model.fc.in_features
            self.model.fc = torch.nn.Linear(nb_features, self.nb_classes)
        elif hasattr(self.model, "classifier"):
            logger.info("reconnecting classifier layer for outputting %d classes..." % self.nb_classes)
            nb_features = self.model.classifier.in_features
            self.model.classifier = torch.nn.Linear(nb_features, self.nb_classes)
        else:
            raise AssertionError("could not reconnect fully connected layer for new classes")
