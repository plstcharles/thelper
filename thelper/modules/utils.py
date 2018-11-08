import logging
from abc import ABC
from abc import abstractmethod
import os

import numpy as np
import torch
import torch.nn

import thelper
import thelper.modules
import thelper.tasks
import thelper.utils

logger = logging.getLogger(__name__)


def load_model(config, task, save_dir=None):
    """Instantiates a model based on a provided task object.

    The model must be specified as a type-param pair in the provided dictionary. It should derive from
    :class:`thelper.modules.Module`, or be instantiable through :class:`thelper.modules.ExternalModule` (or
    one of its specialized classes). The provided task object will be used to make sure that the model has
    the required input/output layers for the requested objective.

    Usage examples inside a session configuration file::

        # ...
        # the content of this field will be passed in this function as 'config'
        "model": {
            # the type provides the class name to instantiate an object from
            "type": "thelper.modules.mobilenet.MobileNetV2",
            # the parameters listed below are passed to the model's constructor
            "params": [
                # ...
            ]
        # ...

    Args:
        config: a dictionary that provides the model type and its constructor's parameters.
        task: a task object that will be passed to the model's constructor in order to specialize it.
        save_dir: if not ``None``, a log file containing model information will be created there.

    Returns:
        The instantiated model, compatible with the interface of both :class:`thelper.modules.Module`
        and ``torch.nn.Module``.

    .. seealso::
        :class:`thelper.modules.Module`
        :class:`thelper.modules.ExternalModule`
        :class:`thelper.tasks.Task`
    """
    if not issubclass(type(task), thelper.tasks.Task):
        raise AssertionError("bad task type passed to load_model")
    if save_dir is not None:
        modules_logger_path = os.path.join(save_dir, "logs", "modules.log")
        modules_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        modules_logger_fh = logging.FileHandler(modules_logger_path)
        modules_logger_fh.setFormatter(modules_logger_format)
        thelper.modules.logger.addHandler(modules_logger_fh)
        thelper.modules.logger.info("created modules log for session '%s'" % config["name"])
    thelper.modules.logger.debug("loading model")
    if "model" not in config or not config["model"]:
        raise AssertionError("config missing 'model' field")
    model_config = config["model"]
    if "type" not in model_config or not model_config["type"]:
        raise AssertionError("model config missing 'type' field")
    model_type = thelper.utils.import_class(model_config["type"])
    if "params" not in model_config:
        raise AssertionError("model config missing 'params' field")
    params = thelper.utils.keyvals2dict(model_config["params"])
    if issubclass(model_type, Module):
        model = model_type(task=task, **params)
    else:
        if type(task) == thelper.tasks.Classification:
            model = thelper.modules.utils.ExternalClassifModule(model_type, task=task, module_config=params)
        else:
            model = thelper.modules.utils.ExternalModule(model_type, task=task, config=params)
    if hasattr(model, "summary"):
        model.summary()
    return model


class Module(torch.nn.Module, ABC):
    """Model inteface used to hold a task object.

    This interface is built on top of ``torch.nn.Module`` and should remain fully compatible with it.

    All models used in the framework should derive from this interface, and therefore expect a task object
    as the first argument of their constructor. Their implementation may decide to ignore this object
    when building their internal layers, but using it should help specialize the network by specifying e.g.
    the number of classes to support.

    .. seealso::
        :func:`thelper.modules.load_model`
        :class:`thelper.tasks.Task`
    """

    def __init__(self, task):
        """Receives a task object to hold internally for model specialization."""
        super().__init__()
        if task is None or not isinstance(task, thelper.tasks.Task):
            raise AssertionError("task must derive from thelper.tasks.Task")
        self.task = task

    def _get_derived_name(self):
        return str(self.__class__.__qualname__)

    @abstractmethod
    def forward(self, *input):
        """Transforms an input tensor in order to generate a prediction."""
        raise NotImplementedError

    def summary(self):
        """Prints a summary of the model using the ``thelper.modules`` logger."""
        params = filter(lambda p: p.requires_grad, self.parameters())
        count = sum([np.prod(p.size()) for p in params])
        logger.info("module '%s' parameter count: %d" % (self._get_derived_name(), count))
        logger.info(self)


class ExternalModule(Module):
    """Model inteface used to hold a task object for an external implementation.

    This interface is built on top of ``torch.nn.Module`` and should remain fully compatible with it. It is
    automatically used when instantiating a model via :func:`thelper.modules.load_model` that is not derived
    from :class:`thelper.modules.Module`. Its only purpose is to hold the task object, and redirect
    :func:`thelper.modules.Module.forward: to the actual model's transformation function. It can also be
    specialized to automatically adapt some external models after their construction using the knowledge
    contained in the task object.

    .. seealso::
        :class:`thelper.modules.Module`
        :class:`thelper.modules.ExternalClassifModule`
        :func:`thelper.modules.load_model`
        :class:`thelper.tasks.Task`
    """

    def __init__(self, model_type, task, config=None):
        """Receives a task object to hold internally for model specialization."""
        super().__init__(task)
        logger.info("instantiating external module '%s'..." % str(model_type))
        self.model_type = model_type
        self.task = task
        self.model = model_type(**config)
        if not hasattr(self.model, "forward"):
            raise AssertionError("external module must implement 'forward' method")

    def _get_derived_name(self):
        return thelper.utils.get_caller_name(0).rsplit(".", 1)[0]

    def forward(self, *input):
        """Transforms an input tensor in order to generate a prediction."""
        return self.model(*input)

    def summary(self):
        """Prints a summary of the model using the ``thelper.modules`` logger."""
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        count = sum([np.prod(p.size()) for p in params])
        logger.info("module '%s' parameter count: %d" % (self._get_derived_name(), count))
        logger.info(self.model)


class ExternalClassifModule(ExternalModule):
    """External model interface specialization for classification tasks.

    This interface will try to 'rewire' the last fully connected layer of the models it instantiates to match
    the number of classes to predict defined in the task object.

    .. seealso::
        :class:`thelper.modules.Module`
        :class:`thelper.modules.ExternalModule`
        :func:`thelper.modules.load_model`
        :class:`thelper.tasks.Task`
    """

    def __init__(self, model_type, task, module_config=None):
        """Receives a task object to hold internally for model specialization, and tries to rewire the last 'fc' layer."""
        super().__init__(model_type, task, config=module_config)
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
