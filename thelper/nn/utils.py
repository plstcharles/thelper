"""Neural network utility functions and classes.

This module contains base interfaces and utility functions used to define and instantiate
neural network models.
"""

import inspect
import logging
import os
from abc import abstractmethod

import numpy as np
import torch
import torch.nn

import thelper
import thelper.nn
import thelper.tasks
import thelper.utils

logger = logging.getLogger(__name__)


def create_model(config, task, save_dir=None, ckptdata=None):
    """Instantiates a model based on a provided task object.

    The configuration must be given as a dictionary object. This dictionary will be parsed for a 'model' field.
    This field is expected to be a dictionary itself. It may then specify a type to instantiate as well as the
    parameters to provide to that class constructor, or a path to a checkpoint from which a model should be loaded.

    All models must derive from :class:`thelper.nn.utils.Module`, or they must be instantiable through
    :class:`thelper.nn.utils.ExternalModule` (or one of its specialized classes). The provided task object will
    be used to make sure that the model has the required input/output layers for the requested objective.

    If checkpoint data is provided by the caller, the weights it contains will be loaded into the returned model.

    Usage examples inside a session configuration file::

        # ...
        # the function will look for a 'model' field in the provided config dict
        "model": {
            # the type provides the class name to instantiate an object from
            "type": "thelper.nn.mobilenet.MobileNetV2",
            # the parameters listed below are passed to the model's constructor
            "params": [
                # ...
            ]
        # ...

    Args:
        config: a session dictionary that provides a 'model' field containing a dictionary.
        task: a task object that will be passed to the model's constructor in order to specialize it. Can be
            ``None`` if a checkpoint is provided, and if the previous task is wanted instead of a new one.
        save_dir: if not ``None``, a log file containing model information will be created there.
        ckptdata: raw checkpoint data loaded via ``torch.load()``; the model will be given its previous state.

    Returns:
        The instantiated model, compatible with the interface of both :class:`thelper.nn.utils.Module`
        and ``torch.nn.Module``.

    .. seealso::
        | :class:`thelper.nn.utils.Module`
        | :class:`thelper.nn.utils.ExternalModule`
        | :class:`thelper.tasks.utils.Task`
        | :func:`thelper.utils.load_checkpoint`
    """
    if save_dir is not None:
        modules_logger_path = os.path.join(save_dir, "logs", "modules.log")
        modules_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        modules_logger_fh = logging.FileHandler(modules_logger_path)
        modules_logger_fh.setFormatter(modules_logger_format)
        logger.addHandler(modules_logger_fh)
    logger.debug("loading model")
    model_config = None
    if config is not None:
        model_config = config["model"] if "model" in config else config
    if model_config is not None and "ckptdata" in model_config:
        if ckptdata is not None:
            logger.warning("config asked to reload ckpt from path, but ckpt already loaded from elsewhere")
        else:
            logger.debug("model config asked for an older model to be loaded through a checkpoint")
            if not isinstance(model_config["ckptdata"], str):
                raise AssertionError("unexpected model config ckptdata field type (should be path)")
            map_location = thelper.utils.get_key_def("map_location", model_config, "cpu")
            ckptdata = thelper.utils.load_checkpoint(model_config["ckptdata"], map_location=map_location)
        if "type" in model_config or "params" in model_config:
            logger.warning("should not provide 'type' or 'params' fields in model config if loading a checkpoint")
    new_task, model, model_type, model_params, model_state = None, None, None, None, None
    if ckptdata is not None:
        # if checkpoint available, instantiate old model, load weights, and reconfigure for new task
        if "name" not in ckptdata or not isinstance(ckptdata["name"], str):
            raise AssertionError("invalid checkpoint, cannot reload previous session name")
        new_task = task  # the 'new task' will later be applied to specialize the model, once it is loaded
        supported_model_types = (torch.nn.Module, thelper.nn.Module, torch.jit.ScriptModule, dict)
        if "model" not in ckptdata or not isinstance(ckptdata["model"], supported_model_types):
            raise AssertionError("invalid checkpoint, cannot reload previous model state")
        if isinstance(ckptdata["model"], torch.jit.ScriptModule):
            logger.debug("loading model trace from session '%s'" % ckptdata["name"])
            model = ckptdata["model"]
        elif isinstance(ckptdata["model"], (torch.nn.Module, thelper.nn.Module)):
            logger.debug("loading model object directly from session '%s'" % ckptdata["name"])
            model = ckptdata["model"]
            if hasattr(model, "get_name") and model.get_name() != ckptdata["model_type"]:
                raise AssertionError("old model type mismatch with ckptdata type")
        elif isinstance(ckptdata["model"], dict):
            logger.debug("loading model type/params from session '%s'" % ckptdata["name"])
            model_state = ckptdata["model"]
            if "task" not in ckptdata or not isinstance(ckptdata["task"], (thelper.tasks.Task, dict, str)):
                raise AssertionError("invalid checkpoint, cannot reload previous model task")
            task = thelper.tasks.create_task(ckptdata["task"]) if isinstance(ckptdata["task"], (dict, str)) else ckptdata["task"]
            if "model_type" not in ckptdata or not isinstance(ckptdata["model_type"], str):
                raise AssertionError("invalid checkpoint, cannot reload previous model type")
            model_type = ckptdata["model_type"]
            if isinstance(model_type, str):
                model_type = thelper.utils.import_class(model_type)
            if "model_params" not in ckptdata or not isinstance(ckptdata["model_params"], dict):
                raise AssertionError("invalid checkpoint, cannot reload previous model params")
            model_params = ckptdata["model_params"]
            if "config" not in ckptdata or not isinstance(ckptdata["config"], dict):
                raise AssertionError("invalid checkpoint, cannot reload previous session config")
            old_config = ckptdata["config"]
            if "model" not in old_config or not isinstance(old_config["model"], dict):
                raise AssertionError("invalid checkpoint, cannot reload previous model config")
            old_model_config = old_config["model"]
            if "type" in old_model_config:
                old_model_type = old_model_config["type"]
                if isinstance(old_model_type, str):
                    old_model_type = thelper.utils.import_class(old_model_type)
                if old_model_type != model_type:
                    raise AssertionError("old model config 'type' field mismatch with ckptdata type")
    else:
        if model_config is None:
            raise AssertionError("must provide model config and/or checkpoint data to create a model")
        if "task" in model_config and isinstance(model_config["task"], (thelper.tasks.Task, dict, str)):
            new_task = thelper.tasks.create_task(model_config["task"]) \
                if isinstance(model_config["task"], (dict, str)) else model_config["task"]
        if not isinstance(task, thelper.tasks.Task) and not isinstance(new_task, thelper.tasks.Task):
            raise AssertionError("bad task type passed to create_model")
        logger.debug("loading model type/params current config")
        if "type" not in model_config or not model_config["type"]:
            raise AssertionError("model config missing 'type' field")
        model_type = model_config["type"]
        if isinstance(model_config["type"], str):
            model_type = thelper.utils.import_class(model_type)
        model_params = thelper.utils.get_key_def("params", model_config, {})
        model_state = thelper.utils.get_key_def(["weights", "state", "state_dict"], model_config, None)
        if model_state is not None and isinstance(model_state, str) and os.path.isfile(model_state):
            model_state = torch.load(model_state)
    if model is None:
        # if model not already loaded from checkpoint, instantiate it fully from type/params/task
        if model_type is None or model_params is None:
            raise AssertionError("messed up logic above")
        logger.debug("model_type = %s" % str(model_type))
        logger.debug("model_params = %s" % str(model_params))
        if inspect.isclass(model_type) and issubclass(model_type, thelper.nn.utils.Module):
            model = model_type(task=task, **model_params)
        else:
            if type(task) == thelper.tasks.Detection:
                model = ExternalDetectModule(model_type, task=task, **model_params)
            elif type(task) == thelper.tasks.Classification:
                model = ExternalClassifModule(model_type, task=task, **model_params)
            else:
                model = ExternalModule(model_type, task=task, **model_params)
        if model_state is not None:
            logger.debug("loading state dictionary from checkpoint into model")
            model.load_state_dict(model_state)
    if new_task is not None:
        if hasattr(model, "task"):
            logger.debug("previous model task = %s" % str(model.task))
        if hasattr(model, "set_task"):
            logger.debug("refreshing model for new task = %s" % str(new_task))
            model.set_task(new_task)
        else:
            logger.warning("model missing 'set_task' interface function")
    if hasattr(model, "config") and model.config is None:
        model.config = model_params
    if hasattr(model, "summary"):
        model.summary()
    return model


class Module(torch.nn.Module):
    """Model interface used to hold a task object.

    This interface is built on top of ``torch.nn.Module`` and should remain fully compatible with it.

    All models used in the framework should derive from this interface, and therefore expect a task object as
    the first argument of their constructor. Their implementation may decide to ignore this task object when
    building their internal layers, but using it should help specialize the network by specifying e.g. the
    number of classes to support.

    .. seealso::
        | :func:`thelper.nn.utils.create_model`
        | :class:`thelper.tasks.utils.Task`
    """

    def __init__(self, task, **kwargs):
        """Receives a task object to hold internally for model specialization."""
        super().__init__()
        if task is None or not isinstance(task, thelper.tasks.Task):
            raise AssertionError("task must derive from thelper.tasks.Task")
        self.task = task
        self.config = kwargs

    @abstractmethod
    def forward(self, *input):
        """Transforms an input tensor in order to generate a prediction."""
        raise NotImplementedError

    @abstractmethod
    def set_task(self, task):
        """Adapts the model to support a new task, replacing layers if needed."""
        raise NotImplementedError

    def summary(self):
        """Prints a summary of the model using the ``thelper.nn`` logger."""
        params = filter(lambda p: p.requires_grad, self.parameters())
        count = sum([np.prod(p.size()) for p in params])
        logger.info("module '%s' parameter count: %d" % (self.get_name(), count))
        logger.info(self)

    def get_name(self):
        """Returns the name of this module (by default, its fully qualified class name)."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__


class ExternalModule(Module):
    """Model inteface used to hold a task object for an external implementation.

    This interface is built on top of ``torch.nn.Module`` and should remain fully compatible with it. It is
    automatically used when instantiating a model via :func:`thelper.nn.utils.create_model` that is not derived
    from :class:`thelper.nn.utils.Module`. Its only purpose is to hold the task object, and redirect
    :func:`thelper.nn.utils.Module.forward` to the actual model's transformation function. It can also be
    specialized to automatically adapt some external models after their construction using the knowledge
    contained in the task object.

    .. seealso::
        | :class:`thelper.nn.utils.Module`
        | :class:`thelper.nn.utils.ExternalClassifModule`
        | :func:`thelper.nn.utils.create_model`
        | :class:`thelper.tasks.utils.Task`
    """

    def __init__(self, model_type, task, **kwargs):
        """Receives a task object to hold internally for model specialization."""
        super().__init__(task=task, **kwargs)
        logger.info("instantiating external module '%s'..." % str(model_type))
        self.model_type = model_type
        self.model = model_type(**kwargs)
        if not hasattr(self.model, "forward"):
            raise AssertionError("external module must implement 'forward' method")

    def load_state_dict(self, state_dict, strict=True):
        """Loads the state dict of an external model."""
        self.model.load_state_dict(state_dict=state_dict, strict=strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Returns the state dict of the external model."""
        return self.model.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def forward(self, *input, **kwargs):
        """Transforms an input tensor in order to generate a prediction."""
        return self.model(*input, **kwargs)

    def set_task(self, task):
        """Stores the new task internally.

        Note that since this external module handler is generic, it does not know what to do with the task,
        so it just assumes that the model is already set up. Specialized external module handlers will instead
        attempt to modify the model they wrap.
        """
        if task is None or not isinstance(task, thelper.tasks.Task):
            raise AssertionError("task must derive from thelper.tasks.Task")
        self.task = task

    def summary(self):
        """Prints a summary of the model using the ``thelper.nn`` logger."""
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        count = sum([np.prod(p.size()) for p in params])
        logger.info("module '%s' parameter count: %d" % (self.get_name(), count))
        logger.info(self.model)

    def get_name(self):
        """Returns the name of this module (by default, the fully qualified class name of the external model)."""
        return self.model_type.__module__ + "." + self.model_type.__qualname__


class ExternalClassifModule(ExternalModule):
    """External model interface specialization for classification tasks.

    This interface will try to 'rewire' the last fully connected layer of the models it instantiates to match
    the number of classes to predict defined in the task object.

    .. seealso::
        | :class:`thelper.nn.utils.Module`
        | :class:`thelper.nn.utils.ExternalModule`
        | :func:`thelper.nn.utils.create_model`
        | :class:`thelper.tasks.classif.Classification`
    """

    def __init__(self, model_type, task, **kwargs):
        """Receives a task object to hold internally for model specialization, and tries to rewire the last 'fc' layer."""
        super().__init__(model_type, task, **kwargs)
        self.nb_classes = None
        self.set_task(task)

    def set_task(self, task):
        """Rewires the last fully connected layer of the wrapped network to fit the given number of classification targets."""
        if type(task) != thelper.tasks.Classification:
            raise AssertionError("task passed to ExternalClassifModule should be 'thelper.tasks.Classification'")
        self.nb_classes = len(self.task.class_names)
        import torchvision
        if hasattr(self.model, "fc") and isinstance(self.model.fc, torch.nn.Linear):
            if self.model.fc.out_features != self.nb_classes:
                logger.info("reconnecting fc layer for outputting %d classes..." % self.nb_classes)
                nb_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(nb_features, self.nb_classes)
        elif hasattr(self.model, "classifier") and isinstance(self.model.classifier, torch.nn.Linear):
            if self.model.classifier.out_features != self.nb_classes:
                logger.info("reconnecting classifier layer for outputting %d classes..." % self.nb_classes)
                nb_features = self.model.classifier.in_features
                self.model.classifier = torch.nn.Linear(nb_features, self.nb_classes)
        elif isinstance(self.model, torchvision.models.squeezenet.SqueezeNet):
            if self.model.num_classes != self.nb_classes:
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(p=0.5),
                    torch.nn.Conv2d(512, self.nb_classes, kernel_size=1),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.AvgPool2d(13, stride=1)
                )
                self.model.num_classes = self.nb_classes
        else:
            logger.warning("could not reconnect fully connected layer for new classes; hope your model is already compatible...")


class ExternalDetectModule(ExternalModule):
    """External model interface specialization for object detection tasks.

    This interface will try to 'rewire' the last fully connected layer of the models it instantiates to match
    the number of classes to predict defined in the task object.

    .. seealso::
        | :class:`thelper.nn.utils.Module`
        | :class:`thelper.nn.utils.ExternalModule`
        | :func:`thelper.nn.utils.create_model`
        | :class:`thelper.tasks.detect.Detection`
    """

    def __init__(self, model_type, task, **kwargs):
        """Receives a task object to hold internally for model specialization, and tries to rewire the last 'fc' layer."""
        super().__init__(model_type, task, **kwargs)
        self.nb_classes = None
        self.set_task(task)

    def set_task(self, task):
        """Rewires the last fully connected layer of the wrapped network to fit the given number of classification targets."""
        if type(task) != thelper.tasks.Detection:
            raise AssertionError("task passed to ExternalClassifModule should be 'thelper.tasks.Detection'")
        self.nb_classes = len(self.task.class_names)
        import torchvision
        if hasattr(self.model, "fc") and isinstance(self.model.fc, torch.nn.Linear):
            if self.model.fc.out_features != self.nb_classes:
                logger.info("reconnecting fc layer for outputting %d classes..." % self.nb_classes)
                nb_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(nb_features, self.nb_classes)
        elif hasattr(self.model, "classifier") and isinstance(self.model.classifier, torch.nn.Linear):
            if self.model.classifier.out_features != self.nb_classes:
                logger.info("reconnecting classifier layer for outputting %d classes..." % self.nb_classes)
                nb_features = self.model.classifier.in_features
                self.model.classifier = torch.nn.Linear(nb_features, self.nb_classes)
        elif hasattr(self.model, "roi_heads") and hasattr(self.model.roi_heads, "box_predictor"):
            # note: this part requires torchvision version >= 0.3
            if isinstance(self.model.roi_heads.box_predictor, torchvision.models.detection.faster_rcnn.FastRCNNPredictor):
                if self.model.roi_heads.box_predictor.cls_score.out_features != self.nb_classes:
                    self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                        self.model.roi_heads.box_predictor.cls_score.in_features, self.nb_classes)
            else:
                logger.warning("unexpected box predictor type (missing impl)")  # @@@@@@ TODO
        else:
            logger.warning("could not reconnect fully connected layer for new classes; hope your model is already compatible...")
