"""Optimization/metrics utility module.

This module contains utility functions and tools used by the other modules of this package.
"""
import copy
import inspect
import logging

import torch

import thelper.optim.metrics
import thelper.utils

logger = logging.getLogger(__name__)


def create_loss_fn(config, model, loader=None, uploader=None):
    """Instantiates and returns the loss function to use for training.

    The default way to specify the loss function to use is to provide a callable type to instantiate
    as well as its initialization parameters. For example::

        # ...
        "loss": {
            # if we want to use PyTorch's cross entropy loss:
            #  >>> loss_fn = torch.nn.CrossEntropyLoss(**params)
            #  >>> ...
            #  >>> loss = loss_fn(pred, target)
            "type": "torch.nn.CrossEntropyLoss"
            "params": {
                "weight": [ ... ],
                "reduction": "mean",
                # ...
            }
        },
        # ...

    The loss function can also be queried from a member function of the model class, as such::

        # ...
        "loss": {
            # to query the model for the loss function:
            #  >>> loss_fn = model.get_loss_fn(**params)
            #  >>> ...
            #  >>> loss = loss_fn(pred, target)
            "model_getter": "get_loss_fn"
            "params": {
                # ...
            }
        },
        # ...

    If the model is supposed to compute its own loss, we suggest creating a specialized trainer class. In that
    case only, the 'loss' field can be omitted from the session configuration file.

    If the task is related to image classification or semantic segmentation, the classes can be weighted based
    on extra parameters in the loss configuration. The strategy used to compute the weights is related to the
    one in :class:`thelper.data.samplers.WeightedSubsetRandomSampler`. The exact parameters that are expected
    for class reweighting are the following:

    - ``weight_distribution`` (mandatory, toggle): the dictionary of weights assigned to each class, or the
      rebalancing strategy to use. If omitted entirely, no class weighting will be performed.
    - ``weight_param_name`` (optional, default="weight"): name of the constructor parameter that expects the
      weight list.
    - ``weight_max`` (optional, default=inf): the maximum weight that can be assigned to a class.
    - ``weight_min`` (optional, default=0): the minimum weight that can be assigned to a class.
    - ``weight_norm`` (optional, default=True): specifies whether the weights should be normalized or not.

    This function also supports an extra special parameter if the task is related to semantic segmentation:
    ``ignore_index``. If this parameter is found and not ``None`` (integer), then the loss function will ignore
    the given value when computing the loss of a sample. The exact parameters that are expected in this case are
    the following:

    - ``ignore_index_param_name`` (optional, default="ignore_index"): name of the constructor parameter that expects
      the ignore value.
    - ``ignore_index_label_name`` (optional, default="dontcare"): name of the label to pass the ignore value from.

    """
    # todo: add flag to toggle loss comp in validation? (add to trainer config maybe?)
    logger.debug("loading loss function")
    if isinstance(model, torch.nn.DataParallel):
        model = model.module  # to avoid interface getter issues
    if uploader is None:
        def uploader(x):
            return x
    if not isinstance(config, dict):
        raise AssertionError("config should be provided as a dictionary")
    if "model_getter" in config and "type" in config:
        raise AssertionError("loss config cannot have both 'model_attrib_name' and 'type' fields")
    if "model_getter" in config:
        model_getter_name = config["model_getter"]
        if not isinstance(model_getter_name, str):
            raise AssertionError("unexpected model getter name type")
        if not hasattr(model, model_getter_name) or not callable(getattr(model, model_getter_name)):
            raise AssertionError("invalid model getter attribute")
        loss_type = getattr(model, model_getter_name)
    elif "type" in config:
        loss_type = thelper.utils.import_class(config["type"])
    else:
        raise AssertionError("loss config missing 'type' or 'model_attrib_name' field")
    loss_params = thelper.utils.get_key_def(["params", "parameters"], config, {})
    loss_params = copy.deepcopy(loss_params)  # required here, we might add some parameters below
    if thelper.utils.str2bool(thelper.utils.get_key_def("weight_classes", config, False)) or \
       thelper.utils.get_key_def("weight_distribution", config, None) is not None:
        if not thelper.utils.str2bool(thelper.utils.get_key_def("weight_classes", config, True)):
            raise AssertionError("'weight_classes' now deprecated, set 'weight_distribution' directly to toggle on")
        if not isinstance(model.task, (thelper.tasks.Classification, thelper.tasks.Segmentation)):
            raise AssertionError("task type does not support class weighting")
        weight_param_name = "weight"
        if "weight_param_name" in config:
            weight_param_name = config["weight_param_name"]
        if "weight_distribution" not in config:
            raise AssertionError("missing 'weight_distribution' field in loss config")
        weight_distrib = config["weight_distribution"]
        if isinstance(weight_distrib, dict):
            for label, weight in weight_distrib.items():
                if label not in model.task.class_names:
                    raise AssertionError("weight distribution label '%s' not in dataset class list" % label)
                if not isinstance(weight, float):
                    raise AssertionError("expected weight distrib map to provide weights as floats directly")
        elif isinstance(weight_distrib, str):
            weight_max = float("inf")
            if "weight_max" in config:
                weight_max = float(config["weight_max"])
            weight_min = 0
            if "weight_min" in config:
                weight_min = float(config["weight_min"])
            if weight_distrib != "uniform":
                if loader is None or not loader:
                    raise AssertionError("cannot get class sizes, no training data available")
                label_sizes_map = model.task.get_class_sizes(loader.dataset)
                weight_norm = False
            else:
                label_sizes_map = {label: -1 for label in model.task.class_names}  # counts don't matter
                weight_norm = True
            if "weight_norm" in config:
                weight_norm = thelper.utils.str2bool(config["weight_norm"])
            weight_distrib = thelper.data.utils.get_class_weights(label_sizes_map, weight_distrib, invmax=True,
                                                                  maxw=weight_max, minw=weight_min, norm=weight_norm)
        else:
            raise AssertionError("unexpected weight distribution strategy (should be map or string)")
        weight_list_str = "weight_distribution: {"
        for label, weight in weight_distrib.items():
            weight_list_str += "\n  \"%s\": %s," % (label, weight)
        logger.info(weight_list_str + "\n}")
        weight_list = [weight_distrib[label] if label in weight_distrib else 1.0 for label in model.task.class_names]
        loss_params[weight_param_name] = uploader(torch.FloatTensor(weight_list))
    if isinstance(model.task, thelper.tasks.Segmentation):
        ignore_index_param_name = thelper.utils.get_key_def("ignore_index_param_name", config, "ignore_index")
        ignore_index_label_name = thelper.utils.get_key_def("ignore_index_label_name", config, "dontcare")
        loss_sig = inspect.signature(loss_type)
        if ignore_index_param_name in loss_sig.parameters:
            if ignore_index_label_name != "dontcare":
                loss_params[ignore_index_param_name] = model.task.class_indices[ignore_index_label_name]
            else:
                loss_params[ignore_index_param_name] = model.task.dontcare
    loss = loss_type(**loss_params)
    return loss


def create_metrics(config):
    """Instantiates and returns the metrics defined in the configuration dictionary.

    All arguments are expected to be handed in through the configuration via a dictionary named 'params'.
    """
    if not isinstance(config, dict):
        raise AssertionError("config should be provided as a dictionary")
    metrics = {}
    for name, metric_config in config.items():
        if not isinstance(metric_config, dict):
            raise AssertionError("metric config should be provided as a dictionary")
        if "type" not in metric_config or not metric_config["type"]:
            raise AssertionError("metric config missing 'type' field")
        metric_type = thelper.utils.import_class(metric_config["type"])
        metric_params = thelper.utils.get_key_def(["params", "parameters"], metric_config, {})
        metric = metric_type(**metric_params)
        if not isinstance(metric, thelper.optim.metrics.Metric):
            raise AssertionError("invalid metric type, must derive from 'thelper.optim.metrics.Metric'")
        goal = getattr(metric, "goal", None)
        if not callable(goal):
            raise AssertionError("expected metric to define 'goal' based on parent interface")
        metrics[name] = metric
    return metrics


def create_optimizer(config, model):
    """Instantiates and returns the optimizer to use for training.

    By default, the optimizer will be instantiated with the model parameters given as the first argument
    of its constructor. All supplementary arguments are expected to be handed in through the configuration
    via a dictionary named 'params'.
    """
    logger.debug("loading optimizer")
    if not isinstance(config, dict):
        raise AssertionError("config should be provided as a dictionary")
    if "type" not in config or not config["type"]:
        raise AssertionError("optimizer config missing 'type' field")
    optimizer_type = thelper.utils.import_class(config["type"])
    optimizer_params = thelper.utils.get_key_def(["params", "parameters"], config, {})
    optimizer = optimizer_type(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)
    return optimizer


def create_scheduler(config, optimizer):
    """Instantiates and returns the learning rate scheduler to use for training.

    All arguments are expected to be handed in through the configuration via a dictionary named 'params'.
    """
    logger.debug("loading scheduler")
    if not isinstance(config, dict):
        raise AssertionError("config should be provided as a dictionary")
    if "type" not in config or not config["type"]:
        raise AssertionError("scheduler config missing 'type' field")
    scheduler_type = thelper.utils.import_class(config["type"])
    scheduler_params = thelper.utils.get_key_def(["params", "parameters"], config, {})
    scheduler = scheduler_type(optimizer, **scheduler_params)
    scheduler_step_metric = None
    if "step_metric" in config:
        scheduler_step_metric = config["step_metric"]
    return scheduler, scheduler_step_metric


def get_lr(optimizer):
    """Returns the optimizer's learning rate, or 0 if not found."""
    for param_group in optimizer.param_groups:
        if "lr" in param_group:
            return param_group["lr"]
    return 0
