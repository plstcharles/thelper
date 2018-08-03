import logging

import thelper.modules.utils
from thelper.modules.utils import Module

logger = logging.getLogger("thelper.modules")


def load_model(config, task):
    if not issubclass(type(task), thelper.tasks.Task):
        raise AssertionError("bad task type passed to load_model")
    if "type" not in config or not config["type"]:
        raise AssertionError("model config missing 'type' field")
    model_type = thelper.utils.import_class(config["type"])
    if "params" not in config:
        raise AssertionError("model config missing 'params' field")
    params = thelper.utils.keyvals2dict(config["params"])
    name = config["name"] if "name" in config else None
    if issubclass(model_type, Module):
        model = model_type(task=task, name=name, **params)
    else:
        if type(task) == thelper.tasks.Classification:
            model = thelper.modules.utils.ExternalClassifModule(model_type, task=task, name=name, module_config=params)
        else:
            model = thelper.modules.utils.ExternalModule(model_type, task=task, name=name, config=params)
    return model
