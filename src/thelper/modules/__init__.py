import logging

import thelper.modules.utils
from thelper.modules.utils import Module

logger = logging.getLogger("thelper.modules")


def load_model(config, task, save_dir=None):
    if not issubclass(type(task), thelper.tasks.Task):
        raise AssertionError("bad task type passed to load_model")
    if save_dir is not None:
        modules_logger_path = os.path.join(save_dir, "logs", "modules.log")
        modules_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        modules_logger_fh = logging.FileHandler(modules_logger_path)
        modules_logger_fh.setFormatter(modules_logger_format)
        thelper.modules.logger.addHandler(modules_logger_fh)
        thelper.modules.logger.info("created modules log for session '%s'" % config["session_name"])
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
    name = model_config["name"] if "name" in model_config else None
    if issubclass(model_type, Module):
        model = model_type(task=task, name=name, **params)
    else:
        if type(task) == thelper.tasks.Classification:
            model = thelper.modules.utils.ExternalClassifModule(model_type, task=task, name=name, module_config=params)
        else:
            model = thelper.modules.utils.ExternalModule(model_type, task=task, name=name, config=params)
    if hasattr(model, "summary"):
        model.summary()
    return model
