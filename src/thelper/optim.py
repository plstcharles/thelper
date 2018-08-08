import logging
from collections import deque
from abc import ABC, abstractmethod

import torch
import sklearn.metrics
import pandas as pd

import thelper.utils

logger = logging.getLogger(__name__)


def load_loss(config):
    if "type" not in config or not config["type"]:
        raise AssertionError("loss config missing 'type' field")
    loss_type = thelper.utils.import_class(config["type"])
    if "params" not in config:
        raise AssertionError("loss config missing 'params' field")
    params = thelper.utils.keyvals2dict(config["params"])
    loss = loss_type(**params)
    return loss


def load_metrics(config):
    if not isinstance(config, dict):
        raise AssertionError("metrics config should be provided as dict")
    metrics = {}
    for name, metric_config in config.items():
        if "type" not in metric_config or not metric_config["type"]:
            raise AssertionError("metric config missing 'type' field")
        metric_type = thelper.utils.import_class(metric_config["type"])
        if "params" not in metric_config:
            raise AssertionError("metric config missing 'params' field")
        params = thelper.utils.keyvals2dict(metric_config["params"])
        metric = metric_type(**params)
        goal = getattr(metric, "goal", None)
        if not callable(goal):
            raise AssertionError("expected metric to define 'goal' based on parent interface")
        metrics[name] = metric
    return metrics


def load_optimization(model, config):
    if not isinstance(config, dict):
        raise AssertionError("optimization config should be provided as dict")
    if "optimizer" not in config or not config["optimizer"]:
        raise AssertionError("optimization config missing 'optimizer' field")
    optimizer_config = config["optimizer"]
    if "type" not in optimizer_config or not optimizer_config["type"]:
        raise AssertionError("optimizer config missing 'type' field")
    optimizer_type = thelper.utils.import_class(optimizer_config["type"])
    optimizer_params = thelper.utils.keyvals2dict(optimizer_config["params"]) if "params" in optimizer_config else None
    optimizer = optimizer_type(filter(lambda p: p.requires_grad, model.parameters()), **optimizer_params)
    scheduler = None
    scheduler_step = 1
    if "scheduler" in config and config["scheduler"]:
        scheduler_config = config["scheduler"]
        if "type" not in scheduler_config or not scheduler_config["type"]:
            raise AssertionError("scheduler config missing 'type' field")
        scheduler_type = thelper.utils.import_class(scheduler_config["type"])
        scheduler_params = thelper.utils.keyvals2dict(scheduler_config["params"]) if "params" in scheduler_config else None
        scheduler = scheduler_type(optimizer, **scheduler_params)
        scheduler_step = scheduler_config["step"] if "step" in scheduler_config else 1
    return optimizer, scheduler, scheduler_step


class Metric(ABC):
    # 'goal' values for optimization (minimum/maximum)
    minimize = float("-inf")
    maximize = float("inf")

    def __init__(self, name, tbx_eval_iter=False):
        if not name:
            raise AssertionError("metric name must not be empty (lookup might fail)")
        self.logger = thelper.utils.get_class_logger()
        self.name = name
        self.tbx_eval_iter = tbx_eval_iter  # defines whether the metric can be evaluated each iter by tensorboard

    @abstractmethod
    def accumulate(self, pred, gt):
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    def needs_reset(self):
        return True  # override and return false if reset not needed

    @abstractmethod
    def goal(self):
        # should return 'minimize' or 'maximize' only
        raise NotImplementedError


class CategoryAccuracy(Metric):

    def __init__(self, top_k=1, max_accum=None, tbx_eval_iter=True):
        super().__init__("CategoryAccuracy", tbx_eval_iter=tbx_eval_iter)
        self.top_k = top_k
        self.max_accum = max_accum
        self.correct = deque()
        self.total = deque()
        self.warned_eval_bad = False

    def accumulate(self, pred, gt):
        top_k = pred.topk(self.top_k, 1)[1]
        true_k = gt.view(len(gt), 1).expand_as(top_k)
        self.correct.append(top_k.eq(true_k).float().sum().item())
        self.total.append(len(pred))
        if self.max_accum and len(self.correct) > self.max_accum:
            self.correct.popleft()
            self.total.popleft()

    def eval(self):
        if len(self.total) == 0 or sum(self.total) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                self.logger.warning("metric '%s' eval result invalid (set as 0.0), no results accumulated" % self.name)
            return 0.0
        return (float(sum(self.correct)) / float(sum(self.total))) * 100

    def reset(self):
        self.correct = deque()
        self.total = deque()

    def needs_reset(self):
        return self.max_accum is None

    def goal(self):
        return Metric.maximize

    def summary(self):
        self.logger.info("metric '%s' with top_k=%d" % (self.name, self.top_k))


class BinaryAccuracy(Metric):

    def __init__(self, max_accum=None, tbx_eval_iter=True):
        super().__init__("BinaryAccuracy", tbx_eval_iter=tbx_eval_iter)
        self.max_accum = max_accum
        self.correct = deque()
        self.total = deque()
        self.warned_eval_bad = False

    def accumulate(self, pred, gt):
        pred = pred.topk(1, 1)[1].view(len(gt))
        if pred.size() != gt.size():
            raise AssertionError("pred and gt should have similar size")
        self.correct.append(pred.eq(gt).float().sum().item())
        self.total.append(len(pred))
        if self.max_accum and len(self.correct) > self.max_accum:
            self.correct.popleft()
            self.total.popleft()

    def eval(self):
        if len(self.total) == 0 or sum(self.total) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                self.logger.warning("metric '%s' eval result invalid (set as 0.0), no results accumulated" % self.name)
            return 0.0
        return (float(sum(self.correct)) / float(sum(self.total))) * 100

    def reset(self):
        self.correct = deque()
        self.total = deque()

    def needs_reset(self):
        return self.max_accum is None

    def goal(self):
        return Metric.maximize

    def summary(self):
        self.logger.info("metric '%s'" % self.name)


class ClassifReport(Metric):

    def __init__(self, class_map=None, sample_weight=None, digits=2):
        super().__init__("ClassifReport")

        def gen_report(y_true, y_pred, _class_map):
            if not _class_map:
                res = sklearn.metrics.classification_report(y_true, y_pred,
                                                            sample_weight=sample_weight,
                                                            digits=digits)
            else:
                _y_true = [_class_map[classid] for classid in y_true]
                _y_pred = [_class_map[classid] if classid in _class_map else "<unset>" for classid in y_pred]
                res = sklearn.metrics.classification_report(_y_true, _y_pred,
                                                            sample_weight=sample_weight,
                                                            digits=digits)
            return "\n" + res

        self.report = gen_report
        self.class_map = class_map
        if class_map and not isinstance(class_map, dict):
            raise AssertionError("unexpected class map type")
        self.pred = None
        self.gt = None

    def set_class_map(self, class_map):
        if class_map and not isinstance(class_map, dict):
            raise AssertionError("unexpected class map type")
        self.class_map = class_map

    def accumulate(self, pred, gt):
        if self.pred is None:
            self.pred = pred.topk(1, 1)[1].view(len(gt))
            self.gt = gt.view(len(gt)).clone()
        else:
            self.pred = torch.cat((self.pred, pred.topk(1, 1)[1].view(len(gt))), 0)
            self.gt = torch.cat((self.gt, gt.view(len(gt))), 0)

    def eval(self):
        return self.report(self.gt.numpy(), self.pred.numpy(), self.class_map)

    def reset(self):
        self.pred = None
        self.gt = None

    def goal(self):
        return None  # means this class should not be used for monitoring

    def summary(self):
        self.logger.info("classification report '%s'" % self.name)


class ConfusionMatrix(Metric):

    def __init__(self, percentage=False, class_map=None):
        super().__init__("ConfusionMatrix")

        def gen_matrix(y_true, y_pred, _class_map):
            if not _class_map:
                res = pd.crosstab(pd.Series(y_true), pd.Series(y_pred), margins=True)
            else:
                _y_true = pd.Series([thelper.utils.truncstr(_class_map[classid]) for classid in y_true])
                _y_pred = pd.Series([thelper.utils.truncstr(_class_map[classid]) if classid in _class_map else "<unset>" for classid in y_pred])
                res = pd.crosstab(_y_true, _y_pred, rownames=["True"], colnames=["Predicted"], margins=True)
            if percentage:
                return "\n" + res.apply(lambda r: 100.0 * r / r.sum()).to_string()
            return "\n" + res.to_string()

        self.matrix = gen_matrix
        self.class_map = class_map
        if class_map and not isinstance(class_map, dict):
            raise AssertionError("unexpected class map type")
        self.pred = None
        self.gt = None

    def set_class_map(self, class_map):
        if class_map and not isinstance(class_map, dict):
            raise AssertionError("unexpected class map type")
        self.class_map = class_map

    def accumulate(self, pred, gt):
        if self.pred is None:
            self.pred = pred.topk(1, 1)[1].view(len(gt))
            self.gt = gt.view(len(gt)).clone()
        else:
            self.pred = torch.cat((self.pred, pred.topk(1, 1)[1].view(len(gt))), 0)
            self.gt = torch.cat((self.gt, gt.view(len(gt))), 0)

    def eval(self):
        return self.matrix(self.gt.numpy(), self.pred.numpy(), self.class_map)

    def reset(self):
        self.pred = None
        self.gt = None

    def goal(self):
        return None  # means this class should not be used for monitoring

    def summary(self):
        self.logger.info("confusion matrix '%s'" % self.name)
