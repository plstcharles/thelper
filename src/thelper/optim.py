import logging
from abc import ABC,abstractmethod

import torch
import sklearn.metrics

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
    if not isinstance(config,dict):
        raise AssertionError("metrics config should be provided as dict")
    metrics = {}
    for name,metric_config in config.items():
        if "type" not in metric_config or not metric_config["type"]:
            raise AssertionError("metric config missing 'type' field")
        metric_type = thelper.utils.import_class(metric_config["type"])
        if "params" not in metric_config:
            raise AssertionError("metric config missing 'params' field")
        params = thelper.utils.keyvals2dict(metric_config["params"])
        metric = metric_type(**params)
        goal = getattr(metric,"goal",None)
        if not callable(goal):
            raise AssertionError("expected metric to define 'goal' based on parent interface")
        metrics[name] = metric
    return metrics


def load_optimization(model,config):
    if not isinstance(config,dict):
        raise AssertionError("optimization config should be provided as dict")
    if "optimizer" not in config or not config["optimizer"]:
        raise AssertionError("optimization config missing 'optimizer' field")
    optimizer_config = config["optimizer"]
    if "type" not in optimizer_config or not optimizer_config["type"]:
        raise AssertionError("optimizer config missing 'type' field")
    optimizer_type = thelper.utils.import_class(optimizer_config["type"])
    optimizer_params = thelper.utils.keyvals2dict(optimizer_config["params"]) if "params" in optimizer_config else None
    optimizer = optimizer_type(model.parameters(),**optimizer_params)
    scheduler = None
    scheduler_step = 1
    if "scheduler" in config and config["scheduler"]:
        scheduler_config = config["scheduler"]
        if "type" not in scheduler_config or not scheduler_config["type"]:
            raise AssertionError("scheduler config missing 'type' field")
        scheduler_type = thelper.utils.import_class(scheduler_config["type"])
        scheduler_params = thelper.utils.keyvals2dict(scheduler_config["params"]) if "params" in scheduler_config else None
        scheduler = scheduler_type(optimizer,**scheduler_params)
        scheduler_step = scheduler_config["step"] if "step" in scheduler_config else 1
    return optimizer,scheduler,scheduler_step


class Metric(ABC):
    # 'goal' values for optimization (minimum/maximum)
    minimize = float("-inf")
    maximize = float("inf")

    def __init__(self,name):
        if not name:
            raise AssertionError("metric name must not be empty (lookup might fail)")
        self.logger = thelper.utils.get_class_logger()
        self.name = name

    @abstractmethod
    def accumulate(self,pred,gt):
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError

    @abstractmethod
    def goal(self):
        # should return 'minimize' or 'maximize' only
        raise NotImplementedError


class CategoryAccuracy(Metric):
    def __init__(self,top_k=1):
        super().__init__("CategoryAccuracy")
        self.top_k = top_k
        self.correct = 0
        self.total = 0

    def accumulate(self,pred,gt):
        top_k = pred.topk(self.top_k,1)[1]
        true_k = gt.view(len(gt),1).expand_as(top_k)
        self.correct += top_k.eq(true_k).float().sum().item()
        self.total += len(pred)
        return self.eval()

    def eval(self):
        return (float(self.correct)/float(self.total))*100

    def reset(self):
        self.correct = 0
        self.total = 0

    def goal(self):
        return Metric.maximize

    def summary(self):
        self.logger.info("metric '%s' with top_k=%d"%(self.name,self.top_k))


class ClassifReport(Metric):
    def __init__(self,labels=None,target_names=None,sample_weight=None,digits=2):
        super().__init__("ClassifReport")

        def gen_report(y_true,y_pred):
            return sklearn.metrics.classification_report(y_true,y_pred,
                                                         labels=labels,
                                                         target_names=target_names,
                                                         sample_weight=sample_weight,
                                                         digits=digits)

        self.report = gen_report
        self.pred = None
        self.gt = None

    def accumulate(self,pred,gt):
        if self.pred is None:
            self.pred = pred.clone()
            self.gt = gt.clone()
        else:
            self.pred = torch.cat(self.pred,pred)
            self.gt = torch.cat(self.gt,gt)

    def eval(self):
        return self.report(self.gt.numpy(),self.pred.numpy())

    def reset(self):
        self.pred = None
        self.gt = None

    def goal(self):
        return None  # means this class should not be used for monitoring

    def summary(self):
        self.logger.info("classification report '%s'"%self.name)


class ConfusionMatrix(Metric):
    def __init__(self,labels=None,sample_weight=None):
        super().__init__("ConfusionMatrix")

        def gen_matrix(y_true,y_pred):
            return sklearn.metrics.confusion_matrix(y_true,y_pred,
                                                    labels=labels,
                                                    sample_weight=sample_weight)

        self.matrix = gen_matrix
        self.pred = None
        self.gt = None

    def accumulate(self,pred,gt):
        if self.pred is None:
            self.pred = pred.clone()
            self.gt = gt.clone()
        else:
            self.pred = torch.cat(self.pred,pred)
            self.gt = torch.cat(self.gt,gt)

    def eval(self):
        return self.matrix(self.gt.numpy(),self.pred.numpy())

    def reset(self):
        self.pred = None
        self.gt = None

    def goal(self):
        return None  # means this class should not be used for monitoring

    def summary(self):
        self.logger.info("confusion matrix '%s'"%self.name)


class BinaryAccuracy(Metric):
    def __init__(self):
        super().__init__("BinaryAccuracy")
        self.correct = 0
        self.total = 0

    def accumulate(self,pred,gt):
        if pred.size()!=gt.size():
            raise AssertionError("pred and gt should have similar size")
        pred_round = pred.round().long()
        self.correct += pred_round.eq(gt).float().sum().item()
        self.total += len(pred)
        return self.eval()

    def eval(self):
        return (float(self.correct)/float(self.total))*100

    def reset(self):
        self.correct = 0
        self.total = 0

    def goal(self):
        return Metric.maximize

    def summary(self):
        self.logger.info("metric '%s'"%self.name)
