import logging
import copy
from collections import deque
from abc import ABC, abstractmethod

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
    if "scheduler" in config and config["scheduler"]:
        scheduler_config = config["scheduler"]
        if "type" not in scheduler_config or not scheduler_config["type"]:
            raise AssertionError("scheduler config missing 'type' field")
        scheduler_type = thelper.utils.import_class(scheduler_config["type"])
        scheduler_params = thelper.utils.keyvals2dict(scheduler_config["params"]) if "params" in scheduler_config else None
        scheduler = scheduler_type(optimizer, **scheduler_params)
    return optimizer, scheduler


class Metric(ABC):
    # 'goal' values for optimization (minimum/maximum)
    minimize = float("-inf")
    maximize = float("inf")

    def __init__(self, name):
        if not name:
            raise AssertionError("metric name must not be empty (lookup might fail)")
        self.name = name

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
        return True  # override and return false if reset not needed between epochs

    @abstractmethod
    def goal(self):
        # should return 'minimize' or 'maximize' if scalar, and None otherwise
        raise NotImplementedError

    def is_scalar(self):
        curr_goal = self.goal()
        return curr_goal == Metric.minimize or curr_goal == Metric.maximize


class CategoryAccuracy(Metric):

    def __init__(self, top_k=1, max_accum=None):
        super().__init__("CategoryAccuracy")
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
                logger.warning("metric '%s' eval result invalid (set as 0.0), no results accumulated" % self.name)
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
        logger.info("metric '%s' with top_k=%d" % (self.name, self.top_k))


class BinaryAccuracy(Metric):

    def __init__(self, max_accum=None):
        super().__init__("BinaryAccuracy")
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
                logger.warning("metric '%s' eval result invalid (set as 0.0), no results accumulated" % self.name)
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
        logger.info("metric '%s'" % self.name)


class ExternalMetric(Metric):

    def __init__(self, metric_name, metric_params, metric_type,
                 target_name=None, target_label=None, goal=None,
                 class_map=None, max_accum=None):
        name = str(metric_name).rsplit(".", 1)[-1]
        if target_name:
            name += "_" + str(target_name)
        super().__init__(name)
        if not isinstance(metric_type, str) or (
                metric_type != "classif_top1" and
                metric_type != "classif_scores" and
                metric_type != "regression"):
            raise AssertionError("unknown metric type for '%s'" % self.name)
        self.metric_goal = None
        if goal is not None:
            if isinstance(goal, str) and "max" in goal.lower():
                self.metric_goal = Metric.maximize
            elif isinstance(goal, str) and "min" in goal.lower():
                self.metric_goal = Metric.minimize
            else:
                raise AssertionError("unexpected goal type for '%s'" % self.name)
        self.metric_type = metric_type
        self.metric = thelper.utils.import_class(metric_name)
        if metric_params:
            self.metric_params = thelper.utils.keyvals2dict(metric_params["params"])
        else:
            self.metric_params = {}
        if "classif" in metric_type:
            self.target_name = target_name
            self.target_label = target_label
            self.class_map = class_map
            if class_map is not None and not isinstance(class_map, dict):
                raise AssertionError("unexpected class map type")
            if class_map is not None:
                if self.target_label is not None and self.target_name is not None:
                    if class_map[self.target_label] != self.target_name:
                        raise AssertionError("target label '{}' did not match with name '{}' in class map".format(self.target_label, self.target_name))
                elif self.target_name is None and self.target_label is not None:
                    self.target_name = class_map[self.target_label]
                elif self.target_label is None and self.target_name is not None:
                    for tgt_lbl, tgt_name in class_map.items():
                        if tgt_name == self.target_name:
                            self.target_label = tgt_lbl
                            break
                    if self.target_label is None:
                        raise AssertionError("could not find target name '%s' in provided class mapping" % self.target_name)
        # elif "regression" in metric_type: missing impl for custom handling
        self.max_accum = max_accum
        self.pred = deque()
        self.gt = deque()

    def set_class_map(self, class_map):
        if "classif" in self.metric_type:
            if not isinstance(class_map, dict):
                raise AssertionError("unexpected class map type")
            if self.target_label is not None and self.target_name is not None:
                if class_map[self.target_label] != self.target_name:
                    raise AssertionError("target label '{}' did not match with name '{}' in class map".format(self.target_label, self.target_name))
            elif self.target_name is None and self.target_label is not None:
                self.target_name = class_map[self.target_label]
            elif self.target_label is None and self.target_name is not None:
                for tgt_lbl, tgt_name in class_map.items():
                    if tgt_name == self.target_name:
                        self.target_label = tgt_lbl
                        break
                if self.target_label is None:
                    raise AssertionError("could not find target name '%s' in provided class mapping" % self.target_name)
            self.class_map = class_map

    def accumulate(self, pred, gt):
        if "classif" in self.metric_type:
            if self.target_name is not None and self.target_label is None:
                raise AssertionError("could not map target name '%s' to target label, missing class map" % self.target_name)
            elif self.target_label is not None:
                pred_label = pred.topk(1, 1)[1].view(len(gt))
                y_true, y_pred = [], []
                if self.metric_type == "classif_top1":
                    must_keep = [y_pred == self.target_label or y_true == self.target_label for y_pred, y_true in zip(pred_label, gt)]
                    for idx in range(len(must_keep)):
                        if must_keep[idx]:
                            y_true.append(gt[idx].item() == self.target_label)
                            y_pred.append(pred_label[idx].item() == self.target_label)
                else:  # self.metric_type == "classif_scores"
                    for idx in range(len(gt)):
                        y_true.append(gt[idx].item() == self.target_label)
                        y_pred.append(pred[idx, self.target_label].item())
                self.gt.append(y_true)
                self.pred.append(y_pred)
            else:
                if self.metric_type == "classif_top1":
                    self.gt.append([gt[idx].item() for idx in range(len(pred.numel()))])
                    self.pred.append([pred[idx].item() for idx in range(len(pred.numel()))])
                else:  # self.metric_type == "classif_scores"
                    raise AssertionError("score-based classification analyses (e.g. roc auc) must specify target label")
        elif self.metric_type == "regression":
            raise NotImplementedError
        else:
            raise AssertionError("unknown metric type for '%s'" % self.name)
        while self.max_accum and len(self.gt) > self.max_accum:
            self.gt.popleft()
            self.pred.popleft()

    def eval(self):
        if "classif" in self.metric_type:
            y_true = [label for labels in self.gt for label in labels]
            y_pred = [label for labels in self.pred for label in labels]
            if len(y_true) != len(y_pred):
                raise AssertionError("list flattening failed")
            return self.metric(y_true, y_pred, **self.metric_params)
        else:
            raise NotImplementedError

    def reset(self):
        self.gt = deque()
        self.pred = deque()

    def needs_reset(self):
        return self.max_accum is None

    def goal(self):
        return self.metric_goal

    def summary(self):
        if self.target_name:
            logger.info("external metric '%s' for target = '%s'" % (self.name, self.target_name))
        else:
            logger.info("external metric '%s' for all targets" % self.name)


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
        logger.info("classification report '%s'" % self.name)


class ConfusionMatrix(Metric):

    def __init__(self, class_map=None):
        super().__init__("ConfusionMatrix")

        def gen_matrix(y_true, y_pred, _class_map, _class_list):
            if not _class_map:
                res = sklearn.metrics.confusion_matrix(y_true, y_pred)
            else:
                _y_true = [_class_map[classid] for classid in y_true]
                _y_pred = [_class_map[classid] if classid in _class_map else "<unset>" for classid in y_pred]
                res = sklearn.metrics.confusion_matrix(_y_true, _y_pred, labels=_class_list)
            return res

        self.matrix = gen_matrix
        self.class_map = None
        self.class_list = None
        if class_map is not None:
            self.set_class_map(class_map)
        self.pred = None
        self.gt = None

    def set_class_map(self, class_map):
        if not isinstance(class_map, dict):
            raise AssertionError("unexpected class map type")
        if len(class_map) < 2:
            raise AssertionError("class map should have at least two elements")
        self.class_map = copy.copy(class_map)
        nb_classes = max(class_map.keys()) + 1
        self.class_map[nb_classes] = "<unset>"
        self.class_list = ["<unknown>"] * nb_classes + ["<unset>"]
        for idx, name in self.class_map.items():
            self.class_list[idx] = name

    def accumulate(self, pred, gt):
        if self.pred is None:
            self.pred = pred.topk(1, 1)[1].view(len(gt))
            self.gt = gt.view(len(gt)).clone()
        else:
            self.pred = torch.cat((self.pred, pred.topk(1, 1)[1].view(len(gt))), 0)
            self.gt = torch.cat((self.gt, gt.view(len(gt))), 0)

    def eval(self):
        confmat = self.matrix(self.gt.numpy(), self.pred.numpy(), self.class_map, self.class_list)
        if self.class_list:
            return "\n" + thelper.utils.stringify_confmat(confmat, self.class_list)
        else:
            return "\n" + str(confmat)

    def get_tbx_image(self):
        confmat = self.matrix(self.gt.numpy(), self.pred.numpy(), self.class_map, self.class_list)
        if self.class_list:
            fig = thelper.utils.draw_confmat(confmat, self.class_list)
            array = thelper.utils.fig2array(fig)
            return array
        else:
            raise NotImplementedError

    def reset(self):
        self.pred = None
        self.gt = None

    def goal(self):
        return None  # means this class should not be used for monitoring

    def summary(self):
        logger.info("confusion matrix '%s'" % self.name)
