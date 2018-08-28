import logging
import copy
from collections import deque
from abc import ABC, abstractmethod

import torch
import torch.nn
import torch.nn.functional
import sklearn.metrics

import thelper.utils

logger = logging.getLogger(__name__)


class Metric(ABC):
    # 'goal' values for optimization (minimum/maximum)
    minimize = float("-inf")
    maximize = float("inf")

    @abstractmethod
    def accumulate(self, pred, gt, meta=None):
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
        self.top_k = top_k
        self.max_accum = max_accum
        self.correct = deque()
        self.total = deque()
        self.warned_eval_bad = False

    def accumulate(self, pred, gt, meta=None):
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
                logger.warning("category accuracy eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return (float(sum(self.correct)) / float(sum(self.total))) * 100

    def reset(self):
        self.correct = deque()
        self.total = deque()

    def needs_reset(self):
        return self.max_accum is None

    def set_max_accum(self, max_accum):
        self.max_accum = max_accum

    def goal(self):
        return Metric.maximize


class BinaryAccuracy(Metric):

    def __init__(self, max_accum=None):
        self.max_accum = max_accum
        self.correct = deque()
        self.total = deque()
        self.warned_eval_bad = False

    def accumulate(self, pred, gt, meta=None):
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
                logger.warning("binary accuracy eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return (float(sum(self.correct)) / float(sum(self.total))) * 100

    def reset(self):
        self.correct = deque()
        self.total = deque()

    def needs_reset(self):
        return self.max_accum is None

    def set_max_accum(self, max_accum):
        self.max_accum = max_accum

    def goal(self):
        return Metric.maximize


class ExternalMetric(Metric):

    def __init__(self, metric_name, metric_params, metric_type,
                 target_name=None, target_label=None, goal=None,
                 class_names=None, max_accum=None, force_softmax=True):
        if not isinstance(metric_type, str) or (
                metric_type != "classif_top1" and
                metric_type != "classif_scores" and
                metric_type != "regression"):
            raise AssertionError("unknown metric type '%s'" % str(metric_type))
        self.metric_goal = None
        if goal is not None:
            if isinstance(goal, str) and "max" in goal.lower():
                self.metric_goal = Metric.maximize
            elif isinstance(goal, str) and "min" in goal.lower():
                self.metric_goal = Metric.minimize
            else:
                raise AssertionError("unexpected goal type for '%s'" % str(metric_name))
        self.metric_type = metric_type
        self.metric = thelper.utils.import_class(metric_name)
        if metric_params:
            self.metric_params = thelper.utils.keyvals2dict(metric_params["params"])
        else:
            self.metric_params = {}
        if "classif" in metric_type:
            self.target_name = target_name
            self.target_label = target_label
            self.class_names = None
            if class_names is not None:
                self.set_class_names(class_names)
            if metric_type == "classif_scores":
                self.force_softmax = force_softmax  # only useful in this case
        # elif "regression" in metric_type: missing impl for custom handling
        self.max_accum = max_accum
        self.pred = deque()
        self.gt = deque()

    def set_class_names(self, class_names):
        if "classif" in self.metric_type:
            if not isinstance(class_names, list):
                raise AssertionError("expected list for class names")
            if len(class_names) < 2:
                raise AssertionError("not enough classes in provided class list")
            if self.target_name is not None and self.target_name not in class_names:
                raise AssertionError("could not find target name '%s' in class names list" % str(self.target_name))
            if self.target_label is not None and not isinstance(self.target_label, int):
                raise AssertionError("expected target label type to be int")
            if self.target_label is not None and (self.target_label < 0 or self.target_label >= len(class_names)):
                raise AssertionError("target label '%d' is out of range for given class names list" % int(self.target_label))
            if self.target_name is not None and self.target_label is not None and class_names[self.target_label] != self.target_name:
                raise AssertionError("target label '{}' did not match with name '{}' in class names list".format(self.target_label, self.target_name))
            elif self.target_name is None and self.target_label is not None:
                self.target_name = class_names[self.target_label]
            elif self.target_label is None and self.target_name is not None:
                self.target_label = class_names.index(self.target_name)
            self.class_names = class_names
        else:
            raise AssertionError("unexpected class list with metric type other than classif")

    def accumulate(self, pred, gt, meta=None):
        if "classif" in self.metric_type:
            if self.target_name is not None and self.target_label is None:
                raise AssertionError("could not map target name '%s' to target label, missing class list" % self.target_name)
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
                    if self.force_softmax:
                        with torch.no_grad():
                            pred = torch.nn.functional.softmax(pred, dim=1)
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
            raise AssertionError("unknown metric type '%s'" % str(self.metric_type))
        while self.max_accum and len(self.gt) > self.max_accum:
            self.gt.popleft()
            self.pred.popleft()

    def eval(self):
        if "classif" in self.metric_type:
            y_gt = [gt for gts in self.gt for gt in gts]
            y_pred = [pred for preds in self.pred for pred in preds]
            if len(y_gt) != len(y_pred):
                raise AssertionError("list flattening failed")
            return self.metric(y_gt, y_pred, **self.metric_params)
        else:
            raise NotImplementedError

    def reset(self):
        self.gt = deque()
        self.pred = deque()

    def needs_reset(self):
        return self.max_accum is None

    def set_max_accum(self, max_accum):
        self.max_accum = max_accum

    def goal(self):
        return self.metric_goal


class ClassifReport(Metric):

    def __init__(self, class_names=None, sample_weight=None, digits=4):

        def gen_report(y_true, y_pred, _class_names):
            if not _class_names:
                res = sklearn.metrics.classification_report(y_true, y_pred,
                                                            sample_weight=sample_weight,
                                                            digits=digits)
            else:
                _y_true = [_class_names[classid] for classid in y_true]
                _y_pred = [_class_names[classid] if (0 <= classid < len(_class_names)) else "<unset>" for classid in y_pred]
                res = sklearn.metrics.classification_report(_y_true, _y_pred,
                                                            sample_weight=sample_weight,
                                                            digits=digits)
            return "\n" + res

        self.report = gen_report
        self.class_names = class_names
        if class_names and not isinstance(class_names, list):
            raise AssertionError("expected class names to be list")
        self.pred = None
        self.gt = None

    def set_class_names(self, class_names):
        if class_names and not isinstance(class_names, list):
            raise AssertionError("expected class names to be list")
        if len(class_names) < 2:
            raise AssertionError("class list should have at least two elements")
        self.class_names = class_names

    def accumulate(self, pred, gt, meta=None):
        if self.pred is None:
            self.pred = pred.topk(1, 1)[1].view(len(gt))
            self.gt = gt.view(len(gt)).clone()
        else:
            self.pred = torch.cat((self.pred, pred.topk(1, 1)[1].view(len(gt))), 0)
            self.gt = torch.cat((self.gt, gt.view(len(gt))), 0)

    def eval(self):
        return self.report(self.gt.numpy(), self.pred.numpy(), self.class_names)

    def reset(self):
        self.pred = None
        self.gt = None

    def goal(self):
        return None  # means this class should not be used for monitoring


class ConfusionMatrix(Metric):

    def __init__(self, class_names=None):

        def gen_matrix(y_true, y_pred, _class_names):
            if not _class_names:
                res = sklearn.metrics.confusion_matrix(y_true, y_pred)
            else:
                _y_true = [_class_names[classid] for classid in y_true]
                _y_pred = [_class_names[classid] if (0 <= classid < len(_class_names)) else "<unset>" for classid in y_pred]
                res = sklearn.metrics.confusion_matrix(_y_true, _y_pred, labels=_class_names)
            return res

        self.matrix = gen_matrix
        self.class_names = None
        if class_names is not None:
            self.set_class_names(class_names)
        self.pred = None
        self.gt = None

    def set_class_names(self, class_names):
        if not isinstance(class_names, list):
            raise AssertionError("expected class names to be list")
        if len(class_names) < 2:
            raise AssertionError("class list should have at least two elements")
        self.class_names = copy.deepcopy(class_names)
        self.class_names.append("<unset>")

    def accumulate(self, pred, gt, meta=None):
        if self.pred is None:
            self.pred = pred.topk(1, 1)[1].view(len(gt))
            self.gt = gt.view(len(gt)).clone()
        else:
            self.pred = torch.cat((self.pred, pred.topk(1, 1)[1].view(len(gt))), 0)
            self.gt = torch.cat((self.gt, gt.view(len(gt))), 0)

    def eval(self):
        confmat = self.matrix(self.gt.numpy(), self.pred.numpy(), self.class_names)
        if self.class_names:
            return "\n" + thelper.utils.stringify_confmat(confmat, self.class_names)
        else:
            return "\n" + str(confmat)

    def get_tbx_image(self):
        confmat = self.matrix(self.gt.numpy(), self.pred.numpy(), self.class_names)
        if self.class_names:
            fig = thelper.utils.draw_confmat(confmat, self.class_names)
            array = thelper.utils.fig2array(fig)
            return array
        else:
            raise NotImplementedError

    def reset(self):
        self.pred = None
        self.gt = None

    def goal(self):
        return None  # means this class should not be used for monitoring


class ROCCurve(Metric):

    def __init__(self, target_name=None, target_label=None, class_names=None, force_softmax=True,
                 log_params=None, sample_weight=None, drop_intermediate=True):
        self.target_name = target_name
        self.target_label = target_label
        self.class_names = None
        if class_names is not None:
            self.set_class_names(class_names)
        self.force_softmax = force_softmax
        self.log_params = log_params
        if log_params is not None:
            if not isinstance(log_params, dict):
                raise AssertionError("unexpected log params type (expected dict)")
            if "fpr_threshold" not in log_params and "tpr_threshold" not in log_params:
                raise AssertionError("missing log 'fpr_threshold' or 'tpr_threshold' field for logging in params")
            if "fpr_threshold" in log_params and "tpr_threshold" in log_params:
                raise AssertionError("must specify only 'fpr_threshold' or 'tpr_threshold' field for logging in params")
            if "fpr_threshold" in log_params:
                self.log_fpr_threshold = float(log_params["fpr_threshold"])
                if self.log_fpr_threshold < 0 or self.log_fpr_threshold > 1:
                    raise AssertionError("bad log fpr threshold (should be in [0,1]")
                self.log_tpr_threshold = None
            elif "tpr_threshold" in log_params:
                self.log_tpr_threshold = float(log_params["tpr_threshold"])
                if self.log_tpr_threshold < 0 or self.log_tpr_threshold > 1:
                    raise AssertionError("bad log tpr threshold (should be in [0,1]")
                self.log_fpr_threshold = None
            if "meta_keys" not in log_params:
                raise AssertionError("missing log 'meta_keys' field for logging in params")
            self.log_meta_keys = log_params["meta_keys"]
            if not isinstance(self.log_meta_keys, list):
                raise AssertionError("unexpected log meta keys params type (expected list)")

        def gen_curve(y_true, y_score, _class_names, _target_label, _sample_weight=sample_weight, _drop_intermediate=drop_intermediate):
            if _class_names is None or not _class_names:
                if _target_label is not None:
                    raise AssertionError("got positive label, but no class list (even at run time)")
                res = sklearn.metrics.roc_curve(y_true, y_score, sample_weight=_sample_weight, drop_intermediate=_drop_intermediate)
            else:
                if _target_label is None:
                    raise AssertionError("missing positive target label at run time")
                _y_true = [classid == _target_label for classid in y_true]
                _y_score = y_score[..., _target_label]
                res = sklearn.metrics.roc_curve(_y_true, _y_score, sample_weight=sample_weight, drop_intermediate=drop_intermediate)
            return res

        def gen_auc(y_true, y_score, _class_names, _target_label, _sample_weight=sample_weight):
            if _class_names is None or not _class_names:
                if _target_label is not None:
                    raise AssertionError("got positive label, but no class list (even at run time)")
                res = sklearn.metrics.roc_auc_score(y_true, y_score, sample_weight=sample_weight)
            else:
                if _target_label is None:
                    raise AssertionError("missing positive target label at run time")
                _y_true = [classid == _target_label for classid in y_true]
                _y_score = y_score[..., _target_label]
                res = sklearn.metrics.roc_auc_score(_y_true, _y_score, sample_weight=sample_weight)
            return res

        self.curve = gen_curve
        self.auc = gen_auc
        self.score = None
        self.true = None
        self.meta = None  # needed if outputting tbx txt

    def set_class_names(self, class_names):
        if not isinstance(class_names, list):
            raise AssertionError("expected list for class names")
        if len(class_names) < 2:
            raise AssertionError("not enough classes in provided class list")
        if self.target_name is not None and self.target_name not in class_names:
            raise AssertionError("could not find target name '%s' in class names list" % str(self.target_name))
        if self.target_label is not None and not isinstance(self.target_label, int):
            raise AssertionError("expected target label type to be int")
        if self.target_label is not None and (self.target_label < 0 or self.target_label >= len(class_names)):
            raise AssertionError("target label '%d' is out of range for given class names list" % int(self.target_label))
        if self.target_name is not None and self.target_label is not None and class_names[self.target_label] != self.target_name:
            raise AssertionError("target label '{}' did not match with name '{}' in class names list".format(self.target_label, self.target_name))
        elif self.target_name is None and self.target_label is not None:
            self.target_name = class_names[self.target_label]
        elif self.target_label is None and self.target_name is not None:
            self.target_label = class_names.index(self.target_name)
        self.class_names = class_names

    def accumulate(self, pred, gt, meta=None):
        if self.force_softmax:
            with torch.no_grad():
                pred = torch.nn.functional.softmax(pred, dim=1)
        if self.score is None:
            self.score = pred.clone()
            self.true = gt.clone()
        else:
            self.score = torch.cat((self.score, pred), 0)
            self.true = torch.cat((self.true, gt), 0)
        if self.log_params:  # do not log meta parameters unless requested
            if meta is None or not meta:
                raise AssertionError("sample metadata is required, logging is activated")
            _meta = {key: meta[key] for key in self.log_meta_keys}
            if self.meta is None:
                self.meta = copy.deepcopy(_meta)
            else:
                for key in self.log_meta_keys:
                    if isinstance(_meta[key], list):
                        self.meta[key] += _meta[key]
                    elif isinstance(_meta[key], torch.Tensor):
                        self.meta[key] = torch.cat((self.meta[key], _meta[key]), 0)
                    else:
                        raise AssertionError("missing impl for meta concat w/ type '%s'" % str(type(_meta[key])))

    def eval(self):
        return self.auc(self.true.numpy(), self.score.numpy(), self.class_names, self.target_label)

    def get_tbx_image(self):
        fpr, tpr, t = self.curve(self.true.numpy(), self.score.numpy(), self.class_names, self.target_label)
        fig = thelper.utils.draw_roc_curve(fpr, tpr)
        array = thelper.utils.fig2array(fig)
        return array

    def get_tbx_text(self):
        if self.log_params is None:
            return None  # do not generate log text unless requested
        if self.meta is None or not self.meta:
            return None
        if self.class_names is None or not self.class_names:
            raise AssertionError("missing class list for logging, current impl only supports named outputs")
        _fpr, _tpr, _t = self.curve(self.true.numpy(), self.score.numpy(), self.class_names, self.target_label, _drop_intermediate=False)
        threshold = None
        for fpr, tpr, t in zip(_fpr, _tpr, _t):
            if self.log_fpr_threshold is not None and self.log_fpr_threshold <= fpr:
                threshold = t
                break
            elif self.log_tpr_threshold is not None and self.log_tpr_threshold <= tpr:
                threshold = t
                break
        if threshold is None:
            raise AssertionError("bad fpr/tpr threshold, could not find cutoff for pred scores")
        res = "sample_idx,classid,classname,pred_score"
        for key in self.log_meta_keys:
            res += "," + str(key)
        res += "\n"
        for sample_idx in range(self.true.numel()):
            true = self.true[sample_idx].item() == self.target_label
            score = self.score[sample_idx, self.target_label].item()
            if true and score <= threshold:
                res += "{:8d},{:4d},{:>10s},{:2.4f}".format(sample_idx, self.target_label, self.target_name, score)
                for key in self.log_meta_keys:
                    val = self.meta[key][sample_idx]
                    if isinstance(val, torch.Tensor) and val.numel() == 1:
                        res += "," + str(val.item())
                    else:
                        res += "," + str(val)
                res += "\n"
        return res

    def reset(self):
        self.score = None
        self.true = None
        self.meta = None

    def goal(self):
        return None  # means this class should not be used for monitoring
