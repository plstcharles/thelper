"""Metrics module.

This module contains classes that implement metrics used to monitor training sessions and evaluate models.
These metrics should all inherit from :class:`thelper.optim.metrics.Metric` to allow them to be dynamically
instantiated by the framework from a configuration file, and evaluated automatically inside a training
session. For more information on this, refer to :class:`thelper.train.base.Trainer`.
"""

import copy
import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, AnyStr, Callable, Dict, List, Optional  # noqa: F401

import numpy as np
import sklearn.metrics
import torch
import torch.nn
import torch.nn.functional

import thelper.utils

logger = logging.getLogger(__name__)


class Metric(ABC):
    """Abstract metric interface.

    This interface defines basic functions required so that :class:`thelper.train.base.Trainer` can
    figure out how to instantiate, update, reset, and optimize a given metric while training/evaluating
    a model.

    Not all metrics are required to be 'optimizable'; in other words, they do not always need to
    return a scalar value and define a goal. For example, a class can derive from this interface
    and simply accumulate predictions to log them or to produce a graph. In such cases, the class
    would simply need to override the ``goal`` method to return ``None``. Then, the trainer would
    still update the object periodically with predictions, but it will not try to monitor the output
    of its ``eval`` function.
    """

    minimize = float("-inf")
    """Possible return value of the ``goal`` function for scalar metrics."""

    maximize = float("inf")
    """Possible return value of the ``goal`` function for scalar metrics."""

    @abstractmethod
    def accumulate(self, pred, gt, meta=None):
        """Receives the latest prediction and groundtruth tensors from the training session.

        The data given here is used to update the internal state of the metric. For example, a
        classification accuracy metric would accumulate the correct number of predictions in
        comparison to groundtruth labels. The meta values are also provided in case the metric
        can use one of them to produce a better output.

        Args:
            pred: model prediction tensor forwarded by the trainer.
            gt: groundtruth tensor forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata tensors forwarded by the trainer.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        """Returns the metric's evaluation result.

        This can be a scalar, a string, or any other type of object. If it is a scalar, the
        metric should also define a goal (minimize, maximize) so that the trainer can monitor
        whether the metric is improving over time. If it is a string, it will be printed in the
        console at the end of every epoch. Otherwise, it is simply logged and eventually returned
        at the end of the training session.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Toggles a reset of the metric's internal state.

        Some metrics might rely on an internal state to smooth out their evaluation results using
        e.g. a moving average, or to keep track of the progression through a dataset. A reset can
        thus be necessary when the dataset changes (e.g. at the end of an epoch). This function is
        called automatically by the trainer in such cases."""
        raise NotImplementedError

    def needs_reset(self):
        """Returns whether the metric needs to be reset between every training epoch or not.

        For example, a metric that computes the prediction accuracy over an entire dataset might
        need to be reset every epoch (it would thus return ``True``). However, if it is implemented
        with a moving average and used to monitor prediction accuracy at each iteration, then
        resetting it every epoch might cause spikes in the results. In this case, returning ``False``
        would be best.

        Note that even if a metric always returns ``False`` here, it might still be reset by the
        trainer if the dataset is switched (e.g. while phasing from training to validation, or to
        testing).
        """
        return True

    @abstractmethod
    def goal(self):
        """Returns the scalar optimization goal of the metric, if available.

        The returned goal can be the ``minimize`` or ``maximize`` members of ``thelper.optim.metrics.Metric``
        if the class's evaluation returns a scalar value, and ``None`` otherwise. The trainer will
        check this value to see if monitoring the metric's evaluation result progression is possible.
        """
        raise NotImplementedError

    def anti_goal(self):
        """Returns the opposite of the scalar optimization goal of the metric, if available."""
        if not self.is_scalar():
            raise AssertionError("undefined anti goal behavior when metric is not scalar")
        return Metric.maximize if self.goal() == Metric.minimize else Metric.minimize

    def is_scalar(self):
        """Returns whether the metric evaluates to a scalar based on its goal."""
        return self.goal() == Metric.minimize or self.goal() == Metric.maximize

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this metric."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + "()"


class CategoryAccuracy(Metric):
    r"""Classification accuracy metric interface.

    This is a scalar metric used to monitor the multi-label prediction accuracy of a model. By default,
    it works in ``top-k`` mode, meaning that the evaluation result is given by:

    .. math::
      \text{accuracy} = \frac{\text{nb. correct predictions}}{\text{nb. total predictions}} \cdot 100

    When :math:`k>1`, a 'correct' prediction is obtained if any of the model's top :math:`k` predictions
    (i.e. the :math:`k` predictions with the highest score) match the groundtruth label. Otherwise, if
    :math:`k=1`, then only the top prediction is compared to the groundtruth label.

    This metric's goal is to maximize its value :math:`\in [0,100]` (a percentage is returned).

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "top_5_accuracy": {
                # this type is used to instantiate the accuracy metric
                "type": "thelper.optim.metrics.CategoryAccuracy",
                # these parameters are passed to the wrapper's constructor
                "params": {
                    # the top prediction count to check for a match with the groundtruth
                    "top_k": 5
                }
            },
            # ...
        }
        # ...

    Attributes:
        top_k: number of top predictions to consider when matching with the groundtruth (default=1).
        max_accum: if using a moving average, this is the window size to use (default=None).
        correct: total number of correct predictions stored using a queue for window-based averaging.
        total: total number of predictions stored using a queue for window-based averaging.
        warned_eval_bad: toggles whether the division-by-zero warning has been flagged or not.
    """

    def __init__(self, top_k=1, max_accum=None):
        """Receives the number of predictions to consider for matches (top-k) and the moving average
        window size (max_accum).

        Note that by default, even if max_accum is not provided here, it can still be set by the
        trainer at runtime through the :func:`thelper.optim.metrics.CategoryAccuracy.set_max_accum` function.
        This is fairly useful as the total size of the training dataset is unlikely to be known when
        metrics are instantiated.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise AssertionError("invalid top-k value")
        if max_accum is not None and (not isinstance(max_accum, int) or max_accum <= 0):
            raise AssertionError("invalid max accumulation value for moving average")
        self.top_k = top_k
        self.max_accum = max_accum
        self.correct = deque()
        self.total = deque()
        self.warned_eval_bad = False

    def accumulate(self, pred, gt, meta=None):
        """Receives the latest class prediction and groundtruth labels from the training session.

        The inputs are expected to still be in ``torch.Tensor`` format, but must be located on the
        CPU. This function computes and accumulate the number of correct and total predictions in
        the queues, popping them if the maximum window length is reached.

        Args:
            pred: model class predictions forwarded by the trainer.
            gt: groundtruth labels forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata forwarded by the trainer (unused).
        """
        if gt is None or gt.numel() == 0:
            return  # only accumulating results when groundtruth available
        if pred.dim() != gt.dim() + 1:
            raise AssertionError("prediction/gt tensors dim mismatch (should be BxCx... and Bx...")
        if pred.shape[0] != gt.shape[0]:
            raise AssertionError("prediction/gt tensors batch size mismatch")
        if pred.dim() > 2 and pred.shape[2:] != gt.shape[1:]:
            raise AssertionError("prediction/gt tensors array size mismatch")
        top_k = pred.topk(self.top_k, dim=1)[1].view(pred.shape[0], self.top_k, -1).numpy()
        true_k = gt.view(gt.shape[0], 1, -1).expand(-1, self.top_k, -1).numpy()
        self.correct.append(np.any(np.equal(top_k, true_k), axis=1).sum(dtype=np.int64))
        self.total.append(gt.numel())
        if self.max_accum and len(self.correct) > self.max_accum:
            self.correct.popleft()
            self.total.popleft()

    def eval(self):
        """Returns the current accuracy (in percentage) based on the accumulated prediction counts.

        Will issue a warning if no predictions have been accumulated yet.
        """
        if len(self.total) == 0 or sum(self.total) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("category accuracy eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return (float(sum(self.correct)) / float(sum(self.total))) * 100

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying prediction count queues."""
        self.correct = deque()
        self.total = deque()

    def needs_reset(self):
        """If the metric is currently operating in moving average mode, then it does not need to
        be reset (returns ``False``); else returns ``True``."""
        return self.max_accum is None

    def set_max_accum(self, max_accum):
        """Sets the moving average window size.

        This is fairly useful as the total size of the training dataset is unlikely to be known when
        metrics are instantiated. The current implementation of :class:`thelper.train.base.Trainer`
        will look for this member function and automatically call it with the dataset size when it is
        available.
        """
        self.max_accum = max_accum

    def goal(self):
        """Returns the scalar optimization goal of this metric (maximization)."""
        return Metric.maximize


class BinaryAccuracy(Metric):
    r"""Binary classification accuracy metric interface.

    This is a scalar metric used to monitor the binary prediction accuracy of a model. The evaluation
    result is given by:

    .. math::
      \text{accuracy} = \frac{\text{TN} + \text{TP}}{\text{TN} + \text{TP} + \text{FN} + \text{FP}} \cdot 100,

    where TN = True Negative, TP = True Positive, FN = False Negative, and FP = False Positive.

    This metric's goal is to maximize its value :math:`\in [0,100]` (a percentage is returned).

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "accuracy": {
                # this type is used to instantiate the accuracy metric
                "type": "thelper.optim.metrics.BinaryAccuracy",
                # there are no useful parameters to give to the constructor
                "params": {}
            },
            # ...
        }
        # ...

    Attributes:
        max_accum: if using a moving average, this is the window size to use (default=None).
        correct: total number of correct predictions stored using a queue for window-based averaging.
        total: total number of predictions stored using a queue for window-based averaging.
        warned_eval_bad: toggles whether the division-by-zero warning has been flagged or not.
    """

    def __init__(self, max_accum=None):
        """Receives the moving average window size (max_accum).

        Note that by default, even if max_accum is not provided here, it can still be set by the
        trainer at runtime through the :func:`thelper.optim.metrics.BinaryAccuracy.set_max_accum` function.
        This is fairly useful as the total size of the training dataset is unlikely to be known when
        metrics are instantiated.
        """
        self.max_accum = max_accum
        self.correct = deque()
        self.total = deque()
        self.warned_eval_bad = False

    def accumulate(self, pred, gt, meta=None):
        """Receives the latest class prediction and groundtruth labels from the training session.

        The inputs are expected to still be in ``torch.Tensor`` format, but must be located on the
        CPU. This function computes and accumulate the number of correct and total predictions in
        the queues, popping them if the maximum window length is reached.

        Args:
            pred: model class predictions forwarded by the trainer.
            gt: groundtruth labels forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata forwarded by the trainer (unused).
        """
        if gt is None or gt.numel() == 0:
            return  # only accumulating results when groundtruth available
        if pred.dim() != gt.dim() + 1:
            raise AssertionError("prediction/gt tensors dim mismatch (should be BxCx... and Bx...)")
        if pred.shape[0] != gt.shape[0]:
            raise AssertionError("prediction/gt tensors batch size mismatch")
        if pred.dim() > 2 and pred.shape[2:] != gt.shape[1:]:
            raise AssertionError("prediction/gt tensors array size mismatch")
        top = pred.topk(1, dim=1)[1].view(pred.shape[0], 1, -1).numpy()
        true = gt.view(gt.shape[0], 1, -1).numpy()
        self.correct.append(np.any(np.equal(top, true), axis=1).sum(dtype=np.int64))
        self.total.append(gt.numel())
        if self.max_accum and len(self.correct) > self.max_accum:
            self.correct.popleft()
            self.total.popleft()

    def eval(self):
        """Returns the current accuracy (in percentage) based on the accumulated prediction counts.

        Will issue a warning if no predictions have been accumulated yet.
        """
        if len(self.total) == 0 or sum(self.total) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("binary accuracy eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return (float(sum(self.correct)) / float(sum(self.total))) * 100

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying prediction count queues."""
        self.correct = deque()
        self.total = deque()

    def needs_reset(self):
        """If the metric is currently operating in moving average mode, then it does not need to
        be reset (returns ``False``); else returns ``True``."""
        return self.max_accum is None

    def set_max_accum(self, max_accum):
        """Sets the moving average window size.

        This is fairly useful as the total size of the training dataset is unlikely to be known when
        metrics are instantiated. The current implementation of :class:`thelper.train.base.Trainer`
        will look for this member function and automatically call it with the dataset size when it is
        available.
        """
        self.max_accum = max_accum

    def goal(self):
        """Returns the scalar optimization goal of this metric (maximization)."""
        return Metric.maximize


class MeanAbsoluteError(Metric):
    r"""Mean absolute error metric interface.

    This is a scalar metric used to monitor the mean absolute deviation (or error) for a model's
    predictions. This regression metric can be described as:

    .. math::
        e(x, y) = E = \{e_1,\dots,e_N\}^\top, \quad
        e_n = \left| x_n - y_n \right|,

    where :math:`N` is the batch size. If ``reduction`` is not ``'none'``, then:

    .. math::
        \text{MAE}(x, y) =
        \begin{cases}
            \operatorname{mean}(E), & \text{if reduction } = \text{mean.}\\
            \operatorname{sum}(E),  & \text{if reduction } = \text{sum.}
        \end{cases}

    `x` and `y` are tensors of arbitrary shapes with a total of `n` elements each.

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "mae": {
                # this type is used to instantiate the error metric
                "type": "thelper.optim.metrics.MeanAbsoluteError",
                "params": {
                    "reduction": "mean"
                }
            },
            # ...
        }
        # ...

    Todo: add support for 'dont care' target value?

    Attributes:
        max_accum: if using a moving average, this is the window size to use (default=None).
        reduction: string representing the tensor reduction strategy to use.
        errors: queue of error values stored for window-based averaging.
        warned_eval_bad: toggles whether the division-by-zero warning has been flagged or not.
    """

    def __init__(self, reduction="mean", max_accum=None):
        """Receives the reduction strategy and moving average window size (max_accum).

        Note that by default, even if max_accum is not provided here, it can still be set by the
        trainer at runtime through the :func:`thelper.optim.metrics.BinaryAccuracy.set_max_accum` function.
        This is fairly useful as the total size of the training dataset is unlikely to be known when
        metrics are instantiated.
        """
        self.reduction = reduction
        self.max_accum = max_accum
        self.errors = deque()
        self.warned_eval_bad = False
        if reduction != "mean":
            raise NotImplementedError  # will only be supported with PyTorch version > 1.0.0

    def accumulate(self, pred, target, meta=None):
        """Receives the latest predictions and target values from the training session.

        The inputs are expected to still be in ``torch.Tensor`` format, but must be located on the
        CPU. This function computes and accumulate the error in the queue, popping it if the maximum
        window length is reached.

        Args:
            pred: model prediction values forwarded by the trainer.
            target: target prediction values forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata forwarded by the trainer (unused).
        """
        if target is None or target.numel() == 0:
            return  # only accumulating results when groundtruth available
        if pred.shape != target.shape:
            raise AssertionError("prediction/gt tensors shape mismatch")
        self.errors.append(torch.nn.functional.l1_loss(pred, target).numpy())  # add reduction here w/ PyTorch v1.0.0
        if self.max_accum and len(self.errors) > self.max_accum:
            self.errors.popleft()

    def eval(self):
        """Returns the current (average) mean absolute error based on the accumulated values.

        Will issue a warning if no predictions have been accumulated yet.
        """
        if len(self.errors) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("mean absolute error eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return np.mean(np.asarray(list(self.errors)), axis=0, dtype=np.float32)

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying the error queue."""
        self.errors = deque()

    def needs_reset(self):
        """If the metric is currently operating in moving average mode, then it does not need to
        be reset (returns ``False``); else returns ``True``."""
        return self.max_accum is None

    def set_max_accum(self, max_accum):
        """Sets the moving average window size.

        This is fairly useful as the total size of the training dataset is unlikely to be known when
        metrics are instantiated. The current implementation of :class:`thelper.train.base.Trainer`
        will look for this member function and automatically call it with the dataset size when it is
        available.
        """
        self.max_accum = max_accum

    def goal(self):
        """Returns the scalar optimization goal of this metric (minimization)."""
        return Metric.minimize


class MeanSquaredError(Metric):
    r"""Mean squared error metric interface.

    This is a scalar metric used to monitor the mean squared deviation (or error) for a model's
    predictions. This regression metric can be described as:

    .. math::
        e(x, y) = E = \{e_1,\dots,e_N\}^\top, \quad
        e_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If ``reduction`` is not ``'none'``, then:

    .. math::
        \text{MSE}(x, y) =
        \begin{cases}
            \operatorname{mean}(E), & \text{if reduction } = \text{mean.}\\
            \operatorname{sum}(E),  & \text{if reduction } = \text{sum.}
        \end{cases}

    `x` and `y` are tensors of arbitrary shapes with a total of `n` elements each.

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "mse": {
                # this type is used to instantiate the error metric
                "type": "thelper.optim.metrics.MeanSquaredError",
                "params": {
                    "reduction": "mean"
                }
            },
            # ...
        }
        # ...

    Todo: add support for 'dont care' target value?

    Attributes:
        max_accum: if using a moving average, this is the window size to use (default=None).
        reduction: string representing the tensor reduction strategy to use.
        errors: queue of error values stored for window-based averaging.
        warned_eval_bad: toggles whether the division-by-zero warning has been flagged or not.
    """

    def __init__(self, reduction="mean", max_accum=None):
        """Receives the reduction strategy and moving average window size (max_accum).

        Note that by default, even if max_accum is not provided here, it can still be set by the
        trainer at runtime through the :func:`thelper.optim.metrics.BinaryAccuracy.set_max_accum` function.
        This is fairly useful as the total size of the training dataset is unlikely to be known when
        metrics are instantiated.
        """
        self.reduction = reduction
        self.max_accum = max_accum
        self.errors = deque()
        self.warned_eval_bad = False
        if reduction != "mean":
            raise NotImplementedError  # will only be supported with PyTorch version > 1.0.0

    def accumulate(self, pred, target, meta=None):
        """Receives the latest predictions and target values from the training session.

        The inputs are expected to still be in ``torch.Tensor`` format, but must be located on the
        CPU. This function computes and accumulate the error in the queue, popping it if the maximum
        window length is reached.

        Args:
            pred: model prediction values forwarded by the trainer.
            target: target prediction values forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata forwarded by the trainer (unused).
        """
        if target is None or target.numel() == 0:
            return  # only accumulating results when groundtruth available
        if pred.shape != target.shape:
            raise AssertionError("prediction/gt tensors shape mismatch")
        self.errors.append(torch.nn.functional.mse_loss(pred, target).numpy())  # add reduction here w/ PyTorch v1.0.0
        if self.max_accum and len(self.errors) > self.max_accum:
            self.errors.popleft()

    def eval(self):
        """Returns the current (average) mean squared error based on the accumulated values.

        Will issue a warning if no predictions have been accumulated yet.
        """
        if len(self.errors) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("mean squared error eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return np.mean(np.asarray(list(self.errors)), axis=0, dtype=np.float32)

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying the error queue."""
        self.errors = deque()

    def needs_reset(self):
        """If the metric is currently operating in moving average mode, then it does not need to
        be reset (returns ``False``); else returns ``True``."""
        return self.max_accum is None

    def set_max_accum(self, max_accum):
        """Sets the moving average window size.

        This is fairly useful as the total size of the training dataset is unlikely to be known when
        metrics are instantiated. The current implementation of :class:`thelper.train.base.Trainer`
        will look for this member function and automatically call it with the dataset size when it is
        available.
        """
        self.max_accum = max_accum

    def goal(self):
        """Returns the scalar optimization goal of this metric (minimization)."""
        return Metric.minimize


class ExternalMetric(Metric):
    r"""External metric wrapping interface.

    This interface is used to wrap external metrics and use them in the training framework. The metrics
    of ``sklearn.metrics`` are good candidates that have been used extensively with this interface in
    the past, but those of other libraries might also be compatible.

    Along with the name of the class to import and its constructor's parameters, the user must provide
    a handling mode that specifies how prediction and groundtruth data should be handled in this wrapper.
    Also, extra arguments such as target label names, goal information, and window sizes can be provided
    for specific use cases related to the selected handling mode.

    For now, two metric handling modes (both related to classification) are supported:

      * ``classif_best``: the wrapper will accumulate the predicted and groundtruth classification \
        labels forwarded by the trainer and provide them to the external metric for evaluation. If \
        a target label name is specified, then only classifications related to that label will be \
        accumulated. This is the handling mode required for count-based classification metrics such \
        as accuracy, F-Measure, precision, recall, etc.

      * ``classif_score``: the wrapper will accumulate the prediction score of the targeted label \
        along with a boolean that indicates whether this label was the groundtruth label or not. This \
        is the handling mode required for score-based classification metrics such as when computing \
        the area under the ROC curve (AUC).

    Usage examples inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the first example metric; it is used for lookup/printing only
            "f1_score_reject": {
                # this type is used to instantiate the wrapper
                "type": "thelper.optim.metrics.ExternalMetric",
                # these parameters are passed to the wrapper's constructor
                "params": {
                    # the external class to import
                    "metric_name": "sklearn.metrics.f1_score",
                    # the parameters passed to the external class's constructor
                    "metric_params": [],
                    # the wrapper metric handling mode
                    "metric_type": "classif_best",
                    # the target class name (note: dataset-specific)
                    "target_name": "reject",
                    # the goal type of the external metric
                    "goal": "max"
                }
            },
            # this is the name of the second example metric; it is used for lookup/printing only
            "roc_auc_accept": {
                # this type is used to instantiate the wrapper
                "type": "thelper.optim.metrics.ExternalMetric",
                # these parameters are passed to the wrapper's constructor
                "params": {
                    # the external class to import
                    "metric_name": "sklearn.metrics.roc_auc_score",
                    # the parameters passed to the external class's constructor
                    "metric_params": [],
                    # the wrapper metric handling mode
                    "metric_type": "classif_score",
                    # the target class name (note: dataset-specific)
                    "target_name": "accept",
                    # the goal type of the external metric
                    "goal": "max"
                }
            },
            # ...
        }
        # ...

    Attributes:
        metric_goal: goal of the external metric, used for monitoring. Can be ``min``, ``max``, or ``None``.
        metric_type: handling mode of the external metric. Can only be one of the predetermined values.
        metric: type of the external metric that will be instantiated when ``eval`` is called.
        metric_params: dictionary of parameters passed to the external metric on instantiation.
        target_name: name of the targeted label. Used only in handling modes related to classification.
        target_idx: index of the targeted label. Used only in handling modes related to classification.
        class_names: holds the list of class label names provided by the dataset parser. If it is not
            provided when the constructor is called, it will be set by the trainer at runtime.
        force_softmax: specifies whether a softmax operation should be applied to the prediction scores
            obtained from the trainer. Only used with the "classif_score" handling mode.
        max_accum: if using a moving average, this is the window size to use (default=None).
        pred: queue used to store predictions-related values for window-based averaging.
        gt: queue used to store groundtruth-related values for window-based averaging.
    """

    def __init__(self, metric_name, metric_type, metric_params=None, target_name=None,
                 goal=None, class_names=None, max_accum=None, force_softmax=True):
        """Receives all necessary arguments for wrapper initialization and external metric instantiation.

        See :class:`thelper.optim.metrics.ExternalMetric` for information on arguments.
        """
        if not isinstance(metric_name, str):
            raise AssertionError("metric_name must be fully qualifiied class name to import")
        if metric_params is not None and not isinstance(metric_params, (list, dict)):
            raise AssertionError("metric_params must be dictionary")
        supported_handling_types = [
            "classif_top1", "classif_best",  # the former is for backwards-compat with the latter
            "classif_scores", "classif_score",  # the former is for backwards-compat with the latter
            "regression",  # missing impl, work in progress
        ]
        if not isinstance(metric_type, str) or metric_type not in supported_handling_types:
            raise AssertionError("unknown metric type '%s'" % str(metric_type))
        if metric_type == "classif_top1":
            metric_type = "classif_best"  # they are identical, just overwrite for backwards compat
        if metric_type == "classif_scores":
            metric_type = "classif_score"  # they are identical, just overwrite for backwards compat
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
        self.metric_params = metric_params if metric_params is not None else {}
        if "classif" in metric_type:
            self.target_name = target_name
            self.target_idx = None
            self.class_names = None
            if class_names is not None:
                self.set_class_names(class_names)
            if metric_type == "classif_score":
                self.force_softmax = force_softmax  # only useful in this case
        # elif "regression" in metric_type: missing impl for custom handling
        self.max_accum = max_accum
        self.pred = deque()
        self.gt = deque()

    def set_class_names(self, class_names):
        """Sets the class label names that must be predicted by the model.

        This is only useful in metric handling modes related to classification. The goal of having
        class names here is to translate a target class label (provided in the constructor) into a
        target class index. This is required as predictions are not mapped to their original names
        (in string format) before being forwarded to this object by the trainer.

        The current implementation of :class:`thelper.train.base.Trainer` will automatically call
        this function at runtime if it is available, and provide the dataset's classes as a list of
        strings.
        """
        if "classif" in self.metric_type:
            if not isinstance(class_names, list):
                raise AssertionError("expected list for class names")
            if len(class_names) < 2:
                raise AssertionError("not enough classes in provided class list")
            if self.target_name is not None:
                if self.target_name not in class_names:
                    raise AssertionError("could not find target name '%s' in class names list" % str(self.target_name))
                self.target_idx = class_names.index(self.target_name)
            self.class_names = class_names
        else:
            raise AssertionError("unexpected class list with metric type other than classif")

    def accumulate(self, pred, gt, meta=None):
        """Receives the latest prediction and groundtruth tensors from the training session.

        The handling of the data received here will depend on the current handling mode.

        Args:
            pred: model prediction tensor forwarded by the trainer.
            gt: groundtruth tensor forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata tensors forwarded by the trainer.
        """
        if "classif" in self.metric_type:
            if gt is None:
                return  # only accumulating results when groundtruth available
            if self.target_name is not None and self.target_idx is None:
                raise AssertionError("could not map target name '%s' to target idx, missing class list" % self.target_name)
            elif self.target_idx is not None:
                pred_label = pred.topk(1, 1)[1].view(len(gt))
                y_true, y_pred = [], []
                if self.metric_type == "classif_best":
                    must_keep = [y_pred == self.target_idx or y_true == self.target_idx for y_pred, y_true in zip(pred_label, gt)]
                    for idx in range(len(must_keep)):
                        if must_keep[idx]:
                            y_true.append(gt[idx].item() == self.target_idx)
                            y_pred.append(pred_label[idx].item() == self.target_idx)
                else:  # self.metric_type == "classif_score"
                    if self.force_softmax:
                        with torch.no_grad():
                            pred = torch.nn.functional.softmax(pred, dim=1)
                    for idx in range(len(gt)):
                        y_true.append(gt[idx].item() == self.target_idx)
                        y_pred.append(pred[idx, self.target_idx].item())
                self.gt.append(y_true)
                self.pred.append(y_pred)
            else:
                if self.metric_type == "classif_best":
                    self.gt.append([gt[idx].item() for idx in range(len(pred.numel()))])
                    self.pred.append([pred[idx].item() for idx in range(len(pred.numel()))])
                else:  # self.metric_type == "classif_score"
                    raise AssertionError("score-based classification analyses (e.g. roc auc) must specify target label")
        elif self.metric_type == "regression":
            raise NotImplementedError
        else:
            raise AssertionError("unknown metric type '%s'" % str(self.metric_type))
        while self.max_accum and len(self.gt) > self.max_accum:
            self.gt.popleft()
            self.pred.popleft()

    def eval(self):
        """Returns the external metric's evaluation result."""
        if "classif" in self.metric_type:
            y_gt = [gt for gts in self.gt for gt in gts]
            y_pred = [pred for preds in self.pred for pred in preds]
            if len(y_gt) != len(y_pred):
                raise AssertionError("list flattening failed")
            if isinstance(self.metric_params, list):
                return self.metric(y_gt, y_pred, *self.metric_params)
            elif isinstance(self.metric_params, dict):
                return self.metric(y_gt, y_pred, **self.metric_params)
            else:
                return self.metric(y_gt, y_pred, self.metric_params)
        else:
            raise NotImplementedError

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying pred/gt queues."""
        self.gt = deque()
        self.pred = deque()

    def needs_reset(self):
        """If the metric is currently operating in moving average mode, then it does not need to
        be reset (returns ``False``); else returns ``True``."""
        return self.max_accum is None

    def set_max_accum(self, max_accum):
        """Sets the moving average window size.

        This is fairly useful as the total size of the training dataset is unlikely to be known when
        metrics are instantiated. The current implementation of :class:`thelper.train.base.Trainer`
        will look for this member function and automatically call it with the dataset size when it is
        available.
        """
        self.max_accum = max_accum

    def goal(self):
        """Returns the scalar optimization goal of this metric (user-defined)."""
        return self.metric_goal


class ClassifReport(Metric):
    """Classification report interface.

    This class provides a simple interface to ``sklearn.metrics.classification_report`` so that all
    count-based metrics can be reported at once under a string-based representation. Note that since
    the evaluation result is a string, this metric cannot be used to directly monitor training
    progression, and thus returns ``None`` in :func:`thelper.optim.metrics.ClassifReport.goal`.

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "classifreport": {
                # this type is used to instantiate the classification report metric
                "type": "thelper.optim.metrics.ClassifReport",
                # we do not need to provide any parameters to the constructor, defaults are fine
                "params": {}
            },
            # ...
        }
        # ...

    Attributes:
        report: report generator function, called at evaluation time to generate the output string.
        class_names: holds the list of class label names provided by the dataset parser. If it is not
            provided when the constructor is called, it will be set by the trainer at runtime.
        pred: queue used to store the top-1 (best) predicted class indices at each iteration.
        gt: queue used to store the groundtruth class indices at each iteration.
    """

    def __init__(self, class_names=None, sample_weight=None, digits=4):
        """Receives the optional class names and arguments passed to the report generator function.

        Args:
            class_names: holds the list of class label names provided by the dataset parser. If it is not
                provided when the constructor is called, it will be set by the trainer at runtime.
            sample_weight: sample weights, forwarded to ``sklearn.metrics.classification_report``.
            digits: metrics output digit count, forwarded to ``sklearn.metrics.classification_report``.
        """

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
        """Sets the class label names that must be predicted by the model.

        The current implementation of :class:`thelper.train.base.Trainer` will automatically
        call this function at runtime if it is available, and provide the dataset's classes as a
        list of strings.
        """
        if class_names and not isinstance(class_names, list):
            raise AssertionError("expected class names to be list")
        if len(class_names) < 2:
            raise AssertionError("class list should have at least two elements")
        self.class_names = class_names

    def accumulate(self, pred, gt, meta=None):
        """Receives the latest class prediction and groundtruth labels from the training session.

        Args:
            pred: model class predictions forwarded by the trainer (in ``torch.Tensor`` format).
            gt: groundtruth labels forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata forwarded by the trainer (unused).
        """
        if gt is None:
            return  # only accumulating results when groundtruth available
        if self.pred is None:
            self.pred = pred.topk(1, 1)[1].view(len(gt))
            self.gt = gt.view(len(gt)).clone()
        else:
            self.pred = torch.cat((self.pred, pred.topk(1, 1)[1].view(len(gt))), 0)
            self.gt = torch.cat((self.gt, gt.view(len(gt))), 0)

    def eval(self):
        """Returns the classification report as a multi-line print-friendly string."""
        if self.pred is None:
            return "<UNAVAILABLE>"
        return self.report(self.gt.numpy(), self.pred.numpy(), self.class_names)

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying queues."""
        self.pred = None
        self.gt = None

    def goal(self):
        """Returns ``None``, as this class should not be used to directly monitor the training progress."""
        return None


class ConfusionMatrix(Metric):
    """Confusion matrix report interface.

    This class provides a simple interface to ``sklearn.metrics.confusion_matrix`` so that a full
    confusion matrix can be easily reported under a string-based representation. Note that since
    the evaluation result is a string, this metric cannot be used to directly monitor training
    progression, and thus returns ``None`` in :func:`thelper.optim.metrics.ConfusionMatrix.goal`.

    It also offers a tensorboardX-compatible output image that can be saved locally or posted to
    tensorboard for browser-based visualization.

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "confmat": {
                # this type is used to instantiate the confusion matrix report metric
                "type": "thelper.optim.metrics.ConfusionMatrix",
                # we do not need to provide any parameters to the constructor, defaults are fine
                "params": {}
            },
            # ...
        }
        # ...

    Attributes:
        matrix: report generator function, called at evaluation time to generate the output string.
        class_names: holds the list of class label names provided by the dataset parser. If it is not
            provided when the constructor is called, it will be set by the trainer at runtime.
        draw_normalized: defines whether rendered confusion matrices should be normalized or not.
        pred: queue used to store the top-1 (best) predicted class indices at each iteration.
        gt: queue used to store the groundtruth class indices at each iteration.
    """

    def __init__(self, class_names=None, draw_normalized=True):
        """Receives the optional class label names used to decorate the output string.

        Args:
            class_names: holds the list of class label names provided by the dataset parser. If it is not
                provided when the constructor is called, it will be set by the trainer at runtime.
            draw_normalized: defines whether rendered confusion matrices should be normalized or not.
        """

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
        self.draw_normalized = draw_normalized

    def set_class_names(self, class_names):
        """Sets the class label names that must be predicted by the model.

        The current implementation of :class:`thelper.train.base.Trainer` will automatically
        call this function at runtime if it is available, and provide the dataset's classes as a
        list of strings.
        """
        if not isinstance(class_names, list):
            raise AssertionError("expected class names to be list")
        if len(class_names) < 2:
            raise AssertionError("class list should have at least two elements")
        self.class_names = copy.deepcopy(class_names)
        self.class_names.append("<unset>")

    def accumulate(self, pred, gt, meta=None):
        """Receives the latest class prediction and groundtruth labels from the training session.

        Args:
            pred: model class predictions forwarded by the trainer (in ``torch.Tensor`` format).
            gt: groundtruth labels forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata forwarded by the trainer (unused).
        """
        if gt is None:
            return  # only accumulating results when groundtruth available
        if self.pred is None:
            self.pred = pred.topk(1, 1)[1].view(len(gt))
            self.gt = gt.view(len(gt)).clone()
        else:
            self.pred = torch.cat((self.pred, pred.topk(1, 1)[1].view(len(gt))), 0)
            self.gt = torch.cat((self.gt, gt.view(len(gt))), 0)

    def eval(self):
        """Returns the confusion matrix as a multi-line print-friendly string."""
        if self.pred is None:
            return "<UNAVAILABLE>"
        confmat = self.matrix(self.gt.numpy(), self.pred.numpy(), self.class_names)
        if self.class_names:
            return "\n" + thelper.utils.stringify_confmat(confmat, self.class_names)
        else:
            return "\n" + str(confmat)

    def render(self):
        """Returns the confusion matrix as a numpy-compatible RGBA image drawn by pyplot."""
        if self.pred is None:
            return None
        confmat = self.matrix(self.gt.numpy(), self.pred.numpy(), self.class_names)
        if self.class_names:
            try:
                fig = thelper.utils.draw_confmat(confmat, self.class_names, normalize=self.draw_normalized)
                array = thelper.utils.fig2array(fig)
                return array
            except AttributeError:
                logger.warning("failed to render confusion matrix figure (caught exception)")
                # return None if rendering fails (probably due to matplotlib on displayless server)
                return None
        else:
            raise NotImplementedError

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying queues."""
        self.pred = None
        self.gt = None

    def goal(self):
        """Returns ``None``, as this class should not be used to directly monitor the training progress."""
        return None


class ROCCurve(Metric):
    """Receiver operating characteristic curve computation interface.

    This class provides an interface to ``sklearn.metrics.roc_curve`` and ``sklearn.metrics.roc_auc_score``
    that can produce various types of ROC-related information including the area under the curve (AUC), the
    false positive and negative rates for various operating points, the ROC curve itself as an image (also
    compatible with tensorboardX), and CSV files containing the metadata of badly predicted samples.

    By default, evaluating this metric returns a print-friendly string containing the AUC score. If a target
    operating point is set, it will instead return the false positive/negative prediction rate of the model
    at that point (also as a print-friendly string). Since this evaluation result is not a scalar, this metric
    cannot be directly used to monitor the progression of a model during a training session, and thus returns
    ``None`` in :func:`thelper.optim.metrics.ROCCurve.goal`.

    Usage examples inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the first example; it will output the AUC of the "reject" class
            "roc_reject_auc": {
                # this type is used to instantiate the ROC metric
                "type": "thelper.optim.metrics.ROCCurve",
                # these parameters are passed to the constructor
                "params": {
                    # the name of the class to evaluate
                    "target_name": "reject"
                }
            },
            # this is the name of the second example; it will output the FPR at TPR=0.99
            "roc_reject_0.99tpr": {
                # this type is used to instantiate the ROC metric
                "type": "thelper.optim.metrics.ROCCurve",
                # these parameters are passed to the constructor
                "params": {
                    # the name of the class to evaluate
                    "target_name": "reject",
                    # the target true positive rate (TPR) operating point
                    "target_tpr": 0.99
                }
            },
            # ...
        }
        # ...

    Attributes:
        target_inv: used to target all classes except the named one(s); experimental!
        target_name: name of targeted class to generate the roc curve/auc information for.
        target_tpr: target operating point in terms of true positive rate (provided in constructor).
        target_fpr: target operating point in terms of false positive rate (provided in constructor).
        target_idx: index of the targeted class, mapped from target_name using the class_names list.
        class_names: holds the list of class label names provided by the dataset parser. If it is not
            provided when the constructor is called, it will be set by the trainer at runtime.
        force_softmax: specifies whether a softmax operation should be applied to the prediction scores
            obtained from the trainer.
        log_params: dictionary of extra parameters used to control the logging of bad predictions.
        log_fpr_threshold: used to deduce the fpr operating threshold to use for logging bad predictions.
        log_tpr_threshold: used to deduce the tpr operating threshold to use for logging bad predictions.
        log_meta_keys: list of metadata fields to copy in the log for each bad prediction.
        curve: roc curve generator function, called at evaluation time to generate the output string.
        auc: auc score generator function, called at evaluation time to generate the output string.
        score: queue used to store prediction score values for window-based averaging.
        true: queue used to store groundtruth label values for window-based averaging.
        meta: dictionary of metadata queues used for logging.
    """

    def __init__(self, target_name, target_tpr=None, target_fpr=None, class_names=None,
                 force_softmax=True, log_params=None, sample_weight=None, drop_intermediate=True):
        """Receives the target class/operating point info, log parameters, and roc computation arguments.

        Args:
            target_name: name of targeted class to generate the roc curve/auc information for.
            target_tpr: target operating point in terms of true positive rate (provided in constructor).
            target_fpr: target operating point in terms of false positive rate (provided in constructor).
            class_names: holds the list of class label names provided by the dataset parser. If it is not
                provided when the constructor is called, it will be set by the trainer at runtime.
            force_softmax: specifies whether a softmax operation should be applied to the prediction scores
                obtained from the trainer.
            log_params: dictionary of extra parameters used to control the logging of bad predictions.
            sample_weight: passed to ``sklearn.metrics.roc_curve`` and ``sklearn.metrics.roc_auc_score``.
            drop_intermediate: passed to ``sklearn.metrics.roc_curve``.
        """
        if target_name is None:
            raise AssertionError("must provide a target (class) name for ROC metric")
        self.target_inv = False
        if isinstance(target_name, str) and target_name[0] == "!":
            self.target_inv = True
            self.target_name = target_name.split("!", 1)[1]
        else:
            self.target_name = target_name
        self.target_tpr, self.target_fpr = None, None
        if target_tpr is not None and target_fpr is not None:
            raise AssertionError("must specify only one of target_fpr and target_tpr, not both")
        if target_tpr is not None or target_fpr is not None:
            target_xpr = target_tpr if target_tpr is not None else target_fpr
            if not isinstance(target_xpr, float):
                raise AssertionError("expected float type for target operating point")
            if target_xpr < 0 or target_xpr > 1:
                raise AssertionError("invalid target operation point value (must be in [0,1])")
            if target_tpr is not None:
                self.target_tpr = target_tpr
            if target_fpr is not None:
                self.target_fpr = target_fpr
        self.target_idx = None
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

        def gen_curve(y_true, y_score, _target_idx, _target_inv, _sample_weight=sample_weight, _drop_intermediate=drop_intermediate):
            if _target_idx is None:
                raise AssertionError("missing positive target idx at run time")
            _y_true, _y_score = [], []
            for sample_idx, label_idx in enumerate(y_true):
                _y_true.append(label_idx != _target_idx if _target_inv else label_idx == _target_idx)
                _y_score.append(1 - y_score[sample_idx, _target_idx] if _target_inv else y_score[sample_idx, _target_idx])
            res = sklearn.metrics.roc_curve(_y_true, _y_score, sample_weight=_sample_weight, drop_intermediate=_drop_intermediate)
            return res

        def gen_auc(y_true, y_score, _target_idx, _target_inv, _sample_weight=sample_weight):
            if _target_idx is None:
                raise AssertionError("missing positive target idx at run time")
            _y_true, _y_score = [], []
            for sample_idx, label_idx in enumerate(y_true):
                _y_true.append(label_idx != _target_idx if _target_inv else label_idx == _target_idx)
                _y_score.append(1 - y_score[sample_idx, _target_idx] if _target_inv else y_score[sample_idx, _target_idx])
            res = sklearn.metrics.roc_auc_score(_y_true, _y_score, sample_weight=_sample_weight)
            return res

        self.curve = gen_curve
        self.auc = gen_auc
        self.score = None
        self.true = None
        self.meta = None  # needed if outputting tbx txt

    def set_class_names(self, class_names):
        """Sets the class label names that must be predicted by the model.

        This allows the target class name to be mapped to a target class index.

        The current implementation of :class:`thelper.train.base.Trainer` will automatically
        call this function at runtime if it is available, and provide the dataset's classes as a
        list of strings.
        """
        if not isinstance(class_names, list):
            raise AssertionError("expected list for class names")
        if len(class_names) < 2:
            raise AssertionError("not enough classes in provided class list")
        if self.target_name not in class_names:
            raise AssertionError("could not find target name '%s' in class names list" % str(self.target_name))
        self.target_idx = class_names.index(self.target_name)
        self.class_names = class_names

    def accumulate(self, pred, gt, meta=None):
        """Receives the latest prediction scores and groundtruth label indices from the trainer.

        Args:
            pred: class prediction scores forwarded by the trainer.
            gt: groundtruth labels forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata tensors forwarded by the trainer (used for logging, if activated).
        """
        if gt is None:
            return  # only accumulating results when groundtruth available
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
            if meta is None:
                raise AssertionError("sample metadata is required, logging is activated")
            _meta = {key: meta[key] if key in meta else None for key in self.log_meta_keys}
            if self.meta is None:
                self.meta = copy.deepcopy(_meta)
            else:
                for key in self.log_meta_keys:
                    if isinstance(_meta[key], list):
                        self.meta[key] += _meta[key]
                    elif isinstance(_meta[key], torch.Tensor):
                        self.meta[key] = torch.cat((self.meta[key], _meta[key]), 0)
                    elif not (_meta[key] is None and self.meta[key] is None):
                        raise AssertionError("missing impl for meta concat w/ type '%s'" % str(type(_meta[key])))

    def eval(self):
        """Returns a print-friendly string containing the evaluation result (AUC/TPR/FPR).

        If no target operating point is set, the returned string contains the AUC for the target class. If a
        target TPR is set, the returned string contains the FPR for that operating point. If a target FPR is set,
        the returned string contains the TPR for that operating point.
        """
        if self.score is None:
            return "<UNAVAILABLE>"
        # if we did not specify a target operating point in terms of true/false positive rate, return AUC
        if self.target_tpr is None and self.target_fpr is None:
            return "AUC = %.5f" % self.auc(self.true.numpy(), self.score.numpy(), self.target_idx, self.target_inv)
        # otherwise, find the opposite rate at the requested target operating point
        _fpr, _tpr, _thrs = self.curve(self.true.numpy(), self.score.numpy(), self.target_idx, self.target_inv, _drop_intermediate=False)
        for fpr, tpr, thrs in zip(_fpr, _tpr, _thrs):
            if self.target_tpr is not None and tpr >= self.target_tpr:
                return "for target tpr = %.5f, fpr = %.5f at threshold = %f" % (self.target_tpr, fpr, thrs)
            elif self.target_fpr is not None and fpr >= self.target_fpr:
                return "for target fpr = %.5f, tpr = %.5f at threshold = %f" % (self.target_fpr, tpr, thrs)
        # if we did not find a proper rate match above, return worse possible value
        if self.target_tpr is not None:
            return "for target tpr = %.5f, fpr = 1.0 at threshold = min" % self.target_tpr
        elif self.target_fpr is not None:
            return "for target fpr = %.5f, tpr = 0.0 at threshold = max" % self.target_fpr

    def render(self):
        """Returns the ROC curve as a numpy-compatible RGBA image drawn by pyplot."""
        if self.score is None:
            return None
        fpr, tpr, t = self.curve(self.true.numpy(), self.score.numpy(), self.target_idx, self.target_inv)
        try:
            fig = thelper.utils.draw_roc_curve(fpr, tpr)
            array = thelper.utils.fig2array(fig)
            return array
        except AttributeError:
            logger.warning("failed to render confusion matrix figure (caught exception)")
            # return None if rendering fails (probably due to matplotlib on displayless server)
            return None

    def print(self):
        """Returns the logged metadata of badly predicted samples if logging is activated, and ``None`` otherwise.

        The returned object is a print-friendly CSV string that can be consumed directly by tensorboardX. Note
        that this string might be very long if the dataset is large (i.e. it will contain one line per sample).
        """
        if self.log_params is None:
            return None  # do not generate log text unless requested
        if self.meta is None or not self.meta or self.score is None:
            return None
        if self.class_names is None or not self.class_names:
            raise AssertionError("missing class list for logging, current impl only supports named outputs")
        _fpr, _tpr, _t = self.curve(self.true.numpy(), self.score.numpy(), self.target_idx, self.target_inv, _drop_intermediate=False)
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
        res = "sample_idx,gt_label_idx,gt_label_name,gt_label_score,pred_label_idx,pred_label_name,pred_label_score"
        for key in self.log_meta_keys:
            res += "," + str(key)
        res += "\n"
        for sample_idx in range(self.true.numel()):
            gt_label_idx = self.true[sample_idx].item()
            scores = self.score[sample_idx, :].tolist()
            gt_label_score = scores[gt_label_idx]
            if (self.target_inv and gt_label_idx != self.target_idx and (1 - gt_label_score) <= threshold) or \
               (not self.target_inv and gt_label_idx == self.target_idx and gt_label_score <= threshold):
                pred_label_score = max(scores)
                pred_label_idx = scores.index(pred_label_score)
                res += "{},{},{},{:2.4f},{},{},{:2.4f}".format(
                    sample_idx,
                    gt_label_idx,
                    self.class_names[gt_label_idx],
                    gt_label_score,
                    pred_label_idx,
                    self.class_names[pred_label_idx],
                    pred_label_score,
                )
                for key in self.log_meta_keys:
                    val = None
                    if key in self.meta and self.meta[key] is not None:
                        val = self.meta[key][sample_idx]
                    if isinstance(val, torch.Tensor) and val.numel() == 1:
                        res += "," + str(val.item())
                    else:
                        res += "," + str(val)
                res += "\n"
        return res

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying queues."""
        self.score = None
        self.true = None
        self.meta = None

    def goal(self):
        """Returns ``None``, as this class should not be used to directly monitor the training progress."""
        return None


class ClassifLogger(Metric):
    """Classification output logger.

    This class provides a simple logging interface for accumulating and saving the predictions of a classifier.
    Note that since the evaluation result is always ``None``, this metric cannot be used to directly monitor
    training progression, and thus also returns ``None`` in :func:`thelper.optim.metrics.ClassifLogger.goal`.

    It also optionally offers tensorboardX-compatible output images that can be saved locally or posted to
    tensorboard for browser-based visualization.

    Usage examples inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "logger": {
                # this type is used to instantiate the confusion matrix report metric
                "type": "thelper.optim.metrics.ClassifLogger",
                "params": {
                    # log the three 'best' predictions for each sample
                    "top_k": 3,
                    # keep updating a set of 10 samples for visualization via tensorboardX
                    "viz_count": 10
                }
            },
            # ...
        }
        # ...

    Attributes:
        top_k: number of 'best' predictions to keep for each sample (along with the gt label).
        class_names: holds the list of class label names provided by the dataset parser. If it is not
            provided when the constructor is called, it will be set by the trainer at runtime.
        viz_count: number of tensorboardX images to generate and update at each epoch.
        meta_keys: list of metadata fields to copy in the log for each prediction.
        force_softmax: specifies whether a softmax operation should be applied to the prediction scores
            obtained from the trainer.
        score: queue used to store prediction score values for logging.
        true: queue used to store groundtruth label values for logging.
        meta: dictionary of metadata queues used for logging.
    """

    def __init__(self, top_k=1, class_names=None, viz_count=0, meta_keys=None, force_softmax=True):
        """Receives the logging parameters & the optional class label names used to decorate the log.

        Args:
            top_k: number of 'best' predictions to keep for each sample (along with the gt label).
            class_names: holds the list of class label names provided by the dataset parser. If it is not
                provided when the constructor is called, it will be set by the trainer at runtime.
            viz_count: number of tensorboardX images to generate and update at each epoch.
            meta_keys: list of metadata fields to copy in the log for each prediction.
            force_softmax: specifies whether a softmax operation should be applied to the prediction scores
                obtained from the trainer.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            raise AssertionError("invalid top-k value")
        self.top_k = top_k
        self.class_names = None
        if class_names is not None:
            self.set_class_names(class_names)
        if not isinstance(viz_count, int) or top_k < 0:
            raise AssertionError("invalid viz_count value")
        self.viz_count = viz_count
        if meta_keys is not None and not isinstance(meta_keys, list):
            raise AssertionError("unexpected log meta keys params type (expected list)")
        self.meta_keys = meta_keys
        self.force_softmax = force_softmax
        self.score = None
        self.true = None
        self.meta = None

    def set_class_names(self, class_names):
        """Sets the class label names that must be predicted by the model.

        This allows the target class name to be mapped to a target class index.

        The current implementation of :class:`thelper.train.base.Trainer` will automatically
        call this function at runtime if it is available, and provide the dataset's classes as a
        list of strings.
        """
        if not isinstance(class_names, list):
            raise AssertionError("expected list for class names")
        if len(class_names) < 2:
            raise AssertionError("not enough classes in provided class list")
        if self.top_k > len(class_names):
            raise AssertionError(f"cannot log top-{self.top_k} predictions with only {len(class_names)} classes")
        self.class_names = class_names

    def accumulate(self, pred, gt, meta=None):
        """Receives the latest prediction scores and groundtruth label indices from the trainer.

        Args:
            pred: class prediction scores forwarded by the trainer.
            gt: groundtruth labels forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata tensors forwarded by the trainer (used for logging).
        """
        if self.force_softmax:
            with torch.no_grad():
                pred = torch.nn.functional.softmax(pred, dim=1)
        if gt is None:
            # if gt is missing, set labels as -1; will be reinterpreted when creating log
            gt = torch.from_numpy(np.full((pred.shape[0], 1), -1, dtype=np.int64))
        if self.score is None:
            self.score = pred.clone()
            self.true = gt.clone()
        else:
            self.score = torch.cat((self.score, pred), 0)
            self.true = torch.cat((self.true, gt), 0)
        if self.meta_keys is not None and self.meta_keys:
            if meta is None:
                raise AssertionError("sample metadata is required, logging is activated")
            _meta = {key: meta[key] if key in meta else None for key in self.meta_keys}
            if self.meta is None:
                self.meta = copy.deepcopy(_meta)
            else:
                for key in self.meta_keys:
                    if isinstance(_meta[key], list):
                        self.meta[key] += _meta[key]
                    elif isinstance(_meta[key], torch.Tensor):
                        self.meta[key] = torch.cat((self.meta[key], _meta[key]), 0)
                    else:
                        raise AssertionError("missing impl for meta concat w/ type '%s'" % str(type(_meta[key])))

    def eval(self):
        """Returns ``None``, as this class only produces log files in the session directory."""
        return None

    def render(self):
        """Returns an image of predicted outputs as a numpy-compatible RGBA image drawn by pyplot."""
        if self.viz_count == 0:
            return None
        raise NotImplementedError  # TODO

    def print(self):
        """Returns the logged metadata of predicted samples.

        The returned object is a print-friendly CSV string that can be consumed directly by tensorboardX. Note
        that this string might be very long if the dataset is large (i.e. it will contain one line per sample).
        """
        if self.score is None:
            return None
        if self.class_names is None or not self.class_names:
            raise AssertionError("missing class list for logging, current impl only supports named outputs")
        res = "sample_idx,gt_label_idx,gt_label_name,gt_label_score"
        for k in range(self.top_k):
            res += ",pred_label_idx_%d,pred_label_name_%d,pred_label_score_%d" % (k, k, k)
        if self.meta_keys is not None and self.meta_keys:
            for key in self.meta_keys:
                res += "," + str(key)
        res += "\n"
        for sample_idx in range(self.true.numel()):
            gt_label_idx = self.true[sample_idx].item()
            scores = np.asarray(self.score[sample_idx, :].tolist())
            sorted_score_idxs = np.argsort(scores)[::-1]
            sorted_scores = scores[sorted_score_idxs]
            res += "{},{},{},{:2.4f}".format(
                sample_idx,
                gt_label_idx,
                self.class_names[gt_label_idx] if gt_label_idx >= 0 else "n/a",
                scores[gt_label_idx] if gt_label_idx >= 0 else 0
            )
            for k in range(self.top_k):
                res += ",{},{},{:2.4f}".format(
                    sorted_score_idxs[k],
                    self.class_names[sorted_score_idxs[k]],
                    sorted_scores[k]
                )
            if self.meta_keys is not None and self.meta_keys:
                for key in self.meta_keys:
                    val = None
                    if key in self.meta and self.meta[key] is not None:
                        val = self.meta[key][sample_idx]
                    if isinstance(val, torch.Tensor) and val.numel() == 1:
                        res += "," + str(val.item())
                    else:
                        res += "," + str(val)
            res += "\n"
        return res

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying queues."""
        self.score = None
        self.true = None
        self.meta = None

    def goal(self):
        """Returns ``None``, as this class should not be used to directly monitor the training progress."""
        return None


class RawPredictions(Metric):
    """Raw predictions storage.

    This class provides a simple interface for accumulating and saving the raw predictions of a classifier.
    Note that since the evaluation result is always ``None``, this metric cannot be used to directly monitor
    training progression, and thus also returns ``None`` in :func:`thelper.optim.metrics.ClassifLogger.goal`.

    It also optionally offers a callback functionality on each accumulated prediction to execute additional
    operations (ex: call a function to update external processes from ``thelper`` package).

    Usage examples inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "predictions": {
                # this type is used to instantiate the confusion matrix report metric
                "type": "thelper.optim.metrics.RawPredictions",
                "params": [
                    # call 'my_function' located within 'external_module' after each added prediction
                    {"name": "callback", "value": "external_module.my_function"},
                ]
            },
            # ...
        }
        # ...

    Attributes:
        callback: callable to be executed after each accumulated prediction
    """

    def __init__(self, callback=None):
        # type: (Optional[Callable]) -> None
        super().__init__()
        self.predictions = list()   # type: List[Dict[AnyStr, Any]]
        if callback is not None and not callable(callback):
            raise TypeError("Callback is not callable, got {!s}.".format(type(callback)))
        self.callback = callback or (lambda *args, **kwargs: None)  # do nothing if None to simplify calls

    @staticmethod
    def _to_py(element):
        if isinstance(element, torch.Tensor):
            return element.tolist()
        return element

    def accumulate(self, pred, gt, meta=None):
        """Receives the latest prediction and groundtruth tensors (each batch) from the session.

        Args:
            pred: model prediction tensor forwarded by the trainer for a given sample batch.
            gt: groundtruth tensor forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata tensors forwarded by the trainer.
        """

        # convert batch tensor to per-sample class predictions
        samples_predictions = [{
            "predictions": self._to_py(p),
        } for p in pred]

        # convert ground truths to per-sample labels
        for i, label in enumerate(self._to_py(gt)):
            samples_predictions[i]["gt"] = label

        # transfer meta information to corresponding samples according to available details
        n_samples = len(samples_predictions)
        if isinstance(meta, dict):
            for meta_key in meta:
                if isinstance(meta[meta_key], list):
                    n_meta = len(meta[meta_key])
                    # list of tensors where each index in every tensor corresponds to a sample index
                    if n_meta != n_samples:
                        meta_info = [self._to_py(t) for t in meta[meta_key]]
                        for j, s in enumerate(samples_predictions):
                            # transfer corresponding sample index info to matching tensor prediction,
                            # or transfer whole meta info if index correspondence cannot be established
                            s[meta_key] = [meta_info[i][j] if hasattr(meta_info[i], '__len__') else meta_info[i]
                                           for i in range(len(meta_info))]
                    # list of elements or tensor of samples size
                    elif n_meta == n_samples:
                        meta_info = self._to_py(meta[meta_key])
                        for i, s in enumerate(samples_predictions):
                            s[meta_key] = meta_info[i]
                else:
                    # each sample gets the same info
                    for s in samples_predictions:
                        s[meta_key] = meta[meta_key]

        self.predictions.extend(samples_predictions)
        self.callback()

    def reset(self):
        self.__init__(callback=self.callback)

    def eval(self):
        """
        Returns the raw predictions as received and accumulated through batch iterations.
        Indices of predictions match the order in which samples where received during ``accumulate`` calls.
        """
        return self.predictions

    def goal(self):
        """Returns ``None``, as this class should not be used to directly monitor the training progress."""
        return None


class PSNR(Metric):
    r"""Peak Signal-to-Noise Ratio (PSNR) metric interface.

    This is a scalar metric used to monitor the change in quality of a signal (or image) following a
    transformation. For more information, see its definition on `[Wikipedia]`__.

    .. __: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    The PSNR (in decibels, dB) between a modified signal :math:`x` and its original version :math:`y` is
    defined as:

    .. math::
        \text{PSNR}(x, y) = 10 * \log_{10} \Bigg( \frac{R^2}{\text{MSE}(x, y)} \Bigg)

    where :math:`\text{MSE}(x, y)` returns the mean squared error (see :class:`thelper.optim.metrics.MeanSquaredError`
    for more information), and :math:`R` is the maximum possible value for a single element in the input signal
    (i.e. its maximum "range").

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "psnr": {
                # this type is used to instantiate the metric
                "type": "thelper.optim.metrics.PSNR",
                "params": {
                    "data_range": "255"
                }
            },
            # ...
        }
        # ...

    Attributes:
        max_accum: if using a moving average, this is the window size to use (default=None).
        data_range: maximum value of an element in the analyzed signal.
        decompose_batch: toggles whether batches should be decomposed along 0-th dim or not.
    """

    def __init__(self, max_accum=None, data_range=1.0, decompose_batch=True):
        """Receives all necessary initialization arguments to compute signal PSNRs,

        See :class:`thelper.optim.metrics.PSNR` for information on arguments.
        """
        self.max_accum = max_accum
        self.psnrs = deque()  # will contain lists to avoid merging batch PSNRs early
        self.warned_eval_bad = False
        self.data_range = data_range
        self.decompose_batch = decompose_batch

    def accumulate(self, pred, target, meta=None):
        """Receives the latest predictions and target values from the training session.

        The inputs are expected to still be in ``torch.Tensor`` format, but must be located on the
        CPU. This function computes and accumulate the PSNR value in a queue, popping it if the maximum
        window length is reached.

        Args:
            pred: model prediction values forwarded by the trainer.
            target: target prediction values forwarded by the trainer (can be ``None`` if unavailable).
            meta: metadata forwarded by the trainer (unused).
        """
        if target is None or target.numel() == 0:
            return  # only accumulating results when groundtruth available
        if pred.shape != target.shape:
            raise AssertionError("prediction/gt tensors shape mismatch")
        if not self.decompose_batch:
            mse = np.mean(np.square(pred.numpy() - target.numpy()), dtype=np.float64)
            psnr = 10 * np.log10(self.data_range / mse)
            self.psnrs.append([psnr])
        else:
            pred = pred.view(pred.shape[0], -1)
            target = target.view(target.shape[0], -1)
            mse = np.mean(np.square(pred.numpy() - target.numpy()), axis=1, dtype=np.float64)
            self.psnrs.append([10 * np.log10(self.data_range / e) for e in mse])
        if self.max_accum and len(self.psnrs) > self.max_accum:
            self.psnrs.popleft()

    def eval(self):
        """Returns the current (average) PSNR based on the accumulated values.

        Will issue a warning if no predictions have been accumulated yet.
        """
        if len(self.psnrs) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("psnr eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return np.mean(np.array([psnr for psnr_list in self.psnrs for psnr in psnr_list]))

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying value queues."""
        self.psnrs = deque()

    def needs_reset(self):
        """If the metric is currently operating in moving average mode, then it does not need to
        be reset (returns ``False``); else returns ``True``."""
        return self.max_accum is None

    def set_max_accum(self, max_accum):
        """Sets the moving average window size.

        This is fairly useful as the total size of the training dataset is unlikely to be known when
        metrics are instantiated. The current implementation of :class:`thelper.train.base.Trainer`
        will look for this member function and automatically call it with the dataset size when it is
        available.
        """
        self.max_accum = max_accum

    def goal(self):
        """Returns the scalar optimization goal of this metric (maximization)."""
        return Metric.maximize
