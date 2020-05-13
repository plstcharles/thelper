"""Metrics module.

This module contains classes that implement metrics used to monitor training sessions and evaluate models.
These metrics should all inherit from :class:`thelper.optim.metrics.Metric` to allow them to be dynamically
instantiated by the framework from a configuration file, and evaluated automatically inside a training
session. For more information on this, refer to :class:`thelper.train.base.Trainer`.
"""

import logging
from abc import abstractmethod
from typing import Any, AnyStr, Optional  # noqa: F401

import numpy as np
import sklearn.metrics
import torch

import thelper.concepts
import thelper.utils
from thelper.ifaces import ClassNamesHandler, PredictionConsumer

logger = logging.getLogger(__name__)


class Metric(PredictionConsumer):
    """Abstract metric interface.

    This interface defines basic functions required so that :class:`thelper.train.base.Trainer` can
    figure out how to instantiate, update, and optimize a given metric while training/evaluating a model.

    All metrics, by definition, must be 'optimizable'. This means that they should return a scalar value
    when 'evaluated' and define an optimal goal (-inf or +inf). If this is not possible, then the class
    should probably be derived using the more generic :class:`thelper.ifaces.PredictionConsumer`
    instead.
    """

    minimize = float("-inf")
    """Possible value of the ``goal`` attribute of this metric."""

    maximize = float("inf")
    """Possible value of the ``goal`` attribute of this metric."""

    @abstractmethod
    def update(self,         # see `thelper.typedefs.IterCallbackParams` for more info
               task,         # type: thelper.tasks.utils.Task
               input,        # type: thelper.typedefs.InputType
               pred,         # type: thelper.typedefs.AnyPredictionType
               target,       # type: thelper.typedefs.AnyTargetType
               sample,       # type: thelper.typedefs.SampleType
               loss,         # type: Optional[float]
               iter_idx,     # type: int
               max_iters,    # type: int
               epoch_idx,    # type: int
               max_epochs,   # type: int
               output_path,  # type: AnyStr
               **kwargs,     # type: Any
               ):            # type: (...) -> None
        """Receives the latest prediction and groundtruth tensors from the training session.

        The data given here will be "consumed" internally, but it should NOT be modified. For example,
        a classification accuracy metric might accumulate the correct number of predictions in comparison
        to groundtruth labels, but never alter those predictions. The iteration/epoch indices may be
        used to 'reset' the internal state of this object when needed (for example, at the start of each
        new epoch).

        Remember that input, prediction, and target tensors received here will all have a batch dimension!

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        """Returns the metric's evaluation result.

        The returned value should be a scalar. As a model improves, this scalar should get closer
        to the optimization goal (defined through the 'goal' attribute). This value will be queried
        at the end of each training epoch by the trainer.
        """
        raise NotImplementedError

    @property
    def goal(self):
        """Returns the scalar optimization goal of the metric.

        The returned goal can be the ``minimize`` or ``maximize`` members of ``thelper.optim.metrics.Metric``
        if the class's evaluation returns a scalar value, and ``None`` otherwise. The trainer will
        check this value to see if monitoring the metric's evaluation result progression is possible.
        """
        raise NotImplementedError

    @property
    def live_eval(self):
        """Returns whether this metric can/should be evaluated at every backprop iteration or not.

        By default, this returns ``True``, but implementations that are quite slow may return ``False``.
        """
        return True


@thelper.concepts.classification
@thelper.concepts.segmentation
class Accuracy(Metric):
    r"""Classification accuracy metric interface.

    This is a scalar metric used to monitor the label prediction accuracy of a model. By default,
    it works in ``top-k`` mode, meaning that the evaluation result is given by:

    .. math::
      \text{accuracy} = \frac{\text{nb. correct predictions}}{\text{nb. total predictions}} \cdot 100

    When :math:`k>1`, a 'correct' prediction is obtained if any of the model's top :math:`k` predictions
    (i.e. the :math:`k` predictions with the highest score) match the groundtruth label. Otherwise, if
    :math:`k=1`, then only the top prediction is compared to the groundtruth label. Note that for
    binary classification problems, :math:`k` should always be set to 1.

    This metric's goal is to maximize its value :math:`\in [0,100]` (a percentage is returned).

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "top_5_accuracy": {
                # this type is used to instantiate the accuracy metric
                "type": "thelper.optim.metrics.Accuracy",
                # these parameters are passed to the wrapper's constructor
                "params": {
                    # the top prediction count to check for a match with the groundtruth
                    "top_k": 5
                }
            },
            # ...
        }
        # ...

    Todo: add support for 'dont care' target value?

    Attributes:
        top_k: number of top predictions to consider when matching with the groundtruth (default=1).
        max_win_size: maximum moving average window size to use (default=None, which equals dataset size).
        correct: total number of correct predictions stored using an array for window-based averaging.
        total: total number of predictions stored using an array for window-based averaging.
        warned_eval_bad: toggles whether the division-by-zero warning has been flagged or not.
    """

    def __init__(self, top_k=1, max_win_size=None):
        """Receives the number of predictions to consider for matches (``top_k``) and the moving average
        window size (``window_size``).

        Note that by default, if ``max_win_size`` is not provided here, the value given to ``max_iters`` on
        the first update call will be used instead to fix the sliding window length. In any case, the
        smallest of ``max_iters`` and ``max_win_size`` will be used to determine the actual window size.
        """
        assert isinstance(top_k, int) and top_k > 0, "invalid top-k value"
        assert max_win_size is None or (isinstance(max_win_size, int) and max_win_size > 0), \
            "invalid max sliding window size (should be positive integer)"
        self.top_k = top_k
        self.max_win_size = max_win_size
        self.correct = None  # will be instantiated on first iter
        self.total = None  # will be instantiated on first iter
        self.warned_eval_bad = False

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this metric."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(top_k={repr(self.top_k)}, max_win_size={repr(self.max_win_size)})"

    def update(self,         # see `thelper.typedefs.IterCallbackParams` for more info
               task,         # type: thelper.tasks.utils.Task
               input,        # type: thelper.typedefs.InputType
               pred,         # type: thelper.typedefs.ClassificationPredictionType
               target,       # type: thelper.typedefs.ClassificationTargetType
               sample,       # type: thelper.typedefs.SampleType
               loss,         # type: Optional[float]
               iter_idx,     # type: int
               max_iters,    # type: int
               epoch_idx,    # type: int
               max_epochs,   # type: int
               output_path,  # type: AnyStr
               **kwargs,     # type: Any
               ):            # type: (...) -> None
        """Receives the latest class prediction and groundtruth labels from the training session.

        This function computes and accumulate the number of correct and total predictions in
        the internal arrays, cycling over the iteration index if the maximum window length is reached.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to metric update function"
        curr_win_size = max_iters if self.max_win_size is None else min(self.max_win_size, max_iters)
        if self.correct is None or self.correct.size != curr_win_size:
            # each 'iteration' will have a corresponding bin with counts for that batch
            self.correct = np.zeros(curr_win_size, dtype=np.int64)
            self.total = np.zeros(curr_win_size, dtype=np.int64)
        curr_idx = iter_idx % curr_win_size
        if target is None or target.numel() == 0:
            # only accumulate results when groundtruth is available
            self.correct[curr_idx] = 0
            self.total[curr_idx] = 0
            return
        assert pred.dim() == target.dim() + 1, "prediction/gt tensors dim mismatch (should be BxCx[...] and Bx[...])"
        assert pred.shape[0] == target.shape[0], "prediction/gt tensors batch size mismatch"
        assert pred.dim() <= 2 or pred.shape[2:] == target.shape[1:], "prediction/gt tensors array size mismatch"
        top_k = pred.topk(self.top_k, dim=1)[1].view(pred.shape[0], self.top_k, -1).numpy()
        true_k = target.view(target.shape[0], 1, -1).expand(-1, self.top_k, -1).numpy()
        self.correct[curr_idx] = np.any(np.equal(top_k, true_k), axis=1).sum(dtype=np.int64)
        self.total[curr_idx] = target.numel()

    def eval(self):
        """Returns the current accuracy (in percentage) based on the accumulated prediction counts.

        Will issue a warning if no predictions have been accumulated yet.
        """
        if self.total is None or self.total.size == 0 or np.sum(self.total) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("category accuracy eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return (float(np.sum(self.correct)) / float(np.sum(self.total))) * 100

    def reset(self):
        """Toggles a reset of the metric's internal state, deallocating count arrays."""
        self.correct = None
        self.total = None

    @property
    def goal(self):
        """Returns the scalar optimization goal of this metric (maximization)."""
        return Metric.maximize


@thelper.concepts.regression
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
        max_win_size: maximum moving average window size to use (default=None, which equals dataset size).
        reduction: string representing the tensor reduction strategy to use.
        errors: array of error values stored for window-based averaging.
        warned_eval_bad: toggles whether the division-by-zero warning has been flagged or not.
    """

    def __init__(self, reduction="mean", max_win_size=None):
        """Receives the reduction strategy and the moving average window size (``window_size``).

        Note that by default, if ``max_win_size`` is not provided here, the value given to ``max_iters`` on
        the first update call will be used instead to fix the sliding window length. In any case, the
        smallest of ``max_iters`` and ``max_win_size`` will be used to determine the actual window size.
        """
        assert max_win_size is None or (isinstance(max_win_size, int) and max_win_size > 0), \
            "invalid max sliding window size (should be positive integer)"
        assert reduction != "none", "metric must absolutely return a scalar, must reduce"
        self.reduction = reduction
        self.max_win_size = max_win_size
        self.errors = None  # will be instantiated on first iter
        self.warned_eval_bad = False

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this metric."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(reduction={repr(self.reduction)}, max_win_size={repr(self.max_win_size)})"

    def update(self,         # see `thelper.typedefs.IterCallbackParams` for more info
               task,         # type: thelper.tasks.utils.Task
               input,        # type: thelper.typedefs.InputType
               pred,         # type: thelper.typedefs.RegressionPredictionType
               target,       # type: thelper.typedefs.RegressionTargetType
               sample,       # type: thelper.typedefs.SampleType
               loss,         # type: Optional[float]
               iter_idx,     # type: int
               max_iters,    # type: int
               epoch_idx,    # type: int
               max_epochs,   # type: int
               output_path,  # type: AnyStr
               **kwargs,     # type: Any
               ):            # type: (...) -> None
        """Receives the latest predictions and target values from the training session.

        This function computes and accumulates the L1 distance between predictions and targets in the
        internal array, cycling over the iteration index if the maximum window length is reached.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to metric update function"
        curr_win_size = max_iters if self.max_win_size is None else min(self.max_win_size, max_iters)
        if self.errors is None or self.errors.size != curr_win_size:
            # each 'iteration' will have a corresponding bin with the average L1 loss for that batch
            self.errors = np.asarray([None] * curr_win_size)
        curr_idx = iter_idx % curr_win_size
        if target is None or target.numel() == 0:
            # only accumulate results when groundtruth is available
            self.errors[curr_idx] = None
            return
        assert pred.shape == target.shape, "prediction/gt tensors shape mismatch"
        self.errors[curr_idx] = torch.nn.functional.l1_loss(pred, target, reduction=self.reduction).item()

    def eval(self):
        """Returns the current (average) mean absolute error based on the accumulated values.

        Will issue a warning if no predictions have been accumulated yet.
        """
        if self.errors is None or self.errors.size == 0 or len([d for d in self.errors if d is not None]) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("mean absolute error eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return np.mean([d for d in self.errors if d is not None])

    def reset(self):
        """Toggles a reset of the metric's internal state, deallocating the errors array."""
        self.errors = None

    @property
    def goal(self):
        """Returns the scalar optimization goal of this metric (minimization)."""
        return Metric.minimize


@thelper.concepts.regression
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
        max_win_size: maximum moving average window size to use (default=None, which equals dataset size).
        reduction: string representing the tensor reduction strategy to use.
        errors: array of error values stored for window-based averaging.
        warned_eval_bad: toggles whether the division-by-zero warning has been flagged or not.
    """

    def __init__(self, reduction="mean", max_win_size=None):
        """Receives the reduction strategy and the moving average window size (``window_size``).

        Note that by default, if ``max_win_size`` is not provided here, the value given to ``max_iters`` on
        the first update call will be used instead to fix the sliding window length. In any case, the
        smallest of ``max_iters`` and ``max_win_size`` will be used to determine the actual window size.
        """
        assert max_win_size is None or (isinstance(max_win_size, int) and max_win_size > 0), \
            "invalid max sliding window size (should be positive integer)"
        assert reduction != "none", "metric must absolutely return a scalar, must reduce"
        self.reduction = reduction
        self.max_win_size = max_win_size
        self.errors = None  # will be instantiated on first iter
        self.warned_eval_bad = False

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this metric."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(reduction={repr(self.reduction)}, max_win_size={repr(self.max_win_size)})"

    def update(self,         # see `thelper.typedefs.IterCallbackParams` for more info
               task,         # type: thelper.tasks.utils.Task
               input,        # type: thelper.typedefs.InputType
               pred,         # type: thelper.typedefs.RegressionPredictionType
               target,       # type: thelper.typedefs.RegressionTargetType
               sample,       # type: thelper.typedefs.SampleType
               loss,         # type: Optional[float]
               iter_idx,     # type: int
               max_iters,    # type: int
               epoch_idx,    # type: int
               max_epochs,   # type: int
               output_path,  # type: AnyStr
               **kwargs,     # type: Any
               ):            # type: (...) -> None
        """Receives the latest predictions and target values from the training session.

        This function computes and accumulates the mean squared error between predictions and targets in
        the internal array, cycling over the iteration index if the maximum window length is reached.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to metric update function"
        curr_win_size = max_iters if self.max_win_size is None else min(self.max_win_size, max_iters)
        if self.errors is None or self.errors.size != curr_win_size:
            # each 'iteration' will have a corresponding bin with the average MSE loss for that batch
            self.errors = np.asarray([None] * curr_win_size)
        curr_idx = iter_idx % curr_win_size
        if target is None or target.numel() == 0:
            # only accumulate results when groundtruth is available
            self.errors[curr_idx] = None
            return
        assert pred.shape == target.shape, "prediction/gt tensors shape mismatch"
        self.errors[curr_idx] = torch.nn.functional.mse_loss(pred, target, reduction=self.reduction).item()

    def eval(self):
        """Returns the current (average) mean squared error based on the accumulated values.

        Will issue a warning if no predictions have been accumulated yet.
        """
        if self.errors is None or self.errors.size == 0 or len([d for d in self.errors if d is not None]) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("mean squared error eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return np.mean([d for d in self.errors if d is not None])

    def reset(self):
        """Toggles a reset of the metric's internal state, deallocating the errors array."""
        self.errors = None

    @property
    def goal(self):
        """Returns the scalar optimization goal of this metric (minimization)."""
        return Metric.minimize


@thelper.concepts.classification
@thelper.concepts.segmentation
class ExternalMetric(Metric, ClassNamesHandler):
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
                    "metric_params": {},
                    # the wrapper metric handling mode
                    "metric_type": "classif_best",
                    # the target class name (note: dataset-specific)
                    "target_name": "reject",
                    # the goal type of the external metric
                    "metric_goal": "max"
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
                    "metric_params": {},
                    # the wrapper metric handling mode
                    "metric_type": "classif_score",
                    # the target class name (note: dataset-specific)
                    "target_name": "accept",
                    # the goal type of the external metric
                    "metric_goal": "max"
                }
            },
            # ...
        }
        # ...

    Attributes:
        metric_goal: goal of the external metric, used for monitoring. Can be ``min`` or ``max``.
        metric_type: handling mode of the external metric. Can only be one of the predetermined values.
        metric: type of the external metric that will be instantiated when ``eval`` is called.
        metric_params: dictionary of parameters passed to the external metric on instantiation.
        target_name: name of the targeted label. Used only in handling modes related to classification.
        target_idx: index of the targeted label. Used only in handling modes related to classification.
        class_names: holds the list of class label names provided by the dataset parser. If it is not
            provided when the constructor is called, it will be set by the trainer at runtime.
        force_softmax: specifies whether a softmax operation should be applied to the prediction scores
            obtained from the trainer. Only used with the "classif_score" handling mode.
        max_win_size: maximum moving average window size to use (default=None, which equals dataset size).
        pred: queue used to store predictions-related values for window-based averaging.
        target: queue used to store groundtruth-related values for window-based averaging.
    """

    def __init__(self, metric_name, metric_type, metric_goal, metric_params=None, target_name=None,
                 class_names=None, max_win_size=None, force_softmax=True, live_eval=True):
        """Receives all necessary arguments for wrapper initialization and external metric instantiation.

        See :class:`thelper.optim.metrics.ExternalMetric` for information on arguments.
        """
        assert isinstance(metric_name, str), "metric_name must be fully qualifiied class name to import"
        assert metric_params is None or isinstance(metric_params, dict), "metric_params must be dictionary"
        supported_handling_types = [
            "classif_top1", "classif_best",  # the former is for backwards-compat with the latter
            "classif_scores", "classif_score",  # the former is for backwards-compat with the latter
            "regression",  # missing impl, work in progress @@@ TODO
        ]
        assert isinstance(metric_type, str) and metric_type in supported_handling_types, \
            f"unknown metric type {repr(metric_type)}"
        if metric_type == "classif_top1":
            metric_type = "classif_best"  # they are identical, just overwrite for backwards compat
        if metric_type == "classif_scores":
            metric_type = "classif_score"  # they are identical, just overwrite for backwards compat
        assert metric_goal is not None and metric_goal in ["max", "min"], "unexpected goal type"
        self.metric_goal = Metric.maximize if metric_goal == "max" else Metric.minimize
        self.metric_type = metric_type
        self.metric_name = metric_name
        self.metric = thelper.utils.import_class(metric_name)
        self.metric_params = metric_params if metric_params is not None else {}
        self.target_name = target_name
        self.target_idx = None
        self.force_softmax = None
        if metric_type == "classif_score":
            self.force_softmax = force_softmax  # only useful in this case
        # elif "regression" in metric_type: missing impl for custom handling @@@
        assert max_win_size is None or (isinstance(max_win_size, int) and max_win_size > 0), \
            "invalid max sliding window size (should be positive integer)"
        self.max_win_size = max_win_size
        self.pred = None  # will be instantiated on first iter
        self.target = None  # will be instantiated on first iter
        self._live_eval = live_eval  # could be 'False' for external impls that are pretty slow to eval
        ClassNamesHandler.__init__(self, class_names)

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this metric."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(metric_name={repr(self.metric_name)}, metric_type={repr(self.metric_type)}, " + \
            f"metric_goal={'min' if self.goal == Metric.minimize else 'max'}, " + \
            f"metric_params={repr(self.metric_params)}, target_name={repr(self.target_name)}, " + \
            f"class_names={repr(self.class_names)}, max_win_size={repr(self.max_win_size)}, " + \
            f"force_softmax={repr(self.force_softmax)})"

    @ClassNamesHandler.class_names.setter
    def class_names(self, class_names):
        """Sets the class label names that must be predicted by the model.

        This is only useful in metric handling modes related to classification. The goal of having
        class names here is to translate a target class label (provided in the constructor) into a
        target class index. This is required as predictions are not mapped to their original names
        (in string format) before being forwarded to this object by the trainer.
        """
        if "classif" in self.metric_type:
            ClassNamesHandler.class_names.fset(self, class_names)
            if self.target_name is not None and self.class_names is not None:
                assert self.target_name in self.class_indices, \
                    f"could not find target name {repr(self.target_name)} in class names list"
                self.target_idx = self.class_indices[self.target_name]
            else:
                self.target_idx = None

    def update(self,         # see `thelper.typedefs.IterCallbackParams` for more info
               task,         # type: thelper.tasks.utils.Task
               input,        # type: thelper.typedefs.InputType
               pred,         # type: thelper.typedefs.AnyTargetType
               target,       # type: thelper.typedefs.AnyPredictionType
               sample,       # type: thelper.typedefs.SampleType
               loss,         # type: Optional[float]
               iter_idx,     # type: int
               max_iters,    # type: int
               epoch_idx,    # type: int
               max_epochs,   # type: int
               output_path,  # type: AnyStr
               **kwargs,     # type: Any
               ):            # type: (...) -> None
        """Receives the latest predictions and target values from the training session.

        The handling of the data received here will depend on the current metric's handling mode.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to metric update function"
        curr_win_size = max_iters if self.max_win_size is None else min(self.max_win_size, max_iters)
        if self.pred is None or self.pred.size != curr_win_size:
            # each 'iteration' will have a corresponding bin with counts for that batch
            self.pred = np.asarray([None] * curr_win_size)
            self.target = np.asarray([None] * curr_win_size)
        curr_idx = iter_idx % curr_win_size
        if "classif" in self.metric_type:
            if hasattr(task, "class_names") and task.class_names != self.class_names:
                self.class_names = task.class_names
            if target is None or target.numel() == 0:
                # only accumulate results when groundtruth is available
                self.pred[curr_idx] = None
                self.target[curr_idx] = None
                return
            assert self.target_name is None or self.target_idx is not None, \
                f"could not map target name '{self.target_name}' to target idx, missing class list"
            assert pred.shape[0] == target.shape[0], "prediction/gt tensors batch size mismatch"
            if self.target_idx is not None:
                y_true, y_pred = [], []
                if self.metric_type == "classif_best":
                    assert pred.dim() == 2 and target.dim() == 1, "current ext metric implementation only supports batched 1D outputs"
                    pred_label = pred.topk(1, dim=1)[1].view(pred.shape[0])
                    assert pred_label.numel() == target.numel(), "pred/target classification element count mismatch"
                    must_keep = [y_pred == self.target_idx or y_true == self.target_idx for y_pred, y_true in zip(pred_label, target)]
                    for idx, keep in enumerate(must_keep):
                        if keep:
                            y_true.append(target[idx].item() == self.target_idx)
                            y_pred.append(pred_label[idx].item() == self.target_idx)
                else:  # self.metric_type == "classif_score"
                    if self.force_softmax:
                        with torch.no_grad():
                            pred = torch.nn.functional.softmax(pred, dim=1)
                    if pred.dim() == 2 and target.dim() == 1:
                        for idx, tgt in enumerate(target):
                            y_true.append(tgt.item() == self.target_idx)
                            y_pred.append(pred[idx, self.target_idx].item())
                    else:
                        assert pred.dim() > 2 and target.dim() == pred.dim() - 1 and pred.shape[2:] == target.shape[1:]
                        y_true = (target.reshape(-1) == self.target_idx).cpu().numpy()
                        y_pred = pred[:, self.target_idx, ...].reshape(-1).cpu().numpy()
                self.target[curr_idx] = y_true
                self.pred[curr_idx] = y_pred
            else:
                assert self.metric_type != "classif_score", "score-based classif analysis (e.g. roc auc) must specify target label"
                if self.metric_type == "classif_best":
                    self.target[curr_idx] = [target[idx].item() for idx in range(pred.numel())]
                    self.pred[curr_idx] = [pred[idx].item() for idx in range(pred.numel())]
        else:  # if self.metric_type == "regression":
            raise NotImplementedError

    def eval(self):
        """Returns the external metric's evaluation result."""
        if "classif" in self.metric_type:
            assert self.target.size == self.pred.size, "internal window size mismatch"
            pred, target = zip(*[(pred, target) for preds, targets in zip(self.pred, self.target)
                                 if targets is not None for pred, target in zip(preds, targets)])
            return self.metric(np.stack(target, axis=0), np.stack(pred, axis=0), **self.metric_params)
        else:  # if self.metric_type == "regression":
            raise NotImplementedError

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying pred/target queues."""
        self.pred = None
        self.target = None

    @property
    def goal(self):
        """Returns the scalar optimization goal of this metric (user-defined)."""
        return self.metric_goal

    @property
    def live_eval(self):
        """Returns whether this metric can/should be evaluated at every backprop iteration or not.

        By default, this returns ``True``, but implementations that are quite slow may return ``False``.
        """
        return self._live_eval


@thelper.concepts.classification
@thelper.concepts.segmentation
class ROCCurve(Metric, ClassNamesHandler):
    """Receiver operating characteristic (ROC) computation interface.

    This class provides an interface to ``sklearn.metrics.roc_curve`` and ``sklearn.metrics.roc_auc_score``
    that can produce various types of ROC-related information including the area under the curve (AUC), the
    false positive and negative rates for various operating points, and the ROC curve itself as an image
    (also compatible with tensorboardX).

    By default, evaluating this metric returns the Area Under the Curve (AUC). If a target operating point is
    set, it will instead return the false positive/negative prediction rate of the model at that point.

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
        curve: roc curve generator function, called at evaluation time to generate the output string.
        auc: auc score generator function, called at evaluation time to generate the output string.
        score: queue used to store prediction score values for window-based averaging.
        true: queue used to store groundtruth label values for window-based averaging.
    """

    def __init__(self, target_name, target_tpr=None, target_fpr=None, class_names=None,
                 force_softmax=True, sample_weight=None, drop_intermediate=True):
        """Receives the target class/operating point info, log parameters, and roc computation arguments.

        Args:
            target_name: name of targeted class to generate the roc curve/auc information for.
            target_tpr: target operating point in terms of true positive rate (provided in constructor).
            target_fpr: target operating point in terms of false positive rate (provided in constructor).
            class_names: holds the list of class label names provided by the dataset parser. If it is not
                provided when the constructor is called, it will be set by the trainer at runtime.
            force_softmax: specifies whether a softmax operation should be applied to the prediction scores
                obtained from the trainer.
            sample_weight: passed to ``sklearn.metrics.roc_curve`` and ``sklearn.metrics.roc_auc_score``.
            drop_intermediate: passed to ``sklearn.metrics.roc_curve``.
        """
        assert target_name is not None, "must provide a target (class) name for ROC metric"
        self.target_inv = False
        if isinstance(target_name, str) and target_name[0] == "!":
            self.target_inv = True
            self.target_name = target_name.split("!", 1)[1]
        else:
            self.target_name = target_name
        self.target_tpr, self.target_fpr = None, None
        assert target_tpr is None or target_fpr is None, "must specify only one of target_fpr and target_tpr, not both"
        if target_tpr is not None or target_fpr is not None:
            target_xpr = target_tpr if target_tpr is not None else target_fpr
            assert isinstance(target_xpr, float), "expected float type for target operating point"
            assert 0 <= target_xpr <= 1, "invalid target operation point value (must be in [0,1])"
            if target_tpr is not None:
                self.target_tpr = target_tpr
            else:  # if target_fpr is not None
                self.target_fpr = target_fpr
        self.target_idx = None
        self.force_softmax = force_softmax
        self.sample_weight = sample_weight
        self.drop_intermediate = drop_intermediate

        def gen_curve(y_true, y_score, _target_idx, _target_inv, _sample_weight=sample_weight, _drop_intermediate=drop_intermediate):
            assert _target_idx is not None, "missing positive target idx at run time"
            _y_true, _y_score = [], []
            for sample_idx, label_idx in enumerate(y_true):
                _y_true.append(label_idx != _target_idx if _target_inv else label_idx == _target_idx)
                _y_score.append(1 - y_score[sample_idx, _target_idx] if _target_inv else y_score[sample_idx, _target_idx])
            res = sklearn.metrics.roc_curve(_y_true, _y_score, sample_weight=_sample_weight, drop_intermediate=_drop_intermediate)
            return res

        def gen_auc(y_true, y_score, _target_idx, _target_inv, _sample_weight=sample_weight):
            assert _target_idx is not None, "missing positive target idx at run time"
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
        ClassNamesHandler.__init__(self, class_names)

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this metric."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(target_name={repr(self.target_name)}, target_tpr={repr(self.target_tpr)}, " + \
            f"target_fpr={repr(self.target_fpr)}, class_names={repr(self.class_names)}, " + \
            f"force_softmax={repr(self.force_softmax)}, sample_weight={repr(self.sample_weight)}, " + \
            f"drop_intermediate={repr(self.drop_intermediate)})"

    @ClassNamesHandler.class_names.setter
    def class_names(self, class_names):
        """Sets the class label names that must be predicted by the model."""
        ClassNamesHandler.class_names.fset(self, class_names)
        if self.target_name is not None and self.class_names is not None:
            assert self.target_name in self.class_indices, \
                f"could not find target name {repr(self.target_name)} in class names list"
            self.target_idx = self.class_indices[self.target_name]
        else:
            self.target_idx = None

    def update(self,         # see `thelper.typedefs.IterCallbackParams` for more info
               task,         # type: thelper.tasks.utils.Task
               input,        # type: thelper.typedefs.InputType
               pred,         # type: thelper.typedefs.AnyPredictionType
               target,       # type: thelper.typedefs.AnyTargetType
               sample,       # type: thelper.typedefs.SampleType
               loss,         # type: Optional[float]
               iter_idx,     # type: int
               max_iters,    # type: int
               epoch_idx,    # type: int
               max_epochs,   # type: int
               output_path,  # type: AnyStr
               **kwargs,     # type: Any
               ):            # type: (...) -> None
        """Receives the latest predictions and target values from the training session.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert isinstance(task, thelper.tasks.Classification), "roc curve only impl for classif tasks"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to metric update function"
        if self.score is None or self.score.size != max_iters:
            self.score = np.asarray([None] * max_iters)
            self.true = np.asarray([None] * max_iters)
        if task.class_names != self.class_names:
            self.class_names = task.class_names
        if target is None or target.numel() == 0:
            # only accumulate results when groundtruth is available
            self.score[iter_idx] = None
            self.true[iter_idx] = None
            return
        assert pred.dim() == 2 or target.dim() == 1, "current classif report impl only supports batched 1D outputs"
        assert pred.shape[0] == target.shape[0], "prediction/gt tensors batch size mismatch"
        assert pred.shape[1] == len(self.class_names), "unexpected prediction class dimension size"
        if self.force_softmax:
            with torch.no_grad():
                pred = torch.nn.functional.softmax(pred, dim=1)
        self.score[iter_idx] = pred.numpy()
        self.true[iter_idx] = target.numpy()

    def eval(self):
        """Returns the evaluation result (AUC/TPR/FPR).

        If no target operating point is set, the returned value is the AUC for the target class. If a
        target TPR is set, the returned value is the FPR for that operating point. If a target FPR is set,
        the returned value is the TPR for that operating point.
        """
        if self.score is None or self.true is None:
            return None
        score, true = zip(*[(score, true) for scores, trues in zip(self.score, self.true)
                            if trues is not None for score, true in zip(scores, trues)])
        # if we did not specify a target operating point in terms of true/false positive rate, return AUC
        if self.target_tpr is None and self.target_fpr is None:
            return self.auc(np.stack(true, axis=0), np.stack(score, axis=0), self.target_idx, self.target_inv)
        # otherwise, find the opposite rate at the requested target operating point
        _fpr, _tpr, _thrs = self.curve(np.stack(true, axis=0), np.stack(score, axis=0), self.target_idx,
                                       self.target_inv, _drop_intermediate=False)
        for fpr, tpr, thrs in zip(_fpr, _tpr, _thrs):
            if self.target_tpr is not None and tpr >= self.target_tpr:
                # print("for target tpr = %.5f, fpr = %.5f at threshold = %f" % (self.target_tpr, fpr, thrs))
                return fpr
            elif self.target_fpr is not None and fpr >= self.target_fpr:
                # print("for target fpr = %.5f, tpr = %.5f at threshold = %f" % (self.target_fpr, tpr, thrs))
                return tpr
        # if we did not find a proper rate match above, return worse possible value
        if self.target_tpr is not None:
            # print("for target tpr = %.5f, fpr = 1.0 at threshold = min" % self.target_tpr)
            return 1.0
        else:  # if self.target_fpr is not None:
            # print("for target fpr = %.5f, tpr = 0.0 at threshold = max" % self.target_fpr)
            return 0.0

    def render(self):
        """Returns the ROC curve as a numpy-compatible RGBA image drawn by pyplot."""
        if self.score is None:
            return None
        score, true = zip(*[(score, true) for scores, trues in zip(self.score, self.true)
                            if trues is not None for score, true in zip(scores, trues)])
        fpr, tpr, t = self.curve(np.stack(true, axis=0), np.stack(score, axis=0), self.target_idx, self.target_inv)
        try:
            fig, ax = thelper.draw.draw_roc_curve(fpr, tpr)
            array = thelper.draw.fig2array(fig)
            return array
        except AttributeError as e:
            logger.warning(f"failed to render roc curve; caught exception:\n{str(e)}")
            # return None if rendering fails (probably due to matplotlib on displayless server)
            return None

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying queues."""
        self.score = None
        self.true = None

    @property
    def goal(self):
        """Returns the scalar optimization goal of this metric (variable based on target op point)."""
        # if we did not specify a target operating point in terms of true/false positive rate, return AUC
        if self.target_tpr is None and self.target_fpr is None:
            return Metric.maximize  # AUC must be maximized
        if self.target_tpr is not None:
            return Metric.minimize  # fpr must be minimized
        else:  # if self.target_fpr is not None:
            return Metric.maximize  # tpr must be maximized

    @property
    def live_eval(self):
        """Returns whether this metric can/should be evaluated at every backprop iteration or not."""
        return False  # some operating modes might be pretty slow, check back impl later


@thelper.concepts.regression
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
        max_win_size: maximum moving average window size to use (default=None, which equals dataset size).
        data_range: maximum value of an element in the target signal.
        psnrs: array of psnr values stored for window-based averaging.
        warned_eval_bad: toggles whether the division-by-zero warning has been flagged or not.
    """

    def __init__(self, data_range=1.0, max_win_size=None):
        """Receives all necessary initialization arguments to compute signal PSNRs,

        See :class:`thelper.optim.metrics.PSNR` for information on arguments.
        """
        self.max_win_size = max_win_size
        self.psnrs = None  # will be instantiated on first iter
        self.warned_eval_bad = False
        self.data_range = data_range

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this metric."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(data_range={repr(self.data_range)}, max_win_size={repr(self.max_win_size)})"

    def update(self,         # see `thelper.typedefs.IterCallbackParams` for more info
               task,         # type: thelper.tasks.utils.Task
               input,        # type: thelper.typedefs.InputType
               pred,         # type: thelper.typedefs.RegressionPredictionType
               target,       # type: thelper.typedefs.RegressionTargetType
               sample,       # type: thelper.typedefs.SampleType
               loss,         # type: Optional[float]
               iter_idx,     # type: int
               max_iters,    # type: int
               epoch_idx,    # type: int
               max_epochs,   # type: int
               output_path,  # type: AnyStr
               **kwargs,     # type: Any
               ):            # type: (...) -> None
        """Receives the latest predictions and target values from the training session.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to metric update function"
        curr_win_size = max_iters if self.max_win_size is None else min(self.max_win_size, max_iters)
        if self.psnrs is None or self.psnrs.size != curr_win_size:
            # each 'iteration' will have a corresponding bin with the psnr for that batch
            self.psnrs = np.asarray([None] * curr_win_size)
        curr_idx = iter_idx % curr_win_size
        if target is None or target.numel() == 0:
            # only accumulate results when groundtruth is available
            self.psnrs[curr_idx] = None
            return
        assert pred.shape == target.shape, "prediction/gt tensors shape mismatch"
        mse = np.mean(np.square(pred.numpy() - target.numpy()), dtype=np.float64)
        self.psnrs[curr_idx] = 10 * np.log10(self.data_range / mse)

    def eval(self):
        """Returns the current (average) PSNR based on the accumulated values.

        Will issue a warning if no predictions have been accumulated yet.
        """
        if self.psnrs is None or self.psnrs.size == 0 or len([v for v in self.psnrs if v is not None]) == 0:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("psnr eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        return np.mean([v for v in self.psnrs if v is not None])

    def reset(self):
        """Toggles a reset of the metric's internal state, deallocating the psnrs array."""
        self.psnrs = None

    @property
    def goal(self):
        """Returns the scalar optimization goal of this metric (maximization)."""
        return Metric.maximize


@thelper.concepts.detection
class AveragePrecision(Metric):
    r"""Object detection average precision score from PascalVOC.

    This metric is computed based on the evaluator function implemented in :mod:`thelper.optim.eval`.
    It can target a single class at a time, or produce the mean average precision for all classes.

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "mAP": {
                # this type is used to instantiate the AP metric
                "type": "thelper.optim.metrics.AveragePrecision",
                # these parameters are passed to the wrapper's constructor
                "params": {
                    # no parameters means we will compute the mAP
                }
            },
            # ...
        }
        # ...

    Attributes:
        target_class: name of the class to target; if 'None', will compute mAP instead of AP.
        iou_threshold: Intersection Over Union (IOU) threshold for true/false positive classification.
        method: the evaluation method to use; can be the the latest & official PASCAL VOC toolkit
            approach ("all-points"), or the 11-point approach ("11-points") described in the original
            paper ("The PASCAL Visual Object Classes(VOC) Challenge").
        max_win_size: maximum moving average window size to use (default=None, which equals dataset size).
        preds: array holding the predicted bounding boxes for all input samples.
        targets: array holding the target bounding boxes for all input samples.
    """

    def __init__(self, target_class=None, iou_threshold=0.5, method="all-points", max_win_size=None):
        """Initializes metric attributes.

        Note that by default, if ``max_win_size`` is not provided here, the value given to ``max_iters`` on
        the first update call will be used instead to fix the sliding window length. In any case, the
        smallest of ``max_iters`` and ``max_win_size`` will be used to determine the actual window size.
        """
        assert max_win_size is None or (isinstance(max_win_size, int) and max_win_size > 0), \
            "invalid max sliding window size (should be positive integer)"
        self.target_class = target_class
        self.iou_threshold = iou_threshold
        self.method = method
        self.max_win_size = max_win_size
        self.preds = None  # will be instantiated on first iter
        self.targets = None  # will be instantiated on first iter
        self.task = None

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this metric."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(target_class={repr(self.target_class)}, max_win_size={repr(self.max_win_size)})"

    def update(self,         # see `thelper.typedefs.IterCallbackParams` for more info
               task,         # type: thelper.tasks.utils.Task
               input,        # type: thelper.typedefs.InputType
               pred,         # type: thelper.typedefs.DetectionPredictionType
               target,       # type: thelper.typedefs.DetectionTargetType
               sample,       # type: thelper.typedefs.SampleType
               loss,         # type: Optional[float]
               iter_idx,     # type: int
               max_iters,    # type: int
               epoch_idx,    # type: int
               max_epochs,   # type: int
               output_path,  # type: AnyStr
               **kwargs):    # type: (...) -> None
        """Receives the latest bbox predictions and targets from the training session.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to metric update function"
        curr_win_size = max_iters if self.max_win_size is None else min(self.max_win_size, max_iters)
        if self.preds is None or self.preds.size != curr_win_size:
            # each 'iteration' will have a corresponding bin with counts for that batch
            self.preds = np.asarray([None] * curr_win_size)
            self.targets = np.asarray([None] * curr_win_size)
        curr_idx = iter_idx % curr_win_size
        self.task = task  # keep reference for eval only
        if target is None or len(target) == 0:
            # only accumulate results when groundtruth is available (should we though? affects false negative count)
            self.preds[curr_idx] = None
            self.targets[curr_idx] = None
            return
        if not pred:
            pred = [[]] * len(target)
        assert isinstance(pred, list) and isinstance(target, list)
        assert all([isinstance(b, list) and
                    all([isinstance(p, thelper.tasks.detect.BoundingBox) for p in b]) for b in pred])
        assert all([isinstance(b, list) and
                    all([isinstance(t, thelper.tasks.detect.BoundingBox) for t in b]) for b in target])
        self.preds[curr_idx] = pred
        self.targets[curr_idx] = target

    def eval(self):
        """Returns the current accuracy (in percentage) based on the accumulated prediction counts.

        Will issue a warning if no predictions have been accumulated yet.
        """
        assert self.targets.size == self.preds.size, "internal window size mismatch"
        pred, target = zip(*[(pred, target) for preds, targets in zip(self.preds, self.targets)
                             if targets for pred, target in zip(preds, targets)])
        # maybe need to concat?
        pred, target = np.concatenate(pred), np.concatenate(target)  # possible due to image ids
        if len(pred) == 0:  # no predictions made by model
            return float("nan")
        metrics = thelper.optim.eval.compute_pascalvoc_metrics(pred, target, self.task,
                                                               self.iou_threshold, self.method)
        if self.target_class is None:
            # compute mAP wrt classes that have at least one positive sample
            return np.mean([m["AP"] for m in metrics.values() if m["total positives"] > 0])
        return metrics[self.target_class]["AP"]

    def reset(self):
        """Toggles a reset of the metric's internal state, deallocating bbox arrays."""
        self.preds = None
        self.targets = None

    @property
    def goal(self):
        """Returns the scalar optimization goal of this metric (maximization)."""
        return Metric.maximize

    @property
    def live_eval(self):
        """Returns whether this metric can/should be evaluated at every backprop iteration or not."""
        return False  # the current PascalVOC implementation is preeetty slow with lots of bboxes


@thelper.concepts.segmentation
class IntersectionOverUnion(Metric):
    r"""Computes the intersection over union over image classes.

    It can target a single class at a time, or produce the mean IoU (mIoU) for a number of classes. It can
    also average IoU scores from each images, or sum up all intersection and union areas and compute a
    global score.

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example metric; it is used for lookup/printing only
            "mIoU": {
                # this type is used to instantiate the IoU metric
                "type": "thelper.optim.metrics.IntersectionOverUnion",
                # these parameters are passed to the wrapper's constructor
                "params": {
                    # no parameters means we will compute the mIoU with global scoring
                }
            },
            # ...
        }
        # ...

    Attributes:
        target_names: name(s) of the class(es) to target; if 'None' or list, will compute mIoU instead of IoU.
        max_win_size: maximum moving average window size to use (default=None, which equals dataset size).
        inters: array holding the intesection areas or IoU scores for all input samples.
        unions: array holding the union areas for all input samples.
    """

    def __init__(self, target_names=None, global_score=False, max_win_size=None):
        """Initializes metric attributes.

        Note that by default, if ``max_win_size`` is not provided here, the value given to ``max_iters`` on
        the first update call will be used instead to fix the sliding window length. In any case, the
        smallest of ``max_iters`` and ``max_win_size`` will be used to determine the actual window size.
        """
        assert max_win_size is None or (isinstance(max_win_size, int) and max_win_size > 0), \
            "invalid max sliding window size (should be positive integer)"
        if target_names is not None and not isinstance(target_names, (list, np.ndarray, torch.Tensor)):
            target_names = [target_names]
        self.target_names = target_names
        self.target_idxs = None  # will be updated at runtime
        self.global_score = global_score
        self.max_win_size = max_win_size
        self.inters = None  # will be instantiated on first iter
        self.unions = None  # will be instantiated on first iter
        self.task = None
        self.warned_eval_bad = False

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this metric."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(target_names={repr(self.target_names)}, global_score={repr(self.global_score)}, max_win_size={repr(self.max_win_size)})"

    def update(self,         # see `thelper.typedefs.IterCallbackParams` for more info
               task,         # type: thelper.tasks.utils.Task
               input,        # type: thelper.typedefs.InputType
               pred,         # type: thelper.typedefs.SegmentationPredictionType
               target,       # type: thelper.typedefs.SegmentationTargetType
               sample,       # type: thelper.typedefs.SampleType
               loss,         # type: Optional[float]
               iter_idx,     # type: int
               max_iters,    # type: int
               epoch_idx,    # type: int
               max_epochs,   # type: int
               output_path,  # type: AnyStr
               **kwargs):    # type: (...) -> None
        """Receives the latest bbox predictions and targets from the training session.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to metric update function"
        curr_win_size = max_iters if self.max_win_size is None else min(self.max_win_size, max_iters)
        if self.inters is None or self.inters.size != curr_win_size:
            # each 'iteration' will have a corresponding bin with counts for that batch
            self.inters = np.asarray([None] * curr_win_size)
            self.unions = np.asarray([None] * curr_win_size)
        curr_idx = iter_idx % curr_win_size
        if task is not None:
            assert isinstance(task, thelper.tasks.Segmentation), "unexpected task type with IoU metric"
            if self.target_names is not None:
                assert all([n in task.class_names for n in self.target_names]), \
                    "missing iou target in task class names"
                self.target_idxs = [task.class_indices[n] for n in self.target_names]
            else:
                self.target_idxs = list(task.class_indices.values())
            self.task = task  # keep reference for eval only
        if target is None or len(target) == 0:
            # only accumulate results when groundtruth is available (should we though? affects false negative count)
            self.inters[curr_idx] = None
            self.unions[curr_idx] = None
            return
        assert pred.dim() == target.dim() + 1 or pred.dim() == target.dim(), \
            "prediction/gt tensors dim mismatch (should be BxCx[...] and Bx[...])"
        if pred.dim() == target.dim():
            assert target.shape[1] == 1, "unexpected channel count (>1) for target tensor"
            target = torch.squeeze(target, dim=1)
        assert pred.dim() == target.dim() + 1, "prediction/gt tensors dim mismatch (should be BxCx[...] and Bx[...])"
        assert pred.shape[0] == target.shape[0], "prediction/gt tensors batch size mismatch"
        assert pred.dim() <= 2 or pred.shape[2:] == target.shape[1:], "prediction/gt tensors array size mismatch"
        with torch.no_grad():
            pred_labels = pred.topk(1, dim=1)[1].view(pred.shape[0], -1).cpu().numpy()
            true_labels = target.view(target.shape[0], -1).cpu().numpy()
        assert self.task is not None, "task object necessary at this point since we need to refer to dontcare value"
        assert self.target_idxs, "messed up something internally..."
        if self.global_score:
            inters_count_map, union_count_map = {}, {}
            for target_idx in self.target_idxs:
                inters = np.logical_and(pred_labels == target_idx, true_labels == target_idx)
                inters_count_map[target_idx] = np.count_nonzero(inters)
                if self.task.dontcare is not None:
                    valid_preds = np.logical_and(pred_labels == target_idx, true_labels != self.task.dontcare)
                    union = np.logical_or(valid_preds, true_labels == target_idx)
                else:
                    union = np.logical_or(pred_labels == target_idx, true_labels == target_idx)
                union_count_map[target_idx] = np.count_nonzero(union)
            self.inters[curr_idx] = inters_count_map
            self.unions[curr_idx] = union_count_map
        else:
            bious = [thelper.optim.eval.compute_mask_iou(pred_labels[b], true_labels[b], self.target_idxs, self.task.dontcare)
                     for b in range(pred.shape[0])]
            self.inters[curr_idx] = {tidx: [ious[tidx] for ious in bious] for tidx in self.target_idxs}
            self.unions[curr_idx] = None

    def eval(self):
        """Returns the current IoU ratio based on the accumulated counts.

        Will issue a warning if no predictions have been accumulated yet.
        """
        assert self.inters.size == self.unions.size, "internal window size mismatch"
        if self.target_idxs is None:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("iou eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        accum_pairs = {}
        valid = False
        for tidx in self.target_idxs:
            accum_pairs[tidx] = [], []
            for i, u in zip(self.inters, self.unions):
                if i is not None or u is not None:
                    accum_pairs[tidx][0].append(i[tidx] if i is not None else None)
                    accum_pairs[tidx][1].append(u[tidx] if u is not None else None)
                    valid = valid or i is not None
        if not valid:
            if not self.warned_eval_bad:
                self.warned_eval_bad = True
                logger.warning("iou eval result invalid (set as 0.0), no results accumulated")
            return 0.0
        if self.global_score:
            tot_inters = {tidx: sum(accum_pairs[tidx][0]) for tidx in self.target_idxs}
            tot_union = {tidx: sum(accum_pairs[tidx][1]) for tidx in self.target_idxs}
            iou_map = {tidx: (tot_inters[tidx] / tot_union[tidx]) if tot_union[tidx] != 0 else 0.0 for tidx in self.target_idxs}
        else:
            iou_map = {}
            for tidx in self.target_idxs:
                ious = []
                for batch_ious in accum_pairs[tidx][0]:
                    for iou in batch_ious:
                        ious.append(iou)
                iou_map[tidx] = 0.0 if not ious else np.mean(ious)
        # could add per-class IoU scores to some log before averaging below...
        return np.array(list(iou_map.values())).mean()

    def reset(self):
        """Toggles a reset of the metric's internal state, deallocating bbox arrays."""
        self.inters = None
        self.unions = None

    @property
    def goal(self):
        """Returns the scalar optimization goal of this metric (maximization)."""
        return Metric.maximize
