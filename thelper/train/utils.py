"""Training/evaluation utilities module.

This module contains utilities and tools used to instantiate training sessions. It also contains
the prediction consumer interface used by metrics and loggers to receive iteration data during
training. See :mod:`thelper.optim.metrics` for more information on metrics.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import AnyStr, Optional  # noqa: F401

import cv2 as cv
import numpy as np
import sklearn.metrics
import torch

import thelper.utils

logger = logging.getLogger(__name__)


class PredictionConsumer(ABC):
    """Abstract model prediction consumer class.

    This interface defines basic functions required so that :class:`thelper.train.base.Trainer` can
    figure out how to instantiate and update a model prediction consumer. The most notable class derived
    from this interface is :class:`thelper.optim.metrics.Metric` which is used to monitor the
    improvement of a model during a training session. Other prediction consumers defined in
    :mod:`thelper.train.utils` will instead log predictions to local files, create graphs, etc.
    """

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this consumer."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + "()"

    def reset(self):
        """Resets the internal state of the consumer.

        May be called for example by the trainer between two evaluation epochs. The default implementation
        does nothing, and if a reset behavior is needed, it should be implemented by the derived class.
        """
        pass

    @abstractmethod
    def update(self,  # see `thelper.typedefs.IterCallbackParams` for more info
               task,  # type: thelper.tasks.utils.Task
               input,  # type: thelper.typedefs.InputType
               pred,  # type: thelper.typedefs.PredictionType
               target,  # type: thelper.typedefs.TargetType
               sample,  # type: thelper.typedefs.SampleType
               loss,  # type: Optional[float]
               iter_idx,  # type: int
               max_iters,  # type: int
               epoch_idx,  # type: int
               max_epochs,  # type: int
               **kwargs):  # type: (...) -> None
        """Receives the latest prediction and groundtruth tensors from the training session.

        The data given here will be "consumed" internally, but it should NOT be modified. For example,
        a classification accuracy metric would accumulate the correct number of predictions in comparison
        to groundtruth labels, while a plotting logger would add new corresponding dots to a curve.

        Remember that input, prediction, and target tensors received here will all have a batch dimension!

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        raise NotImplementedError


class PredictionCallback(PredictionConsumer):
    """Callback function wrapper compatible with the consumer interface.

    This interface is used to hide user-defined callbacks into the list of prediction consumers given
    to trainer implementations. The callbacks must always be compatible with the list of arguments
    defined by `thelper.typedefs.IterCallbackParams`, but may also receive extra arguments defined in
    advance and passed to the constructor of this class.

    Attributes:
        callback_func: user-defined function to call on every update from the trainer.
        callback_kwargs: user-defined extra arguments to provide to the callback function.
    """

    def __init__(self, callback_func, callback_kwargs=None):
        assert callback_func is not None and \
            (isinstance(callback_func, str) or callable(callback_func)), \
            "invalid callback function, must be importable string or callable object"
        if isinstance(callback_func, str):
            callback_func = thelper.utils.import_function(callback_func)
        thelper.utils.check_func_signature(callback_func, thelper.typedefs.IterCallbackParams)
        assert callback_kwargs is None or \
            (isinstance(callback_kwargs, dict) and
             not any([p in callback_kwargs for p in thelper.typedefs.IterCallbackParams])), \
            "invalid callback kwargs (must be dict, and not contain overlap with default args)"
        self.callback_func = callback_func
        self.callback_kwargs = callback_kwargs

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this consumer."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(callback_func={repr(self.callback_func)}, callback_kwargs={repr(self.callback_kwargs)})"

    def update(self, *args, **kwargs):
        """Forwards the latest prediction data from the training session to the user callback."""
        return self.callback_func(*args, **kwargs, **self.callback_kwargs)


class ClassifLogger(PredictionConsumer):
    """Classification output logger.

    This class provides a simple logging interface for accumulating and saving the predictions of a classifier.
    By default, all predictions will be logged. However, a confidence threshold can be set to focus on "hard"
    samples if necessary. It also optionally offers tensorboardX-compatible output images that can be saved
    locally or posted to tensorboard for browser-based visualization.

    Usage examples inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example consumer; it is used for lookup/printing only
            "logger": {
                # this type is used to instantiate the confusion matrix report object
                "type": "thelper.train.utils.ClassifLogger",
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
        conf_threshold: threshold used to eliminate all but the most uncertain predictions.
        class_names: holds the list of class label names provided by the dataset parser. If it is not
            provided when the constructor is called, it will be set by the trainer at runtime.
        target_name: name of the targeted label (may be 'None' if all classes are used).
        target_idx: index of the targeted label (may be 'None' if all classes are used).
        viz_count: number of tensorboardX images to generate and update at each epoch.
        report_count: number of samples to print in reports (use 'None' if all samples must be printed).
        log_keys: list of metadata field keys to copy from samples into the log for each prediction.
        force_softmax: specifies whether a softmax operation should be applied to the prediction scores
            obtained from the trainer.
        score: array used to store prediction scores for logging.
        true: array used to store groundtruth labels for logging.
        meta: array used to store metadata pulled from samples for logging.
    """

    def __init__(self, top_k=1, conf_threshold=None, class_names=None, target_name=None,
                 viz_count=0, report_count=None, log_keys=None, force_softmax=True):
        """Receives the logging parameters & the optional class label names used to decorate the log."""
        assert isinstance(top_k, int) and top_k > 0, "invalid top-k value"
        assert conf_threshold is None or (isinstance(conf_threshold, float) and 0 < conf_threshold <= 1), \
            "classification confidence threshold should be 'None' or float in ]0, 1]"
        assert isinstance(viz_count, int) and viz_count >= 0, "invalid image count to visualize"
        assert report_count is None or (isinstance(report_count, int) and report_count >= 0), "invalid report sample count"
        assert log_keys is None or isinstance(log_keys, list), "invalid list of sample keys to log"
        self.top_k = top_k
        self.target_name = target_name
        self.target_idx = None
        self.class_names = None
        if class_names is not None:
            self.set_class_names(class_names)
        self.conf_threshold = conf_threshold
        self.viz_count = viz_count
        self.report_count = report_count
        self.log_keys = log_keys if log_keys is not None else []
        self.force_softmax = force_softmax
        self.score = None
        self.true = None
        self.meta = None

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this consumer."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(top_k={repr(self.top_k)}, conf_threshold={repr(self.conf_threshold)}, " + \
            f"class_names={repr(self.class_names)}, target_name={repr(self.target_name)}, " + \
            f"viz_count={repr(self.viz_count)}, report_count={repr(self.report_count)}, " + \
            f"log_keys={repr(self.log_keys)}, force_softmax={repr(self.force_softmax)})"

    def set_class_names(self, class_names):
        """Sets the class label names that must be predicted by the model."""
        assert isinstance(class_names, list), "expected list for class names"
        assert len(class_names) >= 2, "not enough classes in provided class list"
        if self.target_name is not None:
            assert self.target_name in class_names, \
                f"could not find target name {repr(self.target_name)} in class names list"
            self.target_idx = class_names.index(self.target_name)
        self.class_names = class_names

    def update(self,  # see `thelper.typedefs.IterCallbackParams` for more info
               task,  # type: thelper.tasks.utils.Task
               input,  # type: thelper.typedefs.InputType
               pred,  # type: thelper.typedefs.PredictionType
               target,  # type: thelper.typedefs.TargetType
               sample,  # type: thelper.typedefs.SampleType
               loss,  # type: Optional[float]
               iter_idx,  # type: int
               max_iters,  # type: int
               epoch_idx,  # type: int
               max_epochs,  # type: int
               **kwargs):  # type: (...) -> None
        """Receives the latest predictions and target values from the training session.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert isinstance(task, thelper.tasks.Classification), "classif report only impl for classif tasks"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to update function"
        if self.score is None or self.score.size != max_iters:
            self.score = np.asarray([None] * max_iters)
            self.true = np.asarray([None] * max_iters)
            self.meta = {key: np.asarray([None] * max_iters) for key in self.log_keys}
        if task.class_names != self.class_names:
            self.set_class_names(task.class_names)
        if target is None or target.numel() == 0:
            # only accumulate results when groundtruth is available
            self.score[iter_idx] = None
            self.true[iter_idx] = None
            for key, array in self.meta.items():
                array[iter_idx] = None
            return
        assert pred.dim() == 2 or target.dim() == 1, "current classif logger impl only supports batched 1D outputs"
        assert pred.shape[0] == target.shape[0], "prediction/gt tensors batch size mismatch"
        assert pred.shape[1] == len(self.class_names), "unexpected prediction class dimension size"
        if self.force_softmax:
            with torch.no_grad():
                pred = torch.nn.functional.softmax(pred, dim=1)
        self.score[iter_idx] = pred.numpy()
        self.true[iter_idx] = target.numpy()
        for meta_key in self.log_keys:
            assert meta_key in sample, f"could not extract sample field with key {repr(meta_key)}"
            val = sample[meta_key]
            assert isinstance(val, (list, np.ndarray, torch.Tensor)), f"field {repr(meta_key)} should be batched"
            self.meta[meta_key][iter_idx] = val if isinstance(val, list) else val.tolist()

    def render(self):
        """Returns an image of predicted outputs as a numpy-compatible RGBA image drawn by pyplot."""
        if self.viz_count == 0:
            return None
        if self.score is None or self.true is None:
            return None
        raise NotImplementedError  # TODO

    def report(self):
        """Returns the logged metadata of predicted samples.

        The returned object is a print-friendly CSV string that can be consumed directly by tensorboardX. Note
        that this string might be very long if the dataset is large (i.e. it will contain one line per sample).
        """
        if self.report_count is not None and self.report_count == 0:
            return None
        if self.score is None or self.true is None:
            return None
        pack = list(zip(*[(*pack, ) for packs in zip(self.score, self.true, *self.meta.values())
                          if packs[1] is not None for pack in zip(*packs)]))
        logdata = {key: np.stack(val, axis=0) for key, val in zip(["pred", "target", *self.meta.keys()], pack)}
        assert all([len(val) == len(logdata["target"]) for val in logdata.values()]), "messed up unpacking"
        header = "target_name,target_score"
        for k in range(self.top_k):
            header += f",pred_{k + 1}_name,pred_{k + 1}_score"
        for meta_key in self.log_keys:
            header += f",{str(meta_key)}"
        lines = []
        for sample_idx in range(len(logdata["target"])):
            gt_label_idx = int(logdata["target"][sample_idx])
            pred_scores = logdata["pred"][sample_idx]
            sorted_score_idxs = np.argsort(pred_scores)[::-1]
            sorted_scores = pred_scores[sorted_score_idxs]
            if self.conf_threshold is None or pred_scores[gt_label_idx] < self.conf_threshold:
                entry = f"{self.class_names[gt_label_idx]},{pred_scores[gt_label_idx]:2.4f}"
                for k in range(self.top_k):
                    entry += f",{self.class_names[sorted_score_idxs[k]]},{sorted_scores[k]:2.4f}"
                for meta_key in self.log_keys:
                    entry += f",{str(logdata[meta_key][sample_idx])}"
                lines.append(entry)
                if self.report_count is not None and len(lines) >= self.report_count:
                    break
        return "\n".join([header, *lines])

    def reset(self):
        """Toggles a reset of the internal state, emptying storage arrays."""
        self.score = None
        self.true = None
        self.meta = None


class ClassifReport(PredictionConsumer):
    """Classification report interface.

    This class provides a simple interface to ``sklearn.metrics.classification_report`` so that all
    count-based metrics can be reported at once under a string-based representation.

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example consumer; it is used for lookup/printing only
            "classifreport": {
                # this type is used to instantiate the classification report object
                "type": "thelper.train.utils.ClassifReport",
                # we do not need to provide any parameters to the constructor, defaults are fine
                "params": {}
            },
            # ...
        }
        # ...

    Attributes:
        gen_report: report generator function, called at evaluation time to generate the output string.
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
            _y_true = [_class_names[classid] for classid in y_true]
            _y_pred = [_class_names[classid] if (0 <= classid < len(_class_names)) else "<unset>" for classid in y_pred]
            return sklearn.metrics.classification_report(_y_true, _y_pred, sample_weight=sample_weight, digits=digits)

        self.gen_report = gen_report
        self.class_names = None
        if class_names is not None:
            self.set_class_names(class_names)
        self.sample_weight = sample_weight
        self.digits = digits
        self.pred = None
        self.target = None

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this consumer."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(class_names={repr(self.class_names)}, sample_weight={repr(self.sample_weight)}, digits={repr(self.digits)})"

    def set_class_names(self, class_names):
        """Sets the class label names that must be predicted by the model."""
        assert isinstance(class_names, list), "expected list for class names"
        assert len(class_names) >= 2, "not enough classes in provided class list"
        self.class_names = class_names

    def update(self,  # see `thelper.typedefs.IterCallbackParams` for more info
               task,  # type: thelper.tasks.utils.Task
               input,  # type: thelper.typedefs.InputType
               pred,  # type: thelper.typedefs.PredictionType
               target,  # type: thelper.typedefs.TargetType
               sample,  # type: thelper.typedefs.SampleType
               loss,  # type: Optional[float]
               iter_idx,  # type: int
               max_iters,  # type: int
               epoch_idx,  # type: int
               max_epochs,  # type: int
               **kwargs):  # type: (...) -> None
        """Receives the latest predictions and target values from the training session.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert isinstance(task, thelper.tasks.Classification), "classif report only impl for classif tasks"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to update function"
        if self.pred is None or self.pred.size != max_iters:
            self.pred = np.asarray([None] * max_iters)
            self.target = np.asarray([None] * max_iters)
        if task.class_names != self.class_names:
            self.set_class_names(task.class_names)
        if target is None or target.numel() == 0:
            # only accumulate results when groundtruth is available
            self.pred[iter_idx] = None
            self.target[iter_idx] = None
            return
        assert pred.dim() == 2 or target.dim() == 1, "current classif report impl only supports batched 1D outputs"
        assert pred.shape[0] == target.shape[0], "prediction/gt tensors batch size mismatch"
        assert pred.shape[1] == len(self.class_names), "unexpected prediction class dimension size"
        self.pred[iter_idx] = pred.topk(1, dim=1)[1].view(pred.shape[0]).tolist()
        self.target[iter_idx] = target.view(target.shape[0]).tolist()

    def report(self):
        """Returns the classification report as a multi-line print-friendly string."""
        if self.pred is None or self.target is None:
            return None
        pred, target = zip(*[(pred, target) for preds, targets in zip(self.pred, self.target)
                             if targets is not None for pred, target in zip(preds, targets)])
        return "\n" + self.gen_report(np.asarray(target), np.asarray(pred), self.class_names)

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying queues."""
        self.pred = None
        self.target = None


class ConfusionMatrix(PredictionConsumer):
    """Confusion matrix report interface.

    This class provides a simple interface to ``sklearn.metrics.confusion_matrix`` so that a full
    confusion matrix can be easily reported under a string-based representation. It also offers a
    tensorboardX-compatible output image that can be saved locally or posted to tensorboard for
    browser-based visualization.

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example consumer; it is used for lookup/printing only
            "confmat": {
                # this type is used to instantiate the confusion matrix report object
                "type": "thelper.train.utils.ConfusionMatrix",
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
            _y_true = [_class_names[classid] for classid in y_true]
            _y_pred = [_class_names[classid] if (0 <= classid < len(_class_names)) else "<unset>" for classid in y_pred]
            return sklearn.metrics.confusion_matrix(_y_true, _y_pred, labels=_class_names)

        self.matrix = gen_matrix
        self.draw_normalized = draw_normalized
        self.class_names = None
        if class_names is not None:
            self.set_class_names(class_names)
        self.pred = None
        self.target = None

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this consumer."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(class_names={repr(self.class_names)}, draw_normalized={repr(self.draw_normalized)})"

    def set_class_names(self, class_names):
        """Sets the class label names that must be predicted by the model."""
        assert isinstance(class_names, list), "expected list for class names"
        assert len(class_names) >= 2, "not enough classes in provided class list"
        self.class_names = class_names

    def update(self,  # see `thelper.typedefs.IterCallbackParams` for more info
               task,  # type: thelper.tasks.utils.Task
               input,  # type: thelper.typedefs.InputType
               pred,  # type: thelper.typedefs.PredictionType
               target,  # type: thelper.typedefs.TargetType
               sample,  # type: thelper.typedefs.SampleType
               loss,  # type: Optional[float]
               iter_idx,  # type: int
               max_iters,  # type: int
               epoch_idx,  # type: int
               max_epochs,  # type: int
               **kwargs):  # type: (...) -> None
        """Receives the latest predictions and target values from the training session.

        The exact signature of this function should match the one of the callbacks defined in
        :class:`thelper.train.base.Trainer` and specified by ``thelper.typedefs.IterCallbackParams``.
        """
        assert len(kwargs) == 0, "unexpected extra arguments present in update call"
        assert isinstance(task, thelper.tasks.Classification), "confmat only impl for classif tasks"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to update function"
        if self.pred is None or self.pred.size != max_iters:
            self.pred = np.asarray([None] * max_iters)
            self.target = np.asarray([None] * max_iters)
        if task.class_names != self.class_names:
            self.set_class_names(task.class_names)
        if target is None or target.numel() == 0:
            # only accumulate results when groundtruth is available
            self.pred[iter_idx] = None
            self.target[iter_idx] = None
            return
        assert pred.dim() == 2 or target.dim() == 1, "current confmat impl only supports batched 1D outputs"
        assert pred.shape[0] == target.shape[0], "prediction/gt tensors batch size mismatch"
        assert pred.shape[1] == len(self.class_names), "unexpected prediction class dimension size"
        self.pred[iter_idx] = pred.topk(1, dim=1)[1].view(pred.shape[0]).tolist()
        self.target[iter_idx] = target.view(target.shape[0]).tolist()

    def report(self):
        """Returns the confusion matrix as a multi-line print-friendly string."""
        if self.pred is None or self.target is None:
            return None
        pred, target = zip(*[(pred, target) for preds, targets in zip(self.pred, self.target)
                             if targets is not None for pred, target in zip(preds, targets)])
        confmat = self.matrix(np.asarray(target), np.asarray(pred), self.class_names)
        return "\n" + thelper.utils.stringify_confmat(confmat, self.class_names)

    def render(self):
        """Returns the confusion matrix as a numpy-compatible RGBA image drawn by pyplot."""
        if self.pred is None or self.target is None:
            return None
        pred, target = zip(*[(pred, target) for preds, targets in zip(self.pred, self.target)
                             if targets is not None for pred, target in zip(preds, targets)])
        confmat = self.matrix(np.asarray(target), np.asarray(pred), self.class_names)
        try:
            fig, ax = thelper.utils.draw_confmat(confmat, self.class_names, normalize=self.draw_normalized)
            array = thelper.utils.fig2array(fig)
            return array
        except AttributeError as e:
            logger.warning(f"failed to render confusion matrix; caught exception:\n{str(e)}")
            # return None if rendering fails (probably due to matplotlib on displayless server)
            return None

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying queues."""
        self.pred = None
        self.target = None


def create_consumers(config):
    """Instantiates and returns the prediction consumers defined in the configuration dictionary.

    All arguments are expected to be handed in through the configuration via a dictionary named 'params'.
    """
    assert isinstance(config, dict), "config should be provided as a dictionary"
    consumers = {}
    for name, consumer_config in config.items():
        assert isinstance(consumer_config, dict), "consumer config should be provided as a dictionary"
        assert "type" in consumer_config and consumer_config["type"], "consumer config missing 'type' field"
        consumer_type = thelper.utils.import_class(consumer_config["type"])
        consumer_params = thelper.utils.get_key_def(["params", "parameters"], consumer_config, {})
        try:
            consumer = consumer_type(**consumer_params)
        except Exception:
            logger.error(f"failed to create consumer {consumer_config['type']} with params:\n\t{str(consumer_params)}")
            raise
        assert isinstance(consumer, PredictionConsumer), "invalid consumer type, must derive from PredictionConsumer interface"
        consumers[name] = consumer
    return consumers


def create_trainer(session_name,    # type: AnyStr
                   save_dir,        # type: AnyStr
                   config,          # type: thelper.typedefs.ConfigDict
                   model,           # type: thelper.typedefs.ModelType
                   task,            # type: thelper.tasks.Task
                   loaders,         # type: thelper.typedefs.MultiLoaderType
                   ckptdata=None    # type: Optional[thelper.typedefs.CheckpointContentType]
                   ):               # type: (...) -> thelper.train.Trainer
    """Instantiates the trainer object based on the type contained in the config dictionary.

    The trainer type is expected to be in the configuration dictionary's `trainer` field, under the `type` key. For more
    information on the configuration, refer to :class:`thelper.train.base.Trainer`. The instantiated type must be
    compatible with the constructor signature of :class:`thelper.train.base.Trainer`. The object's constructor will
    be given the full config dictionary and the checkpoint data for resuming the session (if available).

    If the trainer type is missing, it will be automatically deduced based on the task object.

    Args:
        session_name: name of the training session used for printing and to create internal tensorboardX directories.
        save_dir: path to the session directory where logs and checkpoints will be saved.
        config: full configuration dictionary that will be parsed for trainer parameters and saved in checkpoints.
        model: model to train/evaluate; should be compatible with :class:`thelper.nn.utils.Module`.
        task: global task interface defining the type of model and training goal for the session.
        loaders: a tuple containing the training/validation/test data loaders (a loader can be ``None`` if empty).
        ckptdata: raw checkpoint to parse data from when resuming a session (if ``None``, will start from scratch).

    Returns:
        The fully-constructed trainer object, ready to begin model training/evaluation.

    .. seealso::
        | :class:`thelper.train.base.Trainer`

    """
    assert "trainer" in config and config["trainer"], "session configuration dictionary missing 'trainer' section"
    trainer_config = config["trainer"]
    if "type" not in trainer_config:
        if isinstance(task, thelper.tasks.Classification):
            trainer_type = thelper.train.ImageClassifTrainer
        elif isinstance(task, thelper.tasks.Detection):
            trainer_type = thelper.train.ObjDetectTrainer
        elif isinstance(task, thelper.tasks.Regression):
            trainer_type = thelper.train.RegressionTrainer
        elif isinstance(task, thelper.tasks.Segmentation):
            trainer_type = thelper.train.ImageSegmTrainer
        else:
            raise AssertionError(f"unknown trainer type required for task '{str(task)}'")
    else:
        trainer_type = thelper.utils.import_class(trainer_config["type"])
    return trainer_type(session_name, save_dir, model, task, loaders, config, ckptdata=ckptdata)


# noinspection PyUnusedLocal
def _draw_wrapper(task,  # type: thelper.tasks.utils.Task
                  input,  # type: thelper.typedefs.InputType
                  pred,  # type: thelper.typedefs.PredictionType
                  target,  # type: thelper.typedefs.TargetType
                  sample,  # type: thelper.typedefs.SampleType
                  loss,  # type: Optional[float]
                  iter_idx,  # type: int
                  max_iters,  # type: int
                  epoch_idx,  # type: int
                  max_epochs,  # type: int
                  # extra params added by callback interface below
                  output_path,  # type: str
                  save,  # type: bool
                  # all extra params will be forwarded to the display call
                  **kwargs):  # type: (...) -> None
    """Wrapper to :func:`thelper.utils.draw` used as a callback entrypoint for trainers."""
    res = thelper.utils.draw(task=task, input=input, pred=pred, target=target, **kwargs)
    if save:
        assert isinstance(res, tuple) and len(res) == 2, "unexpected redraw output (should be 2-elem tuple)"
        if isinstance(res[1], np.ndarray):
            assert "path" in sample and isinstance(sample["path"], list) and len(sample["path"]) == 1, \
                "unexpected draw format (current implementation needs batch size = 1, and path metadata)"
            os.makedirs(output_path, exist_ok=True)
            filepath = os.path.join(output_path, os.path.basename(sample["path"][0]))
            cv.imwrite(filepath, res[1])
        else:
            # we're displaying with matplotlib, and have no clue on how to save the output
            raise NotImplementedError
