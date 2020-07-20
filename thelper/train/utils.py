"""Training/evaluation utilities module.

This module contains utilities and tools used to instantiate training sessions. It also contains
the prediction consumer interface used by metrics and loggers to receive iteration data during
training. See :mod:`thelper.optim.metrics` for more information on metrics.
"""

import json
import logging
import os
from typing import Any, AnyStr, Dict, List, Optional, Union  # noqa: F401

import cv2 as cv
import numpy as np
import sklearn.metrics
import torch

import thelper.concepts
import thelper.ifaces
import thelper.typedefs  # noqa: F401
import thelper.utils
from thelper.ifaces import ClassNamesHandler, FormatHandler, PredictionConsumer
from thelper.optim.eval import compute_bbox_iou
from thelper.tasks.detect import BoundingBox

logger = logging.getLogger(__name__)


class PredictionCallback(PredictionConsumer):
    """Callback function wrapper compatible with the consumer interface.

    This interface is used to hide user-defined callbacks into the list of prediction consumers given
    to trainer implementations. The callbacks must always be compatible with the list of arguments
    defined by ``thelper.typedefs.IterCallbackParams``, but may also receive extra arguments defined in
    advance and passed to the constructor of this class.

    Attributes:
        callback_func: user-defined function to call on every update from the trainer.
        callback_kwargs: user-defined extra arguments to provide to the callback function.
    """

    def __init__(self, callback_func, callback_kwargs=None):
        # type: (thelper.typedefs.IterCallbackType, thelper.typedefs.IterCallbackParams) -> None
        assert callback_kwargs is None or \
            (isinstance(callback_kwargs, dict) and
             not any([p in callback_kwargs for p in thelper.typedefs.IterCallbackParams])), \
            "invalid callback kwargs (must be dict, and not contain overlap with default args)"
        callback_func = thelper.utils.import_function(callback_func, params=callback_kwargs)
        thelper.utils.check_func_signature(callback_func, thelper.typedefs.IterCallbackParams)
        self.callback_func = callback_func
        self.callback_kwargs = callback_kwargs

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this consumer."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(callback_func={repr(self.callback_func)}, callback_kwargs={repr(self.callback_kwargs)})"

    def update(self, *args, **kwargs):
        """Forwards the latest prediction data from the training session to the user callback."""
        return self.callback_func(*args, **kwargs)


@thelper.concepts.classification
class ClassifLogger(PredictionConsumer, ClassNamesHandler, FormatHandler):
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
        conf_threshold: threshold used to eliminate uncertain predictions.
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
        format: output format of the produced log (supports: text, CSV)
    """

    def __init__(self,
                 top_k=1,               # type: int
                 conf_threshold=None,   # type: Optional[thelper.typedefs.Number]
                 class_names=None,      # type: Optional[List[AnyStr]]
                 target_name=None,      # type: Optional[AnyStr]
                 viz_count=0,           # type: int
                 report_count=None,     # type: Optional[int]
                 log_keys=None,         # type: Optional[List[AnyStr]]
                 force_softmax=True,    # type: bool
                 format=None,           # type: Optional[AnyStr]
                 ):                     # type: (...) -> None
        """Receives the logging parameters & the optional class label names used to decorate the log."""
        assert isinstance(top_k, int) and top_k > 0, "invalid top-k value"
        assert conf_threshold is None or (isinstance(conf_threshold, (float, int)) and 0 < conf_threshold <= 1), \
            "classification confidence threshold should be 'None' or float in ]0, 1]"
        assert isinstance(viz_count, int) and viz_count >= 0, "invalid image count to visualize"
        assert report_count is None or (isinstance(report_count, int) and report_count >= 0), \
            "invalid report sample count"
        assert log_keys is None or isinstance(log_keys, list), "invalid list of sample keys to log"
        self.top_k = top_k
        self.target_name = target_name
        self.target_idx = None
        self.conf_threshold = conf_threshold
        self.viz_count = viz_count
        self.report_count = report_count
        self.log_keys = log_keys if log_keys is not None else []
        self.force_softmax = force_softmax
        self.score = None
        self.true = None
        self.meta = None
        ClassNamesHandler.__init__(self, class_names)
        FormatHandler.__init__(self, format)

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this consumer."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(top_k={repr(self.top_k)}, conf_threshold={repr(self.conf_threshold)}, " + \
            f"class_names={repr(self.class_names)}, target_name={repr(self.target_name)}, " + \
            f"viz_count={repr(self.viz_count)}, report_count={repr(self.report_count)}, " + \
            f"log_keys={repr(self.log_keys)}, force_softmax={repr(self.force_softmax)})"

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
            self.class_names = task.class_names
        assert pred.dim() == 2, "current classif logger impl only supports 2D outputs (BxC)"
        assert pred.shape[1] == len(self.class_names), "unexpected prediction class dimension size"
        if target is None or target.numel() == 0:
            self.true[iter_idx] = None
        else:
            assert target.dim() == 1, "gt should be batched (1D) tensor"
            assert pred.shape[0] == target.shape[0], "prediction/gt tensors batch size mismatch"
            self.true[iter_idx] = target.numpy()
        if self.force_softmax:
            with torch.no_grad():
                pred = torch.nn.functional.softmax(pred, dim=1)
        self.score[iter_idx] = pred.numpy()
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

    def report_text(self):
        # type: () -> Optional[AnyStr]
        return self.report_csv()

    def report_json(self):
        # type: () -> Optional[AnyStr]
        csv = self.report_csv()
        if csv is None:
            return None
        csv = csv.splitlines()
        header, data = csv[0], csv[1:]
        headers = header.split(",")
        json_entries = [{k: float(v) if "score" in k else str(v)
                         for k, v in zip(headers, line.split(","))} for line in data]
        return json.dumps(json_entries, sort_keys=False, indent=4)

    def report_csv(self):
        # type: () -> Optional[AnyStr]
        """Returns the logged metadata of predicted samples.

        The returned object is a print-friendly CSV string that can be consumed directly by tensorboardX. Note
        that this string might be very long if the dataset is large (i.e. it will contain one line per sample).
        """
        if isinstance(self.report_count, int) and self.report_count <= 0:
            return None
        if self.score is None or self.true is None:
            return None
        pack = list(zip(*[(*pack, )
                          for packs in zip(self.score, self.true, *self.meta.values())
                          for pack in zip(*packs)]))
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
            if self.conf_threshold is None or gt_label_idx is None or \
                    pred_scores[gt_label_idx] >= self.conf_threshold:
                if gt_label_idx is not None:
                    entry = f"{self.class_names[gt_label_idx]},{pred_scores[gt_label_idx]:2.4f}"
                else:
                    entry = f"<unknown>,{0.0:2.4f}"
                for k in range(self.top_k):
                    entry += f",{self.class_names[sorted_score_idxs[k]]},{sorted_scores[k]:2.4f}"
                for meta_key in self.log_keys:
                    entry += f",{str(logdata[meta_key][sample_idx])}"
                lines.append(entry)
                if isinstance(self.report_count, int) and len(lines) >= self.report_count:
                    break
        return "\n".join([header, *lines])

    def reset(self):
        """Toggles a reset of the internal state, emptying storage arrays."""
        self.score = None
        self.true = None
        self.meta = None


@thelper.concepts.classification
class ClassifReport(PredictionConsumer, ClassNamesHandler, FormatHandler):
    """Classification report interface.

    This class provides a simple interface to ``sklearn.metrics.classification_report`` so that all
    count-based metrics can be reported at once under a string-based representation.

    Usage example inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example consumer; it is used for lookup/printing only
            "report": {
                # this type is used to instantiate the classification report object
                "type": "thelper.train.utils.ClassifReport",
                # we do not need to provide any parameters to the constructor, defaults are fine
                "params": {
                    # optional parameter that will indicate output as JSON is desired, plain 'text' otherwise
                    "format": "json"
                }
            },
            # ...
        }
        # ...

    Attributes:
        class_names: holds the list of class label names provided by the dataset parser. If it is not
            provided when the constructor is called, it will be set by the trainer at runtime.
        pred: queue used to store the top-1 (best) predicted class indices at each iteration.
        format: output format of the produced log (supports: text, JSON)
    """

    def __init__(self, class_names=None, sample_weight=None, digits=4, format=None):
        """Receives the optional class names and arguments passed to the report generator function.

        Args:
            class_names: holds the list of class label names provided by the dataset parser. If it is not
                provided when the constructor is called, it will be set by the trainer at runtime.
            sample_weight: sample weights, forwarded to ``sklearn.metrics.classification_report``.
            digits: metrics output digit count, forwarded to ``sklearn.metrics.classification_report``.
            format: output format of the produced log.
        """
        self.class_names = None
        self.sample_weight = sample_weight
        self.digits = digits
        self.pred = None
        self.target = None
        ClassNamesHandler.__init__(self, class_names)
        FormatHandler.__init__(self, format)

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this consumer."""
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}" + \
               f"(class_names={repr(self.class_names)}, sample_weight={repr(self.sample_weight)}, " + \
               f"digits={repr(self.digits)})"

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
            self.class_names = task.class_names
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

    def gen_report(self, as_dict=False):
        # type: (bool) -> Union[AnyStr, thelper.typedefs.JSON]
        if self.pred is None or self.target is None:
            return None
        pred, target = zip(*[(pred, target) for preds, targets in zip(self.pred, self.target)
                             if targets is not None for pred, target in zip(preds, targets)])
        y_true = np.asarray(target)
        y_pred = np.asarray(pred)
        _y_true = [self.class_names[classid] for classid in y_true]
        _y_pred = [self.class_names[classid] if (0 <= classid < len(self.class_names)) else "<unset>"
                   for classid in y_pred]
        return sklearn.metrics.classification_report(_y_true, _y_pred, sample_weight=self.sample_weight,
                                                     digits=self.digits, output_dict=as_dict)

    def report_text(self):
        # type: () -> Optional[AnyStr]
        """Returns the classification report as a multi-line print-friendly string."""
        return f"\n{self.gen_report(as_dict=False)}"

    def report_json(self):
        # type: () -> Optional[AnyStr]
        """Returns the classification report as a JSON formatted string."""
        return json.dumps(self.gen_report(as_dict=True), indent=4)

    def reset(self):
        """Toggles a reset of the metric's internal state, emptying queues."""
        self.pred = None
        self.target = None


@thelper.concepts.detection
class DetectLogger(PredictionConsumer, ClassNamesHandler, FormatHandler):
    """Detection output logger.

    This class provides a simple logging interface for accumulating and saving the bounding boxes of an
    object detector. By default, all detections will be logged. However, a confidence threshold can be set
    to focus on strong predictions if necessary.

    .. todo::
        It also optionally offers tensorboardX-compatible output images that can be saved
        locally or posted to tensorboard for browser-based visualization.

    Usage examples inside a session configuration file::

        # ...
        # lists all metrics to instantiate as a dictionary
        "metrics": {
            # ...
            # this is the name of the example consumer; it is used for lookup/printing only
            "logger": {
                # this type is used to instantiate the confusion matrix report object
                "type": "thelper.train.utils.DetectLogger",
                "params": {
                    # (optional) log the three 'best' detections for each target
                    "top_k": 3
                }
            },
            # ...
        }
        # ...

    Attributes:
        top_k: number of 'best' detections to keep for each target bbox (along with the target label).
            If omitted, lists all bounding box predictions by the model after applying IoU and confidence thresholds.
        conf_threshold: threshold used to eliminate uncertain predictions (if they support confidence).
            If confidence is not supported by the model bbox predictions, this parameter is ignored.
        iou_threshold: threshold used to eliminate predictions too far from target (regardless of confidence).
            If omitted, will ignore only completely non-overlapping predicted bounding boxes (:math:`IoU=0`).
            If no target bounding boxes are provided (prediction-only), this parameter is ignored.
        class_names: holds the list of class label names provided by the dataset parser. If it is not
            provided when the constructor is called, it will be set by the trainer at runtime.
        target_name: name of the targeted label (may be 'None' if all classes are used).
        target_idx: index of the targeted label (may be 'None' if all classes are used).
        viz_count: number of tensorboardX images to generate and update at each epoch.
        report_count: number of samples to print in reports (use 'None' if all samples must be printed).
        log_keys: list of metadata field keys to copy from samples into the log for each prediction.
        bbox: array used to store prediction bounding boxes for logging.
        true: array used to store groundtruth labels for logging.
        meta: array used to store metadata pulled from samples for logging.
        format: output format of the produced log (supports: text, CSV, JSON)
    """

    def __init__(self,
                 top_k=None,            # type: Optional[int]
                 conf_threshold=None,   # type: Optional[thelper.typedefs.Number]
                 iou_threshold=None,    # type: Optional[thelper.typedefs.Number]
                 class_names=None,      # type: Optional[List[AnyStr]]
                 target_name=None,      # type: Optional[AnyStr]
                 viz_count=0,           # type: int
                 report_count=None,     # type: Optional[int]
                 log_keys=None,         # type: Optional[List[AnyStr]]
                 format=None,           # type: Optional[AnyStr]
                 ):                     # type: (...) -> None
        """Receives the logging parameters & the optional class label names used to decorate the log."""
        assert top_k is None or isinstance(top_k, int) and top_k > 0, "invalid top-k value"
        assert conf_threshold is None or (isinstance(conf_threshold, (float, int)) and 0 <= conf_threshold <= 1), \
            "detection confidence threshold should be 'None' or number in [0, 1]"
        assert iou_threshold is None or (isinstance(iou_threshold, (int, float)) and 0 <= iou_threshold <= 1), \
            "detection IoU threshold should be 'None' or number in [0, 1]"
        assert isinstance(viz_count, int) and viz_count >= 0, "invalid image count to visualize"
        assert report_count is None or (
            isinstance(report_count, int) and report_count >= 0), "invalid report sample count"
        assert log_keys is None or isinstance(log_keys, list), "invalid list of sample keys to log"
        self.top_k = top_k
        self.target_name = target_name
        self.target_idx = None
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.viz_count = viz_count
        self.report_count = report_count
        self.log_keys = log_keys if log_keys is not None else []
        self.bbox = None    # type: Optional[thelper.typedefs.DetectionPredictionType]
        self.true = None    # type: Optional[thelper.typedefs.DetectionTargetType]
        self.meta = None    # type: Optional[Dict[AnyStr, List[Any]]]
        ClassNamesHandler.__init__(self, class_names)
        FormatHandler.__init__(self, format)

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this consumer."""
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}" + \
               f"(top_k={repr(self.top_k)}, conf_threshold={repr(self.conf_threshold)}, " + \
               f"class_names={repr(self.class_names)}, target_name={repr(self.target_name)}, " + \
               f"viz_count={repr(self.viz_count)}, report_count={repr(self.report_count)}, " + \
               f"log_keys={repr(self.log_keys)})"

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
               pred,         # type: thelper.typedefs.DetectionPredictionType
               target,       # type: thelper.typedefs.DetectionTargetType
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
        assert isinstance(task, thelper.tasks.Detection), "detect report only impl for detection tasks"
        assert iter_idx is not None and max_iters is not None and iter_idx < max_iters, \
            "bad iteration indices given to update function"
        if self.bbox is None or self.bbox.size != max_iters:
            self.bbox = np.asarray([None] * max_iters)
            self.true = np.asarray([None] * max_iters)
            self.meta = {key: np.asarray([None] * max_iters) for key in self.log_keys}
        if task.class_names != self.class_names:
            self.class_names = task.class_names
        if target is None or len(target) == 0 or all(len(t) == 0 for t in target):
            target = [None] * len(pred)   # simplify unpacking during report generation
        else:
            assert len(pred) == len(target), "prediction/target bounding boxes list batch size mismatch"
            for gt in target:
                assert all(isinstance(bbox, BoundingBox) for bbox in gt), \
                    "detect logger only supports 2D lists of bounding box targets"
        for det in pred:
            assert all(isinstance(bbox, BoundingBox) for bbox in det), \
                "detect logger only supports 2D lists of bounding box predictions"
        self.bbox[iter_idx] = pred
        self.true[iter_idx] = target
        for meta_key in self.log_keys:
            assert meta_key in sample, f"could not extract sample field with key {repr(meta_key)}"
            val = sample[meta_key]
            assert isinstance(val, (list, np.ndarray, torch.Tensor)), f"field {repr(meta_key)} should be batched"
            self.meta[meta_key][iter_idx] = val if isinstance(val, list) else val.tolist()

    def render(self):
        """Returns an image of predicted outputs as a numpy-compatible RGBA image drawn by pyplot."""
        if self.viz_count == 0:
            return None
        if self.bbox is None or self.true is None:
            return None
        raise NotImplementedError  # TODO

    def group_bbox(self,
                   target_bboxes,   # type: List[Optional[BoundingBox]]
                   detect_bboxes,   # type: List[BoundingBox]
                   ):               # type: (...) -> List[Dict[AnyStr, Union[BoundingBox, float, None]]]
        """Groups a sample's detected bounding boxes with target bounding boxes according to configuration parameters.

        Returns a list of detections grouped by target(s) with following format::

            [
                {
                    "target": <associated-target-bbox>,
                    "detect": [
                        {
                            "bbox": <detection-bbox>,
                            "iou": <IoU(detect-bbox, target-bbox)>
                        },
                        ...
                    ]
                },
                ...
            ]

        The associated target bounding box and :math:`IoU` can be ``None`` if no target was provided
        (ie: during inference). In this case, the returned list will have only a single element with all detections
        associated to it. A single element list can also be returned if only one target was specified for this sample.

        When multiple ground truth targets were provided, the returned list will have the same length and ordering as
        these targets. The associated detected bounding boxes will depend on IoU between target/detection combinations.

        All filtering thresholds specified as configuration parameter will be applied for the returned list. Detected
        bounding boxes will also be sorted by highest confidence (if available) or by highest IoU as fallback.
        """
        # remove low confidence and sort by highest
        group_bboxes = sorted(
            [bbox for bbox in (detect_bboxes if detect_bboxes else []) if isinstance(bbox, BoundingBox)],
            key=lambda b: b.confidence if b.confidence is not None else 0, reverse=True)
        if self.conf_threshold:
            group_bboxes = [b for b in group_bboxes if b.confidence is not None and b.confidence >= self.conf_threshold]
        sort_by_iou = all(bbox.confidence is None for bbox in group_bboxes)
        # group according to target count
        target_count = len(target_bboxes)
        if target_count == 0:
            group_bboxes = [{"target": None, "detect": [{"bbox": bbox, "iou": None} for bbox in group_bboxes]}]
        elif target_count == 1:
            sorted_detect = [{"bbox": bbox, "iou": compute_bbox_iou(bbox, target_bboxes[0])} for bbox in group_bboxes]
            if sort_by_iou:
                sorted_detect = list(sorted(sorted_detect, key=lambda d: d["iou"], reverse=True))
            group_bboxes = [{"target": target_bboxes[0], "detect": sorted_detect}]
        else:
            # regroup by highest IoU
            target_detects = [[] for _ in range(target_count)]
            for det in group_bboxes:
                # FIXME:
                #  should we do something different if all IoU = 0 (ie: false positive detection)
                #  for now, they will all be stored in the first target, but can be tracked with IoU = 0
                det_target_iou = [compute_bbox_iou(det, t) for t in target_bboxes]
                best_iou_idx = int(np.argmax(det_target_iou))
                target_detects[best_iou_idx].append({"bbox": det, "iou": det_target_iou[best_iou_idx]})
            group_bboxes = [{
                "target": target_bboxes[i],
                "detect": list(
                    sorted(target_detects[i], key=lambda d: d["iou"], reverse=True)
                ) if sort_by_iou else target_detects[i]
            } for i in range(target_count)]
        # apply filters on grouped results
        if self.iou_threshold:
            for grp in group_bboxes:
                grp["detect"] = [d for d in grp["detect"] if d["iou"] is None or d["iou"] >= self.iou_threshold]
        if self.top_k:
            for grp in group_bboxes:
                grp["detect"] = grp["detect"][:self.top_k]
        return list(group_bboxes)

    def gen_report(self):
        # type: () -> Optional[List[Dict[AnyStr, Any]]]
        """Returns the logged metadata of predicted bounding boxes per sample target in a JSON-like structure.

        For every target bounding box, the corresponding *best*-sorted detections are returned.
        Sample metadata is appended to every corresponding sub-target if any where requested.

        If ``report_count`` was specified, the returned report will be limited to that requested amount of targets.

        .. seealso::
            | :meth:`DetectLogger.group_bbox` for formatting, sorting and filtering details.
        """
        if isinstance(self.report_count, int) and self.report_count <= 0:
            return None
        if self.bbox is None or self.true is None:
            return None
        # flatten batches/samples
        pack = list(zip(*[(*pack,)
                          for packs in zip(self.bbox, self.true, *self.meta.values())
                          for pack in zip(*packs)]))
        data = {key: val for key, val in zip(["detect", "target", *self.meta.keys()], pack)}
        assert all([len(val) == len(data["target"]) for val in data.values()]), "messed up unpacking"

        # flatten targets per batches/samples
        all_targets = []
        sample_count = len(data["target"])
        for sample_idx in range(sample_count):
            if isinstance(self.report_count, int) and self.report_count >= len(all_targets):
                logger.warning(f"report max count {self.report_count} reached at {len(all_targets)} targets "
                               f"(sample {sample_idx}/{sample_count} processed)")
                break
            sample_targets = data["target"][sample_idx]
            sample_detects = data["detect"][sample_idx]
            sample_report = self.group_bbox(sample_targets, sample_detects)
            for target in sample_report:
                for k in self.meta:
                    target[k] = self.meta[k][sample_idx]
                target["target"] = {
                    "bbox": target["target"],
                    "class_name": self.class_names[target["target"].class_id]
                }
            all_targets.extend(sample_report)
        # format everything nicely as json
        for item in all_targets:
            if isinstance(item["target"]["bbox"], BoundingBox):
                item["target"].update(item["target"].pop("bbox").json())
                item["target"]["class_name"] = self.class_names[item["target"]["class_id"]]
            for det in item["detect"]:
                if isinstance(det["bbox"], BoundingBox):
                    det.update(det.pop("bbox").json())
                    det["class_name"] = self.class_names[det["class_id"]]
        return all_targets

    def report_json(self):
        # type: () -> Optional[AnyStr]
        """Returns the logged metadata of predicted bounding boxes as a JSON formatted string."""
        report = self.gen_report()
        if not report:
            return None
        return json.dumps(report, indent=4)

    def report_text(self):
        # type: () -> Optional[AnyStr]
        return self.report_csv()

    def report_csv(self):
        # type: () -> Optional[AnyStr]
        r"""Returns the logged metadata of predicted bounding boxes.

        The returned object is a print-friendly CSV string.

        Note that this string might be very long if the dataset is large or if the model tends to generate a lot of
        detections. The string will contain at least :math:`N_sample \cdot N_target` lines and each line will have
        up to :math:`N_bbox` detections, unless limited by configuration parameters.
        """
        report = self.gen_report()
        if not report:
            return None

        none_str = "unknown"

        def patch_none(to_patch, number_format='2.4f'):  # type: (Any, str) -> str
            if to_patch is None:
                return none_str
            if isinstance(to_patch, float):
                s = f"{{:{number_format}}}"
                return s.format(to_patch)
            return str(to_patch)

        header = "sample,target_name,target_bbox"
        for meta_key in self.log_keys:
            header += f",{str(meta_key)}"
        if self.top_k:
            for k in range(self.top_k):
                header += f",detect_{k + 1}_name,detect_{k + 1}_bbox,detect_{k + 1}_conf,detect_{k + 1}_iou"
        else:
            # unknown count total detections (can be variable)
            header += ",detect_name[N],detect_bbox[N],detect_conf[N],detect_iou[N],(...)[N]"
        lines = [""] * len(report)
        for i, result in enumerate(report):
            target = result["target"]
            detect = result["detect"]
            if not target:
                entry = f"{none_str},{none_str},{none_str}"
            else:
                entry = f"{target['image_id']},{patch_none(target['class_name'])},{patch_none(target['bbox'])}"
            for meta_key in self.log_keys:
                entry += f",{str(target[meta_key])}"
            for det in detect:
                entry += f",{det['class_name']},{det['bbox']},{patch_none(det['confidence'])},{patch_none(det['iou'])}"
            lines[i] = entry
        return "\n".join([header, *lines])

    def reset(self):
        """Toggles a reset of the internal state, emptying storage arrays."""
        self.bbox = None
        self.true = None
        self.meta = None


@thelper.concepts.classification
@thelper.concepts.segmentation
class ConfusionMatrix(PredictionConsumer, ClassNamesHandler):
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
                "params": {
                    # optional parameter that will indicate output as JSON is desired, plain 'text' otherwise
                    "format": "json"
                }
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
        self.pred = None
        self.target = None
        ClassNamesHandler.__init__(self, class_names)

    def __repr__(self):
        """Returns a generic print-friendly string containing info about this consumer."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + \
            f"(class_names={repr(self.class_names)}, draw_normalized={repr(self.draw_normalized)})"

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
            self.class_names = task.class_names
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
            fig, ax = thelper.draw.draw_confmat(confmat, self.class_names, normalize=self.draw_normalized)
            array = thelper.draw.fig2array(fig)
            return array
        except AttributeError as e:
            logger.warning(f"failed to render confusion matrix; caught exception:\n{str(e)}")
            # return None if rendering fails (probably due to matplotlib on display-less server)
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
        assert isinstance(consumer, PredictionConsumer), \
            "invalid consumer type, must derive from PredictionConsumer interface"
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
def _draw_wrapper(task,         # type: thelper.tasks.utils.Task
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
                  # extra params added by display callback interface below
                  save,         # type: bool
                  # all extra params will be forwarded to the display call
                  **kwargs,     # type: Any
                  # see `thelper.typedefs.IterCallbackParams` for more info
                  ):            # type: (...) -> None
    """Wrapper to :func:`thelper.draw.draw` used as a callback entrypoint for trainers."""
    res = thelper.draw.draw(task=task, input=input, pred=pred, target=target, **kwargs)
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
