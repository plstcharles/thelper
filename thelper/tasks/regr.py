"""Regression task interface module.

This module contains a class that defines the objectives of models/trainers for regression tasks.
"""
import logging

import numpy as np

from thelper.tasks.utils import Task

logger = logging.getLogger(__name__)


class Regression(Task):
    """Interface for n-dimension regression tasks.

    This specialization requests that when given an input tensor, the trained model should
    provide an n-dimensional target prediction. This is a fairly generic task that (unlike
    image classification and semantic segmentation) is not linked to a pre-existing set of
    possible solutions. The task interface is used to carry useful metadata for this task,
    e.g. input/output shapes, types, and min/max values for rounding/saturation.

    Attributes:
        input_shape: a numpy-compatible shape to expect model inputs to be in.
        target_shape: a numpy-compatible shape to expect the predictions to be in.
        target_type: a numpy-compatible type to cast the predictions to (if needed).
        target_min: an n-dim tensor containing minimum target values (if applicable).
        target_max: an n-dim tensor containing maximum target values (if applicable).
        input_key: the key used to fetch input tensors from a sample dictionary.
        target_key: the key used to fetch target (groundtruth) values from a sample dictionary.
        meta_keys: the list of extra keys provided by the data parser inside each sample.

    .. seealso::
        | :class:`thelper.tasks.utils.Task`
        | :class:`thelper.train.regr.RegressionTrainer`
    """

    def __init__(self, input_key, target_key, meta_keys=None, input_shape=None,
                 target_shape=None, target_type=None, target_min=None, target_max=None):
        """Receives and stores the target specs, input tensor key, and the extra (meta) keys produced by the dataset parser(s)."""
        super().__init__(input_key, target_key, meta_keys)
        if target_min is not None:
            if isinstance(target_min, (list, tuple)):
                target_min = np.asarray(target_min)
            if not isinstance(target_min, np.ndarray):
                raise AssertionError("target_min should be passed as list/tuple/ndarray")
        if target_max is not None:
            if isinstance(target_max, (list, tuple)):
                target_max = np.asarray(target_max)
            if not isinstance(target_max, np.ndarray):
                raise AssertionError("target_max should be passed as list/tuple/ndarray")
        if target_type is not None:
            if isinstance(target_min, np.ndarray) and target_min.dtype != target_type:
                raise AssertionError("invalid target min dtype")
            if isinstance(target_max, np.ndarray) and target_max.dtype != target_type:
                raise AssertionError("invalid target max dtype")
        if target_shape is not None:
            if isinstance(target_min, np.ndarray) and target_min.shape != target_shape:
                raise AssertionError("invalid target min shape")
            if isinstance(target_max, np.ndarray) and target_max.shape != target_shape:
                raise AssertionError("invalid target max shape")
        if isinstance(target_min, np.ndarray) and isinstance(target_max, np.ndarray):
            if target_min.shape != target_max.shape:
                raise AssertionError("target min/max shape mismatch")
        self.input_shape = input_shape
        if self.input_shape is not None:
            if isinstance(self.input_shape, list):
                self.input_shape = tuple(self.input_shape)
            if not isinstance(self.input_shape, tuple) or not all([isinstance(v, int) for v in self.input_shape]):
                raise AssertionError("unexpected input shape type (should be tuple of integers)")
        self.target_shape = target_shape
        if self.target_shape is not None:
            if isinstance(self.target_shape, list):
                self.target_shape = tuple(self.target_shape)
            if not isinstance(self.target_shape, tuple) or not all([isinstance(v, int) for v in self.target_shape]):
                raise AssertionError("unexpected target shape type (should be tuple of integers)")
        self.target_type = target_type
        if self.target_type is not None:
            if isinstance(self.target_type, str):
                import thelper.utils
                self.target_type = thelper.utils.import_class(self.target_type)
            if not issubclass(self.target_type, np.generic):
                raise AssertionError("target type should be a numpy-compatible type")
        self.target_min = target_min
        self.target_max = target_max

    def get_input_shape(self):
        """Returns the shape of inputs to be processed by the model."""
        return self.input_shape

    def get_target_shape(self):
        """Returns the shape of outputs to be predicted by the model."""
        return self.target_shape

    def get_target_type(self):
        """Returns the type of outputs to be predicted by the model."""
        return self.target_type

    def get_target_min(self):
        """Returns the minimum target value(s) to be predicted by the model."""
        return self.target_min

    def get_target_max(self):
        """Returns the maximum target value(s) to be predicted by the model."""
        return self.target_max

    def check_compat(self, other, exact=False):
        """Returns whether the current task is compatible with the provided one or not.

        This is useful for sanity-checking, and to see if the inputs/outputs of two models
        are compatible. If ``exact = True``, all fields will be checked for exact (perfect)
        compatibility (in this case, matching meta keys).
        """
        if isinstance(other, Regression):
            # if both tasks are related to regression: all non-None keys and specs must match
            return (self.get_input_key() == other.get_input_key() and
                    (self.get_gt_key() is None or other.get_gt_key() is None or
                     self.get_gt_key() == other.get_gt_key()) and
                    (self.get_input_shape() is None or other.get_input_shape() is None or
                     self.get_input_shape() == other.get_input_shape()) and
                    (self.get_target_shape() is None or other.get_target_shape() is None or
                     self.get_target_shape() == other.get_target_shape()) and
                    (self.get_target_type() is None or other.get_target_type() is None or
                     self.get_target_type() == other.get_target_type()) and
                    (self.get_target_min() is None or other.get_target_min() is None or
                     self.get_target_min() == other.get_target_min()) and
                    (self.get_target_max() is None or other.get_target_max() is None or
                     self.get_target_max() == other.get_target_max()) and
                    (not exact or (self.get_meta_keys() == other.get_meta_keys())))
        elif type(other) == Task:
            # if 'other' simply has no gt, compatibility rests on input key only
            return not exact and self.get_input_key() == other.get_input_key() and other.get_gt_key() is None
        return False

    def get_compat(self, other):
        """Returns a task instance compatible with the current task and the given one."""
        if not self.check_compat(other):
            raise AssertionError("cannot create compatible task instance between:\n"
                                 "\tself: %s\n\tother: %s" % (str(self), str(other)))
        meta_keys = list(set(self.get_meta_keys() + other.get_meta_keys()))
        return Regression(self.get_input_key(), self.get_gt_key(),
                          meta_keys=meta_keys, input_shape=self.get_input_shape(),
                          target_shape=self.get_target_shape(), target_type=self.get_target_type(),
                          target_min=self.get_target_min(), target_max=self.get_target_max())

    def __repr__(self):
        """Creates a print-friendly representation of a segmentation task."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + ": " + str({
            "input_key": self.get_input_key(),
            "target_key": self.get_gt_key(),
            "meta_keys": self.get_meta_keys(),
            "input_shape": self.get_input_shape(),
            "target_shape": self.get_target_shape(),
            "target_type": self.get_target_type(),
            "target_min": self.get_target_min(),
            "target_max": self.get_target_max()
        })
