"""Explicit Tester definitions from existing Trainers."""

import thelper.concepts
from thelper.infer.base import Tester, make_tester_from_trainer
from thelper.train.classif import ImageClassifTrainer
from thelper.train.detect import ObjDetectTrainer
from thelper.train.regr import RegressionTrainer
from thelper.train.segm import ImageSegmTrainer


@thelper.concepts.classification
@make_tester_from_trainer(ImageClassifTrainer)
class ImageClassifTester(Tester):
    """Session runner specialized for testing of image classification task with safeguard against model training.

    .. seealso::
        | :class:`thelper.train.base.Tester`
        | :class:`thelper.train.base.Trainer`
        | :class:`thelper.train.classif.ImageClassifTrainer`
    """


@thelper.concepts.detection
@make_tester_from_trainer(ObjDetectTrainer)
class ObjDetectTester(Tester):
    """Session runner specialized for testing of object detection task with safeguard against model training.

    .. seealso::
        | :class:`thelper.train.base.Tester`
        | :class:`thelper.train.base.Trainer`
        | :class:`thelper.train.detect.ObjDetectTrainer`
    """


@thelper.concepts.regression
@make_tester_from_trainer(RegressionTrainer)
class RegressionTester(Tester):
    """Session runner specialized for testing of regression task with safeguard against model training.

    .. seealso::
        | :class:`thelper.train.base.Tester`
        | :class:`thelper.train.base.Trainer`
        | :class:`thelper.train.regr.RegressionTrainer`
    """


@thelper.concepts.segmentation
@make_tester_from_trainer(ImageSegmTrainer)
class ImageSegmTester(Tester):
    """Session runner specialized for testing of image segmentation task with safeguard against model training.

    .. seealso::
        | :class:`thelper.train.base.Tester`
        | :class:`thelper.train.base.Trainer`
        | :class:`thelper.train.segm.ImageSegmTrainer`
    """
