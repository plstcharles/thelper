"""
Typing definitions for thelper.
"""

from typing import Any, AnyStr, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING  # noqa: F401
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import io

if TYPE_CHECKING:
    from thelper.tasks.utils import Task

    ArrayType = np.ndarray
    ArrayShapeType = Union[List[int], Tuple[int]]
    OneOrManyArrayType = Union[List[ArrayType], ArrayType]
    LabelColorMapType = Union[ArrayType, Dict[int, ArrayType]]
    LabelIndex = AnyStr
    LabelType = AnyStr
    LabelDict = Dict[LabelIndex, LabelType]
    LabelList = List[LabelType]
    DrawingType = Union[Tuple[plt.Figure, plt.Axes], None]

    # iteration callbacks should have the following signature:
    #   func(sample, pred, iter_idx, max_iters, epoch_idx, max_epochs)
    SampleType = Dict[Union[AnyStr, int], Tensor]
    PredictionType = Tensor
    IterCallbackType = Optional[Callable[[SampleType, Task, PredictionType, int, int, int, int], None]]
    IterCallbackParams = ["sample", "task", "pred", "iter_idx", "max_iters", "epoch_idx", "max_epochs"]

    ConfigIndex = AnyStr
    ConfigValue = Union[AnyStr, bool, float, int]
    ConfigDict = Dict[ConfigIndex, ConfigValue]

    CheckpointLoadingType = Union[AnyStr, io.FileIO]
    CheckpointContentType = Dict[AnyStr, Any]
    MapLocationType = Union[Callable, AnyStr, Dict[AnyStr, AnyStr]]
