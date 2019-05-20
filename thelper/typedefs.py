"""
Typing definitions for thelper.
"""

import io
from typing import TYPE_CHECKING, Any, AnyStr, Callable, Dict, List, Optional, Tuple, Union  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:

    from thelper.tasks.utils import Task
    from thelper.nn.utils import Module
    from thelper.data.loaders import DataLoader

else:

    class Task:
        pass

    class Module(torch.nn.Module):
        pass

    class DataLoader(torch.utils.data.DataLoader):
        pass

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
SampleType = Dict[Union[AnyStr, int], torch.Tensor]
PredictionType = torch.Tensor
IterCallbackType = Optional[Callable[[SampleType, Task, PredictionType, int, int, int, int], None]]
IterCallbackParams = ["sample", "task", "pred", "iter_idx", "max_iters", "epoch_idx", "max_epochs"]

ConfigIndex = AnyStr
ConfigValue = Union[AnyStr, bool, float, int, List[Any], Dict[Any, Any]]
ConfigDict = Dict[ConfigIndex, ConfigValue]

CheckpointLoadingType = Union[AnyStr, io.FileIO]
CheckpointContentType = Dict[AnyStr, Any]
MapLocationType = Union[Callable, AnyStr, Dict[AnyStr, AnyStr]]

ModelType = Module
LoaderType = DataLoader
MultiLoaderType = Tuple[Optional[LoaderType], Optional[LoaderType], Optional[LoaderType]]
