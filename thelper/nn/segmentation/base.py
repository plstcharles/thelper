import torch

import thelper.nn


class SegmModelBase(thelper.nn.utils.Module):
    """
    Base wrapper class for specialized segmentation models.
    """

    model_cls = None    # type: type
    in_channels = None  # type: int

    def __init__(self, task, pretrained=False):
        """
        .. note::
            -
        """
        # note: parameter "num" goes from 0 (for EfficientNet-b0) to 7 (for EfficientNet-b7)"""
        # note: must always forward args to base class to keep backup
        super().__init__(task, **{k: v for k, v in vars().items()
                                  if k not in ["self", "task", "__class__"]})
        self.num_classes = None
        self.model = None  # will be instantiated in set_task using model_cls
        self.pretrained = pretrained
        self.set_task(task)

    def forward(self, x):
        return self.model(x)

    def set_task(self, task):
        assert isinstance(task, thelper.tasks.Segmentation), \
            "invalid task ({} currently only supports Segmentation)".format(type(self).__name__)
        num_classes = len(task.class_names)
        self.model = self.model_cls(pretrained=self.pretrained)
        if num_classes != self.num_classes:
            # Only the last layer is reinit, may all the classifier part should be reinit
            self.model.classifier[4] = torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=num_classes,
                kernel_size=(1, 1),
                stride=(1, 1),
            )
            self.num_classes = num_classes
