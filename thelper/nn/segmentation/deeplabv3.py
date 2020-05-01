import torch
import torchvision
import thelper.nn

class deeplabv3_resnet50(thelper.nn.utils.Module):

    # note: this class is just a thin wrapper for deeplabv3_resnet101 (torchvision > 0.6)

    def __init__(self, task, pretrained=False):
        # note: parameter "num" goes from 0 (for EfficientNet-b0) to 7 (for EfficientNet-b7)"""
        # note: must always forward args to base class to keep backup
        super().__init__(task, **{k: v for k, v in vars().items() if k not in ["self", "task", "__class__"]})
        self.num_classes = None
        self.model = None  # will be instantiated in set_tack
        self.pretrained = pretrained
        self.set_task(task)

    def forward(self, x):
        return self.model(x)

    def set_task(self, task):
        assert isinstance(task, thelper.tasks.Segmentation), "invalid task (EfficientNet currently only supports classif)"
        self.num_classes = len(task.class_names)
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=self.pretrained)
        if num_classes != self.num_classes:
            # Only the last layer is reinit, may all the classifier part should be reinit
            self.model.classifier[4] = torch.nn.Conv2d(in_channels=256,out_channels=num_classes,kernel_size=(1,1), stride=(1,1))
            self.num_classes = num_classes

class deeplabv3_resnet101(thelper.nn.utils.Module):

    # note: this class is just a thin wrapper for deeplabv3_resnet101 (torchvision > 0.6)

    def __init__(self, task, pretrained=False):
        # note: parameter "num" goes from 0 (for EfficientNet-b0) to 7 (for EfficientNet-b7)"""
        # note: must always forward args to base class to keep backup
        super().__init__(task, **{k: v for k, v in vars().items() if k not in ["self", "task", "__class__"]})
        self.num_classes = None
        self.model = None  # will be instantiated in set_tack
        self.pretrained = pretrained
        self.set_task(task)

    def forward(self, x):
        return self.model(x)

    def set_task(self, task):
        assert isinstance(task, thelper.tasks.Segmentation), "invalid task (EfficientNet currently only supports classif)"
        num_classes = len(task.class_names)
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=self.pretrained)
        if num_classes != self.num_classes:
            # Only the last layer is reinit, may all the classifier part should be reinit
            self.model.classifier[4] = torch.nn.Conv2d(in_channels=256,out_channels=num_classes,kernel_size=(1,1), stride=(1,1))
            self.num_classes = num_classes

