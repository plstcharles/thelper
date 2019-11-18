import efficientnet_pytorch
import torch

import thelper.nn


class EfficientNet(thelper.nn.utils.Module):
    # note: this class is just a thin wrapper for Luke Melas-Kyriazi's PyTorch adaptation of EfficientNet;
    # see https://github.com/lukemelas/EfficientNet-PyTorch for more information on the port
    # see https://arxiv.org/abs/1905.11946 for the original paper

    def __init__(self, task, num, pretrained=False):
        # note: parameter "num" goes from 0 (for EfficientNet-b0) to 7 (for EfficientNet-b7)"""
        # note: must always forward args to base class to keep backup
        super().__init__(task, **{k: v for k, v in vars().items() if k not in ["self", "task", "__class__"]})
        assert 0 <= num <= 7, "num must have a value between 0 and 7"
        self.num = num
        self.num_classes = None
        self.model = None  # will be instantiated in set_tack
        self.pretrained = pretrained
        self.set_task(task)

    def forward(self, x):
        return self.model(x)

    def set_task(self, task):
        assert isinstance(task, thelper.tasks.Classification), "invalid task (EfficientNet currently only supports classif)"
        num_classes = len(task.class_names)
        if num_classes != self.num_classes:
            if self.model is not None:
                thelper.nn.logger.warning("efficient net does not currently handle post-instantiation task updates;"
                                          " model weights will be lost!")
            if self.pretrained is True:
                self.model = efficientnet_pytorch.EfficientNet.from_pretrained(f"efficientnet-b{self.num}", num_classes)
            else:
                self.model = efficientnet_pytorch.EfficientNet.from_name(f"efficientnet-b{self.num}", num_classes)


class FCEfficientNet(EfficientNet):
    # note: this wrapper is very similar to the resnet fully convolutional model reshaper

    def __init__(self, task, ckptdata, map_location="cpu", avgpool_size=0):

        if isinstance(ckptdata, str):
            ckptdata = thelper.utils.load_checkpoint(ckptdata, map_location=map_location)
        model_type = ckptdata["model_type"]
        if model_type != "thelper.nn.efficientnet.EfficientNet":
            raise AssertionError("cannot convert non-EfficientNet model to fully conv with this impl")
        model_params = ckptdata["model_params"]
        if isinstance(ckptdata["task"], str):
            old_model_task = thelper.tasks.create_task(ckptdata["task"])
        else:
            old_model_task = ckptdata["task"]
        self.task = None
        super().__init__(old_model_task, **model_params)
        self.avgpool_size = avgpool_size
        self.load_state_dict(ckptdata["model"], strict=False)  # assumes model always stored as weight dict
        self.finallayer = torch.nn.Conv2d(self.model._fc.in_features, self.model._fc.out_features, kernel_size=1)
        self.finallayer.weight = torch.nn.Parameter(self.model._fc.weight.view(self.model._fc.out_features,
                                                                               self.model._fc.in_features, 1, 1))
        self.finallayer.bias = torch.nn.Parameter(self.model._fc.bias)
        self.set_task(task)

    def forward(self, x):
        x = self.model.extract_features(x)
        if self.avgpool_size > 0:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=self.avgpool_size, stride=1)
        x = self.finallayer(x)
        return x

    def set_task(self, task):
        assert isinstance(task, (thelper.tasks.Segmentation, thelper.tasks.Classification)), \
            "missing impl for non-segm/classif task type"
        num_classes = len(task.class_names)
        if self.model._fc.out_features != num_classes:
            self.model._fc = torch.nn.Linear(self.model._fc.in_features, num_classes)
            self.finallayer = torch.nn.Conv2d(self.model._fc.in_features, self.model._fc.out_features, kernel_size=1)
            self.finallayer.weight = torch.nn.Parameter(self.model._fc.weight.view(self.model._fc.out_features,
                                                                                   self.model._fc.in_features, 1, 1))
            self.finallayer.bias = torch.nn.Parameter(self.model._fc.bias)
        self.task = task
