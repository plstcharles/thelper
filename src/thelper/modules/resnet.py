import torch.nn
import torch.utils.model_zoo
import torchvision.models.resnet

import thelper

resnet_model_names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
resnet_model_depths = [18, 34, 50, 101, 152]
resnet_model_layers = [[2, 2, 2, 2],
                       [3, 4, 6, 3],
                       [3, 4, 6, 3],
                       [3, 4, 23, 3],
                       [3, 8, 36, 3]]


class ResNet(thelper.modules.Module):
    def __init__(self, task, name=None, depth=34, fine_tune=False, freeze_lower=True):
        super().__init__(task, name=name)
        # for more info on layers, see torchvision.models.resnet module source
        if depth not in resnet_model_depths:
            raise AssertionError("unknown resnet depth type")
        depth_idx = resnet_model_depths.index(depth)
        layers = resnet_model_layers[depth_idx]
        name = resnet_model_names[depth_idx]
        self.model = torchvision.models.ResNet(block=torchvision.models.resnet.BasicBlock, layers=layers)
        if fine_tune:
            self.model.load_state_dict(torch.utils.model_zoo.load_url(torchvision.models.resnet.model_urls[name]))
            if freeze_lower:
                for param in self.model.parameters():
                    param.requires_grad = False
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, task.get_nb_classes())

    def forward(self, x):
        return self.model(x)
