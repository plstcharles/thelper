import numpy as np
import torch

import thelper


class LeNet(thelper.nn.Module):
    """LeNet CNN implementation.

    See http://yann.lecun.com/exdb/lenet/ for more information.

    This is NOT a modern architecture; it is only provided here for tutorial purposes.
    """

    def __init__(self, task, input_shape=(1, 28, 28), conv1_filters=6, conv2_filters=16,
                 hidden1_size=120, hidden2_size=84, output_size=10):
        super().__init__(task)
        padding = 2 if input_shape[1] == 28 else 0
        self.baseline = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_shape[0], out_channels=conv1_filters, kernel_size=5, stride=1, padding=padding),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=conv1_filters, out_channels=conv2_filters, kernel_size=5, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.classifier_input_size = conv2_filters * 5 * 5
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.classifier_input_size, out_features=hidden1_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden1_size, out_features=hidden2_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden2_size, out_features=output_size),
        )

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        feature_maps = self.baseline(x)
        embeddings = feature_maps.view(-1, self.classifier_input_size)
        logits = self.classifier(embeddings)
        return logits

    def set_task(self, task):
        assert isinstance(task, thelper.tasks.Classification), "missing impl for non-classif task type"
        num_classes = len(task.class_names)
        if num_classes != self.classifier[4].out_features:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.classifier_input_size, out_features=self.classifier[0].out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.classifier[2].in_features, out_features=self.classifier[2].out_features),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.classifier[4].in_features, out_features=num_classes),
            )
        self.task = task
