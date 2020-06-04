import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """

    .. note::
        Contributed by Mario Beaulieu <mario.beaulieu@crim.ca>.

    .. seealso::
        | `Focal Loss for Dense Object Detection <https://arxiv.org/abs/1708.02002>`_,
          *Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár*, arXiv article.
    """

    def __init__(self, gamma=2, alpha=0.5, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        if self.alpha is not None:
            logpt *= self.alpha
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss
