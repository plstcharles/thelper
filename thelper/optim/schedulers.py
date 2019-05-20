"""Schedulers module.

This module contains classes used for scheduling learning rate changes while training a model. All
classes defined here should derive from ``torch.optim.lr_scheduler._LRScheduler`` to remain torch-
compatible.
"""

import bisect
import logging

import torch.nn.functional

logger = logging.getLogger(__name__)


class CustomStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Sets the learning rate of each parameter group using a dictionary of preset scaling factors
    for epoch-based milestones.

    This class can be useful for tuning the learning rate scheduling behavior of a training session
    beyond what is already possible using PyTorch's existing LR scheduler classes.  Note that all
    epoch indices are assumed to be 0-based.

    Usage example in Python::

        # Assuming the optimizer uses lr = 0.05, we hard-code a slow startup...
        # lr = 0.00625   if epoch < 2        (1/8 scale before epoch 2)
        # lr = 0.0125    if 2 <= epoch < 3   (1/4 scale before epoch 3)
        # lr = 0.025     if 3 <= epoch < 4   (1/2 scale before epoch 4)
        # lr = 0.05      if 4 <= epoch < 30  (default scale between epoch 4 and 30)
        # lr = 0.005     if 30 <= epoch < 80 (1/10 scale past epoch 30)
        # lr = 0.0005    if epoch >= 80      (1/100 scale past epoch 80)
        scheduler = CustomStepLR(optimizer, milestones={
            0: 1/8,
            2: 1/4,
            3: 1/2,
            4: 1,
            30: 0.1,
            80: 0.01
        })
        for epoch in range(100):
            scheduler.step(epoch)
            train(...)
            validate(...)

    Usage example inside a session configuration file::

        # ...
        # lists the model optimization parameters for the training session
        "optimization": {
            # lists the optimizer arguments (type, parameters, LR, ...)
            "optimizer": {
                # ...
            },
            # lists the scheduler arguments (field can be omitted if no scheduler is needed)
            "scheduler": {
                # the type used to instantiate the scheduler
                "type": "thelper.optim.schedulers.CustomStepLR",
                # the parameters passed to the scheduler's constructor
                "params": {
                    # by default, the optimizer is passed automatically;
                    # we only need to specify the extra parameters here
                    "milestones": {
                        "1": 1,  # after epoch 1, scale the LR by 1
                        "10": 0.1, # after epoch 10, scale the LR by 0.1
                        "20": 0.01,  # ... and so on
                        "30": 0.001,
                        "40": 0.0001
                    }
                }
            }
        },
        # ...

    Attributes:
        stages: list of epochs where a new scaling factor is to be applied.
        scales: list of scaling factors to apply at each stage.
        milestones: original milestones map provided in the constructor.

    .. seealso::
        | :class:`thelper.train.base.Trainer`
    """

    def __init__(self, optimizer, milestones, last_epoch=-1):
        """Receives the optimizer, milestone scaling factor, and initialization state.

        If the milestones do not include the first epoch (idx = 0), then its scaling factor is set
        to 1. When last_epoch is -1, the training is assumed to start from scratch.

        Args:
            optimizer: Wrapped optimizer (PyTorch-compatible object).
            milestones: Map of epoch indices tied to scaling factors. Keys must be increasing.
            last_epoch: The index of last epoch. Default: -1.
        """
        if not isinstance(milestones, dict):
            raise AssertionError("milestones should be provided as a dictionary")
        self.stages = []
        if len(milestones) > 0:
            if isinstance(list(milestones.keys())[0], str):  # fixup for json-based config loaders
                self.stages = [int(key) for key in milestones.keys()]
            elif isinstance(list(milestones.keys())[0], int):
                self.stages = list(milestones.keys())
            else:
                raise AssertionError("milestone stages should be epoch indices (integers)")
            if self.stages != sorted(self.stages):
                raise AssertionError("milestone stages should be increasing integers")
            if not isinstance(list(milestones.values())[0], (float, int)):
                raise AssertionError("milestone scaling factors should be int/float")
        self.scales = [float(scale) for scale in milestones.values()]
        if 0 not in self.stages:
            self.stages.insert(0, int(1))
            self.scales.insert(0, float(1))
        self.milestones = milestones
        super().__init__(optimizer, last_epoch)

    def _get_stage_idx(self, epoch):
        if epoch in self.stages:
            return self.stages.index(epoch)
        return max(bisect.bisect_right(self.stages, epoch) - 1, 0)

    def get_lr(self):
        """Returns the learning rate to use given the current epoch and scaling factors."""
        scale = self.scales[self._get_stage_idx(self.last_epoch)]
        return [base_lr * scale for base_lr in self.base_lrs]
