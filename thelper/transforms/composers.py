"""Transformation composers module.

All transforms should aim to be compatible with both numpy arrays and
PyTorch tensors. By default, images are processed using ``__call__``,
meaning that for a given transformation ``t``, we apply it via::

    image_transformed = t(image)

All important parameters for an operation should also be passed in the
constructor and exposed in the operation's ``__repr__`` function so that
external parsers can discover exactly how to reproduce their behavior. For
now, these representations are used for debugging more than anything else.
"""

import bisect
import logging

import torchvision.utils

import thelper.utils

logger = logging.getLogger(__name__)


class Compose(torchvision.transforms.Compose):
    """Composes several transforms together (with support for invert ops).

    This interface is fully compatible with ``torchvision.transforms.Compose``.

    .. seealso::
        | :class:`thelper.transforms.composers.CustomStepCompose`
    """

    def __init__(self, transforms):
        """Forwards the list of transformations to the base class."""
        assert isinstance(transforms, list) and transforms, "expected transforms to be provided as a non-empty list"
        if all([isinstance(stage, dict) for stage in transforms]):
            transforms = thelper.transforms.load_transforms(transforms, avoid_transform_wrapper=True)
            transforms = transforms.transforms if isinstance(transforms, Compose) else transforms
            transforms = transforms if isinstance(transforms, list) else [transforms]
        super(Compose, self).__init__(transforms)

    def invert(self, sample):
        """Tries to invert the transformations applied to a sample.

        Will throw if one of the transformations cannot be inverted.
        """
        for t in reversed(self.transforms):
            assert hasattr(t, "invert"), f"missing invert op for transform = {repr(t)}"
            sample = t.invert(sample)
        return sample

    def __getitem__(self, idx):
        """Returns the idx-th operation wrapped by the composer."""
        assert 0 <= idx < len(self.transforms), "operation index is out of range"
        return self.transforms[idx]

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + "(transforms=[\n\t" + \
            ",\n\t".join([repr(t) for t in self.transforms]) + "\n])"

    def set_seed(self, seed):
        """Sets the internal seed to use for stochastic ops."""
        if self.transforms is not None:
            for t in self.transforms:
                if hasattr(t, "set_seed") and callable(t.set_seed):
                    t.set_seed(seed)

    def set_epoch(self, epoch=0):
        """Sets the current epoch number in order to change the behavior of some suboperations."""
        assert isinstance(epoch, int) and epoch >= 0, "invalid epoch value"
        if self.transforms is not None:
            for t in self.transforms:
                if hasattr(t, "set_epoch") and callable(t.set_epoch):
                    t.set_epoch(epoch)


class CustomStepCompose(torchvision.transforms.Compose):
    """Composes several transforms together based on an epoch schedule.

    This interface is fully compatible with ``torchvision.transforms.Compose``. It can be useful if
    some operations should change their behavior over the course of a training session. Note that
    all epoch indices are assumed to be 0-based.

    Usage example in Python::

        # We will scale the resolution of input patches based on an arbitrary schedule
        # dsize = (16, 16)   if epoch < 2         (16x16 patches before epoch 2)
        # dsize = (32, 32)   if 2 <= epoch < 4    (32x32 patches before epoch 4)
        # dsize = (64, 64)   if 4 <= epoch < 8    (64x64 patches before epoch 8)
        # dsize = (112, 112) if 8 <= epoch < 12   (112x112 patches before epoch 12)
        # dsize = (160, 160) if 12 <= epoch < 15  (160x160 patches before epoch 15)
        # dsize = (196, 196) if 15 <= epoch < 18  (196x196 patches before epoch 18)
        # dsize = (224, 224) if epoch >= 18       (224x224 patches past epoch 18)
        transforms = CustomStepCompose(milestones={
            0: thelper.transforms.Resize(dsize=(16, 16)),
            2: thelper.transforms.Resize(dsize=(32, 32)),
            4: thelper.transforms.Resize(dsize=(64, 64)),
            8: thelper.transforms.Resize(dsize=(112, 112)),
            12: thelper.transforms.Resize(dsize=(160, 160)),
            15: thelper.transforms.Resize(dsize=(196, 196)),
            18: thelper.transforms.Resize(dsize=(224, 224)),
        })
        for epoch in range(100):
            transforms.set_epoch(epoch)
            for sample in loader:
                sample = transforms(sample)
                train(...)

    Attributes:
        stages: list of epochs where a new scaling factor is to be applied.
        transforms: list of transformation to apply at each stage.
        milestones: original milestones map provided in the constructor.
        epoch: index of the current epoch.

    .. seealso::
        | :class:`thelper.transforms.composers.Compose`
    """

    def __init__(self, milestones, last_epoch=-1):
        """Receives the milestone stages (or stage lists), and the initialization state.

        If the milestones do not include the first epoch (idx = 0), then no transform will be applied
        until the next specified epoch index. When last_epoch is -1, the training is assumed to start
        from scratch.

        Args:
            milestones: Map of epoch indices tied to transformation stages. Keys must be increasing.
            last_epoch: The index of last epoch. Default: -1.
        """
        assert isinstance(milestones, dict), "milestones should be provided as a dictionary"
        self.transforms, self.stages = [], []
        if len(milestones) > 0:
            assert all([isinstance(k, int) for k in milestones.keys()]) or \
                all([isinstance(k, str) for k in milestones.keys()]), \
                "milestone stages should all be indices (integers) or strings"
            if isinstance(list(milestones.keys())[0], str):  # fixup for json-based config loaders
                self.stages = [int(key) for key in milestones.keys()]
            elif isinstance(list(milestones.keys())[0], int):
                self.stages = list(milestones.keys())
            assert self.stages == sorted(self.stages), "milestone stages should be increasing integers"
        for transforms in milestones.values():
            assert isinstance(transforms, list) or hasattr(transforms, "__call__"), \
                "stage transformations should be callable"
            if isinstance(transforms, list) and all([isinstance(t, dict) for t in transforms]):
                transforms = thelper.transforms.load_transforms(transforms, avoid_transform_wrapper=True)
            self.transforms.append(transforms)
        if 0 not in self.stages:
            self.stages.insert(0, int(0))
            self.transforms.insert(0, [])
        self.milestones = milestones  # kept only for debug/display purposes
        super(CustomStepCompose, self).__init__(self.transforms)
        self.epoch = last_epoch + 1

    def _get_stage_idx(self, epoch):
        if epoch in self.stages:
            return self.stages.index(epoch)
        return max(bisect.bisect_right(self.stages, epoch) - 1, 0)

    def invert(self, sample):
        """Tries to invert the transformations applied to a sample.

        Will throw if one of the transformations cannot be inverted.
        """
        transforms = self.transforms[self._get_stage_idx(self.epoch)]
        transforms = transforms if isinstance(transforms, list) else [transforms]
        for t in reversed(transforms):
            assert hasattr(t, "invert"), f"missing invert op for transform = {repr(t)}"
            sample = t.invert(sample)
        return sample

    def __call__(self, img):
        """Applies the current stage of transformation operations to a sample."""
        transforms = self.transforms[self._get_stage_idx(self.epoch)]
        transforms = transforms if isinstance(transforms, list) else [transforms]
        for t in transforms:
            img = t(img)
        return img

    def __getitem__(self, idx):
        """Returns the idx-th operation wrapped by the composer."""
        assert 0 <= idx < len(self.transforms), "operation index is out of range"
        return self.transforms[idx]

    def __repr__(self):
        """Provides print-friendly output for class attributes."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__ + "(milestones={\n\t" + \
            ",\n\t".join([f"{str(k)} :{repr(v)}" for k, v in self.milestones.items()]) + "\n})"

    def set_seed(self, seed):
        """Sets the internal seed to use for stochastic ops."""
        transforms = self.transforms[self._get_stage_idx(self.epoch)]
        transforms = transforms if isinstance(transforms, list) else [transforms]
        for t in transforms:
            if hasattr(t, "set_seed") and callable(t.set_seed):
                t.set_seed(seed)

    def set_epoch(self, epoch=0):
        """Sets the current epoch number in order to change the behavior of some suboperations."""
        assert isinstance(epoch, int) and epoch >= 0, "invalid epoch value"
        transforms = self.transforms[self._get_stage_idx(self.epoch)]
        transforms = transforms if isinstance(transforms, list) else [transforms]
        for t in transforms:
            if hasattr(t, "set_epoch") and callable(t.set_epoch):
                t.set_epoch(epoch)
        self.epoch = epoch

    def step(self, epoch=None):  # used for interface compatibility with LRSchedulers only
        """Advances the epoch tracker in order to change the behavior of some suboperations."""
        if epoch is None:
            epoch = self.epoch + 1
        self.set_epoch(epoch=epoch)
