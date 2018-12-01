"""Transformations utilities module.

This module contains utility functions used to instantiate transformation/augmentation ops.
"""

import logging

import torchvision.transforms
import torchvision.utils

import thelper.utils

logger = logging.getLogger(__name__)


def load_transforms(stages):
    """Loads a transformation pipeline from a list of stages.

    Each entry in the provided list will be considered a stage in the pipeline. The ordering of the stages
    is important, as some transformations might not be compatible if taken out of order. The entries must
    each be dictionaries that define an operation, its parameters, and some meta-parameters (detailed below).

    The ``operation`` field of each stage will be used to dynamically import a specific type of operation to
    apply. The ``parameters`` field of each stage will then be used to pass parameters to the constructor of
    this operation. If an operation is identified as ``"Augmentor.Pipeline"``, it will be specially handled. In
    this case, a ``operations`` field in its parameters is required, and it must specify the Augmentor pipeline
    operation names and parameters (as a dictionary). Three additional optional parameter fields can also be set:
    ``input_tensor`` (bool) which specifies whether the previous stage provides a ``torch.Tensor`` to the
    augmentor pipeline (default=False); ``output_tensor`` (bool) which specifies whether the output of the
    augmentor pipeline should be converted into a ``torch.Tensor`` (default=False); and ``linked_fate`` (bool)
    which specifies whether the samples provided in lists should all have the same fate or not (default=True).

    Usage examples inside a session configuration file::

        # ...
        # the 'loaders' field can contain several transformation pipelines
        # (see 'thelper.data.utils.create_loaders' for more information on these pipelines)
        "loaders": {
            # ...
            # the 'base_transforms' operations are applied to all loaded samples
            "base_transforms": [
                {
                    "operation": "...",
                    "params": {
                        ...
                    }
                },
                {
                    "operation": "...",
                    "params": {
                        ...
                    }
                }
            ],
        # ...

    Args:
        stages: a list defining a series of transformations to apply as a single pipeline.

    Returns:
        A transformation pipeline object compatible with the ``torchvision.transforms`` interfaces.

    .. seealso::
        | :class:`thelper.transforms.wrappers.AugmentorWrapper`
        | :func:`thelper.transforms.utils.load_augments`
        | :func:`thelper.data.utils.create_loaders`
    """
    if not isinstance(stages, list):
        raise AssertionError("expected stages to be provided as a list")
    if not stages:
        return None, True  # no-op transform, and dont-care append
    if not isinstance(stages[0], dict):
        raise AssertionError("expected each stage to be provided as a dictionary")
    operations = []
    for stage_idx, stage in enumerate(stages):
        if "operation" not in stage or not stage["operation"]:
            raise AssertionError("stage #%d is missing its operation field" % stage_idx)
        operation_name = stage["operation"]
        if "params" in stage and not isinstance(stage["params"], dict):
            raise AssertionError("stage #%d parameters are not provided as a dictionary" % stage_idx)
        operation_params = stage["params"] if "params" in stage else {}
        if operation_name == "Augmentor.Pipeline":
            import Augmentor
            augp = Augmentor.Pipeline()
            if "operations" not in operation_params:
                raise AssertionError("missing mandatory augmentor pipeline config 'operations' field")
            augp_operations = operation_params["operations"]
            if not isinstance(augp_operations, dict):
                raise AssertionError("augmentor pipeline 'operations' field should contain dictionary")
            for augp_op_name, augp_op_params in augp_operations.items():
                getattr(augp, augp_op_name)(**augp_op_params)
            if "input_tensor" in operation_params and thelper.utils.str2bool(operation_params["input_tensor"]):
                operations.append(torchvision.transforms.ToPILImage())
            linked_fate = thelper.utils.str2bool(operation_params["linked_fate"]) if "linked_fate" in operation_params else True
            operations.append(thelper.transforms.wrappers.AugmentorWrapper(augp, linked_fate))
            if "output_tensor" in operation_params and thelper.utils.str2bool(operation_params["output_tensor"]):
                operations.append(torchvision.transforms.ToTensor())
        else:
            operation_type = thelper.utils.import_class(operation_name)
            operation = operation_type(**operation_params)
            operations.append(operation)
    if len(operations) > 1:
        return thelper.transforms.Compose(operations)
    elif len(operations) == 1:
        return operations[0]
    else:
        return None


def load_augments(config):
    """Loads a data augmentation pipeline.

    An augmentation pipeline is essentially a specialized transformation pipeline that can be appended or
    prefixed to the base transforms defined for all samples. Most importantly, it can increase the number
    of samples in a dataset based on the duplication/tiling of input samples from the dataset parser.

    Usage examples inside a session configuration file::

        # ...
        # the 'loaders' field can contain several augmentation pipelines
        # (see 'thelper.data.utils.create_loaders' for more information on these pipelines)
        "loaders": {
            # ...
            # the 'train_augments' operations are applied to training samples only
            "train_augments": {
                # specifies whether to apply the augmentations before or after the base transforms
                "append": false,
                "transforms": [
                    {
                        # here, we use a single stage, which is actually an augmentor sub-pipeline
                        # that is purely probabilistic (i.e. it does not increase input sample count)
                        "operation": "Augmentor.Pipeline",
                        "params": {
                            "operations": {
                                # the augmentor pipeline defines two operations: rotations and flips
                                "rotate_random_90": {"probability": 0.75},
                                "flip_random": {"probability": 0.75}
                            }
                        }
                    }
                ]
            ],
            # the 'eval_augments' operations are applied to validation/test samples only
            "eval_augments": {
                # specifies whether to apply the augmentations before or after the base transforms
                "append": false,
                "transforms": [
                    # here, we use a combination of a sample duplicator and a random cropper; this
                    # increases the number of samples provided by the dataset parser
                    {
                        "operation": "thelper.transforms.Duplicator",
                        "params": {
                            "count": 10
                        }
                    },
                    {
                        "operation": "thelper.transforms.ImageTransformWrapper",
                        "params": {
                            "operation": "thelper.transforms.RandomResizedCrop",
                            "params": {
                                "output_size": [224, 224],
                                "input_size": [0.1, 1.0],
                                "ratio": 1.0
                            },
                            "linked_fate": false
                        }
                    },
                ]
            ],
        # ...

    Args:
        config: the configuration dictionary defining the meta parameters as well as the list of transformation
            operations of the augmentation pipeline.

    Returns:
        A tuple that consists of a pipeline compatible with the ``torchvision.transforms`` interfaces, and
        a bool specifying whether this pipeline should be appended or prefixed to the base transforms.

    .. seealso::
        | :class:`thelper.transforms.wrappers.AugmentorWrapper`
        | :func:`thelper.transforms.utils.load_transforms`
        | :func:`thelper.data.utils.create_loaders`
    """
    if not isinstance(config, dict):
        raise AssertionError("augmentation config should be provided as dictionary")
    augments = None
    augments_append = False
    if "append" in config:
        augments_append = thelper.utils.str2bool(config["append"])
    if "transforms" in config and config["transforms"]:
        augments = thelper.transforms.load_transforms(config["transforms"])
    return augments, augments_append
