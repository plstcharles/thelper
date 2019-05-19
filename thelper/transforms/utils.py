"""Transformations utilities module.

This module contains utility functions used to instantiate transformation/augmentation ops.
"""

import logging

import torchvision.transforms
import torchvision.utils

import thelper.utils

logger = logging.getLogger(__name__)


def load_transforms(stages, avoid_transform_wrapper=False):
    """Loads a transformation pipeline from a list of stages.

    Each entry in the provided list will be considered a stage in the pipeline. The ordering of the stages
    is important, as some transformations might not be compatible if taken out of order. The entries must
    each be dictionaries that define an operation, its parameters, and some meta-parameters (detailed below).

    The ``operation`` field of each stage will be used to dynamically import a specific type of operation to
    apply. The ``params`` field of each stage will then be used to pass parameters to the constructor of
    this operation.

    If an operation is identified as ``"Augmentor.Pipeline"`` or ``"albumentations.Compose"``, it will be
    specially handled. In both case, the ``params`` field becomes mandatory in the stage dictionary, and it
    must specify the Augmentor or albumentations pipeline operation names and parameters (as a dictionary).
    Two additional optional config fields can then be set for Augmentor pipelines: ``input_tensor`` (bool)
    which specifies whether the previous stage provides a ``torch.Tensor`` to the pipeline (default=False);
    and ``output_tensor`` (bool) which specifies whether the output of the pipeline should be converted into
    a tensor (default=False). For albumentations pipelines, two additional fields are also available, namely
    ``bbox_params`` (dict) and ``keypoint_params`` (dict). For more information on these, refer to the
    documentation of ``albumentations.core.composition.Compose``. Finally, when unpacking dictionaries for
    albumentations pipelines, the keys associated to bounding boxes/masks/keypoints that must be forwarded
    to the composer can be specified via the ``bboxes_key``, ``mask_key``, and ``keypoints_key`` fields.

    All operations can also specify which sample components they should be applied to via the ``target_key``
    field. This field can contain a single key (typically a string), or a list of keys. The operation will
    be applied at runtime to all values which are found in the samples with one of those keys. If no key is
    provided for an operation, it will be applied to all array-like components of the sample. Finally, all
    operations can specify a ``linked_fate`` field (bool) to specify whether the samples provided in lists
    should all have the same fate or not (default=True).

    Usage examples inside a session configuration file::

        # ...
        # the 'loaders' field may contain several transformation pipelines
        # (see 'thelper.data.utils.create_loaders' for more information on these pipelines)
        "loaders": {
            # ...
            # the 'base_transforms' operations are applied to all loaded samples
            "base_transforms": [
                {
                    "operation": "...",
                    "params": {
                        ...
                    },
                    "target_key": [ ... ],
                    "linked_fate": ...
                },
                {
                    "operation": "...",
                    "params": {
                        ...
                    },
                    "target_key": [ ... ],
                    "linked_fate": ...
                },
                ...
            ],
        # ...

    Args:
        stages: a list defining a series of transformations to apply as a single pipeline.

    Returns:
        A transformation pipeline object compatible with the ``torchvision.transforms`` interface.

    .. seealso::
        | :class:`thelper.transforms.wrappers.AlbumentationsWrapper`
        | :class:`thelper.transforms.wrappers.AugmentorWrapper`
        | :class:`thelper.transforms.wrappers.TransformWrapper`
        | :func:`thelper.transforms.utils.load_augments`
        | :func:`thelper.data.utils.create_loaders`
    """
    assert isinstance(stages, list), "expected stages to be provided as a list"
    if not stages:
        return None, True  # no-op transform, and dont-care append
    assert all([isinstance(stage, dict) for stage in stages]), "expected all stages to be provided as dictionaries"
    operations = []
    for stage_idx, stage in enumerate(stages):
        assert "operation" in stage and stage["operation"], f"stage #{stage_idx} is missing its operation field"
        operation_name = stage["operation"]
        operation_params = thelper.utils.get_key_def(["params", "parameters"], stage, {})
        assert isinstance(operation_params, dict), f"stage #{stage_idx} parameters are not provided as a dictionary"
        operation_targets = None
        if "target_key" in stage:
            assert isinstance(stage["target_key"], (list, str, int)), \
                f"stage #{stage_idx} target keys are not provided as a list or string/int"
            operation_targets = stage["target_key"] if isinstance(stage["target_key"], list) else [stage["target_key"]]
        linked_fate = thelper.utils.str2bool(stage["linked_fate"]) if "linked_fate" in stage else True
        if operation_name == "Augmentor.Pipeline":
            import Augmentor
            pipeline = Augmentor.Pipeline()
            assert isinstance(operation_params, dict) and operation_params, \
                "augmentor pipeline 'params' field should contain dictionary of suboperations"
            for pipeline_op_name, pipeline_op_params in operation_params.items():
                getattr(pipeline, pipeline_op_name)(**pipeline_op_params)
            if "input_tensor" in stage and thelper.utils.str2bool(stage["input_tensor"]):
                operations.append(torchvision.transforms.ToPILImage())
            operations.append(thelper.transforms.wrappers.AugmentorWrapper(pipeline, operation_targets, linked_fate))
            if "output_tensor" in stage and thelper.utils.str2bool(stage["output_tensor"]):
                operations.append(torchvision.transforms.ToTensor())
        elif operation_name == "albumentations.Compose":
            assert isinstance(operation_params, dict) and operation_params, \
                "albumentations pipeline 'params' field should contain dictionary of suboperations"
            suboperations = []
            for op_name, op_params in operation_params.items():
                if not op_name.startswith("albumentations."):
                    op_name = "albumentations." + op_name
                op_type = thelper.utils.import_class(op_name)
                suboperations.append(op_type(**op_params))
            probability = thelper.utils.get_key_def("probability", stage, 1.0)
            to_tensor = thelper.utils.get_key_def("to_tensor", stage, None)
            bbox_params = thelper.utils.get_key_def("bbox_params", stage, {})
            add_targets = thelper.utils.get_key_def("add_targets", stage, {})
            bboxes_key = thelper.utils.get_key_def("bboxes_key", stage, "bbox")
            mask_key = thelper.utils.get_key_def("mask_key", stage, "mask")
            keypoints_key = thelper.utils.get_key_def("keypoints_key", stage, "keypoints")
            cvt_kpts_to_bboxes = thelper.utils.str2bool(thelper.utils.get_key_def("cvt_kpts_to_bboxes", stage, False))
            operations.append(thelper.transforms.wrappers.AlbumentationsWrapper(
                transforms=suboperations, to_tensor=to_tensor, bbox_params=bbox_params, add_targets=add_targets,
                image_key=operation_targets, bboxes_key=bboxes_key, mask_key=mask_key, keypoints_key=keypoints_key,
                probability=probability, cvt_kpts_to_bboxes=cvt_kpts_to_bboxes, linked_fate=linked_fate))
        else:
            operation_type = thelper.utils.import_class(operation_name)
            operation = operation_type(**operation_params)
            if not avoid_transform_wrapper and not isinstance(operation, (thelper.transforms.wrappers.TransformWrapper,
                                                                          thelper.transforms.operations.NoTransform,
                                                                          torchvision.transforms.Compose)):
                operations.append(thelper.transforms.wrappers.TransformWrapper(operation,
                                                                               target_keys=operation_targets,
                                                                               linked_fate=linked_fate))
            else:
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
    prefixed to the base transforms defined for all samples. Augmentations are typically used to diversify
    the samples within the training set in order to help model generalization. They can also be applied to
    validation and test samples in order to get multiple responses for the same input so that they can
    be averaged/concatenated into a single output.

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
                            # the augmentor pipeline defines two operations: rotations and flips
                            "rotate_random_90": {"probability": 0.75},
                            "flip_random": {"probability": 0.75}
                        }
                    }
                ]
            },
            # ...
        }
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
    assert isinstance(config, dict), "augmentation config should be provided as dictionary"
    augments = None
    augments_append = False
    if "append" in config:
        augments_append = thelper.utils.str2bool(config["append"])
    if "transforms" in config and config["transforms"]:
        augments = thelper.transforms.load_transforms(config["transforms"])
    return augments, augments_append
