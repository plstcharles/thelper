.. _use-cases:

=========
Use Cases
=========

This section is still under construction. Some example configuration files are available in the
``config`` directory of the repository (`[see them here]`__). For high-level information on generic
parts of the framework, refer to the :ref:`[user guide] <user-guide>`.

.. __: https://github.com/plstcharles/thelper/tree/master/configs


.. _use-cases-dataset-viz:

Dataset/Loader visualization
============================

Section statement here @@@@@@

.. _use-cases-image-classif:

Image classification
====================

Section statement here @@@@@@

.. _use-cases-image-segm:

Image segmentation
==================

Section statement here @@@@@@

.. _use-cases-obj-detect:

Object Detection
================

Section statement here @@@@@@

.. _use-cases-super-res:

Super-resolution
================

Section statement here @@@@@@

.. _use-cases-dataset-annot:

Dataset annotation
==================

Section statement here @@@@@@

.. _use-cases-dataset-rebalance:

Rebalancing a dataset
=====================

Section statement here @@@@@@

.. _use-cases-dataset-export:

Exporting a dataset
===================

Section statement here @@@@@@

.. _use-cases-dataset-augment:

Defining a data augmentation pipeline
=====================================

Section statement here @@@@@@

.. _use-cases-custom-trainer:

Supporting a custom trainer
===========================

Section statement here @@@@@@

.. _use-cases-custom-task:

Supporting a custom task
========================

Section statement here @@@@@@

.. _use-cases-tensorboardx:

Visualizing metrics using ``tensorboardX``
==========================================

Section statement here @@@@@@

.. _use-cases-model-reload:

Manually reloading a model
==========================

Section statement here @@@@@@

.. _use-cases-model-export:

Exporting a model
=================

Once you have trained a model (using the framework or otherwise), you might want to share
it with others. Models are typically exported in two parts: architecture and weights. However,
metadata related to the task the model was built for would be missing with only those two components.
Here, we show a solution for exporting a classification model trained using the framework under ONNX,
TraceScript, or pickle format along with its corresponding index-to-class-name mapping. Further down,
we also give tips on similarly exporting a model trained in another framework.

The advantage of ONNX and TraceScript exports is that whoever reloads your model does not need to have
the class that you used to define the model's architecture at hand. However, this approach might make
fine-tuning or retraining your model more complicated (you should consider it a 'read-only' export).

Models/checkpoints exported this way can be easily reloaded using the framework, and may also be
opened manually by others to extract only the information they need.

So, first off, let's start by training a classification model using the following configuration::

    {
        "name": "classif-cifar10",
        "datasets": {
            "cifar10": {
                "type": "torchvision.datasets.CIFAR10",
                "params": {"root": "data/cifar/train"},
                "task": {
                    "type": "thelper.tasks.Classification",
                    "params": {
                        "class_names": [
                            "airplane", "car", "bird", "cat", "deer",
                            "dog", "frog", "horse", "ship", "truck"
                        ],
                        "input_key": "0", "label_key": "1"
                    }
                }
            }
        },
        "loaders": {
            "batch_size": 32,
            "base_transforms": [
                {
                    "operation": "thelper.transforms.NormalizeMinMax",
                    "params": {
                        "min": [127, 127, 127], "max": [255, 255, 255]
                    }
                },
                {
                    "operation": "thelper.transforms.Resize",
                    "params": {"dsize": [224, 224]}
                },
                {
                    "operation": "torchvision.transforms.ToTensor"
                }
            ],
            "train_split": {"cifar10": 0.9},
            "valid_split": {"cifar10": 0.1}
        },
        "model": {"type": "thelper.nn.resnet.ResNet"},
        "trainer": {
            "epochs": 5,
            "monitor": "accuracy",
            "optimization": {
                "loss": {"type": "torch.nn.CrossEntropyLoss"},
                "optimizer": {"type": "torch.optim.Adam"}
            },
            "metrics": {
                "accuracy": {"type": "thelper.optim.CategoryAccuracy"}
            }
        }
    }

The above configuration essentially means that we will be training a ResNet model with
default settings on CIFAR10 using all 10 classes. You can launch the training process via::

    $ thelper new <PATH_TO_CLASSIF_CIFAR10_CONFIG>.json <PATH_TO_OUTPUT_DIR>

See the :ref:`[user guide] <user-guide-cli-new>` for more information on creating training
sessions. Once that's done, you should obtain a folder named ``classif-cifar10`` in your output
directory that contains training logs as well as checkpoints. To export this model
in a new checkpoint, we will use the following session configuration::

    {
        "name": "export-classif-cifar10",
        "model": {
            "ckptdata": "<PATH_TO_OUTPUT_DIR>/classif-cifar10/checkpoints/ckpt.best.pth"
        },
        "export": {
            "ckpt_name": "test-export.pth",
            "trace_name": "test-export.zip",
            "save_raw": true,
            "trace_input": "torch.rand(1, 3, 224, 224)"
        }
    }

This configuration essentially specifies where to find the 'best' checkpoint for the model we
just trained, and how to export a trace of it. For more information on the export operation, refer
to :ref:`[the user guide] <user-guide-cli-export>`. We now provide the configuration as a JSON to
the CLI one more::

    $ thelper export <PATH_TO_EXPORT_CONFIG>.json <PATH_TO_OUTPUT_DIR>

If everything goes well, ``<PATH_TO_OUTPUT_DIR>/export-classif-cifar10`` should now contain a checkpoint
with the exported model trace and all metadata required to reinstantiate it. Note that as of 2019/06,
PyTorch exports model traces as zip files, meaning you will have to copy two files from the output
session folder. In this case, that would be ``test-export.pth`` and ``test-export.zip``.

Finally, note that if you are attempting to export a model that was trained outside the framework, you
will have to specify which task this model was trained for as well as the type of the model to instantiate
and possibly the path to its weights in the ``model`` field of the configuration above. An example
configuration is given below::

    {
        "name": "export-classif-custom",
        "model": {
            "type": "fully.qualified.name.to.model",
            "params": {
                # here, provide all model constructor parameters
            },
            "weights": "path_to_model_state_dictionary.pth"
        },
        "export": {
            "ckpt_name": "test-export.pth",
            "trace_name": "test-export.zip",
            "save_raw": true,
            "trace_input": "torch.rand(1, 3, 224, 224)"
        }
    }

For more information on model importation, refer to the documentation of :meth:`thelper.nn.utils.create_model`.

`[to top] <#use-cases>`_
