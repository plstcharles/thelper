.. _use-cases:

=========
Use Cases
=========

This section is still under construction. Some example configuration files are available in the
``config`` directory of the repository (`[see them here]`__). For high-level information on generic
parts of the framework, refer to the :ref:`[user guide] <user-guide>`.

.. __: https://github.com/plstcharles/thelper/tree/master/configs


.. _use-cases-image-classif:

Image classification
====================

Building an image classifier is probably the simplest thing you can do with the framework. Here, we
provide an in-depth look at a JSON configuration used to built a 10-class object classification model based
on the CIFAR-10 dataset.

As usual, we must define four different fields in our configuration for everything to work: the datasets,
the data loaders, the model, and the trainer. First off, the dataset. As mentionned before, here we will
work with the CIFAR-10 dataset provided by ``torchvision``. We do so simply to have a configuration
that can run "off-the-shelf", but in reality, you will probably be using your own data. To learn how to
create your own dataset interface and load your own data, see :ref:`[this section] <use-cases-dataset-new>`.
So, for our simple task, we define the ``datasets`` field as follows::

    "datasets": {
        "cifar10": {
            "type": "torchvision.datasets.CIFAR10",
            "params": {"root": "data/cifar/train", "download": true},
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
    }

In summary, we will instantiate a single dataset named "cifar10" based on the ``torchvision.datasets.CIFAR10``
class. The constructor of that class will receive two arguments, namely the (relative) path where to save
the data, and a boolean indicating that the dataset should be downloaded if not already available. More info
on these two arguments can actually be found in the `[PyTorch documentation] <pytorch-cifar10-doc_>`_. Here,
since this dataset interface does not explicitly define a "task", we need to provide one ourselves. Therefore,
we add a "task" field in which we specify that the task type is related to classification, and provide the
task's construction arguments. In this case, this is the list of class names that correspond to the indices
that the dataset interface will be associating to each loaded sample, and the key strings. Note that since
``torchvision.datasets.CIFAR10`` will be loading samples as tuples, so the key strings merely correspond to
tuple indices.

  .. _pytorch-cifar10-doc: https://pytorch.org/docs/stable/torchvision/datasets.html#cifar

Next, we will define how this cute little dataset will be used to load training and/or evaluation data. This
is accomplished through the ``loaders`` fields as follows::

    "loaders": {
        "train_split": {"cifar10": 0.9},
        "valid_split": {"cifar10": 0.1},
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
        ]
    }

In this example, we ask the training and validation data loaders to split the "cifar10" dataset we defined
above using a 90%-10% ratio. By default, this split will be randomized, but repeatable across experiments.
Seeds used to shuffle the data samples will be printed in the logs, and can even be set manually in this
section if needed. Next, we set the batch size for all data loaders to 32, and define base transforms to
apply to all samples. In this case, the transforms will normalize each 8-bit image by min-maxing it, resize
it 224x224 (as required by our model), and finally transform it into a tensor. Many transformation operations
can be defined in the ``loaders`` section, and it may be interesting to visualize the output before trying
to feed it to a model. For more information on how to do so, see :ref:`[this use case] <use-cases-dataset-viz>`.

Next, we will define the model architecture to train. Again, you might want to use your own architecture here.
If so, refer to the :ref:`[relevant use case] <use-cases-custom-model>`. Here, we keep things extremely simple
and rely on the pre-existing ResNet implementation located inside the framework::

    "model": {
        "type": "thelper.nn.resnet.ResNet"
    }

Since this class can be instantiated as-is without having to provide any arguments (it defaults to ResNet-34),
we do not even need to specify a "params" field. Once created, this model will be adapted to our classification
problem by providing it the "task" object we defined in our dataset interface. This means that its default
1000-class output layer will be updated to produce 10 outputs (since we have 10 classes).

Finally, we define the ``trainer`` field to provide all remaining training parameters::

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

Here, we limit the training to 5 epochs. The loss is a traditional cross-entropy, and we use Adam to
update model weights via backprop. The loss function and optimizer could both receive extra parameters (using a
"params" field once more), but we keep the defaults everywhere. Finally, we define a single metric to be
evaluated during training (accuracy), and set it as the "monitoring" metric to use for early stopping.

The complete configuration is shown below::

    {
        "name": "classif-cifar10",
        "datasets": {
            "cifar10": {
                "type": "torchvision.datasets.CIFAR10",
                "params": {"root": "data/cifar/train", "download": true},
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

Once saved to a json file, we will be able to launch the training session via::

    $ thelper new <PATH_TO_CLASSIF_CIFAR10_CONFIG>.json <PATH_TO_OUTPUT_DIR>

The dataset will first be downloaded, split, and passed to data loaders. Then, the model will be
instantiated, and all objects will be given to the trainer to start the session. Right away, some
log files will be created in a new folder named "classif-cifar10" in the directory provided as
the second argument on the command line. Once the training is complete, that folder will
contain the model checkpoints as well as the final evaluation results.

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

.. _use-cases-dataset-new:

Creating a new dataset interface
================================

Section statement here @@@@@@

.. _use-cases-dataset-viz:

Dataset/Loader visualization
============================

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

.. _use-cases-custom-model:

Supporting a custom model
=========================

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
