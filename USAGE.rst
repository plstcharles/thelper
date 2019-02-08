.. _user-guide:

==========
User Guide
==========

This guide provides an overview of the basic functionalities and typical use cases of the thelper
framework. For installation instructions, refer to the installation guide :ref:`[here] <install-guide>`.

Currently, the framework can be used to tackle image classification, image segmentation, and generic
regression tasks out-of-the-box using PyTorch. Direct support for object detection is planned in the
not-so-distant future, and more tasks will follow. The goal of the framework is not to solve those
problems for you; its goal is to facilitate your model exploration and development process. This is
achieved by providing a centralized interface for the control of all your experiment settings, by
offering a simple solution for model checkpointing and fine-tuning, and by providing debugging tools
and visualizations to help you understand your model's behavior. It can also help users working with
GPU clusters by keeping track of their jobs more easily. This framework will not directly give you the
perfect solution for your particular problem, but it will help you discover a solution while enforcing
good reproducibility standards.

If your problem is related to one of the aforementionned tasks, and if you can solve this problem using
a standard model architecture already included in PyTorch or in the framework itself, then you might be
able to train and export a solution without writing a single line of code. Examples of such use-cases
`are detailed below <#use-case-examples>`_. It is however typical to work with a custom model, a custom
trainer, or even a custom task/objective. This is also supported by the framework, as most classes can
be either imported as-is, or they can replace the internal classes of the framework by inheriting their
parent interface.

In the sections below, we first introduce the framework's `Command-Line Interface (CLI)
<#command-line-interface>`_ used to launch jobs, the `session configuration files <#configuration-files>`_
used to define the settings of these jobs, and the `session directories <#session-directories>`_ that
contain job outputs. Then, we 


Command-Line Interface
======================

The Command-Line Interface (CLI) of the framework offers the main entrypoint from which jobs are executed.
A number of different operations are supported; these are detailed in the following subsections, and
listed :ref:`[in the documentation] <thelper:thelper.cli module>`. For now, note that these operations
all rely on a configuration dictionary which is typically parsed from a JSON file. The fields of this
dictionary that are required by each operation are detailed `in the next section <#configuration-files>`_.

Note that using the framework's CLI is not mandatory. If you prefer bypassing it and creating your own
high-level job dispatcher, you can do so by deconstructing one of the already-existing CLI entrypoints,
and by calling the same high-level functions it uses to load the components you need. These might include
for example :meth:`thelper.data.utils.create_loaders` and :meth:`thelper.nn.utils.create_model`. Calling
those functions directly may also be necessary if you intend on embedding the framework inside another
application.


Creating a training session
---------------------------

To create a training session, the ``new`` operation of the CLI is used. This redirects the execution flow
of the CLI to :meth:`thelper.cli.create_session`. The configuration dictionary that is provided must
contain all sections required to train a model, namely ``datasets``, ``loaders``, ``model``, and
``trainer``. It is also mandatory to provide a ``name`` field in the global space for the training session
to be properly identified later on.

No distinction is made at this stage regarding the task that the training session is tackling. The nature
of this task (e.g. image classification) will be deduced from the ``datasets`` section of the configuration
later in the process. This CLI entrypoint can therefore be used to start training sessions for any task.

Finally, note that since starting a training session produces logs and data, the path to a directory where
the output can be created must be provided as the second argument.

Usage from the terminal::

  $ thelper new <PATH_TO_CONFIG_FILE.json> <PATH_TO_SAVE_DIRECTORY>


Resuming a training session
---------------------------

If a previously created training session was halted for any reason, it is possible to resume it with the
``resume`` operation of the CLI. To do so, you must provide either the path to the session directory
created by the training session itself, or the path to a checkpoint that should be loaded directly. If a
directory is given, it will be searched for checkpoints and the latest one will be loaded. The training
session will then be resumed using the loaded model and optimizer state, and subsequent outputs will be
saved in the original session directory.

A session can be resumed with an overriding configuration dictionary adding (for example) new metrics.
If no dictionary is provided at all, the original one contained in the loaded checkpoint will be used.
Compatibility between an overriding configuration dictionary and the original one must be ensured by the
user. A session can also be resumed only to evaluate the (best) trained model performance on the testing
set. This is done by adding the ``--eval-only`` flag at the end of the command line. For more information
on the parameters, see the documentation of :meth:`thelper.cli.resume_session`.

Usage from the terminal::

  $ thelper resume <PATH_TO_SESSION_DIR_OR_CHECKPT> [-m MAP_LOCATION] [-c OVERRIDE_CFG] [...]


Visualizing data
----------------

Visualizing the images that will be forwarded to the model during training after applying data
augmentation operations can be useful to determine whether they still look natural or not. The ``viz``
operation of the CLI allows you to do just this using the dataset parsers or data loaders defined in a
configuration dictionary that would normally be given to the CLI under the ``new`` or ``resume``
operation modes. For more information on this mode, see the documentation of
:meth:`thelper.cli.visualize_data`.

Usage from the terminal::

  $ thelper viz <PATH_TO_CONFIG_FILE.json>


Annotating data
---------------

Lastly, the ``annot`` CLI operation allows the user to browse a dataset and annotate individual
samples from it using a specialized GUI tool. The configuration dictionary that is provided must contain
a ``datasets`` section to define the parsers that load the data, and an ``annotator`` section that defines
the GUI tool settings used to create annotations. During an annotation session, all annotations that are
created by the user will be saved into the session directory. For more information on the parameters,
refer to the documentation of :meth:`thelper.cli.annotate_data`.

Usage from the terminal::

  $ thelper annot <PATH_TO_CONFIG_FILE.json> <PATH_TO_SAVE_DIRECTORY>


Configuration Files
===================

Configuration files are at the heart of the framework. These essentially contain all the settings that
might affect the behavior of a training session, and therefore of a trained model. The framework itself
does not enforce that all parameters must be passed through the configuration file, but it is a good
idea to respect this, as it helps enforce reproducibility. On the other hand, the framework will
automatically skips sections of the configuration file that it does not need to use or that it does not
understand. This is useful when sections or subsections are added for custom needs, or when only a portion
of the configuration is relevant to some use case (for example, the 'visualization' mode of the CLI will
only look at the datasets and data loaders sections).

For now, all configuration files are expected to be in JSON format, but future versions of the framework
will support YAML configurations as well as raw python modules (.py files) that define each section
as a dictionary.


Datasets section
----------------

The ``datasets`` section of the configuration defines the dataset "parsers" that will be instantiated by
the framework, and passed to the data loaders. These are responsible for parsing the structure of a
dataset and providing the total number of samples that it contains. Dataset parsers should also expose a
``__getitem__`` function that returns an individual data sample when queried by index. The dataset parsers
provided in the ``torchvision.datasets`` package are all fully compatible with these requirements.

The configuration section itself should be built like a dictionary of objects to instantiate. The key
associated with each parser is the name that will be used to refer to it in the ``loaders`` section. If
a dataset parser that does not derive from :class:`thelper.data.parsers.Dataset` is needed, you will have
to specify a task object inside its definition. An example configuration based on the CIFAR10 class
provided by ``torchvision`` (`[more info here]`__) is shown below::

    "datasets": {
        "cifar10_train": {  # name of the first dataset parser
            "type": "torchvision.datasets.CIFAR10",
            "params": {  # parameters forwarded to the class constructor
                "root": "data/cifar/train",
                "train": true,
                "download": true
            },
            "task": {  # task defined explicitely due to external class
                "type": "thelper.tasks.Classification",
                "params": { # by default, we just need to know the class names
                    "class_names": [
                        "airplane", "car", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"
                    ],
                    # now, the CIFAR10 class loads samples as tuple...
                    "input_key": "0",  # input = element at index#0 in tuple
                    "label_key": "1"   # label = element at index#1 in tuple
                }
            }
        },
        "cifar10_test": {  # name of the second dataset parser
            "type": "torchvision.datasets.CIFAR10",
            "params": {
                "root": "data/cifar/test",
                "train": false,  # here, fetch test data instead of train data
                "download": true
            },
            "task": {
                # we use the same task info as above, both will be merged
                "type": "thelper.tasks.Classification",
                "params": {
                    "class_names": [
                        "airplane", "car", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"
                    ],
                    "input_key": "0",
                    "label_key": "1"
                }
            }
        }
    }

.. __: https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.CIFAR10

The example above defines two dataset parsers, ``cifar10_train`` and ``cifar10_test``, that could now
be referred to in the ``loaders`` section of a configuration file (`described next <#loaders-section>`_).
For more information on the instantiation of dataset parsers, refer to
:meth:`thelper.data.utils.create_parsers`.


Loaders section
---------------

The ``loaders`` section of the configuration defines all data loader-related settings including data split
ratios, data samplers, batch sizes, base transforms and augmentations, seeds, memory pinning, and async
worker count. The first important concept to understand here is that multiple data parsers (`defined
earlier <#datasets-section>`_) can be combined or split into one or more data loaders. Moreover, there are
exactly three data loaders defined for all experiments: the training data loader, the validation data
loader, and the test data loader. For more information on the fundamental role of each loader, see
`[this link]`__. In short, data loaders are essentially "handlers" that deal with parsers to load data
samples efficiently, and that transform and pack these samples into batches so we can feed them to our
models.

.. __: https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7

Some of the settings defined in this section apply to all three data loaders (e.g. memory pinning, base
data transforms), while others can be specified for each loader individually (e.g. augmentations, batch
size). The meta-settings that should always be set however are the split ratios that define the fraction
of samples from each parser to use in a data loader. As shown in the example below, these ratios allow
us to split a dataset into different loaders automatically, and without any possibility of data leakage
between them. If all seed are also set in this section, then the split will be fixed between experiments,
ensuring that the difference between the performance of two models trained in two different sessions is
never due to a difference in their training data.

Besides, base transformations defined in this section are used to ensure that all samples loaded by
parsers are compatible with the input format expected by the model during training. For example, typical
image classification pipelines expect that images will be forwarded at a resolution of 224x224 pixels,
with each color channel normalized to either the [-1, 1] range, or using pre-computed mean and standard
deviation values. We can define such operations directly using the classes available in the
:mod:`thelper.transforms` module. This is also demonstrated in the example configuration below::

    # note: this example is in line with the ``datasets`` example given earlier
    "loaders": {
        "batch_size": 32,     # pack 32 images per minibatch
        "valid_seed": 0,  
        "test_seed": 0,       # fix all seeds for reproducible experiments
        "torch_seed": 0,      # (otherwise, random seed will be picked and printed in logs)
        "numpy_seed": 0,
        "random_seed": 0,
        "workers": 4,         # use 4 threads to load independent minibatches in parallel
        "base_transforms": [  # defines list of operations to apply to ALL loaded samples
            {   # first, normalize 8-bit images to the [-1, 1] range
                "operation": "thelper.transforms.NormalizeMinMax",
                "params": {
                    "min": [127, 127, 127],
                    "max": [255, 255, 255]
                }
            },
            {   # next, resize the CIFAR10 images to 224x224 for the model
                "operation": "thelper.transforms.Resize",
                "params": {
                    "dsize": [224, 224]
                }
            },
            {   # finally, transform the opencv/numpy arrays to torch.Tensor arrays
                "operation": "torchvision.transforms.ToTensor"
            }
        ],
        # we reserve 20% of the samples from the training data parser for validation
        "train_split": {
            "cifar10_train": 0.8
        },
        "valid_split": {
            "cifar10_train": 0.2
        },
        # we use 100% of the samples from the test data parser for testing
        "test_split": {
            "cifar10_test": 1.0
        }
    }

The example above prepares the CIFAR10 dataset for standard training using a 80%-20% training-validation
split, and keeps all the original CIFAR10 testing data for actual testing. All loaded samples will be
normalized and resized to fit the expected input resolution of a ResNet18 model, detailed in the next
subsection. This example however contains no data augmentation pipelines; refer to the `[relevant sections
further down] <#defining-a-data-augmentation-pipeline>`_ for actual usage examples. Similarly, no sampler
is used above to rebalance the classes; `[see here] <#using-a-data-sampler-to-rebalance-a-dataset>`_ for
a use case. Finally, for more information on other parameters that are not discussed here, refer to
the documentation of :meth:`thelper.data.utils.create_loaders`.


Model section
-------------

Section statement here @@@@@@


Trainer section
---------------

Section statement here @@@@@@


Annotator section
-----------------

Section statement here @@@@@@


Global parameters
-----------------

Section statement here @@@@@@



Session Directories
===================

Overview statement here @@@@@@


Checkpoints
-----------

Section statement here @@@@@@


Session logs
------------

Section statement here @@@@@@


Outputs (TensorboardX, metrics)
-------------------------------

Section statement here @@@@@@




Use Case Examples
=================

Image classification
--------------------

Section statement here @@@@@@


Image segmentation
------------------

Section statement here @@@@@@


Dataset/Loader visualization
----------------------------

Section statement here @@@@@@


Dataset annotation
------------------

Section statement here @@@@@@


Supporting a custom trainer
---------------------------

Section statement here @@@@@@


Supporting a custom task
------------------------

Section statement here @@@@@@


Defining a data augmentation pipeline
-------------------------------------

Section statement here @@@@@@


Using an external augmentation pipeline
---------------------------------------

Section statement here @@@@@@


Using a data sampler to rebalance a dataset
-------------------------------------------

Section statement here @@@@@@


