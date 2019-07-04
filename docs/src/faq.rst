.. _faq:

===
FAQ
===

We answer some of the simple and frequently asked questions about the framework below. If you think of
any other question that should be in this list, send a mail to one of the maintainers, and it will be
added here.


What it is...
-------------

  - This framework is used to simplify the exploration, development, and testing of models that you
    create yourself, or that you import from other libraries or frameworks.

  - This framework is used to enforce good reproducibility standards for your experiments via
    the use of global configuration files and checkpoints.

  - This framework is used to easily swap, split, scale, combine, and augment datasets used in
    your experiments.

  - This framework can help you fine-tune, debug, visualize, and understand the behavior of your
    models more easily.

  - This framework is **NOT** used to obtain off-the-shelf solutions. In most cases, you will
    have to put in some work by at least modifying pre-existing configuration files.


What it supports...
-------------------

  - **PyTorch.** For now, the entire backend is based on the design patterns, interfaces, and
    concepts of the PyTorch library (`[more info] <pytorch_>`_).
  
  - Image classification, segmentation, object detection, super-resolution, and generic regression
    tasks. More types of tasks are planned in the near future.

  - Live evaluation and monitoring of predefined metrics. The framework implements :ref:`[several
    types of metrics] <thelper.optim:thelper.optim.metrics module>`, but custom metrics can also be
    defined and evaluated at run time.

  - Data augmentation. The framework implements basic :ref:`[transformation operations and wrappers]
    <thelper.transforms:thelper.transforms package>` for large augmentation libraries such as
    ``albumentations`` (`[more info] <albumen_>`_).

  - Model fine-tuning and exportation. Models obtained from the ``torchvision`` package (`[more info]
    <torchvis_>`_) or pre-trained using the framework can be loaded and fine-tuned directly for any
    compatible task. They can also be exported in PyTorch-JIT/ONNX format for external inference.

  - Tensorboard. Event logs are generated using ``tensorboardX`` (`[more info] <tbx_>`_) and may
    contain plots, visualizations, histograms, graph module trees and more.

  .. _pytorch: https://pytorch.org/
  .. _albumen: https://github.com/albu/albumentations
  .. _torchvis: https://pytorch.org/docs/stable/torchvision/models.html
  .. _tbx: https://github.com/lanpa/tensorboardX


How do I...
-----------

This section is still a work in progress; see the use case examples :ref:`[here] <use-cases>` for a list
of code snippets and tutorials on how to use the framework. For high-level documentation, refer to the
:ref:`[user guide] <user-guide>`.
