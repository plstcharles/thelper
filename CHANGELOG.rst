.. _changelog:

Changelog
=========

0.3.2 (2019/07/05)
------------------

* Update documentation use cases (model export) & faq
* Cleanup module base class config backup
* Fixed docker build and automated it via travis

0.3.0 - 0.3.1 (2019/06/12)
--------------------------

* Added dockerfile for containerized builds
* Added object detection task & trainer implementations
* Added CLI model/checkpoint export support
* Added CLI dataset splitting/HDF5 support
* Added baseline superresolution implementations
* Added lots of new unit tests & docstrings
* Cleaned up transform & display operations

0.2.8 (2019/03/17)
--------------------------

* Cleaned up build tools & docstrings throughout api
* Added user guide in documentation build
* Update tasks to allow dataset interface override
* Cleaned up trainer output logs
* Added fully convolutional resnet implementation
* Fixup various issues related to fine-tuning via 'resume'

0.2.7 (2019/02/04)
--------------------------

* Updated conda build recipe for python variants w/ auto upload

0.2.6 (2019/01/31)
--------------------------

* Added framework checkpoint/configuration migration utilities
* Fixed minor config parsing backward compatibility issues
* Fixed minor bugs related to query & drawing utilities

0.2.2 - 0.2.5 (2019/01/29)
--------------------------

* Fixed travis-ci matrix configuration
* Added travis-ci deployment step for pypi
* Fixed readthedocs documentation building
* Updated readme shields & front page look
* Cleaned up cli module entrypoint
* Fixed openssl dependency issues for travis tox check jobs
* Updated travis post-deploy to try to fix conda packaging (wip)

0.2.1 (2019/01/24)
-------------------

* Added typedef module & cleaned up parameter inspections
* Cleaned up all drawing utils & added callback support to trainers
* Added support for albumentation pipelines via wrapper
* Updated all trainers/schedulers to rely on 0-based indexing
* Updated travis/rtd configs for auto-deploy & 3.6 support

0.2.0 (2019/01/15)
-------------------

* Added regression/segmentation tasks and trainers
* Added interface for pascalvoc dataset
* Refactored data loaders/parsers and cleaned up data package
* Added lots of new utilities in base trainer implementation
* Added new unit tests for transformations
* Refactored transformations to use wrappers for augments/lists
* Added new samplers with dataset scaling support
* Added baseline implementation for FCN32s
* Added mae/mse metrics implementations
* Added trainer support for loss computation via external members
* Added utils to download/verify/extract files

0.1.1 (2019/01/14)
-------------------

* Minor fixups and updates for CCFB02 compatibility
* Added RawPredictions metric to fetch data from trainers

0.1.0 (2018/11/28)
-------------------

* Fixed readthedocs sphinx auto-build w/ mocking.
* Refactored package structure to avoid env issues.
* Rewrote seeding to allow 100% reproducible sessions.
* Cleaned up config file parameter lists.
* Cleaned up session output vars/logs/images.
* Add support for eval-time augmentation.
* Update transform wrappers for multi-channels & lists.
* Add gui module w/ basic segmentation annotation tool.
* Refactored task interfaces to allow merging.
* Simplified model fine-tuning via checkpoints.

0.0.2 (2018/10/18)
-------------------

* Completed first documentation pass.
* Fixed travis/rtfd builds.
* Fixed device mapping/loading issues.

0.0.1 (2018/10/03)
-------------------

* Initial release (work in progress).
