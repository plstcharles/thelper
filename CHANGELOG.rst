.. _changelog:

Changelog
=========

0.3.12 (2019/09/13)
-------------------

* Fixed potential issue when reinstantiating custom ResNet
* Fixed ClassifLogger prediction logger w/o groundtruth

0.3.11 (2019/09/09)
-------------------

* Add cli/config override for task compatibility mode setting

0.3.10 (2019/09/05)
-------------------

* Cleaned up dependency lists, docstrings
* Fixed bbox iou computation with mixed int/float
* Fixed dontcare label deletion in segmentation task
* Cleaned up training session output directory localization
* Fixed object detection trainer empty bbox lists
* Fixed exponential parsing with pyyaml
* Fixed bbox display when using integer coords values

0.3.9 (2019/08/20)
------------------

* Fixed collate issues for pytorch >= 1.2
* Fixed null-size batch issues
* Cleaned up params#kwargs parsing in trainer
* Added pickled hashed param support utils
* Added support for yaml-based session configuration
* Added concept decorators for metrics/consumer classes
* Cleaned up shared interfaces to fix circular dependencies
* Added detection (bbox) logger class

0.3.8 (2019/08/08)
------------------

* Fixed nn modules constructor args forwarding
* Updated class importer to allow parsing of non-package dirs
* Fixed file-based logging from submodules (e.g. for all data)
* Cleaned and API-fied the CLI entrypoints for external use

0.3.7 (2019/07/31)
------------------

* Fixed travis timeouts on long deploy operations
* Added output path to trainer callback impls
* Added new draw-and-save display callback
* Added togray/tocolor transformation operations
* Cleaned up matplotlib use and show/block across draw functions
* Fixed various dependency and logging issues

0.3.6 (2019/07/26)
------------------

* Fixed torch version checks in custom default collate impl
* Fixed bbox predictions forwarding and evaluation in objdetect
* Refactored metrics/callbacks to clean up trainer impls
* Added pretrained opt to default resnet impl
* Fixed objdetect trainer display and prediction callbacks

0.3.5 (2019/07/23)
------------------

* Refactored metrics/consumers into separate interfaces
* Added unit tests for all metrics/prediction consumers
* Updated trainer callback signatures to include more data
* Updated install doc with links to anaconda/docker hubs
* Cleaned drawing functions args wrt callback refactoring
* Added eval module to optim w/ pascalvoc evaluation funcs

0.3.4 (2019/07/12)
------------------

* Fixed issues when reloading objdet model checkpoints
* Fixed issues when trying to use missing color maps
* Fixed backward compat issues when reloading old tasks
* Cleaned up object detection drawing utilities

0.3.3 (2019/07/09)
------------------

* Fixed travis conda build dependencies & channels

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
