.. _changelog:

Changelog
=========

`Unreleased <http://github.com/plstcharles/thelper/tree/master>`_ (latest)
----------------------------------------------------------------------------------

* Added ``SelectChannels`` transform operator for inplace sample modifications that might need it.

`0.6.1 <http://github.com/plstcharles/thelper/tree/v0.6.1>`_ (2020/07/29)
----------------------------------------------------------------------------------

* Fixed conda package builds for tagged deployments

`0.6.0 <http://github.com/plstcharles/thelper/tree/v0.6.0>`_ (2020/07/28)
----------------------------------------------------------------------------------

* Refactored and cleaned up HDF5 data extraction/parsing classes
* Added dataset interfaces for BigEarthNet, Agri-Vis challenge
* Update classification task to allow multi-label classification
* Added activation layer customization for in-framework ResNet archs
* Updated default move_tensor behavior to be non-blocking
* Added trainer implementation for auto-encoder-type models
* Added Orion reporting support for hyperparameter explorations
* Added SLURM cluster utilities (tmpdir getter, launch scripts)

`0.5.0 <http://github.com/plstcharles/thelper/tree/v0.5.0>`_ (2020/07/21)
----------------------------------------------------------------------------------

* Skip image save call during metric rendering if the provided value is ``None`` as employed by basic logger/reporter.
* Add JSON implementation for ``thelper.train.utils.ClassifLogger``.
* Fix ``concepts`` to handle any variation of upper/lower concept name.

`0.5.0-rc2 <http://github.com/plstcharles/thelper/tree/v0.5.0-rc2>`_ (2020/07/08)
----------------------------------------------------------------------------------

* Employ ``requirements.txt`` within ``conda-env.yml`` to kept dependencies in sync.
* Fixes built docker image not using appropriate dependencies enforced through ``requirements.txt``.

`0.5.0-rc1 <http://github.com/plstcharles/thelper/tree/v0.5.0-rc1>`_ (2020/07/07)
----------------------------------------------------------------------------------

* Fix version comparison check when validating configuration and/or checkpoint against package version.
  Version can now have a release part which was not considered.
* Fix incorrect calculation of sample coordinates in ``thelper.data.geo.parsers.SlidingWindowDataset``.
* Remove ``not_skip = __init__.py`` config option for ``isort`` since ``__init__.py`` is included since ``4.3.5``.
  Also force ``isort<5`` since many import checks break suddenly (e.g.: direct import with `as` alias break).

`0.5.0-rc0 <http://github.com/plstcharles/thelper/tree/v0.5.0-rc0>`_ (2020/04/25)
----------------------------------------------------------------------------------

* Update this changelog to use rst links (renders on github and readthedocs)
* Add ``infer`` mode for classification of geo-referenced rasters
* Add ``Dockerfile-geo`` to build thelper with pre-installed geo packages
* Add geo-related build instructions to travis-ci build steps
* Add auto-documentation of makefile targets and docker related targets

`0.4.7 <http://github.com/plstcharles/thelper/tree/v0.4.7>`_ (2019/11/20)
----------------------------------------------------------------------------------

* Removed optional dependencies from conda build env

`0.4.6 <http://github.com/plstcharles/thelper/tree/v0.4.6>`_ (2019/11/20)
----------------------------------------------------------------------------------

* Travis deploy test w/ split conda/docker stages

`0.4.5 <http://github.com/plstcharles/thelper/tree/v0.4.5>`_ (2019/11/18)
----------------------------------------------------------------------------------

* Split travis deploy stage into two phases
* Fixed draw_segment threshold usage & params lookup
* Fixed FCResNet embedding getter wrt latest pooling update
* Update all matplotlib plots to use 160 dpi by default
* Refactor trainer data/metric writer to save all viz data

`0.4.4 <http://github.com/plstcharles/thelper/tree/v0.4.4>`_ (2019/11/18)
----------------------------------------------------------------------------------

* Added viz pkg w/ t-SNE & UMAP support for in-trainer usage
* Fixed geo pkg documentation build issue related to mocking
* Fixed type and output format checks in numerous metrics
* Updated all callback readers to rely on new utility function
* Cleaned and optimize coordconv implementation
* Added U-Net architecture implementation to nn package
* Added IoU metric implementation
* Added support for SRM kernels and SRM convolutions
* Updated documentation (install, faq, maintenance)
* Added fixed weight sampler to data package
* Added lots of extra unit tests
* Added efficientnet 3rd-party module wrapper
* Fixed potential conflicts in task class names ordering

`0.4.3 <http://github.com/plstcharles/thelper/tree/v0.4.3>`_ (2019/11/06)
----------------------------------------------------------------------------------

* Fixed pytest-mock scope usage in metrics utests

`0.4.2 <http://github.com/plstcharles/thelper/tree/v0.4.2>`_ (2019/11/06)
----------------------------------------------------------------------------------

* Updated common resnet impl to support segmentation heads
* Fixed samples usage for auto-weighting of loss functions
* Cleaned up samples usage in loader factory data splitter
* Add GDL compatibility module to geo package
* Fix segmentation task dontcare default color mapping
* Cleaned up and simplified coordconv implementation
* Update segmentation trainer to use long-typed label maps
* Cleaned up augmentor/albumentations demo configurations

`0.4.1 <http://github.com/plstcharles/thelper/tree/v0.4.1>`_ (2019/10/15)
----------------------------------------------------------------------------------

* Removed travis check in deploy stage for master branch

`0.4.0 <http://github.com/plstcharles/thelper/tree/v0.4.0>`_ (2019/10/11)
----------------------------------------------------------------------------------

* Added geo subpackage
* Added geo vector/raster parsing classes
* Added ogc module for testbed15-specific utilities
* Added testbed15 train/viz configuration files
* Cleaned up makefile targets & coverage usage
* Replaced tox build system with makefile completely
* Merged 3rdparty configs into setup.cfg
* Updated travis to rely on makefile directly

`0.3.14 <http://github.com/plstcharles/thelper/tree/v0.3.14>`_ (2019/09/30)
----------------------------------------------------------------------------------

* Added extra logging calls in trainer and framework utils
* Cleaned up data configuration parsing logger calls
* Bypassed full device check when specific one is requested

`0.3.13 <http://github.com/plstcharles/thelper/tree/v0.3.13>`_ (2019/09/26)
----------------------------------------------------------------------------------

* Moved drawing utilities to new module
* Cleaned up output root/save directory parsing
* Cleaned up potential circular imports
* Moved optional dependency imports inside relevant functions
* Added support for root directory specification via config
* Updated config load/save to make naming optional

`0.3.12 <http://github.com/plstcharles/thelper/tree/v0.3.12>`_ (2019/09/13)
----------------------------------------------------------------------------------

* Fixed potential issue when reinstantiating custom ResNet
* Fixed ClassifLogger prediction logger w/o groundtruth

`0.3.11 <http://github.com/plstcharles/thelper/tree/v0.3.11>`_ (2019/09/09)
----------------------------------------------------------------------------------

* Add cli/config override for task compatibility mode setting

`0.3.10 <http://github.com/plstcharles/thelper/tree/v0.3.10>`_ (2019/09/05)
----------------------------------------------------------------------------------

* Cleaned up dependency lists, docstrings
* Fixed bbox iou computation with mixed int/float
* Fixed dontcare label deletion in segmentation task
* Cleaned up training session output directory localization
* Fixed object detection trainer empty bbox lists
* Fixed exponential parsing with pyyaml
* Fixed bbox display when using integer coords values

`0.3.9 <http://github.com/plstcharles/thelper/tree/v0.3.9>`_ (2019/08/20)
----------------------------------------------------------------------------------

* Fixed collate issues for pytorch >= 1.2
* Fixed null-size batch issues
* Cleaned up params#kwargs parsing in trainer
* Added pickled hashed param support utils
* Added support for yaml-based session configuration
* Added concept decorators for metrics/consumer classes
* Cleaned up shared interfaces to fix circular dependencies
* Added detection (bbox) logger class

`0.3.8 <http://github.com/plstcharles/thelper/tree/v0.3.8>`_ (2019/08/08)
----------------------------------------------------------------------------------

* Fixed nn modules constructor args forwarding
* Updated class importer to allow parsing of non-package dirs
* Fixed file-based logging from submodules (e.g. for all data)
* Cleaned and API-fied the CLI entrypoints for external use

`0.3.7 <http://github.com/plstcharles/thelper/tree/v0.3.7>`_ (2019/07/31)
----------------------------------------------------------------------------------

* Fixed travis timeouts on long deploy operations
* Added output path to trainer callback impls
* Added new draw-and-save display callback
* Added togray/tocolor transformation operations
* Cleaned up matplotlib use and show/block across draw functions
* Fixed various dependency and logging issues

`0.3.6 <http://github.com/plstcharles/thelper/tree/v0.3.6>`_ (2019/07/26)
----------------------------------------------------------------------------------

* Fixed torch version checks in custom default collate impl
* Fixed bbox predictions forwarding and evaluation in objdetect
* Refactored metrics/callbacks to clean up trainer impls
* Added pretrained opt to default resnet impl
* Fixed objdetect trainer display and prediction callbacks

`0.3.5 <http://github.com/plstcharles/thelper/tree/v0.3.5>`_ (2019/07/23)
----------------------------------------------------------------------------------

* Refactored metrics/consumers into separate interfaces
* Added unit tests for all metrics/prediction consumers
* Updated trainer callback signatures to include more data
* Updated install doc with links to anaconda/docker hubs
* Cleaned drawing functions args wrt callback refactoring
* Added eval module to optim w/ pascalvoc evaluation funcs

`0.3.4 <http://github.com/plstcharles/thelper/tree/v0.3.4>`_ (2019/07/12)
----------------------------------------------------------------------------------

* Fixed issues when reloading objdet model checkpoints
* Fixed issues when trying to use missing color maps
* Fixed backward compat issues when reloading old tasks
* Cleaned up object detection drawing utilities

`0.3.3 <http://github.com/plstcharles/thelper/tree/v0.3.3>`_ (2019/07/09)
----------------------------------------------------------------------------------

* Fixed travis conda build dependencies & channels

`0.3.2 <http://github.com/plstcharles/thelper/tree/v0.3.2>`_ (2019/07/05)
----------------------------------------------------------------------------------

* Update documentation use cases (model export) & faq
* Cleanup module base class config backup
* Fixed docker build and automated it via travis

`0.3.1 <http://github.com/plstcharles/thelper/tree/v0.3.1>`_ (2019/06/17)
----------------------------------------------------------------------------------

* Fix metrics RawPredictions not returning predictions during eval
* Fix parsing of checkpoint base path

`0.3.0 <http://github.com/plstcharles/thelper/tree/v0.3.0>`_ (2019/06/12)
----------------------------------------------------------------------------------

* Added dockerfile for containerized builds
* Added object detection task & trainer implementations
* Added CLI model/checkpoint export support
* Added CLI dataset splitting/HDF5 support
* Added baseline superresolution implementations
* Added lots of new unit tests & docstrings
* Cleaned up transform & display operations

`0.2.8 <http://github.com/plstcharles/thelper/tree/v0.2.8>`_ (2019/03/17)
----------------------------------------------------------------------------------

* Cleaned up build tools & docstrings throughout api
* Added user guide in documentation build
* Update tasks to allow dataset interface override
* Cleaned up trainer output logs
* Added fully convolutional resnet implementation
* Fixup various issues related to fine-tuning via 'resume'

`0.2.7 <http://github.com/plstcharles/thelper/tree/v0.2.7>`_ (2019/02/04)
----------------------------------------------------------------------------------

* Updated conda build recipe for python variants w/ auto upload

`0.2.6 <http://github.com/plstcharles/thelper/tree/v0.2.6>`_ (2019/01/31)
----------------------------------------------------------------------------------

* Added framework checkpoint/configuration migration utilities
* Fixed minor config parsing backward compatibility issues
* Fixed minor bugs related to query & drawing utilities

`0.2.5 <http://github.com/plstcharles/thelper/tree/v0.2.5>`_ (2019/01/29)
----------------------------------------------------------------------------------

* Fix travis-ci conda build/env path

`0.2.4 <http://github.com/plstcharles/thelper/tree/v0.2.4>`_ (2019/01/29)
----------------------------------------------------------------------------------

* Fix travis-ci conda channel setup

`0.2.3 <http://github.com/plstcharles/thelper/tree/v0.2.3>`_ (2019/01/29)
----------------------------------------------------------------------------------

* Fix ``openssl`` dependency

`0.2.2 <http://github.com/plstcharles/thelper/tree/v0.2.2>`_ (2019/01/29)
----------------------------------------------------------------------------------

* Fixed travis-ci matrix configuration
* Added travis-ci deployment step for pypi
* Fixed readthedocs documentation building
* Updated readme shields & front page look
* Cleaned up cli module entrypoint
* Fixed openssl dependency issues for travis tox check jobs
* Updated travis post-deploy to try to fix conda packaging (wip)

`0.2.1 <http://github.com/plstcharles/thelper/tree/v0.2.1>`_ (2019/01/24)
----------------------------------------------------------------------------------

* Added typedef module & cleaned up parameter inspections
* Cleaned up all drawing utils & added callback support to trainers
* Added support for albumentation pipelines via wrapper
* Updated all trainers/schedulers to rely on 0-based indexing
* Updated travis/rtd configs for auto-deploy & 3.6 support

`0.2.0 <http://github.com/plstcharles/thelper/tree/v0.2.0>`_ (2019/01/15)
----------------------------------------------------------------------------------

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

`0.1.1 <http://github.com/plstcharles/thelper/tree/v0.1.1>`_ (2019/01/14)
----------------------------------------------------------------------------------

* Minor fixups and updates for CCFB02 compatibility
* Added RawPredictions metric to fetch data from trainers

`0.1.0 <http://github.com/plstcharles/thelper/tree/v0.1.0>`_ (2018/11/28)
----------------------------------------------------------------------------------

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

`0.0.2 <http://github.com/plstcharles/thelper/tree/v0.0.2>`_ (2018/10/18)
----------------------------------------------------------------------------------

* Completed first documentation pass.
* Fixed travis/rtfd builds.
* Fixed device mapping/loading issues.

`0.0.1 <http://github.com/plstcharles/thelper/tree/v0.0.1>`_ (2018/10/03)
----------------------------------------------------------------------------------

* Initial release (work in progress).
