"""Dataset utility functions and tools.

This module contains utility functions and tools used to instantiate data loaders and parsers.
"""

import copy
import json
import logging
import os
import random
import sys
from collections import Counter

import numpy as np
import torch
import torch.utils.data
import torch.utils.data.sampler

import thelper.tasks
import thelper.transforms
import thelper.utils

logger = logging.getLogger(__name__)


def create_loaders(config, save_dir=None):
    """Prepares the task and data loaders for a model trainer based on a provided data configuration.

    This function will parse a configuration dictionary and extract all the information required to
    instantiate the requested dataset parsers. Then, combining the task metadata of all these parsers, it
    will evenly split the available samples into three sets (training, validation, test) to be handled by
    different data loaders. These will finally be returned along with the (global) task object.

    The configuration dictionary is expected to contain two fields: ``loaders``, which specifies all
    parameters required for establishing the dataset split, shuffling seeds, and batch size (these are
    listed and detailed below); and ``datasets``, which lists the dataset parser interfaces to instantiate
    as well as their parameters. For more information on the ``datasets`` field, refer to
    :func:`thelper.data.utils.create_parsers`.

    The parameters expected in the 'loaders' configuration field are the following:

    - ``batch_size`` (mandatory): specifies the (mini)batch size to use in data loaders. Note that
      the framework does not currently test the validity of the provided value, so if you get an
      'out of memory' error at runtime, try reducing it.
    - ``shuffle`` (optional, default=True): specifies whether the data loaders should shuffle
      their samples or not.
    - ``test_seed`` (optional): specifies the RNG seed to use when splitting test data. If no seed
      is specified, the RNG will be initialized with a device-specific or time-related seed.
    - ``valid_seed`` (optional): specifies the RNG seed to use when splitting validation data. If no
      seed is specified, the RNG will be initialized with a device-specific or time-related seed.
    - ``torch_seed`` (optional): specifies the RNG seed to use for torch-related stochastic operations
      (e.g. for data augmentation). If no seed is specified, the RNG will be initialized with a
      device-specific or time-related seed.
    - ``numpy_seed`` (optional): specifies the RNG seed to use for numpy-related stochastic operations
      (e.g. for data augmentation). If no seed is specified, the RNG will be initialized with a
      device-specific or time-related seed.
    - ``random_seed`` (optional): specifies the RNG seed to use for stochastic operations with python's
      'random' package. If no seed is specified, the RNG will be initialized with a device-specific or
      time-related seed.
    - ``workers`` (optional, default=1): specifies the number of threads to use to preload batches in
      parallel; can be 0 (loading will be on main thread), or an integer >= 1.
    - ``pin_memory`` (optional, default=False): specifies whether the data loaders will copy tensors
      into CUDA-pinned memory before returning them.
    - ``drop_last`` (optional, default=False): specifies whether to drop the last incomplete batch
      or not if the dataset size is not a multiple of the batch size.
    - ``sampler`` (optional): specifies a type of sampler and its constructor parameters to be used
      in the data loaders. This can be used for example to help rebalance a dataset based on its
      class distribution. See :mod:`thelper.data.samplers` for more information.
    - ``augments`` (optional): provides a list of transformation operations used to augment all samples
      of a dataset. See :func:`thelper.transforms.utils.load_augments` for more info.
    - ``train_augments`` (optional): provides a list of transformation operations used to augment the
      training samples of a dataset. See :func:`thelper.transforms.utils.load_augments` for more info.
    - ``valid_augments`` (optional): provides a list of transformation operations used to augment the
      validation samples of a dataset. See :func:`thelper.transforms.utils.load_augments` for more info.
    - ``test_augments`` (optional): provides a list of transformation operations used to augment the
      test samples of a dataset. See :func:`thelper.transforms.utils.load_augments` for more info.
    - ``eval_augments`` (optional): provides a list of transformation operations used to augment the
      validation and test samples of a dataset. See :func:`thelper.transforms.utils.load_augments` for more info.
    - ``base_transforms`` (optional): provides a list of transformation operations to apply to all
      loaded samples. This list will be passed to the constructor of all instantiated dataset parsers.
      See :func:`thelper.transforms.utils.load_transforms` for more info.
    - ``train_split`` (optional): provides the proportion of samples of each dataset to hand off to the
      training data loader. These proportions are given in a dictionary format (``name: ratio``).
    - ``valid_split`` (optional): provides the proportion of samples of each dataset to hand off to the
      validation data loader. These proportions are given in a dictionary format (``name: ratio``).
    - ``test_split`` (optional): provides the proportion of samples of each dataset to hand off to the
      test data loader. These proportions are given in a dictionary format (``name: ratio``).
    - ``skip_verif`` (optional, default=True): specifies whether the dataset split should be verified
      if resuming a session by parsing the log files generated earlier.
    - ``skip_split_norm`` (optional, default=False): specifies whether the question about normalizing
      the split ratios should be skipped or not.
    - ``skip_class_balancing`` (optional, default=False): specifies whether the balancing of class
      labels should be skipped in case the task is classification-related.

    Example configuration file::

        # ...
        "loaders": {
            "batch_size": 128,  # batch size to use in data loaders
            "shuffle": true,  # specifies that the data should be shuffled
            "workers": 4,  # number of threads to pre-fetch data batches with
            "sampler": {  # we can use a data sampler to rebalance classes (optional)
                # see e.g. 'thelper.data.samplers.WeightedSubsetRandomSampler'
                # ...
            },
            "train_augments": { # training data augmentation operations
                # see 'thelper.transforms.utils.load_augments'
                # ...
            },
            "eval_augments": { # evaluation (valid/test) data augmentation operations
                # see 'thelper.transforms.utils.load_augments'
                # ...
            },
            "base_transforms": { # global sample transformation operations
                # see 'thelper.transforms.utils.load_transforms'
                # ...
            },
            # finally, we define a 80%-10%-10% split for our data
            # (we could instead use one dataset for training and one for testing)
            "train_split": {
                "dataset_A": 0.8
                "dataset_B": 0.8
            },
            "valid_split": {
                "dataset_A": 0.1
                "dataset_B": 0.1
            },
            "test_split": {
                "dataset_A": 0.1
                "dataset_B": 0.1
            }
            # (note that the dataset names above are defined in the field below)
        },
        "datasets": {
            "dataset_A": {
                # type of dataset interface to instantiate
                "type": "...",
                "params": [
                    # ...
                ]
            },
            "dataset_B": {
                # type of dataset interface to instantiate
                "type": "...",
                "params": [
                    # ...
                ],
                # if it does not derive from 'thelper.data.parsers.Dataset', a task is needed:
                "task": {
                    # this type must derive from 'thelper.tasks.Task'
                    "type": "...",
                    "params": [
                        # ...
                    ]
                }
            },
            # ...
        },
        # ...

    Args:
        config: a dictionary that provides all required data configuration information under two fields,
            namely 'datasets' and 'loaders'.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.

    Returns:
        A 4-element tuple that contains: 1) the global task object to specialize models and trainers with;
        2) the training data loader; 3) the validation data loader; and 4) the test data loader.

    .. seealso::
        | :func:`thelper.data.create_parsers`
        | :func:`thelper.transforms.utils.load_augments`
        | :func:`thelper.transforms.utils.load_transforms`
    """
    logstamp = thelper.utils.get_log_stamp()
    repover = thelper.__version__ + ":" + thelper.utils.get_git_stamp()
    session_name = config["name"] if "name" in config else "session"
    data_logger_dir = None
    if save_dir is not None:
        data_logger_dir = os.path.join(save_dir, "logs")
        os.makedirs(data_logger_dir, exist_ok=True)
        data_logger_path = os.path.join(data_logger_dir, "data.log")
        data_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        data_logger_fh = logging.FileHandler(data_logger_path)
        data_logger_fh.setFormatter(data_logger_format)
        logger.addHandler(data_logger_fh)
        logger.info("created data log for session '%s'" % session_name)
        config_backup_path = os.path.join(data_logger_dir, "config." + logstamp + ".json")
        with open(config_backup_path, "w") as fd:
            json.dump(config, fd, indent=4, sort_keys=False)
    logger.debug("loading data usage config")
    # todo: 'data_config' field is deprecated, might be removed later
    if "data_config" in config:
        logger.warning("using 'data_config' field in configuration dictionary is deprecated; switch it to 'loaders'")
    loaders_config = thelper.utils.get_key(["data_config", "loaders"], config)
    # noinspection PyProtectedMember
    loader_factory = thelper.data.utils._LoaderFactory(loaders_config)
    datasets, task = create_parsers(config, loader_factory.get_base_transforms())
    if not datasets or task is None:
        raise AssertionError("invalid dataset configuration (got empty list)")
    for dataset_name, dataset in datasets.items():
        logger.info("parsed dataset: %s" % str(dataset))
    logger.info("task info: %s" % str(task))
    logger.debug("splitting datasets and creating loaders...")
    train_idxs, valid_idxs, test_idxs = loader_factory.get_split(datasets, task)
    if save_dir is not None:
        with open(os.path.join(data_logger_dir, "task.log"), "a+") as fd:
            fd.write("session: %s-%s\n" % (session_name, logstamp))
            fd.write("version: %s\n" % repover)
            fd.write(str(task) + "\n")
        for dataset_name, dataset in datasets.items():
            dataset_log_file = os.path.join(data_logger_dir, dataset_name + ".log")
            if not loader_factory.skip_verif and os.path.isfile(dataset_log_file):
                logger.info("verifying sample list for dataset '%s'..." % dataset_name)
                with open(dataset_log_file, "r") as fd:
                    log_content = fd.read()
                    if not log_content or log_content[0] != "{":
                        # could not find new style (json) dataset log, cannot easily parse and compare this log
                        logger.warning("cannot verify that old split is similar to new split, log is out-of-date")
                        continue
                    log_content = json.loads(log_content)
                    if "samples" not in log_content or not isinstance(log_content["samples"], list):
                        raise AssertionError("unexpected dataset log content (bad 'samples' field)")
                    samples_old = log_content["samples"]
                    samples_new = dataset.samples if hasattr(dataset, "samples") and isinstance(dataset.samples, list) else []
                    if len(samples_old) != len(samples_new):
                        query_msg = "Old sample list for dataset '%s' mismatch with current sample list; proceed anyway?"
                        answer = thelper.utils.query_yes_no(query_msg, bypass="n")
                        if not answer:
                            logger.error("sample list mismatch with previous run; user aborted")
                            sys.exit(1)
                        break
                    else:
                        breaking = False
                        for set_name, idxs in zip(["train_idxs", "valid_idxs", "test_idxs"],
                                                  [train_idxs[dataset_name], valid_idxs[dataset_name], test_idxs[dataset_name]]):
                            # index values were paired in tuples earlier, 0=idx, 1=label
                            if log_content[set_name] != [idx for idx, _ in idxs]:
                                query_msg = "Old indices list for dataset '%s' mismatch with current indices" \
                                            "list ('%s'); proceed anyway?" % (dataset_name, set_name)
                                answer = thelper.utils.query_yes_no(query_msg, bypass="n")
                                if not answer:
                                    logger.error("indices list mismatch with previous run; user aborted")
                                    sys.exit(1)
                                breaking = True
                                break
                        if not breaking:
                            for idx, (sample_new, sample_old) in enumerate(zip(samples_new, samples_old)):
                                if str(sample_new) != sample_old:
                                    query_msg = "Old sample #%d for dataset '%s' mismatch with current #%d; proceed anyway?" \
                                                "\n\told: %s\n\tnew: %s" % (idx, dataset_name, idx, str(sample_old), str(sample_new))
                                    answer = thelper.utils.query_yes_no(query_msg, bypass="n")
                                    if not answer:
                                        logger.error("sample list mismatch with previous run; user aborted")
                                        sys.exit(1)
                                    break
        for dataset_name, dataset in datasets.items():
            dataset_log_file = os.path.join(data_logger_dir, dataset_name + ".log")
            samples = dataset.samples if hasattr(dataset, "samples") and isinstance(dataset.samples, list) else []
            log_content = {
                "metadata": {
                    "session_name": session_name,
                    "logstamp": logstamp,
                    "version": repover,
                    "dataset": str(dataset),
                },
                "samples": [str(sample) for sample in samples],
                # index values were paired in tuples earlier, 0=idx, 1=label
                "train_idxs": [idx for idx, _ in train_idxs[dataset_name]],
                "valid_idxs": [idx for idx, _ in valid_idxs[dataset_name]],
                "test_idxs": [idx for idx, _ in test_idxs[dataset_name]]
            }
            # now, always overwrite, as it can get too big otherwise
            with open(dataset_log_file, "w") as fd:
                json.dump(log_content, fd, indent=4, sort_keys=False)
    train_loader, valid_loader, test_loader = loader_factory.create_loaders(datasets, train_idxs, valid_idxs, test_idxs)
    return task, train_loader, valid_loader, test_loader


def create_parsers(config, base_transforms=None):
    """Instantiates dataset parsers based on a provided dictionary.

    This function will instantiate dataset parsers as defined in a name-type-param dictionary. If multiple
    datasets are instantiated, this function will also verify their task compatibility and return the global
    task. The dataset interfaces themselves should be derived from :class:`thelper.data.parsers.Dataset`, be
    compatible with :class:`thelper.data.parsers.ExternalDataset`, or should provide a 'task' field specifying
    all the information related to sample dictionary keys and model i/o. An example configuration is provided
    in :func:`thelper.data.utils.create_parsers`.

    The provided configuration will be parsed for a 'datasets' dictionary entry. The keys in this dictionary
    are treated as unique dataset names and are used for lookups. The value associated to each key (or dataset
    name) should be a type-params dictionary that can be parsed to instantiate the dataset interface.

    An example configuration dictionary is given in :func:`thelper.data.utils.create_loaders`.

    Args:
        config: a dictionary that provides unique dataset names and parameters needed for instantiation under
            the 'datasets' field.
        base_transforms: the transform operation that should be applied to all loaded samples, and that
            will be provided to the constructor of all instantiated dataset parsers.

    Returns:
        A 2-element tuple that contains: 1) the list of dataset interfaces/parsers that were instantiated; and
        2) a task object compatible with all of those (see :class:`thelper.tasks.utils.Task` for more information).

    .. seealso::
        | :func:`thelper.data.utils.create_loaders`
        | :class:`thelper.data.parsers.Dataset`
        | :class:`thelper.data.parsers.ExternalDataset`
        | :class:`thelper.tasks.utils.Task`
    """
    if not isinstance(config, dict):
        raise AssertionError("unexpected session config type")
    if "datasets" not in config or not config["datasets"]:
        raise AssertionError("config missing 'datasets' field (must contain dict or str value)")
    config = config["datasets"]  # no need to keep the full config here
    if isinstance(config, str):
        if os.path.isfile(config) and os.path.splitext(config)[1] == ".json":
            config = json.load(open(config))
        else:
            raise AssertionError("'datasets' string should point to valid json file")
    logger.debug("loading datasets templates")
    if not isinstance(config, dict):
        raise AssertionError("invalid datasets config type (must be dictionary)")
    datasets = {}
    tasks = []
    for dataset_name, dataset_config in config.items():
        if "type" not in dataset_config:
            raise AssertionError("missing field 'type' for instantiation of dataset '%s'" % dataset_name)
        dataset_type = thelper.utils.import_class(dataset_config["type"])
        dataset_params = thelper.utils.get_key_def("params", dataset_config, {})
        transforms = None
        if "transforms" in dataset_config and dataset_config["transforms"]:
            transforms = thelper.transforms.load_transforms(dataset_config["transforms"])
            if base_transforms is not None:
                transforms = thelper.transforms.Compose([transforms, base_transforms])
        elif base_transforms is not None:
            transforms = base_transforms
        if issubclass(dataset_type, thelper.data.Dataset):
            # assume that the dataset is derived from thelper.data.parsers.Dataset (it is fully sampling-ready)
            dataset = dataset_type(name=dataset_name, config=dataset_params, transforms=transforms)
            if "task" in dataset_config:
                logger.warning("'task' field detected in dataset '%s' config; will be ignored (interface should provide it)" % dataset_name)
            task = dataset.get_task()
        else:
            if "task" not in dataset_config or not dataset_config["task"]:
                raise AssertionError("external dataset '%s' must define task interface in its configuration dict" % dataset_name)
            task = thelper.tasks.create_task(dataset_config["task"])
            # assume that __getitem__ and __len__ are implemented, but we need to make it sampling-ready
            dataset = thelper.data.ExternalDataset(dataset_name, dataset_type, task, config=dataset_params, transforms=transforms)
        if task is None:
            raise AssertionError("parsed task interface should not be None anymore (old code doing something strange?)")
        tasks.append(task)
        datasets[dataset_name] = dataset
    return datasets, thelper.tasks.create_global_task(tasks)


class _LoaderFactory(object):
    """Factory used for preparing and splitting dataset parsers into usable data loader objects.

    This class is responsible for parsing the parameters contained in the 'loaders' field of a
    configuration dictionary, instantiating the data loaders, and shuffling/splitting the samples.
    An example configuration is presented in :func:`thelper.data.utils.create_loaders`.

    .. seealso::
        | :func:`thelper.data.utils.create_loaders`
        | :func:`thelper.transforms.utils.load_augments`
        | :func:`thelper.transforms.utils.load_transforms`
    """

    def __init__(self, config):
        """Receives and parses the data configuration dictionary."""
        logger.debug("loading data config")
        if not isinstance(config, dict):
            raise AssertionError("input config should be dict")
        if "batch_size" not in config or not config["batch_size"]:
            raise AssertionError("data config missing 'batch_size' field")
        self.batch_size = config["batch_size"]
        logger.debug("loaders will use batch size = %d" % self.batch_size)
        self.shuffle = thelper.utils.str2bool(config["shuffle"]) if "shuffle" in config else True
        if self.shuffle:
            logger.debug("dataset samples will be shuffled according to predefined seeds")
            np.random.seed()  # for seed generation below (if needed); will be reseeded afterwards
        self.test_seed = self._get_seed(["test_seed", "test_split_seed"], config, (int, str))
        self.valid_seed = self._get_seed(["valid_seed", "valid_split_seed"], config, (int, str))
        self.torch_seed = self._get_seed(["torch_seed"], config, int)
        self.numpy_seed = self._get_seed(["numpy_seed"], config, int)
        self.random_seed = self._get_seed(["random_seed"], config, int)
        torch.manual_seed(self.torch_seed)
        torch.cuda.manual_seed_all(self.torch_seed)
        np.random.seed(self.numpy_seed)
        random.seed(self.random_seed)
        self.workers = config["workers"] if "workers" in config and config["workers"] >= 0 else 1
        self.pin_memory = thelper.utils.str2bool(config["pin_memory"]) if "pin_memory" in config else False
        self.drop_last = thelper.utils.str2bool(config["drop_last"]) if "drop_last" in config else False
        if self.drop_last:
            logger.debug("loaders will drop last batch if sample count not multiple of %d" % self.batch_size)
        self.sampler_type = None
        self.train_sampler, self.valid_sampler, self.test_sampler = None, None, None
        if "sampler" in config:
            sampler_config = config["sampler"]
            if sampler_config:
                if "type" not in sampler_config or not sampler_config["type"]:
                    raise AssertionError("missing 'type' field for sampler config")
                self.sampler_type = thelper.utils.import_class(sampler_config["type"])
                self.sampler_params = thelper.utils.get_key_def("params", sampler_config, {})
                logger.debug("will use global sampler with type '%s' and config : %s" % (str(self.sampler_type),
                                                                                         str(self.sampler_params)))
                self.sampler_pass_labels = False
                if "pass_labels" in sampler_config:
                    self.sampler_pass_labels = thelper.utils.str2bool(sampler_config["pass_labels"])
                self.train_sampler = thelper.utils.str2bool(sampler_config["apply_train"]) if "apply_train" in sampler_config else True
                self.valid_sampler = thelper.utils.str2bool(sampler_config["apply_valid"]) if "apply_valid" in sampler_config else False
                self.test_sampler = thelper.utils.str2bool(sampler_config["apply_test"]) if "apply_test" in sampler_config else False
                logger.debug("global sampler will be applied as: %s" % str([self.train_sampler, self.valid_sampler, self.test_sampler]))
        train_augs_targets = ["augments", "trainvalid_augments", "train_augments"]
        valid_augs_targets = ["augments", "trainvalid_augments", "eval_augments", "validtest_augments", "valid_augments"]
        test_augs_targets = ["augments", "eval_augments", "validtest_augments", "test_augments"]
        self.train_augments, self.train_augments_append = self._get_augments(train_augs_targets, "train", config)
        self.valid_augments, self.valid_augments_append = self._get_augments(valid_augs_targets, "valid", config)
        self.test_augments, self.test_augments_append = self._get_augments(test_augs_targets, "test", config)
        self.base_transforms = None
        if "base_transforms" in config and config["base_transforms"]:
            self.base_transforms = thelper.transforms.load_transforms(config["base_transforms"])
            if self.base_transforms:
                logger.debug("base transforms: %s" % str(self.base_transforms))
        self.train_split = self._get_ratios_split("train", config)
        self.valid_split = self._get_ratios_split("valid", config)
        self.test_split = self._get_ratios_split("test", config)
        if not self.train_split and not self.valid_split and not self.test_split:
            raise AssertionError("data config must define a split for at least one loader type (train/valid/test)")
        self.total_usage = Counter(self.train_split) + Counter(self.valid_split) + Counter(self.test_split)
        self.skip_split_norm = thelper.utils.str2bool(config["skip_split_norm"]) if "skip_split_norm" in config else False
        self.skip_class_balancing = thelper.utils.str2bool(config["skip_class_balancing"]) if "skip_class_balancing" in config else False
        for name, usage in self.total_usage.items():
            if usage != 1:
                normalize_ratios = None
                if usage < 0:
                    raise AssertionError("ratio should never be negative...")
                elif 0 < usage < 1 and not self.skip_split_norm:
                    query_msg = "dataset split for '%s' has a ratio sum less than 1; do you want to normalize the split?" % name
                    normalize_ratios = thelper.utils.query_yes_no(query_msg, bypass="n")
                if (normalize_ratios or usage > 1) and usage > 0:
                    if usage > 1:
                        logger.warning("dataset split for '%s' sums to more than 1; will normalize..." % name)
                    if name in self.train_split:
                        self.train_split[name] /= usage
                    if name in self.valid_split:
                        self.valid_split[name] /= usage
                    if name in self.test_split:
                        self.test_split[name] /= usage
        self.skip_verif = thelper.utils.str2bool(config["skip_verif"]) if "skip_verif" in config else True

    @staticmethod
    def _get_seed(prefixes, config, stype):
        key = None
        for prefix in prefixes:
            if prefix in config:
                key = prefix
                break
        if key is not None:
            if not isinstance(config[key], stype):
                raise AssertionError("unexpected value type for field '%s'" % key)
            return config[key]
        seed = np.random.randint(2 ** 16)
        logger.info("setting '%s' to %d" % (key, seed))
        return seed

    @staticmethod
    def _get_ratios_split(prefix, config):
        key = prefix + "_split"
        if key not in config or not config[key]:
            return {}
        split = config[key]
        if any(ratio < 0 or ratio > 1 for ratio in split.values()):
            raise AssertionError("split ratios in '%s' must be in [0,1]" % key)
        return split

    @staticmethod
    def _get_augments(targets, name, config):
        for target in targets:
            if target in config and config[target]:
                augments, augments_append = thelper.transforms.load_augments(config[target])
                if augments:
                    logger.debug("will %s %s augments: %s" % ("append" if augments_append else "prefix", name, str(augments)))
                return augments, augments_append
        return None, False

    def _get_raw_split(self, indices):
        for name in self.total_usage:
            if name not in indices:
                raise AssertionError("dataset '%s' does not exist" % name)
        _indices, train_idxs, valid_idxs, test_idxs = {}, {}, {}, {}
        for name, indices in indices.items():
            _indices[name] = copy.deepcopy(indices)
            train_idxs[name] = []
            valid_idxs[name] = []
            test_idxs[name] = []
        indices = _indices
        if self.shuffle:
            np.random.seed(self.test_seed)  # test idxs will be picked first, then valid+train
            for idxs in indices.values():
                np.random.shuffle(idxs)
        offsets = dict.fromkeys(self.total_usage, 0)
        for loader_idx, (idxs_map, ratio_map) in enumerate(zip([test_idxs, valid_idxs, train_idxs],
                                                               [self.test_split, self.valid_split, self.train_split])):
            for name in self.total_usage.keys():
                if name in ratio_map:
                    count = int(round(ratio_map[name] * len(indices[name])))
                    if count < 0:
                        raise AssertionError("ratios should be non-negative values!")
                    elif count < 1 and len(indices[name]) > 0:
                        logger.warning("split ratio for '%s' too small, sample set will be empty" % name)
                    begidx = offsets[name]
                    endidx = min(begidx + count, len(indices[name]))
                    idxs_map[name] = indices[name][begidx:endidx]
                    offsets[name] = endidx
            if loader_idx == 0 and self.shuffle:
                np.random.seed(self.valid_seed)  # all test idxs are now picked, reshuffle for train/valid
                for name in self.total_usage.keys():
                    trainvalid_idxs = indices[name][offsets[name]:]
                    np.random.shuffle(trainvalid_idxs)
                    indices[name][offsets[name]:] = trainvalid_idxs
                np.random.seed(self.numpy_seed)  # back to default random state for future use
        return train_idxs, valid_idxs, test_idxs

    def get_split(self, datasets, task):
        r"""Returns the train/valid/test sample indices split for a given dataset (name-parser) map.

        Note that the returned indices are unique, possibly shuffled, and never duplicated between sets.
        If the samples have a class attribute (i.e. the task is related to classification), the split
        will respect the initial distribution and apply the ratios within the classes themselves. For
        example, consider a dataset of three classes (:math:`A`, :math:`B`, and :math:`C`) that contains
        100 samples such as:

        .. math::

            |A| = 50,\;|B| = 30,\;|C| = 20

        If we require a 80%-10%-10% ratio distribution for the training, validation, and test loaders
        respectively, the resulting split will contain the following sample counts:

        .. math::

                \text{training loader} = {40A + 24B + 16C}

        .. math::

                \text{validation loader} = {5A + 3B + 2C}

        .. math::

                \text{test loader} = {5A + 3B + 2C}

        Args:
            datasets: the map of datasets to split, where each has a name (key) and a parser (value).
            task: a task object that should be compatible with all provided datasets (can be ``None``).

        Returns:
            A three-element tuple containing the maps of the training, validation, and test sets
            respectively. These maps associate dataset names to a list of sample indices.
        """
        dataset_sizes = {}
        must_split = {}
        global_size = 0
        for dataset_name, dataset in datasets.items():
            if not isinstance(dataset, thelper.data.Dataset) and not isinstance(dataset, thelper.data.ExternalDataset):
                raise AssertionError("unexpected dataset type for '%s'" % dataset_name)
            dataset_sizes[dataset_name] = len(dataset)
            global_size += dataset_sizes[dataset_name]
            # if a single dataset is used in more than a single loader, we cannot skip the rebalancing below
            must_split[dataset_name] = sum([dataset_name in split for split in
                                            [self.train_split, self.valid_split, self.test_split]]) > 1
        global_size = sum(len(dataset) for dataset in datasets.values())
        logger.info("splitting datasets with parsed sizes = %s" % str(dataset_sizes))
        must_split = any(must_split.values())
        if task is not None and isinstance(task, thelper.tasks.Classification) and not self.skip_class_balancing and must_split:
            # note: with current impl, all class sets will be shuffled the same way... (shouldnt matter, right?)
            logger.debug("will split evenly over %d classes..." % len(task.get_class_names()))
            unset_class_key = "<unset>"
            global_class_names = task.get_class_names() + [unset_class_key]  # extra name added for unlabeled samples (if needed!)
            sample_maps = {}
            for dataset_name, dataset in datasets.items():
                if not task.check_compat(dataset.get_task()):
                    raise AssertionError("global task should already have been compatible with all datasets")
                if isinstance(dataset, thelper.data.ExternalDataset):
                    if hasattr(dataset.samples, "samples") and isinstance(dataset.samples.samples, list):
                        sample_maps[dataset_name] = task.get_class_sample_map(dataset.samples.samples, unset_class_key)
                    else:
                        logger.warning(("must fully parse the external dataset '%s' for intra-class shuffling;" % dataset_name) +
                                       " this might take a while! (consider making a dataset interface that can return labels only)")
                        label_key = task.get_gt_key()
                        samples = []
                        for sample in dataset:
                            if label_key not in sample:
                                raise AssertionError("could not find label key ('%s') in sample dict" % label_key)
                            samples.append({label_key: sample[label_key]})
                        sample_maps[dataset_name] = task.get_class_sample_map(samples, unset_class_key)
                elif isinstance(dataset, thelper.data.Dataset):
                    sample_maps[dataset_name] = task.get_class_sample_map(dataset.samples, unset_class_key)
            train_idxs, valid_idxs, test_idxs = {}, {}, {}
            for class_name in global_class_names:
                curr_class_samples, curr_class_size = {}, {}
                for dataset_name in datasets:
                    class_samples = sample_maps[dataset_name][class_name] if class_name in sample_maps[dataset_name] else []
                    samples_pairs = list(zip(class_samples, [class_name] * len(class_samples)))
                    curr_class_samples[dataset_name] = samples_pairs
                    curr_class_size[dataset_name] = len(curr_class_samples[dataset_name])
                    logger.debug("dataset '{}' class '{}' sample count: {} ({}% of local, {}% of total)".format(
                        dataset_name,
                        class_name,
                        curr_class_size[dataset_name],
                        int(100 * curr_class_size[dataset_name] / dataset_sizes[dataset_name]),
                        int(100 * curr_class_size[dataset_name] / global_size)))
                class_train_idxs, class_valid_idxs, class_test_idxs = self._get_raw_split(curr_class_samples)
                for idxs_dict_list, class_idxs_dict_list in zip([train_idxs, valid_idxs, test_idxs],
                                                                [class_train_idxs, class_valid_idxs, class_test_idxs]):
                    for dataset_name in datasets:
                        if dataset_name in idxs_dict_list:
                            idxs_dict_list[dataset_name] += class_idxs_dict_list[dataset_name]
                        else:
                            idxs_dict_list[dataset_name] = class_idxs_dict_list[dataset_name]
        else:  # no balancing to be done
            dataset_indices = {}
            for dataset_name in datasets:
                # note: all indices paired with 'None' below as class is ignored; used for compatibility with code above
                dataset_indices[dataset_name] = list(
                    zip(list(range(dataset_sizes[dataset_name])), [None] * dataset_sizes[dataset_name]))
            train_idxs, valid_idxs, test_idxs = self._get_raw_split(dataset_indices)
        return train_idxs, valid_idxs, test_idxs

    def create_loaders(self, datasets, train_idxs, valid_idxs, test_idxs):
        """Returns the data loaders for the train/valid/test sets based on a prior split.

        This function essentially takes the dataset parser interfaces and indices maps, and instantiates
        data loaders that are ready to produce samples for training or evaluation. Note that the dataset
        parsers will be deep-copied in each data loader, meaning that they should ideally not contain a
        persistent loading state or a large buffer.

        Args:
            datasets: the map of dataset parsers, where each has a name (key) and a parser (value).
            train_idxs: training data samples indices map.
            valid_idxs: validation data samples indices map.
            test_idxs: test data samples indices map.

        Returns:
            A three-element tuple containing the training, validation, and test data loaders, respectively.
        """
        loaders = []
        for idxs_map, (augs, augs_append), sampler_apply in zip([train_idxs, valid_idxs, test_idxs],
                                                                [(self.train_augments, self.train_augments_append),
                                                                 (self.valid_augments, self.valid_augments_append),
                                                                 (self.test_augments, self.test_augments_append)],
                                                                [self.train_sampler, self.valid_sampler, self.test_sampler]):
            loader_sample_idx_offset = 0
            loader_sample_classes = []
            loader_sample_idxs = []
            loader_datasets = []
            for dataset_name, sample_idxs in idxs_map.items():
                if datasets[dataset_name].bypass_deepcopy:
                    dataset = copy.copy(datasets[dataset_name])
                else:
                    dataset = copy.deepcopy(datasets[dataset_name])
                if augs:
                    augs_copy = copy.deepcopy(augs)
                    if dataset.transforms is not None:
                        if augs_append:
                            dataset.transforms = thelper.transforms.Compose([dataset.transforms, augs_copy])
                        else:
                            dataset.transforms = thelper.transforms.Compose([augs_copy, dataset.transforms])
                    else:
                        dataset.transforms = augs_copy
                for sample_idx_idx in range(len(sample_idxs)):
                    # values were paired in tuples earlier, 0=idx, 1=label
                    loader_sample_idxs.append(sample_idxs[sample_idx_idx][0] + loader_sample_idx_offset)
                    loader_sample_classes.append(sample_idxs[sample_idx_idx][1])
                loader_sample_idx_offset += len(dataset)
                loader_datasets.append(dataset)
            if len(loader_datasets) > 0:
                dataset = torch.utils.data.ConcatDataset(loader_datasets) if len(loader_datasets) > 1 else loader_datasets[0]
                if self.sampler_type is not None and sampler_apply:
                    if self.sampler_pass_labels:
                        sampler = self.sampler_type(loader_sample_idxs, loader_sample_classes, **self.sampler_params)
                    else:
                        sampler = self.sampler_type(loader_sample_idxs, **self.sampler_params)
                else:
                    sampler = torch.utils.data.sampler.SubsetRandomSampler(loader_sample_idxs)
                loaders.append(torch.utils.data.DataLoader(dataset,
                                                           batch_size=self.batch_size,
                                                           sampler=sampler,
                                                           num_workers=self.workers,
                                                           pin_memory=self.pin_memory,
                                                           drop_last=self.drop_last,
                                                           worker_init_fn=self._worker_init_fn))
            else:
                loaders.append(None)
        train_loader, valid_loader, test_loader = loaders
        train_samples = len(train_loader) if train_loader else 0
        valid_samples = len(valid_loader) if valid_loader else 0
        test_samples = len(test_loader) if test_loader else 0
        logger.info("initialized loaders with batch counts: train=%d, valid=%d, test=%d" % (train_samples, valid_samples, test_samples))
        return train_loader, valid_loader, test_loader

    def get_base_transforms(self):
        """Returns the (global) sample transformation operations parsed in the data configuration."""
        return self.base_transforms

    def _worker_init_fn(self, worker_id):
        torch.manual_seed(self.torch_seed + worker_id)
        torch.cuda.manual_seed_all(self.torch_seed + worker_id)
        np.random.seed(self.numpy_seed + worker_id)
        random.seed(self.random_seed + worker_id)


def get_class_weights(label_map, stype, invmax, maxw=float('inf'), minw=0.0, norm=True):
    """Returns a map of adjusted class weights based on a given rebalancing strategy.

    Args:
        label_map: map of index lists tied to class labels.
        stype: weighting strategy ('uniform', `linear`, or 'rootX'); see :class:`thelper.data.samplers.WeightedSubsetRandomSampler`
            for more information on these.
        invmax: specifies whether to max-invert the weight vector (thus creating cost factors) or not (default=True).
        maxw: maximum allowed weight value (applied after invmax, if required).
        minw: minimum allowed weight value (applied after invmax, if required).
        norm: specifies whether the returned weights should be normalized (default=True, i.e. normalized).

    Returns:
        Map of adjusted weights tied to class labels.

    .. seealso::
        | :class:`thelper.data.samplers.WeightedSubsetRandomSampler`
    """
    if stype == "uniform":
        label_weights = {label: 1.0 / len(label_map) for label in label_map}
    elif stype == "linear" or "root" in stype:
        if stype == "root" or stype == "linear":
            rpow = 1.0
        else:
            rpow = 1.0 / float(stype.split("root", 1)[1])
        tot_count = sum([len(idxs) for idxs in label_map.values()])
        label_weights = {label: (len(idxs) / tot_count) ** rpow for label, idxs in label_map.items()}
    else:
        raise AssertionError("unknown label weighting strategy")
    if invmax:
        label_weights = {label: max(label_weights.values()) / max(weight, 1e-6) for label, weight in label_weights.items()}
    label_weights = {label: min(max(weight, minw), maxw) for label, weight in label_weights.items()}
    if norm:
        tot_weight = sum([w for w in label_weights.values()])
        label_weights = {label: weight / tot_weight for label, weight in label_weights.items()}
    return label_weights
