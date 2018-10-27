"""Data parsing/handling/splitting module.

This module contains classes and functions whose role is to fetch the data required to train, validate,
and test a model. The :func:`thelper.data.load` function contained herein is responsible for preparing
the data loaders and the task interface for a training session based on a provided configuration.
"""

import copy
import json
import logging
import os
import sys
from abc import ABC
from abc import abstractmethod
from collections import Counter

import numpy as np
import PIL
import PIL.Image
import torch
import torch.utils.data
import torch.utils.data.sampler

import thelper.samplers
import thelper.tasks
import thelper.transforms
import thelper.utils

logger = logging.getLogger(__name__)


def load(config, data_root, save_dir=None):
    """Prepares the task interface and the data loaders based on a provided data configuration.

    This function will parse a configuration dictionary and extract all the information required to
    instantiate the requested dataset parsers. Then, combining the task metadata of all these parsers,
    it will evenly split the available samples into three sets (training, validation, test) to be handled
    by different data loaders. These will finally be returned along with the task interface.

    The configuration dictionary is expected to contain two fields: ``data_config``, which includes
    information about the dataset split, shuffling seeds, and batch size; and ``datasets``, which lists
    the dataset parser interfaces to instantiate as well as their parameters. The former is parsed by
    the :class:`thelper.data.DataConfig` class, and the latter via :func:`thelper.data.load_datasets`.

    Example configuration file::

        # ...
        "data_config": {
            "batch_size": 128,  # batch size to use in data loaders
            "shuffle": true,  # specifies that the data should be shuffled
            "workers": 4,  # number of threads to pre-fetch data batches with
            "sampler": {  # we can use a data sampler to rebalance classes (optional)
                # see e.g. 'thelper.samplers.WeightedSubsetRandomSampler'
                # ...
            },
            "train_augments": { # training data augmentation operations
                # see 'thelper.transforms.load_augments'
                # ...
            },
            "eval_augments": { # evaluation (valid/test) data augmentation operations
                # see 'thelper.transforms.load_augments'
                # ...
            },
            "base_transforms": { # global sample transformation operations
                # see 'thelper.transforms.load_transforms'
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
                # (note: must be derived from 'thelper.data.Dataset')
                "type": "...",
                "params": [
                    # ...
                ]
            },
            "dataset_B": {
                # type of dataset interface to instantiate
                # (note: must be derived from 'thelper.data.Dataset')
                "type": "...",
                "params": [
                    # ...
                ]
            },
            # ...
        },
        # ...

    Args:
        config: a dictionary that provides all required data configuration and dataset parameters; these
            are detailed in :class:`thelper.data.DataConfig` and :func:`thelper.data.load_datasets`.
        data_root: the path to the dataset root directory that will be passed to the dataset interfaces
            for them to figure out where the training/validation/testing data is located. This path may
            be unused if the dataset interfaces already know where to look via config parameters.
        save_dir: the path to the root directory where the session directory should be saved. Note that
            this is not the path to the session directory itself, but its parent, which may also contain
            other session directories.

    Returns:
        A 4-element tuple that contains the global task object to specialize models with and the training,
        validation, and test data loaders (respectively).

    .. seealso::
        :class:`thelper.data.DataConfig`
        :func:`thelper.data.load_datasets`
        :func:`thelper.transforms.load_augments`
        :func:`thelper.transforms.load_transforms`
        :class:`thelper.samplers.WeightedSubsetRandomSampler`
    """
    logstamp = thelper.utils.get_log_stamp()
    repover = thelper.utils.get_git_stamp()
    session_name = config["name"] if "name" in config else "session"
    if save_dir is not None:
        data_logger_path = os.path.join(save_dir, "logs", "data.log")
        data_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        data_logger_fh = logging.FileHandler(data_logger_path)
        data_logger_fh.setFormatter(data_logger_format)
        logger.addHandler(data_logger_fh)
        logger.info("created data log for session '%s'" % session_name)
        config_backup_path = os.path.join(save_dir, "logs", "config." + logstamp + ".json")
        json.dump(config, open(config_backup_path, "w"), indent=4, sort_keys=False)
    logger.debug("loading data usage config")
    if "data_config" not in config or not config["data_config"]:
        raise AssertionError("config missing 'data_config' field")
    data_config = thelper.data.DataConfig(config["data_config"])
    logger.info("parsing datasets configuration")
    if "datasets" not in config or not config["datasets"]:
        raise AssertionError("config missing 'datasets' field (can be dict or str)")
    datasets_config = config["datasets"]
    if isinstance(datasets_config, str):
        if os.path.isfile(datasets_config) and os.path.splitext(datasets_config)[1] == ".json":
            datasets_config = json.load(open(datasets_config))
        else:
            raise AssertionError("'datasets' string should point to valid json file")
    logger.debug("loading datasets templates")
    if not isinstance(datasets_config, dict):
        raise AssertionError("invalid datasets config type")
    datasets, task = load_datasets(datasets_config, data_root, data_config.get_base_transforms())
    for dataset_name, dataset in datasets.items():
        logger.info("dataset '%s' info: %s" % (dataset_name, str(dataset)))
    logger.info("task info: %s" % str(task))
    logger.debug("splitting datasets and creating loaders...")
    train_idxs, valid_idxs, test_idxs = data_config.get_split(datasets, task)
    if save_dir is not None:
        with open(os.path.join(save_dir, "logs", "task.log"), "a+") as fd:
            fd.write("session: %s-%s\n" % (session_name, logstamp))
            fd.write("version: %s\n" % repover)
            fd.write(str(task) + "\n")
        for dataset_name, dataset in datasets.items():
            dataset_log_file = os.path.join(save_dir, "logs", dataset_name + ".log")
            if not data_config.skip_verif and os.path.isfile(dataset_log_file):
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
                        answer = thelper.utils.query_yes_no(
                            "Old sample list for dataset '%s' mismatch with current sample list; proceed anyway?")
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
                                answer = thelper.utils.query_yes_no(
                                    "Old indices list for dataset '%s' mismatch with current indices list ('%s'); proceed anyway?"
                                    % (dataset_name, set_name))
                                if not answer:
                                    logger.error("indices list mismatch with previous run; user aborted")
                                    sys.exit(1)
                                breaking = True
                                break
                        if not breaking:
                            for idx, (sample_new, sample_old) in enumerate(zip(samples_new, samples_old)):
                                if str(sample_new) != sample_old:
                                    answer = thelper.utils.query_yes_no(
                                        "Old sample #%d for dataset '%s' mismatch with current #%d; proceed anyway?"
                                        "\n\told: %s\n\tnew: %s" % (idx, dataset_name, idx, str(sample_old), str(sample_new)))
                                    if not answer:
                                        logger.error("sample list mismatch with previous run; user aborted")
                                        sys.exit(1)
                                    break
        for dataset_name, dataset in datasets.items():
            dataset_log_file = os.path.join(save_dir, "logs", dataset_name + ".log")
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
            json.dump(log_content, open(dataset_log_file, "w"), indent=4, sort_keys=False)
    train_loader, valid_loader, test_loader = data_config.get_loaders(datasets, train_idxs, valid_idxs, test_idxs)
    return task, train_loader, valid_loader, test_loader


def load_datasets(config, data_root, base_transforms=None):
    """Instantiates dataset parsers based on a provided dictionary.

    This function will instantiate dataset parsers as defined in a name-type-param dictionary. If multiple
    datasets are instantiated, this function will also verify their task compatibility and return the combined
    output task (if necessary). The dataset interfaces themselves should be derived from
    :class:`thelper.data.Dataset`, or be compatible with :class:`thelper.data.ExternalDataset`.

    The keys in ``config`` are treated as unique dataset names and are used for lookups (e.g. for splitting
    in :class:`thelper.data.DataConfig`). The value associated to each key (or dataset name) should be a
    type-params dictionary as shown in the example of :func:`thelper.data.load`.

    Args:
        config: a dictionary that provides unique dataset names and parameters needed for instantiation.
        data_root: the path to the dataset root directory that will be passed to the dataset interfaces
            for them to figure out where the training/validation/testing data is located. This path may
            be unused if the dataset interfaces already know where to look via config parameters.
        base_transforms: the transform operation that should be applied to all loaded samples, and that
            will be provided to the constructor of all instantiated dataset parsers.

    Returns:
        A 2-element tuple that contains the list of dataset interfaces/parsers and a task object compatible
        with all of those.

    .. seealso::
        :func:`thelper.data.load`
        :class:`thelper.data.Dataset`
        :class:`thelper.data.ExternalDataset`
    """
    datasets = {}
    tasks = []
    global_class_names = None  # if remains none, task is not classif-related
    for dataset_name, dataset_config in config.items():
        if "type" not in dataset_config:
            raise AssertionError("missing field 'type' for instantiation of dataset '%s'" % dataset_name)
        dataset_type = thelper.utils.import_class(dataset_config["type"])
        if "params" not in dataset_config:
            raise AssertionError("missing field 'params' for instantiation of dataset '%s'" % dataset_name)
        params = thelper.utils.keyvals2dict(dataset_config["params"])
        transforms = None
        if "transforms" in dataset_config and dataset_config["transforms"]:
            transforms = thelper.transforms.load_transforms(dataset_config["transforms"])
            if base_transforms is not None:
                transforms = thelper.transforms.Compose([transforms, base_transforms])
        elif base_transforms is not None:
            transforms = base_transforms
        if issubclass(dataset_type, Dataset):
            # assume that the dataset is derived from thelper.data.Dataset (it is fully sampling-ready)
            dataset = dataset_type(name=dataset_name, root=data_root, config=params, transforms=transforms)
            if "task" in dataset_config:
                logger.warning("'task' field detected in dataset config; will be ignored, since interface should itself define it")
        else:
            # assume that __getitem__ and __len__ are implemented, but we need to make it sampling-ready
            if "task" not in dataset_config or not dataset_config["task"]:
                raise AssertionError("missing field 'task' for instantiation of external dataset '%s'" % dataset_name)
            task_config = dataset_config["task"]
            if "type" not in task_config:
                raise AssertionError("missing field 'type' in task config for instantiation of dataset '%s'" % dataset_name)
            task_type = thelper.utils.import_class(task_config["type"])
            if "params" not in task_config:
                raise AssertionError("missing field 'params' in task config for instantiation of dataset '%s'" % dataset_name)
            task_params = thelper.utils.keyvals2dict(task_config["params"])
            task = task_type(**task_params)
            if not issubclass(task_type, thelper.tasks.Task):
                raise AssertionError("the task type for dataset '%s' must be derived from 'thelper.tasks.Task'" % dataset_name)
            dataset = ExternalDataset(dataset_name, data_root, dataset_type, task, config=params, transforms=transforms)
        task = dataset.get_task()
        if len(tasks) > 0 and task != tasks[0]:
            raise AssertionError("not all datasets have similar task, or sample input/gt keys differ")
        if isinstance(task, thelper.tasks.Classification):
            class_names = task.get_class_names()
            if global_class_names is None:
                if len(datasets) > 0:
                    raise AssertionError("cannot handle mixed classification and non-classification tasks in datasets")
                global_class_names = class_names
            else:
                for class_name in class_names:
                    if class_name not in global_class_names:
                        global_class_names.append(class_name)
        elif global_class_names is not None:
            raise AssertionError("cannot handle mixed classification and non-classification tasks in datasets")
        tasks.append(task)
        datasets[dataset_name] = dataset
    if global_class_names is not None:
        global_task = thelper.tasks.Classification(global_class_names, tasks[0].input_key, tasks[0].label_key, meta_keys=tasks[0].meta_keys)
        return datasets, global_task
    return datasets, tasks[0]


class DataConfig(object):
    """Data configuration helper class used for preparing and splitting datasets.

    This class is responsible for parsing the parameters contained in the 'data_config' field of a
    configuration dictionary, instantiating the data loaders, and shuffling/splitting the samples.
    An example configuration is presented in :func:`thelper.data.load`.

    The parameters it currently looks for in the configuration dictionary are the following:

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
    - ``workers`` (optional, default=1): specifies the number of threads to use to preload batches in
      parallel; can be 0 (loading will be on main thread), or an integer >= 1.
    - ``pin_memory`` (optional, default=False): specifies whether the data loaders will copy tensors
      into CUDA-pinned memory before returning them.
    - ``drop_last`` (optional, default=False): specifies whether to drop the last incomplete batch
      or not if the dataset size is not a multiple of the batch size.
    - ``sampler`` (optional): specifies a type of sampler and its constructor parameters to be used
      in the data loaders. This can be used for example to help rebalance a dataset based on its
      class distribution. See :class:`thelper.samplers.WeightedSubsetRandomSampler` for more info.
    - ``train_augments`` (optional): provides a list of transformation operations used to augment the
      training samples of a dataset. See :func:`thelper.transforms.load_augments` for more info.
    - ``eval_augments`` (optional): provides a list of transformation operations used to augment the
      validation/test samples of a dataset. See :func:`thelper.transforms.load_augments` for more info.
    - ``base_transforms`` (optional): provides a list of transformation operations to apply to all
      loaded samples. This list will be passed to the constructor of all instantiated dataset parsers.
      See :func:`thelper.transforms.load_transforms` for more info.
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

    .. seealso::
        :func:`thelper.data.load`
        :func:`thelper.transforms.load_augments`
        :func:`thelper.transforms.load_transforms`
        :class:`thelper.samplers.WeightedSubsetRandomSampler`
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
        self.valid_seed, self.test_seed, self.torch_seed, self.numpy_seed = None, None, None, None
        if "test_seed" in config or "test_split_seed" in config:
            key = "test_seed" if "test_seed" in config else "test_split_seed"
            if not isinstance(config[key], (int, str)):
                raise AssertionError("unexpected value type for field '%s'" % key)
            self.test_seed = config[key]
        elif self.shuffle:
            self.test_seed = np.random.randint(2 ** 16)
            logger.info("setting test split seed to %d" % self.test_seed)
        if "valid_seed" in config or "valid_split_seed" in config:
            key = "valid_seed" if "valid_seed" in config else "valid_split_seed"
            if not isinstance(config[key], (int, str)):
                raise AssertionError("unexpected value type for field '%s'" % key)
            self.valid_seed = config[key]
        elif self.shuffle:
            self.valid_seed = np.random.randint(2 ** 16)
            logger.info("setting valid split seed to %d" % self.valid_seed)
        if "torch_seed" in config:
            if not isinstance(config["torch_seed"], int):
                raise AssertionError("unexpected value type for field 'torch_seed'")
            self.torch_seed = config["torch_seed"]
        else:
            self.torch_seed = np.random.randint(2 ** 16)
            logger.info("setting torch seed to %d" % self.torch_seed)
        torch.manual_seed(self.torch_seed)
        torch.cuda.manual_seed_all(self.torch_seed)
        if "numpy_seed" in config:
            if not isinstance(config["numpy_seed"], int):
                raise AssertionError("unexpected value type for field 'numpy_seed'")
            self.numpy_seed = config["numpy_seed"]
        else:
            self.numpy_seed = np.random.randint(2 ** 16)
            logger.info("setting numpy seed to %d" % self.numpy_seed)
        np.random.seed(self.numpy_seed)
        self.workers = config["workers"] if "workers" in config and config["workers"] >= 0 else 1
        self.pin_memory = thelper.utils.str2bool(config["pin_memory"]) if "pin_memory" in config else False
        self.drop_last = thelper.utils.str2bool(config["drop_last"]) if "drop_last" in config else False
        if self.drop_last:
            logger.debug("loaders will drop last batch if sample count not multiple of %d" % self.batch_size)
        self.sampler_type = None
        if "sampler" in config:
            sampler_config = config["sampler"]
            if sampler_config:
                if "type" not in sampler_config or not sampler_config["type"]:
                    raise AssertionError("missing 'type' field for sampler config")
                self.sampler_type = thelper.utils.import_class(sampler_config["type"])
                self.sampler_params = thelper.utils.keyvals2dict(sampler_config["params"]) if "params" in sampler_config else None
                logger.debug("will use global sampler with type '%s' and config : %s" % (str(self.sampler_type), str(self.sampler_params)))
                self.sampler_pass_labels = False
                if "pass_labels" in sampler_config:
                    self.sampler_pass_labels = thelper.utils.str2bool(sampler_config["pass_labels"])
                self.sampler_apply = [
                    thelper.utils.str2bool(sampler_config["apply_train"]) if "apply_train" in sampler_config else True,
                    thelper.utils.str2bool(sampler_config["apply_valid"]) if "apply_valid" in sampler_config else False,
                    thelper.utils.str2bool(sampler_config["apply_test"]) if "apply_test" in sampler_config else False,
                ]
                logger.debug("global sampler will be applied to loaders: %s" % str(self.sampler_apply))
        self.train_augments, self.train_augments_append = None, False
        if "train_augments" in config and config["train_augments"]:
            self.train_augments, self.train_augments_append = thelper.transforms.load_augments(config["train_augments"])
            if self.train_augments:
                logger.debug("will %s train augmentations: %s" % ("append" if self.train_augments_append else "prefix", str(self.train_augments)))
        self.eval_augments, self.eval_augments_append = None, False
        if "eval_augments" in config and config["eval_augments"]:
            self.eval_augments, self.eval_augments_append = thelper.transforms.load_augments(config["eval_augments"])
            if self.eval_augments:
                logger.debug("will %s eval augmentations: %s" % ("append" if self.train_augments_append else "prefix", str(self.eval_augments)))
        self.base_transforms = None
        if "base_transforms" in config and config["base_transforms"]:
            self.base_transforms = thelper.transforms.load_transforms(config["base_transforms"])

        def get_ratios_split(prefix, config):
            key = prefix + "_split"
            if key not in config or not config[key]:
                return {}
            split = config[key]
            if any(ratio < 0 or ratio > 1 for ratio in split.values()):
                raise AssertionError("split ratios in '%s' must be in [0,1]" % key)
            return split

        self.train_split = get_ratios_split("train", config)
        self.valid_split = get_ratios_split("valid", config)
        self.test_split = get_ratios_split("test", config)
        if not self.train_split and not self.valid_split and not self.test_split:
            raise AssertionError("data config must define a split for at least one loader type (train/valid/test)")
        self.total_usage = Counter(self.train_split) + Counter(self.valid_split) + Counter(self.test_split)
        self.skip_split_norm = thelper.utils.str2bool(config["skip_split_norm"]) if "skip_split_norm" in config else False
        for name, usage in self.total_usage.items():
            if usage != 1:
                normalize_ratios = None
                if usage < 0:
                    raise AssertionError("ratio should never be negative...")
                elif 0 < usage < 1 and not self.skip_split_norm:
                    normalize_ratios = thelper.utils.query_yes_no(
                        "Dataset split for '%s' has a ratio sum less than 1; do you want to normalize the split?" % name)
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
                    elif count < 1:
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
        """Returns the train/valid/test sample indices split for a given dataset (name-parser) map.

        Note that the returned indices are unique, possibly shuffle, and never duplicated between sets.
        If the samples have a class attribute (i.e. the task is related to classification), the split
        will respect the initial distribution and apply the ratios within the classes themselves.

        Args:
            datasets: the map of datasets to split, where each has a name (key) and a parser (value).
            task: a task object that should be compatible with all provided datasets.

        Returns:
            A three-element tuple containing the maps of the training, validation, and test sets
            respectively. These maps associate dataset names to a list of sample indices.
        """
        dataset_sizes = {}
        global_size = 0
        for dataset_name, dataset in datasets.items():
            if not isinstance(dataset, thelper.data.Dataset) and not isinstance(dataset, thelper.data.ExternalDataset):
                raise AssertionError("unexpected dataset type for '%s'" % dataset_name)
            dataset_sizes[dataset_name] = len(dataset)
            global_size += dataset_sizes[dataset_name]
        global_size = sum(len(dataset) for dataset in datasets.values())
        logger.info("splitting datasets with parsed sizes = %s" % str(dataset_sizes))
        if isinstance(task, thelper.tasks.Classification):
            # note: with current impl, all class sets will be shuffled the same way... (shouldnt matter, right?)
            global_class_names = task.get_class_names()
            logger.info("will split evenly over %d classes..." % len(global_class_names))
            sample_maps = {}
            for dataset_name, dataset in datasets.items():
                if isinstance(dataset, thelper.data.ExternalDataset):
                    if hasattr(dataset.samples, "samples"):
                        sample_maps[dataset_name] = task.get_class_sample_map(dataset.samples.samples)
                    else:
                        logger.warning(("must fully parse the external dataset '%s' for intra-class shuffling;" % dataset_name) +
                                       " this might take a while! (consider making a dataset interface that can return labels only)")
                        label_keys = task.get_gt_key() if isinstance(task.get_gt_key(), list) else [task.get_gt_key()]
                        samples = []
                        for sample in dataset:
                            for key in label_keys:
                                if key in sample:
                                    samples.append({key: sample[key]})
                                    break  # by default, stop after finding first match
                        sample_maps[dataset_name] = task.get_class_sample_map(samples)
                elif isinstance(dataset, thelper.data.Dataset):
                    sample_maps[dataset_name] = task.get_class_sample_map(dataset.samples)
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
        else:  # task is not classif-related, no balancing to be done
            dataset_indices = {}
            for dataset_name in datasets:
                # note: all indices paired with 'None' below as class is ignored; used for compatibility with code above
                dataset_indices[dataset_name] = list(
                    zip(list(range(dataset_sizes[dataset_name])), [None] * len(dataset_sizes[dataset_name])))
            train_idxs, valid_idxs, test_idxs = self._get_raw_split(dataset_indices)
        return train_idxs, valid_idxs, test_idxs

    def get_loaders(self, datasets, train_idxs, valid_idxs, test_idxs):
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
        for loader_idx, idxs_map in enumerate([train_idxs, valid_idxs, test_idxs]):
            loader_sample_idx_offset = 0
            loader_sample_classes = []
            loader_sample_idxs = []
            loader_datasets = []
            for dataset_name, sample_idxs in idxs_map.items():
                if datasets[dataset_name].bypass_deepcopy:
                    dataset = copy.copy(datasets[dataset_name])
                else:
                    dataset = copy.deepcopy(datasets[dataset_name])
                if loader_idx == 0 and self.train_augments:
                    train_augs_copy = copy.deepcopy(self.train_augments)
                    if dataset.transforms is not None:
                        if self.train_augments_append:
                            dataset.transforms = thelper.transforms.Compose([dataset.transforms, train_augs_copy])
                        else:
                            dataset.transforms = thelper.transforms.Compose([train_augs_copy, dataset.transforms])
                    else:
                        dataset.transforms = train_augs_copy
                elif loader_idx != 0 and self.eval_augments:
                    eval_augs_copy = copy.deepcopy(self.eval_augments)
                    if dataset.transforms is not None:
                        if self.eval_augments_append:
                            dataset.transforms = thelper.transforms.Compose([dataset.transforms, eval_augs_copy])
                        else:
                            dataset.transforms = thelper.transforms.Compose([eval_augs_copy, dataset.transforms])
                    else:
                        dataset.transforms = eval_augs_copy
                for sample_idx_idx in range(len(sample_idxs)):
                    # values were paired in tuples earlier, 0=idx, 1=label
                    loader_sample_idxs.append(sample_idxs[sample_idx_idx][0] + loader_sample_idx_offset)
                    loader_sample_classes.append(sample_idxs[sample_idx_idx][1])
                loader_sample_idx_offset += len(dataset)
                loader_datasets.append(dataset)
            if len(loader_datasets) > 0:
                dataset = torch.utils.data.ConcatDataset(loader_datasets) if len(loader_datasets) > 1 else loader_datasets[0]
                if self.sampler_type is not None and self.sampler_apply[loader_idx]:
                    if self.sampler_pass_labels:
                        if self.sampler_params is not None:
                            sampler = self.sampler_type(loader_sample_idxs, loader_sample_classes, **self.sampler_params)
                        else:
                            sampler = self.sampler_type(loader_sample_idxs, loader_sample_classes)
                    else:
                        if self.sampler_params is not None:
                            sampler = self.sampler_type(loader_sample_idxs, **self.sampler_params)
                        else:
                            sampler = self.sampler_type(loader_sample_idxs)
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


class Dataset(torch.utils.data.Dataset, ABC):
    """Abstract dataset parsing interface that holds a task and a list of sample dictionaries.

    This interface helps fix a failure of PyTorch's dataset interface (``torch.utils.data.Dataset``):
    the lack of identity associated with the components of a sample. In short, a data sample loaded by a
    dataset typically contains the input data that should be forwarded to a model and the expected
    prediction of the model (i.e. the 'groundtruth') that will be used to compute the loss. These two
    elements are typically loaded and paired in a tuple that can then be provided to the data loader for
    batching. Problems however arise when the model has multiple inputs or outputs, when the sample needs
    to carry supplemental metadata to simplify debugging, or when transformation operations need to be
    applied only to specific elements of the sample. Here, we fix this issue by specifying that all
    samples are provided to data loaders as dictionaries. The keys of these dictionaries explicitly
    define which value(s) to forward to the model, which value(s) to use for prediction evaluation,
    and which value(s) is(are) only used for debugging. The keys are defined through the task object
    that is generated by the dataset (see :class:`thelper.tasks.Task` for more information).

    To properly use this interface, a derived class must thus implement :func:`thelper.data.Dataset.__getitem__`,
    :func:`thelper.data.Dataset.get_task`, and store its samples as dictionaries in ``self.samples``.

    Attributes:
        name: printable and key-compatible name of the dataset currently being instantiated.
        root: the path to the root directory that is used to figure out where the data is located.
            This path may be unused if the 'config' argument already includes the necessary info.
        config: dictionary of extra parameters that are required by the dataset interface.
        transforms: function or object that should be applied to all loaded samples in order to
            return the data in the requested transformed/augmented state.
        bypass_deepcopy: specifies whether this dataset interface can avoid the (possibly costly)
            deep copy inside :func:`thelper.data.DataConfig.get_loaders`, and instead only use
            a shallow copy. This is false by default, as if the dataset parser contains an
            internal state or a buffer, it would cause problems in multi-threaded data loaders.
        samples: list of dictionaries containing the data that is ready to be forwarded to the
            data loader. Note that relatively costly operations (such as reading images from a disk
            or transforming them) should be delayed until the :func:`thelper.data.Dataset.__getitem__`
            function is called, as they will most likely then be accomplished in a separate thread.

    .. seealso::
        :class:`thelper.data.ExternalDataset`
    """

    def __init__(self, name, root, config=None, transforms=None, bypass_deepcopy=False):
        """Dataset parser constructor.

        In order for derived datasets to be instantiated automatically be the framework from a
        configuration file, the signature of their constructors should match the one shown here.
        This means all required extra parameters must be passed in the 'config' argument, which is
        a dictionary.

        Args:
            name: printable and key-compatible name of the dataset currently being instantiated.
            root: the path to the root directory that is used to figure out where the data is located.
                This path may be unused if the 'config' argument already includes the necessary info.
            config: dictionary of extra parameters that are required by the dataset interface.
            transforms: function or object that should be applied to all loaded samples in order to
                return the data in the requested transformed/augmented state.
            bypass_deepcopy: specifies whether this dataset interface can avoid the (possibly costly)
                deep copy inside :func:`thelper.data.DataConfig.get_loaders`, and instead only use
                a shallow copy. This is false by default, as if the dataset parser contains an
                internal state or a buffer, it would cause problems in multi-threaded data loaders.
        """
        super().__init__()
        if not name:
            raise AssertionError("dataset name must not be empty (lookup might fail)")
        self.name = name
        self.root = root
        self.config = config
        self.transforms = transforms
        self.bypass_deepcopy = bypass_deepcopy  # will determine if we deepcopy in each loader
        self.samples = None  # must be filled by the derived class as a list of dictionaries

    def _get_derived_name(self):
        """Returns a pretty-print version of the derived class's name."""
        dname = str(self.__class__.__qualname__)
        if self.name:
            dname += "." + self.name
        return dname

    def __len__(self):
        """Returns the total number of samples available from this dataset interface."""
        return len(self.samples)

    def __iter__(self):
        """Returns an iterator over the dataset's samples."""
        for idx in range(len(self.samples)):
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        raise NotImplementedError

    @abstractmethod
    def get_task(self):
        """Returns the dataset task object that provides the input/output keys for sample dictionaries."""
        raise NotImplementedError

    def __repr__(self):
        """Returns a print-friendly representation of this dataset."""
        return self._get_derived_name() + " : size=%s, transforms=%s" % (str(len(self)), str(self.transforms))


class ClassificationDataset(Dataset):
    """Classification dataset specialization interface.

    This specialization receives some extra parameters in its constructor and automatically defines
    the dataset task (sample classification) based on those. The derived class must still implement
    :func:`thelper.data.ClassificationDataset.__getitem__`, and it must still store its samples as
    dictionaries in ``self.samples``.

    Attributes:
        task: classification task object containing the key information passed in the constructor.

    .. seealso::
        :class:`thelper.data.Dataset`
    """

    def __init__(self, name, root, class_names, input_key, label_key, meta_keys=None, config=None,
                 transforms=None, bypass_deepcopy=False):
        """Classification dataset parser constructor.

        In order for derived datasets to be instantiated automatically be the framework from a
        configuration file, the signature of their constructors should match the one shown here.
        This means all required extra parameters must be passed in the 'config' argument, which is
        a dictionary.

        Args:
            name: printable and key-compatible name of the dataset currently being instantiated.
            root: the path to the root directory that is used to figure out where the data is located.
                This path may be unused if the 'config' argument already includes the necessary info.
            class_names: list of all class names (or labels) that will be associated with the samples.
            input_key: key used to index the input data in the loaded samples.
            label_key: key used to index the label (or class name) in the loaded samples.
            meta_keys: list of extra keys that will be available in the loaded samples.
            config: dictionary of extra parameters that are required by the dataset interface.
            transforms: function or object that should be applied to all loaded samples in order to
                return the data in the requested transformed/augmented state.
            bypass_deepcopy: specifies whether this dataset interface can avoid the (possibly costly)
                deep copy inside :func:`thelper.data.DataConfig.get_loaders`, and instead only use
                a shallow copy. This is false by default, as if the dataset parser contains an
                internal state or a buffer, it would cause problems in multi-threaded data loaders.
        """
        super().__init__(name, root, config=config, transforms=transforms, bypass_deepcopy=bypass_deepcopy)
        self.task = thelper.tasks.Classification(class_names, input_key, label_key, meta_keys=meta_keys)

    @abstractmethod
    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        raise NotImplementedError

    def get_task(self):
        """Returns the classification task object that provides the input/output keys for sample dictionaries."""
        return self.task


class ExternalDataset(Dataset):
    """External dataset interface.

    This interface allows external classes to be instantiated automatically in the framework through
    a configuration file, as long as they themselves provide implementations for  ``__getitem__`` and
    ``__len__``. This includes all derived classes of ``torch.utils.data.Dataset`` such as
    ``torchvision.datasets.ImageFolder``, and the specialized versions such as ``torchvision.datasets.CIFAR10``.

    Note that for this interface to be compatible with our runtime instantiation rules, the constructor
    needs to receive a fully constructed task object. This object is currently constructed in
    :func:`thelper.data.load_datasets` based on extra parameters; see the code there for more information.

    Attributes:
        dataset_type: type of the external dataset object to instantiate
        task: task object containing the key information passed in the external configuration.
        samples: instantiation of the dataset object itself, faking the presence of a list of samples
        warned_partial_transform: specifies whether the user was warned about partially applying
            transforms to samples without knowing which component is being modified.
        warned_dictionary: specifies whether the user was warned about missing keys in the output
            samples dictionaries.

    .. seealso::
        :class:`thelper.data.Dataset`
    """

    def __init__(self, name, root, dataset_type, task, config=None, transforms=None, bypass_deepcopy=False):
        """External dataset parser constructor.

        Args:
            name: printable and key-compatible name of the dataset currently being instantiated.
            root: the path to the root directory that is used to figure out where the data is located.
                This path may be unused if the 'config' argument already includes the necessary info.
            dataset_type: fully qualified name of the dataset object to instantiate
            task: fully constructed task object providing key information for sample loading.
            config: dictionary of extra parameters that are required by the dataset interface.
            transforms: function or object that should be applied to all loaded samples in order to
                return the data in the requested transformed/augmented state.
            bypass_deepcopy: specifies whether this dataset interface can avoid the (possibly costly)
                deep copy inside :func:`thelper.data.DataConfig.get_loaders`, and instead only use
                a shallow copy. This is false by default, as if the dataset parser contains an
                internal state or a buffer, it would cause problems in multi-threaded data loaders.
        """
        super().__init__(name, root, config=config, transforms=transforms, bypass_deepcopy=bypass_deepcopy)
        logger.info("instantiating external dataset '%s'..." % name)
        if not dataset_type or not hasattr(dataset_type, "__getitem__") or not hasattr(dataset_type, "__len__"):
            raise AssertionError("external dataset type must implement '__getitem__' and '__len__' methods")
        if not issubclass(type(task), thelper.tasks.Task):
            raise AssertionError("task type must be derived from 'thelper.tasks.Task' class")
        self.dataset_type = dataset_type
        self.task = task
        self.samples = dataset_type(**config)
        self.warned_partial_transform = False
        self.warned_dictionary = False

    def _get_derived_name(self):
        """Returns a pretty-print version of the external class's name."""
        dname = str(self.dataset_type)
        if self.name:
            dname += "." + self.name
        return dname

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        sample = self.samples[idx]
        if sample is None:
            # since might have provided an invalid sample count before, it's dangerous to skip empty samples here
            raise AssertionError("invalid sample received in external dataset impl")
        # we will only transform sample contents that are nparrays, PIL images, or torch tensors (might cause issues...)
        warn_partial_transform = False
        warn_dictionary = False
        if isinstance(sample, (list, tuple)):
            out_sample_list = []
            for idx, subs in enumerate(sample):
                if isinstance(subs, (np.ndarray, PIL.Image.Image, torch.Tensor)):
                    out_sample_list.append(self.transforms(subs) if self.transforms else subs)
                else:
                    out_sample_list.append(subs)  # don't transform it, it will probably fail
                    warn_partial_transform = bool(self.transforms)
            out_sample = {str(idx): out_sample_list[idx] for idx in range(len(out_sample_list))}
            warn_dictionary = True
        elif isinstance(sample, dict):
            out_sample = {}
            for key, subs in sample.keys():
                if isinstance(subs, (np.ndarray, PIL.Image.Image, torch.Tensor)):
                    out_sample[key] = self.transforms(subs) if self.transforms else subs
                else:
                    out_sample[key] = subs  # don't transform it, it will probably fail
                    warn_partial_transform = bool(self.transforms)
        elif isinstance(sample, (np.ndarray, PIL.Image.Image, torch.Tensor)):
            out_sample = {"0": self.transforms(sample) if self.transforms else sample}
            warn_dictionary = True
        else:
            # could add checks to see if the sample already behaves like a dict? todo
            raise AssertionError("no clue how to convert given data sample into dictionary")
        if warn_partial_transform and not self.warned_partial_transform:
            logger.warning("blindly transforming sample parts for dataset '%s'; consider using a proper interface" % self.name)
            self.warned_partial_transform = True
        if warn_dictionary and not self.warned_dictionary:
            logger.warning("dataset '%s' not returning samples as dictionaries; will blindly map elements to their indices" % self.name)
            self.warned_dictionary = True
        return out_sample

    def get_task(self):
        """Returns the task object that provides the input/output keys for sample dictionaries."""
        return self.task
