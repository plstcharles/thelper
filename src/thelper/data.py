import logging
import time
import copy
import os
import json
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import PIL
import PIL.Image
import torch
import torch.utils.data
import torch.utils.data.sampler

import thelper.utils
import thelper.tasks
import thelper.samplers
import thelper.transforms

logger = logging.getLogger(__name__)


def load(config, data_root, save_dir=None):
    if save_dir is not None:
        data_logger_path = os.path.join(save_dir, "logs", "data.log")
        data_logger_format = logging.Formatter("[%(asctime)s - %(process)s] %(levelname)s : %(message)s")
        data_logger_fh = logging.FileHandler(data_logger_path)
        data_logger_fh.setFormatter(data_logger_format)
        logger.addHandler(data_logger_fh)
        logger.info("created data log for session '%s'" % config["name"])
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
    datasets, task = load_datasets(datasets_config, data_root)
    logger.info("task info: %s" % str(task))
    if save_dir is not None:
        with open(os.path.join(save_dir, "logs", "task.log"), "w") as fd:
            fd.write(str(task) + "\n")
    for dataset_name, dataset in datasets.items():
        logger.info("dataset '%s' info: %s" % (dataset_name, str(dataset)))
        if save_dir is not None:
            with open(os.path.join(save_dir, "logs", dataset_name + ".log"), "w") as fd:
                fd.write(str(dataset) + "\n")
                if hasattr(dataset, "samples") and isinstance(dataset.samples, list):
                    for idx, sample in enumerate(dataset.samples):
                        fd.write("%d: %s\n" % (idx, str(sample)))
    logger.debug("loading data usage config")
    if "data_config" not in config or not config["data_config"]:
        raise AssertionError("config missing 'data_config' field")
    data_config = thelper.data.DataConfig(config["data_config"])
    logger.debug("splitting datasets and creating loaders")
    train_loader, valid_loader, test_loader = data_config.get_data_split(datasets, task)
    return task, train_loader, valid_loader, test_loader


def load_datasets(config, root):
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
            transforms, _ = thelper.transforms.load_transforms(dataset_config["transforms"])
        if issubclass(dataset_type, Dataset):
            # assume that the dataset is derived from thelper.data.Dataset (it is fully sampling-ready)
            dataset = dataset_type(name=dataset_name, root=root, config=params, transforms=transforms)
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
            dataset = ExternalDataset(dataset_name, root, dataset_type, task, config=params, transforms=transforms)
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

    def __init__(self, config):
        logger.debug("loading data config")
        if not isinstance(config, dict):
            raise AssertionError("input config should be dict")
        if "batch_size" not in config or not config["batch_size"]:
            raise AssertionError("data config missing 'batch_size' field")
        self.batch_size = config["batch_size"]
        logger.debug("loaders will use batch size = %d" % self.batch_size)
        self.shuffle = thelper.utils.str2bool(config["shuffle"]) if "shuffle" in config else False
        if self.shuffle:
            logger.debug("dataset samples will be shuffled according to predefined seeds")
        self.test_seed = config["test_seed"] if "test_seed" in config and isinstance(config["test_seed"], (int, str)) else None
        self.valid_seed = config["valid_seed"] if "valid_seed" in config and isinstance(config["valid_seed"], (int, str)) else None
        if self.shuffle and self.test_seed is None:
            np.random.seed()
            self.test_seed = np.random.randint(2**16)
            logger.debug("setting test seed to %d" % self.test_seed)
        if self.shuffle and self.valid_seed is None:
            np.random.seed()
            self.valid_seed = np.random.randint(2**16)
            logger.debug("setting valid seed to %d" % self.valid_seed)
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
                self.sampler_pass_labels = thelper.utils.str2bool(sampler_config["pass_labels"]) if "pass_labels" in sampler_config else False
                apply_train = thelper.utils.str2bool(sampler_config["apply_train"]) if "apply_train" in sampler_config else True
                apply_valid = thelper.utils.str2bool(sampler_config["apply_valid"]) if "apply_valid" in sampler_config else False
                apply_test = thelper.utils.str2bool(sampler_config["apply_test"]) if "apply_test" in sampler_config else False
                self.sampler_apply = [apply_train, apply_valid, apply_test]
                logger.debug("global sampler will be applied to loaders: %s" % str(self.sampler_apply))
        self.train_augments = None
        self.train_augments_append = False
        if "train_augments" in config and config["train_augments"]:
            self.train_augments, self.train_augments_append = thelper.transforms.load_transforms(config["train_augments"])
            if self.train_augments_append:
                logger.debug("will append train augmentations: %s" % str(self.train_augments))
            else:
                logger.debug("will prepend train augmentations: %s" % str(self.train_augments))

        def get_split(prefix, config):
            key = prefix + "_split"
            if key not in config or not config[key]:
                return {}
            split = config[key]
            if any(ratio < 0 or ratio > 1 for ratio in split.values()):
                raise AssertionError("split ratios in '%s' must be in [0,1]" % key)
            return split

        self.train_split = get_split("train", config)
        self.valid_split = get_split("valid", config)
        self.test_split = get_split("test", config)
        if not self.train_split and not self.valid_split and not self.test_split:
            raise AssertionError("data config must define a split for at least one loader type (train/valid/test)")
        self.total_usage = Counter(self.train_split) + Counter(self.valid_split) + Counter(self.test_split)
        for name, usage in self.total_usage.items():
            if usage != 1:
                normalize_ratios = None
                if usage < 0:
                    raise AssertionError("ratio should never be negative...")
                elif usage > 0 and usage < 1:
                    time.sleep(0.25)  # to make sure all debug/info prints are done, and we see the question
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

    def get_idx_split(self, indices):
        for name in self.total_usage:
            if name not in indices:
                raise AssertionError("dataset '%s' does not exist" % name)
        indices = {name: copy.deepcopy(indices) for name, indices in indices.items()}
        if self.shuffle:
            np.random.seed(self.test_seed)  # test idxs will be picked first, then valid+train
            for idxs in indices.values():
                np.random.shuffle(idxs)
        train_idxs, valid_idxs, test_idxs = {}, {}, {}
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
        return train_idxs, valid_idxs, test_idxs

    def get_data_split(self, datasets, task):
        dataset_sizes = {dataset_name: len(dataset) for dataset_name, dataset in datasets.items()}
        global_size = sum(len(dataset) for dataset in datasets.values())
        logger.info("splitting datasets with parsed sizes = %s" % str(dataset_sizes))
        if isinstance(task, thelper.tasks.Classification):
            # note: with current impl, all classes will be shuffle the same way... (shouldnt matter, right?)
            global_class_names = task.get_class_names()
            logger.info("will split evenly over %d classes..." % len(global_class_names))
            dataset_class_sample_maps = {dataset_name: task.get_class_sample_map(dataset.samples) for dataset_name, dataset in datasets.items()}
            train_idxs, valid_idxs, test_idxs = {}, {}, {}
            for class_name in global_class_names:
                curr_class_samples, curr_class_size = {}, {}
                for dataset_name, dataset in datasets.items():
                    class_samples = dataset_class_sample_maps[dataset_name][class_name] if class_name in dataset_class_sample_maps[dataset_name] else []
                    samples_pairs = list(zip(class_samples, [class_name] * len(class_samples)))
                    curr_class_samples[dataset_name] = samples_pairs
                    curr_class_size[dataset_name] = len(curr_class_samples[dataset_name])
                    logger.debug("dataset '{}' class '{}' sample count: {} ({}% of local, {}% of total)".format(
                        dataset_name,
                        class_name,
                        curr_class_size[dataset_name],
                        int(100 * curr_class_size[dataset_name] / dataset_sizes[dataset_name]),
                        int(100 * curr_class_size[dataset_name] / global_size)))
                class_train_idxs, class_valid_idxs, class_test_idxs = self.get_idx_split(curr_class_samples)
                for idxs_dict_list, class_idxs_dict_list in zip([train_idxs, valid_idxs, test_idxs],
                                                                [class_train_idxs, class_valid_idxs, class_test_idxs]):
                    for dataset_name in datasets:
                        if dataset_name in idxs_dict_list:
                            idxs_dict_list[dataset_name] += class_idxs_dict_list[dataset_name]
                        else:
                            idxs_dict_list[dataset_name] = class_idxs_dict_list[dataset_name]
            # one last intra-dataset shuffle for good mesure, samples of the same class should not be always fed consecutively
            for dataset_name in datasets:
                for idxs_dict_list in [train_idxs, valid_idxs, test_idxs]:
                    np.random.shuffle(idxs_dict_list[dataset_name])
        else:  # task is not classif-related, no balancing to be done
            dataset_indices = {}
            for dataset_name in datasets:
                # note: all indices paired with 'None' below as class is ignored; used for compatibility with code above
                dataset_indices[dataset_name] = list(zip(list(range(dataset_sizes[dataset_name])), [None] * len(dataset_sizes[dataset_name])))
            train_idxs, valid_idxs, test_idxs = self.get_idx_split(dataset_indices)
        loaders = []
        for loader_idx, idxs_map in enumerate([train_idxs, valid_idxs, test_idxs]):
            loader_sample_idx_offset = 0
            loader_sample_classes = []
            loader_sample_idxs = []
            loader_datasets = []
            for dataset_name, sample_idxs in idxs_map.items():
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
                for sample_idx_idx in range(len(sample_idxs)):
                    loader_sample_idxs.append(sample_idxs[sample_idx_idx][0] + loader_sample_idx_offset)
                    loader_sample_classes.append(sample_idxs[sample_idx_idx][1])  # values were paired earlier, 0=idx, 1=label
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
                                                           drop_last=self.drop_last))
            else:
                loaders.append(None)
        train_loader, valid_loader, test_loader = loaders
        train_samples = len(train_loader) if train_loader else 0
        valid_samples = len(valid_loader) if valid_loader else 0
        test_samples = len(test_loader) if test_loader else 0
        logger.info("initialized loaders with batch counts: train=%d, valid=%d, test=%d" % (train_samples, valid_samples, test_samples))
        return train_loader, valid_loader, test_loader


class Dataset(torch.utils.data.Dataset, ABC):

    def __init__(self, name, root, config=None, transforms=None):
        super().__init__()
        if not name:
            raise AssertionError("dataset name must not be empty (lookup might fail)")
        self.name = name
        self.root = root
        self.config = config
        self.transforms = transforms
        self.samples = None  # must be filled by the derived class as a list of dictionaries

    def _get_derived_name(self):
        dname = str(self.__class__.__qualname__)
        if self.name:
            dname += "." + self.name
        return dname

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        for idx in range(len(self.samples)):
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx):
        # note: returned samples should be dictionaries (keys are used in config file to setup model i/o during training)
        raise NotImplementedError

    @abstractmethod
    def get_task(self):
        # the task is up to the derived class to specify (but it must be derived from the thelper.tasks.Task)
        raise NotImplementedError


class ClassificationDataset(Dataset):

    def __init__(self, name, root, class_names, input_key, label_key, meta_keys=None, config=None, transforms=None):
        super().__init__(name, root, config=config, transforms=transforms)
        self.task = thelper.tasks.Classification(class_names, input_key, label_key, meta_keys=meta_keys)

    @abstractmethod
    def __getitem__(self, idx):
        # note: returned samples should be dictionaries (keys are used in config file to setup model i/o during training)
        raise NotImplementedError

    def get_task(self):
        return self.task


class ExternalDataset(Dataset):

    def __init__(self, name, root, dataset_type, task, config=None, transforms=None):
        super().__init__(name, root, config=config, transforms=transforms)
        logger.info("instantiating external dataset '%s'..." % name)
        if not dataset_type or not hasattr(dataset_type, "__getitem__") or not hasattr(dataset_type, "__len__"):
            raise AssertionError("external dataset type must implement '__getitem__' and '__len__' methods")
        if not issubclass(type(task), thelper.tasks.Task):
            raise AssertionError("task type must be derived from 'thelper.tasks.Task' class")
        self.dataset_type = dataset_type
        self.task = task
        self.key_in, self.key_gt = None, None
        self.samples = dataset_type(**config)
        self.warned_partial_transform = False
        self.warned_dictionary = False

    def _get_derived_name(self):
        dname = thelper.utils.get_caller_name(0).rsplit(".", 1)[0]
        if self.name:
            dname += "." + self.name
        return dname

    def __getitem__(self, idx):
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
        return self.task
