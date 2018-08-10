import logging
import time
from abc import ABC, abstractmethod
from collections import Counter
from copy import copy

import numpy as np
import PIL
import PIL.Image
import torch
import torch.utils.data
import torch.utils.data.sampler

import thelper.utils
import thelper.tasks
import thelper.transforms


logger = logging.getLogger(__name__)


class DataConfig(object):

    def __init__(self, config):
        logger.debug("loading data config")
        if not isinstance(config, dict):
            raise AssertionError("input config should be dict")
        if "batch_size" not in config or not config["batch_size"]:
            raise AssertionError("data config missing 'batch_size' field")
        self.batch_size = config["batch_size"]
        self.shuffle = thelper.utils.str2bool(config["shuffle"]) if "shuffle" in config else False
        self.test_seed = config["test_seed"] if "test_seed" in config and isinstance(config["test_seed"], (int, str)) else None
        self.valid_seed = config["valid_seed"] if "valid_seed" in config and isinstance(config["valid_seed"], (int, str)) else None
        self.workers = config["workers"] if "workers" in config and config["workers"] >= 0 else 1
        self.pin_memory = thelper.utils.str2bool(config["pin_memory"]) if "pin_memory" in config else False
        self.drop_last = thelper.utils.str2bool(config["drop_last"]) if "drop_last" in config else False
        self.train_augments = None
        self.train_augments_append = False
        if "train_augments" in config and config["train_augments"]:
            self.train_augments, self.train_augments_append = thelper.transforms.load_transforms(config["train_augments"])

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

    def get_idx_split(self, dataset_map_size):
        logger.debug("loading dataset split & normalizing ratios")
        for name in self.total_usage:
            if name not in dataset_map_size:
                raise AssertionError("dataset '%s' does not exist" % name)
        indices = {name: list(range(size)) for name, size in dataset_map_size.items()}
        if self.shuffle:
            np.random.seed(self.test_seed) # test idxs will be picked first, then valid+train
            for idxs in indices.values():
                np.random.shuffle(idxs)
        train_idxs, valid_idxs, test_idxs = {}, {}, {}
        offsets = dict.fromkeys(self.total_usage, 0)
        for loader_idx, (idxs_map, ratio_map) in enumerate(zip([test_idxs, valid_idxs, train_idxs],
                                                               [self.test_split, self.valid_split, self.train_split])):
            for name in self.total_usage.keys():
                if name in ratio_map:
                    count = int(round(ratio_map[name] * dataset_map_size[name]))
                    if count < 0:
                        raise AssertionError("ratios should be non-negative values!")
                    elif count < 1:
                        logger.warning("split ratio for '%s' too small, sample set will be empty" % name)
                    begidx = offsets[name]
                    endidx = min(begidx + count, dataset_map_size[name])
                    idxs_map[name] = indices[name][begidx:endidx]
                    offsets[name] = endidx
            if loader_idx==0 and self.shuffle:
                np.random.seed(self.valid_seed)  # all test idxs are now picked, reshuffle for train/valid
                for name in self.total_usage.keys():
                    trainvalid_idxs = indices[name][offsets[name]:]
                    np.random.shuffle(trainvalid_idxs)
                    indices[name][offsets[name]:] = trainvalid_idxs
        return train_idxs, valid_idxs, test_idxs

    def get_data_split(self, dataset_templates):
        dataset_size_map = {name: len(dataset) for name, dataset in dataset_templates.items()}
        train_idxs, valid_idxs, test_idxs = self.get_idx_split(dataset_size_map)
        train_data, valid_data, test_data, loaders = [], [], [], []
        for loader_idx, (idxs_map, datasets) in enumerate(zip([train_idxs, valid_idxs, test_idxs],
                                                              [train_data, valid_data, test_data])):
            for name, sample_idxs in idxs_map.items():
                dataset = copy(dataset_templates[name])
                if loader_idx == 0 and self.train_augments:
                    if dataset.transforms is not None:
                        if self.train_augments_append:  # append or not
                            dataset.transforms = thelper.transforms.Compose([dataset.transforms, copy(self.train_augments)])
                        else:
                            dataset.transforms = thelper.transforms.Compose([copy(self.train_augments), dataset.transforms])
                    else:
                        dataset.transforms = copy(self.train_augments)
                dataset.sampler = SubsetRandomSampler(sample_idxs)
                datasets.append(dataset)
            if len(datasets) > 1:
                dataset = torch.utils.data.ConcatDataset(datasets)
                sampler = torch.utils.data.sampler.RandomSampler(dataset)
                loaders.append(torch.utils.data.DataLoader(dataset,
                                                           batch_size=self.batch_size,
                                                           sampler=sampler,
                                                           num_workers=self.workers,
                                                           pin_memory=self.pin_memory,
                                                           drop_last=self.drop_last))
            elif len(datasets) == 1:
                loaders.append(torch.utils.data.DataLoader(datasets[0],
                                                           batch_size=self.batch_size,
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


def load_dataset_templates(config, root):
    templates = {}
    task_out = None
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
            templates[dataset_name] = dataset_type(name=dataset_name, root=root, config=params, transforms=transforms)
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
            templates[dataset_name] = ExternalDataset(dataset_name, root, dataset_type, task, config=params, transforms=transforms)
        if task_out is None:
            task_out = templates[dataset_name].get_task()
        elif task_out != templates[dataset_name].get_task():
            raise AssertionError("not all datasets have similar task, or sample input/gt keys")
    return templates, task_out


class SubsetRandomSampler(torch.utils.data.sampler.SubsetRandomSampler):

    def __init__(self, indices):
        # the difference between this class and pytorch's default one is the __getitem__ member that provides raw indices
        super().__init__(indices)

    def __getitem__(self, idx):
        # we do not reshuffle here, as we cannot know when the cycle is 'reset'; indices should thus come in pre-shuffled
        return self.indices[idx]


class Dataset(torch.utils.data.Dataset, ABC):

    def __init__(self, name, root, config=None, transforms=None):
        super().__init__()
        if not name:
            raise AssertionError("dataset name must not be empty (lookup might fail)")
        self.name = name
        self.root = root
        self.config = config
        self.transforms = transforms
        self._sampler = None
        self.samples = None  # must be filled by the derived class

    # todo: add method to reset sampler shuffling when epoch complete?

    def _get_derived_name(self):
        dname = str(self.__class__.__qualname__)
        if self.name:
            dname += "." + self.name
        return dname

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, newsampler):
        if newsampler is not None:
            sample_iter = iter(newsampler)
            try:
                while True:
                    sample_idx = next(sample_iter)
                    if sample_idx < 0 or sample_idx > len(self.samples):
                        raise AssertionError("sampler provides oob indices for assigned dataset")
            except StopIteration:
                pass
        self._sampler = newsampler

    def total_size(self):
        # bypasses sampler, if one is active
        return len(self.samples)

    def __len__(self):
        # if a sampler is active, return its subset size
        if self.sampler is not None:
            return len(self.sampler)
        return len(self.samples)

    def __iter__(self):
        if not self.sampler:
            for idx in range(len(self.samples)):
                yield self[idx]
        else:
            for idx in self.sampler:
                yield self[idx]

    @abstractmethod
    def __getitem__(self, idx):
        # note: returned samples should be dictionaries (keys are used in config file to setup model i/o during training)
        raise NotImplementedError

    @abstractmethod
    def get_task(self):
        # the task is up to the derived class to specify (but it must be derived from the thelper.tasks.Task)
        raise NotImplementedError


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
        if self.sampler is not None:
            idx = self.sampler[idx]
        sample = self.samples[idx]
        # we will only transform sample contents that are nparrays, PIL images, or torch tensors (might cause issues...)
        if not self.transforms or sample is None:
            return sample
        warn_partial_transform = False
        warn_dictionary = False
        if isinstance(sample, (list, tuple)):
            out_sample_list = []
            for idx, subs in enumerate(sample):
                if isinstance(subs, (np.ndarray, PIL.Image.Image, torch.Tensor)):
                    out_sample_list.append(self.transforms(subs))
                else:
                    out_sample_list.append(subs)  # don't transform it, it will probably fail
                    warn_partial_transform = True
            out_sample = {str(idx): out_sample_list[idx] for idx in range(len(out_sample_list))}
            warn_dictionary = True
        elif isinstance(sample, dict):
            out_sample = {}
            for key, subs in sample.keys():
                if isinstance(subs, (np.ndarray, PIL.Image.Image, torch.Tensor)):
                    out_sample[key] = self.transforms(subs)
                else:
                    out_sample[key] = subs  # don't transform it, it will probably fail
                    warn_partial_transform = True
        elif isinstance(sample, (np.ndarray, PIL.Image.Image, torch.Tensor)):
            out_sample = {"0": self.transforms(sample)}
            warn_dictionary = True
        else:
            raise AssertionError("no clue how to transform given data sample")
        if warn_partial_transform and not self.warned_partial_transform:
            logger.warning("blindly transforming sample parts for dataset '%s'; consider using a proper interface" % self.name)
            self.warned_partial_transform = True
        if warn_dictionary and not self.warned_dictionary:
            logger.warning("dataset '%s' not returning samples as dictionaries; will blindly map elements to their indices" % self.name)
            self.warned_dictionary = True
        return out_sample

    def get_task(self):
        return self.task
