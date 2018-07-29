import logging
import inspect
import os
from abc import abstractmethod
from collections import Counter
from copy import copy

import PIL
import numpy as np
import torch
import torch.utils.data

import thelper.transforms
import thelper.utils

logger = logging.getLogger(__name__)


class DataConfig(object):
    def __init__(self,config):
        self.logger = thelper.utils.get_class_logger()
        self.logger.debug("loading data config")
        if not isinstance(config,dict):
            raise AssertionError("input config should be dict")
        if "batch_size" not in config or not config["batch_size"]:
            raise AssertionError("data config missing 'batch_size' field")
        self.batch_size = config["batch_size"]
        self.shuffle = thelper.utils.str2bool(config["shuffle"]) if "shuffle" in config else False
        self.seed = config["seed"] if "seed" in config and isinstance(config["seed"],(int,str)) else None
        self.workers = config["workers"] if "workers" in config and config["workers"]>=0 else 1
        self.pin_memory = thelper.utils.str2bool(config["pin_memory"]) if "pin_memory" in config else False
        self.drop_last = thelper.utils.str2bool(config["drop_last"]) if "drop_last" in config else False
        self.train_augments = None
        if "train_augments" in config and config["train_augments"]:
            self.train_augments = thelper.transforms.load_transforms(config["train_augments"])
        if "train_split" not in config or not config["train_split"]:
            raise AssertionError("data config missing 'train_split' field")
        self.train_split = config["train_split"]
        if any(ratio<0 or ratio>1 for ratio in self.train_split.values()):
            raise AssertionError("split ratios must be in [0,1]")
        if "valid_split" not in config or not config["valid_split"]:
            raise AssertionError("data config missing 'valid_split' field")
        self.valid_split = config["valid_split"]
        if any(ratio<0 or ratio>1 for ratio in self.valid_split.values()):
            raise AssertionError("split ratios must be in [0,1]")
        if "test_split" not in config or not config["test_split"]:
            raise AssertionError("data config missing 'test_split' field")
        self.test_split = config["test_split"]
        if any(ratio<0 or ratio>1 for ratio in self.test_split.values()):
            raise AssertionError("split ratios must be in [0,1]")
        self.total_usage = Counter(self.train_split)+Counter(self.valid_split)+Counter(self.test_split)
        for name,usage in self.total_usage.items():
            if usage>0 and usage!=1:
                self.logger.warning("dataset split for '%s' does not sum 1; will normalize..."%name)
                self.train_split[name] /= usage
                self.valid_split[name] /= usage
                self.test_split[name] /= usage

    def get_idx_split(self,dataset_map_size):
        self.logger.debug("loading dataset split & normalizing ratios")
        for name in self.total_usage:
            if name not in dataset_map_size:
                raise AssertionError("dataset '%s' does not exist"%name)
        indices = {name:list(range(size)) for name,size in dataset_map_size.items()}
        if self.shuffle:
            np.random.seed(self.seed)
            for idxs in indices.values():
                np.random.shuffle(idxs)
        train_idxs,valid_idxs,test_idxs = {},{},{}
        offsets = dict.fromkeys(self.total_usage,0)
        for name in self.total_usage.keys():
            for idxs_map,ratio_map in zip([train_idxs,valid_idxs,test_idxs],[self.train_split,self.valid_split,self.test_split]):
                if name in ratio_map:
                    count = int(round(ratio_map[name]*dataset_map_size[name]))
                    if count<0:
                        raise AssertionError("ratios should be non-negative values!")
                    elif count<1:
                        self.logger.warning("split ratio for '%s' too small, sample set will be empty"%name)
                    begidx = offsets[name]
                    endidx = min(begidx+count,dataset_map_size[name])
                    idxs_map[name] = indices[name][begidx:endidx]
                    offsets[name] = endidx
        return train_idxs,valid_idxs,test_idxs

    def get_data_split(self,dataset_templates):
        dataset_size_map = {name:len(dataset) for name,dataset in dataset_templates.items()}
        train_idxs,valid_idxs,test_idxs = self.get_idx_split(dataset_size_map)
        train_data,valid_data,test_data,loaders = [],[],[],[]
        for loader_idx,(idxs_map,datasets) in enumerate(zip([train_idxs,valid_idxs,test_idxs],
                                                            [train_data,valid_data,test_data])):
            for name,sample_idxs in idxs_map.items():
                dataset = copy(dataset_templates[name])
                if loader_idx==0 and self.train_augments:
                    if dataset.transforms:
                        if self.train_augments[1]: # append or not
                            dataset.transforms = thelper.transforms.Compose([dataset.transforms,copy(self.train_augments[0])])
                        else:
                            dataset.transforms = thelper.transforms.Compose([copy(self.train_augments[0]),dataset.transforms])
                    else:
                        dataset.transforms = copy(self.train_augments)
                dataset.sampler = SubsetRandomSampler(sample_idxs)
                datasets.append(dataset)
            if len(datasets)>1:
                dataset = torch.utils.data.ConcatDataset(datasets)
                sampler = torch.utils.data.sampler.RandomSampler(dataset)
                loaders.append(torch.utils.data.DataLoader(dataset,
                                                           batch_size=self.batch_size,
                                                           sampler=sampler,
                                                           num_workers=self.workers,
                                                           pin_memory=self.pin_memory,
                                                           drop_last=self.drop_last))
            else:
                loaders.append(torch.utils.data.DataLoader(datasets[0],
                                                           batch_size=self.batch_size,
                                                           num_workers=self.workers,
                                                           pin_memory=self.pin_memory,
                                                           drop_last=self.drop_last))
        train_loader,valid_loader,test_loader = loaders
        return train_loader,valid_loader,test_loader


def load_dataset_templates(config,root):
    templates = {}
    # todo : check compatibility between predicted types? (thru key map?)
    for dataset_name,dataset_config in config.items():
        if "type" not in dataset_config:
            raise AssertionError("missing field 'type' for instantiation of dataset '%s'"%dataset_name)
        type = thelper.utils.import_class(dataset_config["type"])
        if "params" not in dataset_config:
            raise AssertionError("missing field 'params' for instantiation of dataset '%s'"%dataset_name)
        params = thelper.utils.keyvals2dict(dataset_config["params"])
        transforms = None
        if "transforms" in dataset_config and dataset_config["transforms"]:
            transforms,append = thelper.transforms.load_transforms(dataset_config["transforms"])
        if inspect.isclass(type) and issubclass(type,thelper.data.Dataset):
            # assume that the dataset is derived from thelper.data.Dataset (it is fully sampling-ready)
            templates[dataset_name] = type(dataset_name,root,config=params,transforms=transforms)
        else:
            # assume that __getitem__ and __len__ are implemented, but we need to make it sampling-ready
            templates[dataset_name] = thelper.data.ExternalDataset(dataset_name,root,type,config=params,transforms=transforms)
    return templates


class SubsetRandomSampler(torch.utils.data.sampler.SubsetRandomSampler):
    def __init__(self,indices):
        # the difference between this class and pytorch's default one is the __getitem__ member that provides raw indices
        super().__init__(indices)

    def __getitem__(self,idx):
        # we do not reshuffle here, as we cannot know when the cycle is 'reset'; indices should thus come in pre-shuffled
        return self.indices[idx]


class Dataset(torch.utils.data.Dataset):
    def __init__(self,name,root,config=None,transforms=None):
        super().__init__()
        if not name:
            raise AssertionError("dataset name must not be empty (lookup might fail)")
        if not root or not os.path.exists(root) or not os.path.isdir(root):
            raise AssertionError("dataset root folder at '%s' does not exist"%root)
        self.logger = thelper.utils.get_class_logger()
        self.name = name
        self.root = root
        self.config = config
        self.transforms = transforms
        self._sampler = None
        self.samples = None  # must be filled by the derived class

    # todo: add method to reset sampler shuffling when epoch complete?

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self,newsampler):
        if newsampler:
            sample_iter = iter(newsampler)
            try:
                while True:
                    sample_idx = next(sample_iter)
                    if sample_idx<0 or sample_idx>len(self.samples):
                        raise AssertionError("sampler provides oob indices for assigned dataset")
            except StopIteration:
                pass
        self._sampler = newsampler

    def total_size(self):
        # bypasses sampler, if one is active
        return len(self.samples)

    def __len__(self):
        # if a sampler is active, return its subset size
        if self.sampler:
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
    def __getitem__(self,idx):
        # note: returned samples should be dictionaries (keys are used in config file to setup model i/o during training)
        raise NotImplementedError


class ExternalDataset(Dataset):
    def __init__(self,name,root,type,config=None,transforms=None):
        super().__init__(name,root,config=config,transforms=transforms)
        self.logger.info("instantiating external dataset '%s'..."%name)
        if not type or not hasattr(type,"__getitem__") or not hasattr(type,"__len__"):
            raise AssertionError("external dataset type must implement '__getitem__' and '__len__' methods")
        self.type = type
        self.samples = type(**config)
        self.warned_partial_transform = False
        self.warned_dictionary = False

    def __getitem__(self,idx):
        if self.sampler:
            idx = self.sampler[idx]
        sample = self.samples[idx]
        # we will only transform sample contents that are nparrays, PIL images, or torch tensors (might cause issues...)
        if not self.transforms or sample is None:
            return sample
        out_sample = None
        warn_partial_transform = False
        warn_dictionary = False
        if isinstance(sample,(list,tuple)):
            out_sample_list = []
            for idx,subs in enumerate(sample):
                if isinstance(subs,(np.ndarray,PIL.Image.Image,torch.Tensor)):
                    out_sample_list.append(self.transforms(subs))
                else:
                    out_sample_list.append(subs)  # don't transform it, it will probably fail
                    warn_partial_transform = True
            out_sample = {str(idx):out_sample_list[idx] for idx in range(len(out_sample_list))}
            warn_dictionary = True
        elif isinstance(sample,dict):
            out_sample = {}
            for key,subs in sample.keys():
                if isinstance(subs,(np.ndarray,PIL.Image.Image,torch.Tensor)):
                    out_sample[key] = self.transforms(subs)
                else:
                    out_sample[key] = subs  # don't transform it, it will probably fail
                    warn_partial_transform = True
        elif isinstance(sample,(np.ndarray,PIL.Image.Image,torch.Tensor)):
            out_sample = {"0":self.transforms(sample)}
            warn_dictionary = True
        else:
            raise AssertionError("no clue how to transform given data sample")
        if warn_partial_transform and not self.warned_partial_transform:
            self.logger.warning("blindly transforming sample parts for dataset '%s'; consider using a proper interface"%self.name)
            self.warned_partial_transform = True
        if warn_dictionary and not self.warned_dictionary:
            self.logger.warning("dataset '%s' not returning samples as dictionaries; will map elements to their indices"%self.name)
            self.warned_dictionary = True
        return out_sample
