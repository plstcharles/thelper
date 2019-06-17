"""Dataset loaders module.

This module contains a dataset loader specialization used to properly seed samplers and workers.
"""

import copy
import inspect
import logging
import math
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.utils.data
import torch.utils.data.sampler
import tqdm

import thelper.tasks
import thelper.transforms
import thelper.utils

logger = logging.getLogger(__name__)


def default_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size.

    This function is copied from PyTorch's `torch.utils.data.dataloader.default_collate`, but additionally
    supports custom objects from the framework (such as bounding boxes).

    See ``torch.utils.data.DataLoader`` for more information.
    """
    import torchvision
    from torch._six import container_abcs, string_classes, int_classes
    error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
    torchvision_ver = [int(v) for v in torchvision.__version__.split(".")]
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if torchvision_ver[0] > 0 or torchvision_ver[1] >= 3:  # ver >= 0.3
            if torch.utils.data._utils.collate._use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
        else:  # ver < 0.3
            if torch.utils.data.dataloader._use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' and \
            elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if torchvision_ver[0] > 0 or torchvision_ver[1] >= 3:  # ver >= 0.3
                if torch.utils.data._utils.collate.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(error_msg_fmt.format(elem.dtype))
            else:  # ver < 0.3
                import re
                if re.search('[SaUO]', elem.dtype.str) is not None:
                    raise TypeError(error_msg_fmt.format(elem.dtype))
            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            if torchvision_ver[0] > 0 or torchvision_ver[1] >= 3:  # ver >= 0.3
                return torch.utils.data._utils.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
            else:  # ver < 0.3
                return torch.utils.data.dataloader.numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        if isinstance(batch, list) and all([isinstance(l, list) for l in batch]) and \
                all([isinstance(b, thelper.data.BoundingBox) for l in batch for b in l]):
            return batch
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]
    raise TypeError((error_msg_fmt.format(type(batch[0]))))


class DataLoader(torch.utils.data.DataLoader):
    """Specialized data loader used to load minibatches from a dataset parser.

    This specialization handles the seeding of samplers and workers.

    See ``torch.utils.data.DataLoader`` for more information on attributes/methods.
    """
    def __init__(self, *args, seeds=None, epoch=0, collate_fn=default_collate, **kwargs):
        super().__init__(*args, collate_fn=collate_fn, worker_init_fn=self._worker_init_fn, **kwargs)
        self.seeds = {}
        if seeds is not None:
            if not isinstance(seeds, dict):
                raise AssertionError("unexpected seed pack type")
            self.seeds = seeds
        if not isinstance(epoch, int) or epoch < 0:
            raise AssertionError("invalid epoch value")
        self.epoch = epoch
        self.num_workers = kwargs["num_workers"] if "num_workers" in kwargs else 0

    def __iter__(self):
        """Advances the epoch number for the workers initialization function."""
        self.set_epoch(self.epoch)  # preset for all attributes
        if self.num_workers == 0:
            if "torch" in self.seeds:
                torch.manual_seed(self.seeds["torch"] + self.epoch)
                torch.cuda.manual_seed_all(self.seeds["torch"] + self.epoch)
            if "numpy" in self.seeds:
                np.random.seed(self.seeds["numpy"] + self.epoch)
            if "random" in self.seeds:
                random.seed(self.seeds["random"] + self.epoch)
        result = super().__iter__()
        self.epoch += 1
        return result

    def set_epoch(self, epoch=0):
        """Sets the current epoch number in order to offset RNG states for the workers and the sampler."""
        if not isinstance(epoch, int) or epoch < 0:
            raise AssertionError("invalid epoch value")
        self.epoch = epoch
        if self.sampler is not None:
            if hasattr(self.sampler, "set_epoch") and callable(self.sampler.set_epoch):
                self.sampler.set_epoch(self.epoch)
        if hasattr(self, "dataset"):
            if hasattr(self.dataset, "set_epoch") and callable(self.dataset.set_epoch):
                self.dataset.set_epoch(epoch)
            if hasattr(self.dataset, "transforms"):
                if hasattr(self.dataset.transforms, "set_epoch") and callable(self.dataset.transforms.set_epoch):
                    self.dataset.transforms.set_epoch(epoch)

    def _worker_init_fn(self, worker_id):
        """Sets up the RNGs state of each worker based on their unique id and the epoch number."""
        seed_offset = self.num_workers * self.epoch
        if "torch" in self.seeds:
            torch.manual_seed(self.seeds["torch"] + seed_offset + worker_id)
            torch.cuda.manual_seed_all(self.seeds["torch"] + seed_offset + worker_id)
        if "numpy" in self.seeds:
            np.random.seed(self.seeds["numpy"] + seed_offset + worker_id)
        if "random" in self.seeds:
            random.seed(self.seeds["random"] + seed_offset + worker_id)

    @property
    def sample_count(self):
        return len(self.sampler) if self.sampler is not None else len(self.dataset)


class LoaderFactory:
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
        default_batch_size = 0
        if "batch_size" in config:
            if any([v in config for v in ["train_batch_size", "valid_batch_size", "test_batch_size"]]):
                raise AssertionError("specifying 'batch_size' overrides all other (loader-specific) values")
            default_batch_size = int(thelper.utils.get_key("batch_size", config))
        self.train_batch_size = thelper.utils.get_key_def("train_batch_size", config, default_batch_size)
        self.valid_batch_size = thelper.utils.get_key_def("valid_batch_size", config, default_batch_size)
        self.test_batch_size = thelper.utils.get_key_def("test_batch_size", config, default_batch_size)
        logger.debug("loaders will use batch sizes:\n  train = %d\n  valid = %d\n  test = %d" %
                     (self.train_batch_size, self.valid_batch_size, self.test_batch_size))
        self.train_scale = thelper.utils.get_key_def("train_scale", config, 1.0)
        self.valid_scale = thelper.utils.get_key_def("valid_scale", config, 1.0)
        self.test_scale = thelper.utils.get_key_def("test_scale", config, 1.0)
        logger.debug("samplers will use scaling factors:\n  train = %f\n  valid = %f\n  test = %f" %
                     (self.train_scale, self.valid_scale, self.test_scale))
        default_collate_fn = default_collate
        if "collate_fn" in config:
            if any([v in config for v in ["train_collate_fn", "valid_collate_fn", "test_collate_fn"]]):
                raise AssertionError("specifying 'collate_fn' overrides all other (loader-specific) values")
            default_collate_fn = self._get_collate_fn(config["collate_fn"])
        self.train_collate_fn = self._get_collate_fn(thelper.utils.get_key_def("train_collate_fn", config, default_collate_fn))
        self.valid_collate_fn = self._get_collate_fn(thelper.utils.get_key_def("valid_collate_fn", config, default_collate_fn))
        self.test_collate_fn = self._get_collate_fn(thelper.utils.get_key_def("test_collate_fn", config, default_collate_fn))
        self.train_shuffle = thelper.utils.str2bool(thelper.utils.get_key_def(["shuffle", "train_shuffle"], config, True))
        self.valid_shuffle = thelper.utils.str2bool(thelper.utils.get_key_def(["shuffle", "valid_shuffle"], config, False))
        self.test_shuffle = thelper.utils.str2bool(thelper.utils.get_key_def(["shuffle", "test_shuffle"], config, False))
        np.random.seed()  # for seed generation below (if needed); will be reseeded afterwards
        test_seed = self._get_seed(["test_seed", "test_split_seed"], config, (int, str))
        valid_seed = self._get_seed(["valid_seed", "valid_split_seed"], config, (int, str))
        torch_seed = self._get_seed(["torch_seed"], config, int)
        numpy_seed = self._get_seed(["numpy_seed"], config, int)
        random_seed = self._get_seed(["random_seed"], config, int)
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
        np.random.seed(numpy_seed)
        random.seed(random_seed)
        self.seeds = {
            "test": test_seed,
            "valid": valid_seed,
            "torch": torch_seed,
            "numpy": numpy_seed,
            "random": random_seed
        }
        self.workers = config["workers"] if "workers" in config and config["workers"] >= 0 else 1
        self.pin_memory = thelper.utils.str2bool(config["pin_memory"]) if "pin_memory" in config else False
        self.drop_last = thelper.utils.str2bool(config["drop_last"]) if "drop_last" in config else False
        if self.drop_last:
            logger.debug("loaders will drop last batch if sample count not multiple of batch size")
        self.sampler_type = None
        self.train_sampler, self.valid_sampler, self.test_sampler = None, None, None
        sampler_config = thelper.utils.get_key_def("sampler", config, {})
        if sampler_config:
            sampler_type = thelper.utils.get_key("type", sampler_config)
            self.sampler_type = thelper.utils.import_class(sampler_type)
            self.sampler_params = thelper.utils.get_key_def(["params", "parameters"], sampler_config, {})
            logger.debug("will use sampler with type '%s' and config : %s" % (str(self.sampler_type), str(self.sampler_params)))
            self.sampler_pass_labels = thelper.utils.str2bool(thelper.utils.get_key_def("pass_labels", sampler_config, False))
            self.sampler_pass_labels_param_name = thelper.utils.get_key_def("pass_labels_param_name", sampler_config, "labels")
            self.train_sampler = thelper.utils.str2bool(thelper.utils.get_key_def("apply_train", sampler_config, True))
            self.valid_sampler = thelper.utils.str2bool(thelper.utils.get_key_def("apply_valid", sampler_config, False))
            self.test_sampler = thelper.utils.str2bool(thelper.utils.get_key_def("apply_test", sampler_config, False))
            logger.debug("global sampler will be applied as: %s" % str([self.train_sampler, self.valid_sampler, self.test_sampler]))
        train_augs_targets = ["augments", "trainvalid_augments", "train_augments"]
        valid_augs_targets = ["augments", "trainvalid_augments", "eval_augments", "validtest_augments", "valid_augments"]
        test_augs_targets = ["augments", "eval_augments", "validtest_augments", "test_augments"]
        self.train_augments, self.train_augments_append = self._get_augments(train_augs_targets, "train", config)
        self.valid_augments, self.valid_augments_append = self._get_augments(valid_augs_targets, "valid", config)
        self.test_augments, self.test_augments_append = self._get_augments(test_augs_targets, "test", config)
        self.base_transforms = None
        if "base_transforms" in config and config["base_transforms"]:
            logger.debug("loading base transforms...")
            self.base_transforms = thelper.transforms.load_transforms(config["base_transforms"])
            if self.base_transforms:
                logger.debug("base transforms: %s" % str(self.base_transforms))
        self.train_split = self._get_ratios_split("train", config)
        self.valid_split = self._get_ratios_split("valid", config)
        self.test_split = self._get_ratios_split("test", config)
        if not self.train_split and not self.valid_split and not self.test_split:
            raise AssertionError("data config must define a split for at least one loader type (train/valid/test)")
        self.total_usage = Counter(self.train_split) + Counter(self.valid_split) + Counter(self.test_split)
        self.skip_split_norm = thelper.utils.str2bool(thelper.utils.get_key_def("skip_split_norm", config, False))
        self.skip_class_balancing = thelper.utils.str2bool(thelper.utils.get_key_def("skip_class_balancing", config, False))
        for name, usage in self.total_usage.items():
            if usage != 1:
                normalize_ratios = None
                assert usage >= 0
                if 0 < usage < 1 and not math.isclose(usage, 1) and not self.skip_split_norm:
                    query_msg = f"dataset split for {name} has a ratio sum less than 1; do you want to normalize the split?\n\t("
                    query_msg += f"train={self.train_split[name] if name in self.train_split else None}, "
                    query_msg += f"valid={self.valid_split[name] if name in self.valid_split else None}, "
                    query_msg += f"test={self.test_split[name] if name in self.test_split else None})"
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
    def _get_collate_fn(val):
        if val is torch.utils.data.dataloader.default_collate or val is default_collate or callable(val):
            return val
        if isinstance(val, dict):
            collate_fn_type = thelper.utils.get_key("type", val)
            collate_fn_params = thelper.utils.get_key_def("params", val, None)
            return thelper.utils.import_function(collate_fn_type, collate_fn_params)
        elif isinstance(val, str):
            return thelper.utils.import_function(val)
        else:
            raise AssertionError("unexpected collate val type")

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
        logger.info("setting '%s' to %d" % (prefixes[0], seed))
        return seed

    @staticmethod
    def _get_ratios_split(prefix, config):
        key = prefix + "_split"
        if key not in config or not config[key]:
            return {}
        split = config[key]
        assert not any(ratio < 0 for ratio in split.values())
        return split

    @staticmethod
    def _get_augments(targets, name, config):
        logger.debug("loading %s augments..." % name)
        for target in targets:
            if target in config and config[target]:
                augments, augments_append = thelper.transforms.load_augments(config[target])
                if augments:
                    logger.debug("will %s %s augments: %s" % ("append" if augments_append else "prefix", name, str(augments)))
                return augments, augments_append
        return None, False

    def _get_raw_split(self, indices):
        for name in self.total_usage:
            assert name in indices, f"dataset '{name}' does not exist"
        _indices, train_idxs, valid_idxs, test_idxs = {}, {}, {}, {}
        for name, indices in indices.items():
            _indices[name] = copy.deepcopy(indices)
            train_idxs[name] = []
            valid_idxs[name] = []
            test_idxs[name] = []
        indices = _indices
        shuffle = any([self.train_shuffle, self.valid_shuffle, self.test_shuffle])
        if shuffle:
            np.random.seed(self.seeds["test"])  # test idxs will be picked first, then valid+train
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
            if loader_idx == 0 and shuffle:
                np.random.seed(self.seeds["valid"])  # all test idxs are now picked, reshuffle for train/valid
                for name in self.total_usage.keys():
                    trainvalid_idxs = indices[name][offsets[name]:]
                    np.random.shuffle(trainvalid_idxs)
                    indices[name][offsets[name]:] = trainvalid_idxs
        if shuffle:
            np.random.seed(self.seeds["numpy"])  # back to default random state for future use
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
            assert isinstance(dataset, thelper.data.Dataset) or isinstance(dataset, thelper.data.ExternalDataset), \
                f"unexpected dataset type for '{dataset_name}'"
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
            logger.debug("will split evenly over %d classes..." % len(task.class_names))
            unset_class_key = "<unset>"
            global_class_names = task.class_names + [unset_class_key]  # extra name added for unlabeled samples (if needed!)
            sample_maps = {}
            for dataset_name, dataset in datasets.items():
                if isinstance(dataset, thelper.data.ExternalDataset):
                    if hasattr(dataset.samples, "samples") and isinstance(dataset.samples.samples, list):
                        sample_maps[dataset_name] = task.get_class_sample_map(dataset.samples.samples, unset_class_key)
                    else:
                        logger.warning(("must fully parse the external dataset '%s' for intra-class shuffling;" % dataset_name) +
                                       " this might take a while!\n(consider making a dataset interface that can return labels" +
                                       " only, it would greatly speed up the analysis of class distributions)")
                        label_key = task.gt_key
                        # to allow glitch-less tqdm printing after latest logger output
                        sys.stdout.flush()
                        sys.stderr.flush()
                        time.sleep(0.01)
                        samples = []
                        for sample in tqdm.tqdm(dataset):
                            assert label_key in sample, f"could not find label key ('{label_key}') in sample dict"
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
        for idxs_map, (augs, augs_append), shuffle, scale, sampler_apply, batch_size, collate_fn \
                in zip([train_idxs, valid_idxs, test_idxs],
                       [(self.train_augments, self.train_augments_append),
                        (self.valid_augments, self.valid_augments_append),
                        (self.test_augments, self.test_augments_append)],
                       [self.train_shuffle, self.valid_shuffle, self.test_shuffle],
                       [self.train_scale, self.valid_scale, self.test_scale],
                       [self.train_sampler, self.valid_sampler, self.test_sampler],
                       [self.train_batch_size, self.valid_batch_size, self.test_batch_size],
                       [self.train_collate_fn, self.valid_collate_fn, self.test_collate_fn]):
            loader_sample_idx_offset = 0
            loader_sample_classes = []
            loader_sample_idxs = []
            loader_datasets = []
            for dataset_name, sample_idxs in idxs_map.items():
                if not sample_idxs:
                    continue
                # todo: investigate need to copy at all?
                if not datasets[dataset_name].deepcopy:
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
                    sampler_params = {**self.sampler_params}
                    if self.sampler_pass_labels:
                        sampler_params[self.sampler_pass_labels_param_name] = loader_sample_classes
                    sampler_sig = inspect.signature(self.sampler_type)
                    if "seeds" in sampler_sig.parameters:
                        sampler_params["seeds"] = self.seeds
                    if "scale" in sampler_sig.parameters:
                        sampler_params["scale"] = scale
                    else:
                        assert scale == 1.0, f"could not apply scale factor to sample with type '{str(self.sampler_type)}'"
                    sampler = self.sampler_type(loader_sample_idxs, **self.sampler_params)
                else:
                    if shuffle:
                        sampler = thelper.data.SubsetRandomSampler(loader_sample_idxs, seeds=self.seeds, scale=scale)
                    else:
                        assert scale == 1.0, "sequential sampler currently does not handle scale changes (turn on shuffling)"
                        sampler = thelper.data.SubsetSequentialSampler(loader_sample_idxs)
                assert hasattr(sampler, "__len__")
                assert batch_size > 0
                loaders.append(DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,
                                          num_workers=self.workers, collate_fn=collate_fn,
                                          pin_memory=self.pin_memory, drop_last=self.drop_last,
                                          seeds=self.seeds))
            else:
                loaders.append(None)
        train_loader, valid_loader, test_loader = loaders
        logger.info("initialized loaders with batch counts:\n  train=%d\n  valid=%d\n  test=%d" %
                    (len(train_loader) if train_loader else 0,
                     len(valid_loader) if valid_loader else 0,
                     len(test_loader) if test_loader else 0))
        return train_loader, valid_loader, test_loader

    def get_base_transforms(self):
        """Returns the (global) sample transformation operations parsed in the data configuration."""
        return self.base_transforms
