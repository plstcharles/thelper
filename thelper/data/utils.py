"""Dataset utility functions and tools.

This module contains utility functions and tools used to instantiate data loaders and parsers.
"""

import inspect
import json
import logging
import os
import sys

import h5py
import numpy as np
import tqdm

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

    - ``<train_/valid_/test_>batch_size`` (mandatory): specifies the (mini)batch size to use in data
      loaders. If you get an 'out of memory' error at runtime, try reducing it.
    - ``<train_/valid_/test_>collate_fn`` (optional): specifies the collate function to use in data
      loaders. The default one is typically fine, but some datasets might require a custom function.
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
                "params": {
                    # ...
                }
            },
            "dataset_B": {
                # type of dataset interface to instantiate
                "type": "...",
                "params": {
                    # ...
                },
                # if it does not derive from 'thelper.data.parsers.Dataset', a task is needed:
                "task": {
                    # this type must derive from 'thelper.tasks.Task'
                    "type": "...",
                    "params": {
                        # ...
                    }
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
        | :func:`thelper.data.utils.create_parsers`
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
    logger.debug("loading data usage config")
    # todo: 'data_config' field is deprecated, might be removed later
    if "data_config" in config:
        logger.warning("using 'data_config' field in configuration dictionary is deprecated; switch it to 'loaders'")
    loaders_config = thelper.utils.get_key(["data_config", "loaders"], config)
    # noinspection PyProtectedMember
    from thelper.data.loaders import LoaderFactory as LoaderFactory
    loader_factory = LoaderFactory(loaders_config)
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
    all the information related to sample dictionary keys and model i/o.

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
            with open(config) as fd:
                config = json.load(fd)
        else:
            raise AssertionError("'datasets' string should point to valid json file")
    logger.debug("loading datasets templates")
    if not isinstance(config, dict):
        raise AssertionError("invalid datasets config type (must be dictionary)")
    datasets = {}
    tasks = []
    for dataset_name, dataset_config in config.items():
        if isinstance(dataset_config, thelper.data.Dataset):
            dataset = dataset_config
            task = dataset.task
        else:
            logger.debug("loading dataset '%s' configuration..." % dataset_name)
            if "type" not in dataset_config:
                raise AssertionError("missing field 'type' for instantiation of dataset '%s'" % dataset_name)
            dataset_type = thelper.utils.import_class(dataset_config["type"])
            dataset_params = thelper.utils.get_key_def(["params", "parameters"], dataset_config, {})
            transforms = None
            if "transforms" in dataset_config and dataset_config["transforms"]:
                logger.debug("loading custom transforms for dataset '%s'..." % dataset_name)
                transforms = thelper.transforms.load_transforms(dataset_config["transforms"])
                if base_transforms is not None:
                    transforms = thelper.transforms.Compose([transforms, base_transforms])
            elif base_transforms is not None:
                transforms = base_transforms
            if issubclass(dataset_type, thelper.data.Dataset):
                # assume that the dataset is derived from thelper.data.parsers.Dataset (it is fully sampling-ready)
                dataset_sig = inspect.signature(dataset_type)
                if "config" in dataset_sig.parameters:  # pragma: no cover
                    # @@@@ for backward compatibility only, will be removed in v0.3
                    dataset = dataset_type(transforms=transforms, config=dataset_params)
                else:
                    dataset = dataset_type(transforms=transforms, **dataset_params)
                if "task" in dataset_config:
                    logger.warning("'task' field detected in dataset '%s' config; dataset's default task will be ignored" % dataset_name)
                    task = thelper.tasks.create_task(dataset_config["task"])
                else:
                    task = dataset.task
            else:
                if "task" not in dataset_config or not dataset_config["task"]:
                    raise AssertionError("external dataset '%s' must define task interface in its configuration dict" % dataset_name)
                task = thelper.tasks.create_task(dataset_config["task"])
                # assume that __getitem__ and __len__ are implemented, but we need to make it sampling-ready
                dataset = thelper.data.ExternalDataset(dataset_type, task, transforms=transforms, **dataset_params)
        if task is None:
            raise AssertionError("parsed task interface should not be None anymore (old code doing something strange?)")
        tasks.append(task)
        datasets[dataset_name] = dataset
    return datasets, thelper.tasks.create_global_task(tasks)


def create_hdf5(archive_path, task, train_loader, valid_loader, test_loader, compression=None, config_backup=None):
    """Saves the samples loaded from train/valid/test data loaders into an HDF5 archive.

    The loaded minibatches are decomposed into individual samples. The keys provided via the task interface are used
    to fetch elements (input, groundtruth, ...) from the samples, and save them in the archive. The archive will
    contain three groups (`train`, `valid`, and `test`), and each group will contain a dataset for each element
    originally found in the samples.

    Note that the compression operates at the sample level, not at the dataset level. This means that elements of
    each sample will be compressed individually, not as an array. Therefore, if you are trying to compress very
    correlated samples (e.g. frames in a video sequence), this approach will be pretty bad.

    Args:
        archive_path: path pointing where the HDF5 archive should be created.
        task: task object that defines the input, groundtruth, and meta keys tied to elements that should be
            parsed from loaded samples and saved in the HDF5 archive.
        train_loader: training data loader (can be `None`).
        valid_loader: validation data loader (can be `None`).
        test_loader: testing data loader (can be `None`).
        compression: the compression configuration dictionary that will be parsed to determine how sample
            elements should be compressed. If a mapping is missing, that element will not be compressed.
        config_backup: optional session configuration file that should be saved in the HDF5 archive.

    Example compression configuration::

        # the config is given as a dictionary
        {
            # each field is a key that corresponds to an element in each sample
            "key1": {
                # the 'type' identifies the compression approach to use
                # (see thelper.utils.encode_data for more information)
                "type": "jpg",
                # extra parameters might be needed to encode the data
                # (see thelper.utils.encode_data for more information)
                "encode_params": {}
                # these parameters are packed and kept for decoding
                # (see thelper.utils.decode_data for more information)
                "decode_params": {"flags": "cv.IMREAD_COLOR"}
            },
            "key2": {
                # this explicitly means that no encoding should be performed
                "type": "none"
            },
            ...
            # if a key is missing, its elements will not be compressed
        }

    .. seealso::
        | :func:`thelper.cli.split_data`
        | :class:`thelper.data.parsers.HDF5Dataset`
        | :func:`thelper.utils.encode_data`
        | :func:`thelper.utils.decode_data`
    """
    if compression is None:
        compression = {}
    if config_backup is None:
        config_backup = {}
    with h5py.File(archive_path, "w") as fd:
        fd.attrs["source"] = thelper.utils.get_log_stamp()
        fd.attrs["git_sha1"] = thelper.utils.get_git_stamp()
        fd.attrs["version"] = thelper.__version__
        fd.attrs["task"] = str(task)
        fd.attrs["config"] = str(config_backup)
        fd.attrs["compression"] = str(compression)
        dtype = h5py.special_dtype(vlen=np.uint8)
        target_keys = task.keys

        def create_dataset(name, max_len, array_template, compr_args):
            if array_template.ndim > 1:
                dset = fd.create_dataset(name, shape=(max_len,), maxshape=(max_len,), dtype=dtype)
                dset.attrs["orig_dtype"] = str(array_template.dtype)
                dset.attrs["orig_shape"] = array_template.shape[1:]  # removes batch dim
            else:
                assert thelper.utils.is_scalar(array_template[0])
                if "type" in compr_args and compr_args["type"] != "none":
                    raise AssertionError("cannot compress scalar elements")
                if np.issubdtype(array_template.dtype, np.number):
                    dset = fd.create_dataset(name, shape=(max_len,), maxshape=(max_len,), dtype=array_template.dtype)
                else:
                    dset = fd.create_dataset(name, shape=(max_len,), maxshape=(max_len,), dtype=dtype)
                    dset.attrs["orig_dtype"] = str(array_template.dtype)
                    dset.attrs["orig_shape"] = ()
            return dset

        def fill_dataset(dset, dset_idx, array_idx, array, compr, **compr_kwargs):
            if array.ndim > 1:
                dset[dset_idx] = np.frombuffer(thelper.utils.encode_data(array[array_idx], compr, **compr_kwargs), dtype=np.uint8)
            elif not np.issubdtype(array.dtype, np.number):
                sample = array[array_idx].tobytes() if np.issubdtype(array[array_idx].dtype, np.dtype(str).type) else array[array_idx]
                dset[dset_idx] = np.frombuffer(sample, dtype=np.uint8)
            else:
                dset[dset_idx] = array[array_idx]

        def get_compr_args(key, config):
            config = thelper.utils.get_key_def(key, config, default={})
            compr_type = thelper.utils.get_key_def("type", config, default="none")
            encode_params = thelper.utils.get_key_def("encode_params", config, default={})
            return compr_type, encode_params

        for loader, group in [(train_loader, "train"), (valid_loader, "valid"), (test_loader, "test")]:
            if loader is None:
                continue
            max_dataset_len = len(loader) * loader.batch_size
            datasets = {key: None for key in target_keys}
            datasets_len = {key: 0 for key in target_keys}
            datasets_compr = {key: get_compr_args(key, compression) for key in target_keys}
            for batch in tqdm.tqdm(loader, desc=f"packing {group} loader"):
                for key in target_keys:
                    tensor = thelper.utils.to_numpy(batch[key])
                    if datasets[key] is None:
                        datasets[key] = create_dataset(group + "/" + key, max_dataset_len, tensor, datasets_compr[key])
                    for idx in range(tensor.shape[0]):
                        fill_dataset(datasets[key], datasets_len[key], idx, tensor, datasets_compr[key][0], **datasets_compr[key][1])
                        datasets_len[key] += 1
            assert len(set(datasets_len.values())) == 1
            fd[group].attrs["count"] = datasets_len[task.input_key]
            for key in target_keys:
                datasets[key].resize(size=(datasets_len[key],))


def get_class_weights(label_map, stype, invmax, maxw=float('inf'), minw=0.0, norm=True):
    """Returns a map of adjusted class weights based on a given rebalancing strategy.

    Args:
        label_map: map of index lists or sample counts tied to class labels.
        stype: weighting strategy ('uniform', 'linear', or 'rootX'); see :class:`thelper.data.samplers.WeightedSubsetRandomSampler`
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
    if not isinstance(label_map, dict) or any([not isinstance(val, (list, int)) for val in label_map.values()]):
        raise AssertionError("unexpected label map type")
    if stype == "uniform":
        label_weights = {label: 1.0 / len(label_map) for label in label_map}
    elif stype == "linear" or "root" in stype:
        if stype == "root" or stype == "linear":
            rpow = 1.0
        else:
            rpow = 1.0 / float(stype.split("root", 1)[1])
        label_sizes = {label: len(v) if isinstance(v, list) else v for label, v in label_map.items()}
        label_weights = {label: (lsize / sum(label_sizes.values())) ** rpow for label, lsize in label_sizes.items()}
    else:
        raise AssertionError("unknown label weighting strategy")
    if invmax:
        label_weights = {label: max(label_weights.values()) / max(weight, 1e-6) for label, weight in label_weights.items()}
    label_weights = {label: min(max(weight, minw), maxw) for label, weight in label_weights.items()}
    if norm:
        tot_weight = sum([w for w in label_weights.values()])
        label_weights = {label: weight / tot_weight for label, weight in label_weights.items()}
    return label_weights
