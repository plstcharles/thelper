"""Dataset parsers module.

This module contains dataset parser interfaces and base classes that define basic i/o
operations so that the framework can automatically interact with training data.
"""

import inspect
import logging
import os
from abc import abstractmethod

import cv2 as cv
import numpy as np
import PIL
import PIL.Image
import torch
import torch.utils.data

import thelper.tasks
import thelper.utils

logger = logging.getLogger(__name__)


class Dataset(torch.utils.data.Dataset):
    """Abstract dataset parsing interface that holds a task and a list of sample dictionaries.

    This interface helps fix a failure of PyTorch's dataset interface (``torch.utils.data.Dataset``):
    the lack of identity associated with the components of a sample. In short, a data sample loaded by a
    dataset typically contains the input data that should be forwarded to a model as well as the expected
    prediction of the model (i.e. the 'groundtruth') that will be used to compute the loss. These two
    elements are typically paired in a tuple that can then be provided to the data loader for batching.
    Problems however arise when the model has multiple inputs or outputs, when the sample needs to carry
    supplemental metadata to simplify debugging, or when transformation operations need to be applied
    only to specific elements of the sample. Here, we fix this issue by specifying that all samples must
    be provided to data loaders as dictionaries. The keys of these dictionaries explicitly define which
    value(s) should be transformed, which should be forwarded to the model, which are the expected model
    predictions, and which are only used for debugging. The keys are defined via the task object that is
    generated by the dataset or specified via the configuration file (see :class:`thelper.tasks.utils.Task`
    for more information).

    To properly use this interface, a derived class must implement :func:`thelper.data.parsers.Dataset.__getitem__`,
    as well as provide proper ``task`` and ``samples`` attributes. The ``task`` attribute must derive from
    :class:`thelper.tasks.utils.Task`, and ``samples`` must be an array-like object holding already-parsed
    information about the dataset samples (in dictionary format). The length of the ``samples`` array will
    automatically be returned as the size of the dataset in this interface. For class-based datasets, it is
    recommended to parse the classes in the dataset constructor so that external code can directly peek into
    the ``samples`` attribute to see their distribution without having to call ``__getitem__``. This is done
    for example in :func:`thelper.data.loaders.LoaderFactory.get_split` to automatically rebalance classes
    without having to actually load the samples one by one, which speeds up the process dramatically.

    Attributes:
        transforms: function or object that should be applied to all loaded samples in order to
            return the data in the requested transformed/augmented state.
        deepcopy: specifies whether this dataset interface should be deep-copied inside
            :class:`thelper.data.loaders.LoaderFactory` so that it may be shared between
            different threads. This is false by default, as we assume datasets do not contain a state
            or buffer that might cause problems in multi-threaded data loaders.
        samples: list of dictionaries containing the data that is ready to be forwarded to the
            data loader. Note that relatively costly operations (such as reading images from a disk
            or pre-transforming them) should be delayed until the :func:`thelper.data.parsers.Dataset.__getitem__`
            function is called, as they will most likely then be accomplished in a separate thread.
            Once loaded, these samples should never be modified by another part of the framework. For
            example, transformation and augmentation operations will always be applied to copies
            of these samples.
        task: object used to define what keys are used to index the loaded data into sample dictionaries.

    .. seealso::
        | :class:`thelper.data.parsers.ExternalDataset`
    """

    def __init__(self, transforms=None, deepcopy=False):
        """Dataset parser constructor.

        In order for derived datasets to be instantiated automatically by the framework from a configuration
        file, they must minimally accept a 'transforms' argument like the shown one here.

        Args:
            transforms: function or object that should be applied to all loaded samples in order to
                return the data in the requested transformed/augmented state.
            deepcopy: specifies whether this dataset interface should be deep-copied inside
                :class:`thelper.data.loaders.LoaderFactory` so that it may be shared between
                different threads. This is false by default, as we assume datasets do not contain a state
                or buffer that might cause problems in multi-threaded data loaders.
        """
        super(Dataset, self).__init__()
        self.logger = thelper.utils.get_class_logger()
        self.transforms = transforms
        self.deepcopy = deepcopy  # will determine if we deepcopy in each loader
        self.samples = None  # must be set by the derived class as a array-like object of dictionaries
        self.task = None  # must be set by the derived class as a valid task object

    def _get_derived_name(self):
        """Returns a pretty-print version of the derived class's name."""
        return self.__class__.__module__ + "." + self.__class__.__qualname__

    def __len__(self):
        """Returns the total number of samples available from this dataset interface."""
        return len(self.samples)

    def __iter__(self):
        """Returns an iterator over the dataset's samples."""
        for idx in range(self.__len__()):
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        raise NotImplementedError

    @property
    def transforms(self):
        """Returns the transformation operations to apply to this dataset's loaded samples."""
        return self._transforms

    @transforms.setter
    def transforms(self, transforms):
        """Sets the transformation operations to apply to this dataset's loaded samples."""
        assert transforms is None or hasattr(transforms, "__call__") or \
            (isinstance(transforms, list) and all([hasattr(t, "__call__") for t in transforms])), \
            "transformations should be callable (or list of callables)"
        self._transforms = transforms

    @property
    def deepcopy(self):
        """specifies whether this dataset interface should be deep-copied inside
        :class:`thelper.data.loaders.LoaderFactory` so that it may be shared between
        different threads. This is false by default, as we assume datasets do not contain a state
        or buffer that might cause problems in multi-threaded data loaders."""
        return self._deepcopy

    @deepcopy.setter
    def deepcopy(self, deepcopy):
        assert deepcopy is None or isinstance(deepcopy, bool), "deepcopy flag should be boolean"
        self._deepcopy = deepcopy

    @property
    def task(self):
        """Returns the task object associated with this dataset interface."""
        return self._task

    @task.setter
    def task(self, task):
        """Sets the task object associated with this dataset interface."""
        assert task is None or isinstance(task, thelper.tasks.Task), "invalid task"
        self._task = task

    @property
    def samples(self):
        """Returns the list of internal samples held by this dataset interface."""
        return self._samples

    @samples.setter
    def samples(self, samples):
        assert samples is None or isinstance(samples, (list, Dataset)) or hasattr(samples, "__getitem__"), \
            "invalid samples (should be list of dicts, dataset, or it should have '__getitem__' attrib)"
        self._samples = samples

    def _getitems(self, idxs):
        """Returns a list of dictionaries corresponding to the sliced sample indices."""
        if not isinstance(idxs, slice):
            raise AssertionError("unexpected input (should be slice)")
        return [self[idx] for idx in range(*idxs.indices(len(self)))]

    def __repr__(self):
        """Returns a print-friendly representation of this dataset."""
        return self._get_derived_name() + f"(transforms={repr(self.transforms)}, deepcopy={repr(self.deepcopy)})"


class HDF5Dataset(Dataset):
    """HDF5 dataset specialization interface.

    This specialization is compatible with the HDF5 packages made by the CLI's "split" operation. The
    archives it loads contains pre-split datasets that can be reloaded without having to resplit their
    data. The archive also contains useful metadata, and a task interface.

    Attributes:
        archive: file descriptor for the opened hdf5 dataset.
        subset: hdf5 group section representing the targeted set.
        target_args: list decompression args required for each sample key.
        source: source logstamp of the hdf5 dataset.
        git_sha1: framework git tag of the hdf5 dataset.
        version: version of the framework that saved the hdf5 dataset.
        orig_config: configuration used to originally generate the hdf5 dataset.

    .. seealso::
        | :func:`thelper.cli.split_data`
        | :func:`thelper.data.utils.create_hdf5`
    """

    def __init__(self, root, subset="train", transforms=None):
        """HDF5 dataset parser constructor.

        This constructor receives the path to the HDF5 archive as well as a subset indicating which
        section of the archive to load. By default, it loads the training set.
        """
        super(HDF5Dataset, self).__init__(transforms=transforms, deepcopy=False)
        assert subset in ["train", "valid", "test"], f"unrecognized subset '{subset}'"
        import h5py
        self.archive = h5py.File(root, "r")
        self.source = self.archive.attrs["source"]
        self.git_sha1 = self.archive.attrs["git_sha1"]
        self.version = self.archive.attrs["version"]
        self.task = thelper.tasks.create_task(self.archive.attrs["task"])
        self.orig_config = eval(self.archive.attrs["config"])
        compr_config = eval(self.archive.attrs["compression"])
        if subset not in self.archive:
            raise AssertionError(f"subset '{subset}' not found in hdf5 archive")
        self.subset = self.archive[subset]
        sample_count = self.subset.attrs["count"]
        self.samples = [{}] * sample_count
        self.target_args = {}
        for key in self.task.keys:
            dset = self.subset[key]
            assert dset.len() == len(self.samples)
            dtype = dset.attrs["orig_dtype"] if "orig_dtype" in dset.attrs else None
            shape = dset.attrs["orig_shape"] if "orig_shape" in dset.attrs else None
            compr_config = thelper.utils.get_key_def(key, compr_config, default={})
            compr_type = thelper.utils.get_key_def("type", compr_config, default="none")
            compr_kwargs = thelper.utils.get_key_def(["decode_params", "decode_kwargs"], compr_config, default={})
            self.target_args[key] = {"dset": dset, "dtype": dtype, "shape": shape, "compr_type": compr_type, "compr_kwargs": compr_kwargs}

    def _unpack(self, dset, idx, dtype=None, shape=None, compr_type="none", **compr_kwargs):
        if dtype is not None:
            array = np.frombuffer(thelper.utils.decode_data(dset[idx], compr_type, **compr_kwargs), dtype=dtype)
        else:
            array = dset[idx]
        if shape is not None:
            if np.issubdtype(dtype, np.dtype(str).type) and len(shape) == 0:
                array = "".join(array)  # reassemble string if needed
            else:
                array = array.reshape(shape)
        return array

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        if idx >= len(self.samples):
            raise AssertionError("sample index is out-of-range")
        sample = {key: self._unpack(args["dset"], idx, args["dtype"], args["shape"],
                                    args["compr_type"], **args["compr_kwargs"]) for key, args in self.target_args.items()}
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def close(self):
        """Closes the internal HDF5 file."""
        # note: if we dont do it explicitly, it will be done by the garbage collector on destruction, but it might take time...
        self.archive.close()


class ClassificationDataset(Dataset):
    """Classification dataset specialization interface.

    This specialization receives some extra parameters in its constructor and automatically defines
    a :class:`thelper.tasks.classif.Classification` task based on those. The derived class must still
    implement :func:`thelper.data.parsers.ClassificationDataset.__getitem__`, and it must still store its
    samples as dictionaries in ``self.samples`` to behave properly.

    .. seealso::
        | :class:`thelper.data.parsers.Dataset`
    """

    def __init__(self, class_names, input_key, label_key, meta_keys=None, transforms=None, deepcopy=False):
        """Classification dataset parser constructor.

        This constructor receives all extra arguments necessary to build a classification task object.

        Args:
            class_names: list of all class names (or labels) that will be associated with the samples.
            input_key: key used to index the input data in the loaded samples.
            label_key: key used to index the label (or class name) in the loaded samples.
            meta_keys: list of extra keys that will be available in the loaded samples.
            transforms: function or object that should be applied to all loaded samples in order to
                return the data in the requested transformed/augmented state.
            deepcopy: specifies whether this dataset interface should be deep-copied inside
                :class:`thelper.data.loaders.LoaderFactory` so that it may be shared between
                different threads. This is false by default, as we assume datasets do not contain a state
                or buffer that might cause problems in multi-threaded data loaders.
        """
        super(ClassificationDataset, self).__init__(transforms=transforms, deepcopy=deepcopy)
        self.task = thelper.tasks.Classification(class_names, input_key, label_key, meta_keys=meta_keys)

    @abstractmethod
    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        raise NotImplementedError


class SegmentationDataset(Dataset):
    """Segmentation dataset specialization interface.

    This specialization receives some extra parameters in its constructor and automatically defines
    its task (:class:`thelper.tasks.segm.Segmentation`) based on those. The derived class must still
    implement :func:`thelper.data.parsers.SegmentationDataset.__getitem__`, and it must still store its
    samples as dictionaries in ``self.samples`` to behave properly.

    .. seealso::
        | :class:`thelper.data.parsers.Dataset`
    """

    def __init__(self, class_names, input_key, label_map_key, meta_keys=None, dontcare=None, transforms=None, deepcopy=False):
        """Segmentation dataset parser constructor.

        This constructor receives all extra arguments necessary to build a segmentation task object.

        Args:
            class_names: list of all class names (or labels) that must be predicted in the image.
            input_key: key used to index the input image in the loaded samples.
            label_map_key: key used to index the label map in the loaded samples.
            meta_keys: list of extra keys that will be available in the loaded samples.
            transforms: function or object that should be applied to all loaded samples in order to
                return the data in the requested transformed/augmented state.
            deepcopy: specifies whether this dataset interface should be deep-copied inside
                :class:`thelper.data.loaders.LoaderFactory` so that it may be shared between
                different threads. This is false by default, as we assume datasets do not contain a state
                or buffer that might cause problems in multi-threaded data loaders.
        """
        super(SegmentationDataset, self).__init__(transforms=transforms, deepcopy=deepcopy)
        self.task = thelper.tasks.Segmentation(class_names, input_key, label_map_key, meta_keys=meta_keys, dontcare=dontcare)

    @abstractmethod
    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        raise NotImplementedError


class ImageDataset(Dataset):
    """Image dataset specialization interface.

    This specialization is used to parse simple image folders, and it does not fulfill the requirements of any
    specialized task constructors due to the lack of groundtruth data support. Therefore, it returns a basic task
    object (:class:`thelper.tasks.utils.Task`) with no set value for the groundtruth key, and it cannot be used to
    directly train a model. It can however be useful when simply visualizing, annotating, or testing raw data
    from a simple directory structure.

    .. seealso::
        | :class:`thelper.data.parsers.Dataset`
    """

    def __init__(self, root, transforms=None, image_key="image", path_key="path", idx_key="idx"):
        """Image dataset parser constructor.

        This constructor exposes some of the configurable keys used to index sample dictionaries.
        """
        super(ImageDataset, self).__init__(transforms=transforms)
        self.root = root
        if self.root is None or not os.path.isdir(self.root):
            raise AssertionError("invalid input data root '%s'" % self.root)
        self.image_key = image_key
        self.path_key = path_key
        self.idx_key = idx_key
        self.samples = []
        for folder, subfolder, files in os.walk(self.root):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in [".jpg", ".jpeg", ".bmp", ".png", ".ppm", ".pgm", ".tif"]:
                    self.samples.append({self.path_key: os.path.join(folder, file)})
        self.task = thelper.tasks.Task(self.image_key, None, [self.path_key, self.idx_key])

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        if idx >= len(self.samples):
            raise AssertionError("sample index is out-of-range")
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        image_path = sample[self.path_key]
        image = cv.imread(image_path)
        if image is None:
            raise AssertionError("invalid image at '%s'" % image_path)
        sample = {
            self.image_key: image,
            self.idx_key: idx,
            **sample
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class ImageFolderDataset(ClassificationDataset):
    """Image folder dataset specialization interface for classification tasks.

    This specialization is used to parse simple image subfolders, and it essentially replaces the very
    basic ``torchvision.datasets.ImageFolder`` interface with similar functionalities. It it used to provide
    a proper task interface as well as path metadata in each loaded packet for metrics/logging output.

    .. seealso::
        | :class:`thelper.data.parsers.ImageDataset`
        | :class:`thelper.data.parsers.ClassificationDataset`
    """

    def __init__(self, root, transforms=None, image_key="image", label_key="label", path_key="path", idx_key="idx"):
        """Image folder dataset parser constructor."""
        self.root = root
        if self.root is None or not os.path.isdir(self.root):
            raise AssertionError("invalid input data root '%s'" % self.root)
        class_map = {}
        for child in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, child)):
                class_map[child] = []
        if not class_map:
            raise AssertionError("could not find any image folders at '%s'" % self.root)
        image_exts = [".jpg", ".jpeg", ".bmp", ".png", ".ppm", ".pgm", ".tif"]
        self.image_key = image_key
        self.path_key = path_key
        self.idx_key = idx_key
        self.label_key = label_key
        samples = []
        for class_name in class_map:
            class_folder = os.path.join(self.root, class_name)
            for folder, subfolder, files in os.walk(class_folder):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_exts:
                        class_map[class_name].append(len(samples))
                        samples.append({
                            self.path_key: os.path.join(folder, file),
                            self.label_key: class_name
                        })
        old_unsorted_class_names = list(class_map.keys())
        class_map = {k: class_map[k] for k in sorted(class_map.keys()) if len(class_map[k]) > 0}
        if old_unsorted_class_names != list(class_map.keys()):
            # new as of v0.4.4; this may only be an issue for old models trained on windows and ported to linux
            # (this is caused by the way os.walk returns folders in an arbitrary order on some platforms)
            logger.warning("class name ordering changed due to folder name sorting; this may impact the "
                           "behavior of previously-trained models as task class indices may be swapped!")
        if not class_map:
            raise AssertionError("could not locate any subdir in '%s' with images to load" % self.root)
        meta_keys = [self.path_key, self.idx_key]
        super(ImageFolderDataset, self).__init__(class_names=list(class_map.keys()), input_key=self.image_key,
                                                 label_key=self.label_key, meta_keys=meta_keys, transforms=transforms)
        self.samples = samples

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        if idx >= len(self.samples):
            raise AssertionError("sample index is out-of-range")
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        image_path = sample[self.path_key]
        image = cv.imread(image_path)
        if image is None:
            raise AssertionError("invalid image at '%s'" % image_path)
        sample = {
            self.image_key: image,
            self.idx_key: idx,
            **sample
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class SuperResFolderDataset(Dataset):
    """Image folder dataset specialization interface for super-resolution tasks.

    This specialization is used to parse simple image subfolders, and it essentially replaces the very
    basic ``torchvision.datasets.ImageFolder`` interface with similar functionalities. It it used to provide
    a proper task interface as well as path/class metadata in each loaded packet for metrics/logging output.
    """

    def __init__(self, root, downscale_factor=2.0, rescale_lowres=True, center_crop=None, transforms=None,
                 lowres_image_key="lowres_image", highres_image_key="highres_image", path_key="path", idx_key="idx", label_key="label"):
        """Image folder dataset parser constructor."""
        if isinstance(downscale_factor, int):
            downscale_factor = float(downscale_factor)
        if not isinstance(downscale_factor, float) or downscale_factor <= 1.0:
            raise AssertionError("invalid downscale factor (should be greater than one)")
        self.downscale_factor = downscale_factor
        self.rescale_lowres = rescale_lowres
        if center_crop is not None:
            if isinstance(center_crop, int):
                center_crop = (center_crop, center_crop)
            if not isinstance(center_crop, (list, tuple)):
                raise AssertionError("invalid center crop size type")
        self.center_crop = center_crop
        self.root = root
        if self.root is None or not os.path.isdir(self.root):
            raise AssertionError("invalid input data root '%s'" % self.root)
        class_map = {}
        for child in os.listdir(self.root):
            if os.path.isdir(os.path.join(self.root, child)):
                class_map[child] = []
        if not class_map:
            raise AssertionError("could not find any image folders at '%s'" % self.root)
        image_exts = [".jpg", ".jpeg", ".bmp", ".png", ".ppm", ".pgm", ".tif"]
        self.lowres_image_key = lowres_image_key
        self.highres_image_key = highres_image_key
        self.path_key = path_key
        self.idx_key = idx_key
        self.label_key = label_key  # to provide folder names
        samples = []
        for class_name in class_map:
            class_folder = os.path.join(self.root, class_name)
            for folder, subfolder, files in os.walk(class_folder):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in image_exts:
                        class_map[class_name].append(len(samples))
                        samples.append({
                            self.path_key: os.path.join(folder, file),
                            self.label_key: class_name
                        })
        class_map = {k: v for k, v in class_map.items() if len(v) > 0}
        if not class_map:
            raise AssertionError("could not locate any subdir in '%s' with images to load" % self.root)
        meta_keys = [self.path_key, self.idx_key, self.label_key]
        super(SuperResFolderDataset, self).__init__(transforms=transforms)
        self.task = thelper.tasks.SuperResolution(input_key=self.lowres_image_key, target_key=self.highres_image_key, meta_keys=meta_keys)
        self.samples = samples

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        if idx >= len(self.samples):
            raise AssertionError("sample index is out-of-range")
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        image_path = sample[self.path_key]
        image = cv.imread(image_path)
        if image is None:
            raise AssertionError("invalid image at '%s'" % image_path)
        if self.center_crop is not None:
            tl = (image.shape[1] // 2 - self.center_crop[0] // 2,
                  image.shape[0] // 2 - self.center_crop[1] // 2)
            br = (tl[0] + self.center_crop[0], tl[1] + self.center_crop[1])
            image = thelper.draw.safe_crop(image, tl, br)
        scale = 1.0 / self.downscale_factor
        image_lowres = cv.resize(image, dsize=(0, 0), fx=scale, fy=scale)
        if self.rescale_lowres:
            image_lowres = cv.resize(image_lowres, dsize=(image.shape[1], image.shape[0]))
        sample = {
            self.lowres_image_key: image_lowres,
            self.highres_image_key: image,
            self.idx_key: idx,
            **sample
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class ExternalDataset(Dataset):
    """External dataset interface.

    This interface allows external classes to be instantiated automatically in the framework through
    a configuration file, as long as they themselves provide implementations for  ``__getitem__`` and
    ``__len__``. This includes all derived classes of ``torch.utils.data.Dataset`` such as
    ``torchvision.datasets.ImageFolder``, and the specialized versions such as ``torchvision.datasets.CIFAR10``.

    Note that for this interface to be compatible with our runtime instantiation rules, the constructor
    needs to receive a fully constructed task object. This object is currently constructed in
    :func:`thelper.data.utils.create_parsers` based on extra parameters; see the code there for more
    information.

    Attributes:
        dataset_type: type of the internally instantiated or provided dataset object.
        warned_dictionary: specifies whether the user was warned about missing keys in the output
            samples dictionaries.

    .. seealso::
        | :class:`thelper.data.parsers.Dataset`
    """

    def __init__(self, dataset, task, transforms=None, deepcopy=False, **kwargs):
        """External dataset parser constructor.

        Args:
            dataset: fully qualified name of the dataset object to instantiate, or the dataset itself.
            task: fully constructed task object providing key information for sample loading.
            transforms: function or object that should be applied to all loaded samples in order to
                return the data in the requested transformed/augmented state.
            deepcopy: specifies whether this dataset interface should be deep-copied inside
                :class:`thelper.data.loaders.LoaderFactory` so that it may be shared between
                different threads. This is false by default, as we assume datasets do not contain a state
                or buffer that might cause problems in multi-threaded data loaders.
        """
        super(ExternalDataset, self).__init__(transforms=transforms, deepcopy=deepcopy)
        if isinstance(dataset, str):
            dataset = thelper.utils.import_class(dataset)
        if dataset is None or not hasattr(dataset, "__getitem__") or not hasattr(dataset, "__len__"):
            raise AssertionError("external dataset type must implement '__getitem__' and '__len__' methods")
        if inspect.isclass(dataset):
            self.samples = dataset(**kwargs)
        else:
            self.samples = dataset
        if task is None or not isinstance(task, thelper.tasks.Task):
            raise AssertionError("task type must derive from thelper.tasks.Task")
        self.dataset_type = type(self.samples)
        self.task = task
        self.warned_dictionary = False

    def _get_derived_name(self):
        """Returns a pretty-print version of the external class's name."""
        return self.dataset_type.__module__ + "." + self.dataset_type.__qualname__

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        if idx >= len(self.samples):
            raise AssertionError("sample index is out-of-range")
        sample = self.samples[idx]
        if sample is None:
            # since might have provided an invalid sample count before, it's dangerous to skip empty samples here
            raise AssertionError("invalid sample received in external dataset impl")
        warn_dictionary = False
        if isinstance(sample, (list, tuple)):
            out_sample_list = []
            for key_idx, subs in enumerate(sample):
                if isinstance(subs, PIL.Image.Image):
                    subs = np.array(subs)
                out_sample_list.append(subs)
            sample = {str(key_idx): out_sample_list[key_idx] for key_idx in range(len(out_sample_list))}
            warn_dictionary = True
        elif isinstance(sample, (np.ndarray, PIL.Image.Image, torch.Tensor)):
            sample = {"0": sample}
            warn_dictionary = True
        if warn_dictionary and not self.warned_dictionary:
            logger.warning("dataset '%s' not returning samples as dictionaries;"
                           " will blindly map elements to their indices" % self._get_derived_name())
            self.warned_dictionary = True
        if self.transforms:
            sample = self.transforms(sample)
        return sample
