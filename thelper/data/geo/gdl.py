"""Data parsers & utilities for cross-framework compatibility with Geo Deep Learning (GDL).

Geo Deep Learning (GDL) is a machine learning framework initiative for geospatial projects
lead by the wonderful folks at NRCan's CCMEO. See https://github.com/NRCan/geo-deep-learning
for more information.

The classes and functions defined here were used for the exploration of research topics and
for the validation and testing of new software components.
"""

import collections
import logging
import os

import h5py
import numpy as np

import thelper.nn.coordconv
from thelper.data.parsers import SegmentationDataset as BaseSegmentationDataset

logger = logging.getLogger(__name__)


class SegmentationDataset(BaseSegmentationDataset):
    """Semantic segmentation dataset interface for GDL-based HDF5 parsing."""

    def __init__(self, class_names, work_folder, dataset_type, max_sample_count=None,
                 dontcare=None, transforms=None):
        self.dontcare = dontcare
        if isinstance(dontcare, (tuple, list)) and len(dontcare) == 2:
            logger.warning(f"will remap dontcare index from {dontcare[0]} to {dontcare[1]}")
            dontcare = dontcare[1]
        assert dontcare is None or isinstance(dontcare, int), "unexpected dontcare type"
        super().__init__(class_names=class_names, input_key="sat_img", label_map_key="map_img",
                         meta_keys=["metadata"], dontcare=dontcare, transforms=transforms)
        # note: if 'max_sample_count' is None, then it will be read from the dataset at runtime
        self.work_folder = work_folder
        self.max_sample_count = max_sample_count
        self.dataset_type = dataset_type
        self.metadata = []
        self.hdf5_path = os.path.join(self.work_folder, self.dataset_type + "_samples.hdf5")
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            if "metadata" in hdf5_file:
                for i in range(hdf5_file["metadata"].shape[0]):
                    metadata = hdf5_file["metadata"][i, ...]
                    if isinstance(metadata, np.ndarray) and len(metadata) == 1:
                        metadata = metadata[0]
                    if isinstance(metadata, str):
                        if "ordereddict" in metadata:
                            metadata = metadata.replace("ordereddict", "collections.OrderedDict")
                        if metadata.startswith("collections.OrderedDict"):
                            metadata = eval(metadata)
                    self.metadata.append(metadata)
            if self.max_sample_count is None:
                self.max_sample_count = hdf5_file["sat_img"].shape[0]
            self.samples = [{}] * self.max_sample_count

    def _remap_labels(self, map_img):
        # note: will do nothing if 'dontcare' remap mode is not activated in constructor
        if not isinstance(self.dontcare, (tuple, list)):
            return map_img
        # for now, the current implementation only handles the original 'dontcare' as zero
        assert self.dontcare[0] == 0, "missing implementation for non-zero original dontcare value"
        # to keep the impl simple, we just reduce all indices by one and replace -1 by the new value
        assert map_img.dtype == np.int8 or map_img.dtype == np.int16 or map_img.dtype == np.int32
        map_img -= 1
        if self.dontcare[1] != -1:
            map_img[map_img == -1] = self.dontcare[1]
        return map_img

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            sat_img = hdf5_file["sat_img"][index, ...]
            map_img = self._remap_labels(hdf5_file["map_img"][index, ...])
            meta_idx = int(hdf5_file["meta_idx"][index]) if "meta_idx" in hdf5_file else -1
            metadata = None
            if meta_idx != -1:
                metadata = self.metadata[meta_idx]
        sample = {"sat_img": sat_img, "map_img": map_img, "metadata": metadata}
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class MetaSegmentationDataset(SegmentationDataset):
    """Semantic segmentation dataset interface that appends metadata under new tensor layers."""

    metadata_handling_modes = ["const_channel", "scaled_channel"]  # TODO: add more

    def __init__(self, class_names, work_folder, dataset_type, meta_map, max_sample_count=None,
                 dontcare=None, transforms=None):
        assert isinstance(meta_map, dict), "unexpected metadata mapping object type"
        assert all([isinstance(k, str) and v in self.metadata_handling_modes for k, v in meta_map.items()]), \
            "unexpected metadata key type or value handling mode"
        super().__init__(class_names=class_names, work_folder=work_folder, dataset_type=dataset_type,
                         max_sample_count=max_sample_count, dontcare=dontcare, transforms=transforms)
        assert all([isinstance(m, (dict, collections.OrderedDict)) for m in self.metadata]), \
            "cannot use provided metadata object type with meta-mapping dataset interface"
        self.meta_map = meta_map

    @staticmethod
    def get_meta_value(map, key):
        if not isinstance(key, list):
            key = key.split("/")  # subdict indexing split using slash
        assert key[0] in map, f"missing key '{key[0]}' in metadata dictionary"
        val = map[key[0]]
        if isinstance(val, (dict, collections.OrderedDict)):
            assert len(key) > 1, "missing keys to index metadata subdictionaries"
            return MetaSegmentationDataset.get_meta_value(val, key[1:])
        return val

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            sat_img = hdf5_file["sat_img"][index, ...]
            map_img = self._remap_labels(hdf5_file["map_img"][index, ...])
            meta_idx = int(hdf5_file["meta_idx"][index]) if "meta_idx" in hdf5_file else -1
            assert meta_idx != -1, f"metadata unvailable in sample #{index}"
            metadata = self.metadata[meta_idx]
            assert isinstance(metadata, (dict, collections.OrderedDict)), "unexpected metadata type"
        for meta_key, mode in self.meta_map.items():
            meta_val = self.get_meta_value(metadata, meta_key)
            if mode == "const_channel":
                assert np.isscalar(meta_val), "constant channel-wise assignment requires scalar value"
                layer = np.full(sat_img.shape[0:2], meta_val, dtype=np.float32)
                sat_img = np.insert(sat_img, sat_img.shape[2], layer, axis=2)
            elif mode == "scaled_channel":
                assert np.isscalar(meta_val), "scaled channel-wise coords assignment requires scalar value"
                layers = thelper.nn.coordconv.get_coords_map(sat_img.shape[0], sat_img.shape[1]) * meta_val
                sat_img = np.insert(sat_img, sat_img.shape[2], layers, axis=2)
            #else...
        sample = {"sat_img": sat_img, "map_img": map_img, "metadata": metadata}
        if self.transforms:
            sample = self.transforms(sample)
        return sample
