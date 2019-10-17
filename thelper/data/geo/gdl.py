"""Data parsers & utilities for cross-framework compatibility with Geo Deep Learning (GDL).

Geo Deep Learning (GDL) is a machine learning framework initiative for geospatial projects
lead by the wonderful folks at NRCan's CCMEO. See https://github.com/NRCan/geo-deep-learning
for more information.

The classes and functions defined here were used for the exploration of research topics and
for the validation and testing of new software components.
"""

import collections
import os

import h5py
import numpy as np

import thelper.data


class SegmentationDataset(thelper.data.parsers.SegmentationDataset):
    """Semantic segmentation dataset interface for GDL-based HDF5 parsing."""

    def __init__(self, class_names, work_folder, dataset_type, max_sample_count=None,
                 dontcare=None, transforms=None):
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

    def __len__(self):
        return self.max_sample_count

    def __getitem__(self, index):
        with h5py.File(self.hdf5_path, "r") as hdf5_file:
            sat_img = hdf5_file["sat_img"][index, ...]
            map_img = hdf5_file["map_img"][index, ...]
            meta_idx = int(hdf5_file["meta_idx"][index]) if "meta_idx" in hdf5_file else -1
            metadata = None
            if meta_idx != -1:
                metadata = self.metadata[meta_idx]
        sample = {"sat_img": sat_img, "map_img": map_img, "metadata": metadata}
        if self.transforms:
            sample = self.transforms(sample)
        return sample
