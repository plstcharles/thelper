"""
Agricultural Semantic Segentation Challenge Dataset Interface

Original author: David Landry (david.landry@crim.ca)
Updated by Pierre-Luc St-Charles (April 2020)
"""

import copy
import logging
import os
import pprint
import shutil
import typing

import h5py
import numpy as np
import torch.utils
import torch.utils.data
import tqdm

import thelper.data
import thelper.tasks
import thelper.utils
from thelper.data.parsers import Dataset

logger = logging.getLogger(__name__)


class_names = [
    "background",     # optional, depending on task
    "cloud_shadow",
    "double_plant",
    "planter_skip",
    "standing_water",
    "waterway",
    "weed_cluster",
]

approx_weight_map = {
    "background": 0.7754729398810614,
    "cloud_shadow": 0.02987549383646342,
    "double_plant": 0.006768273283806349,
    "planter_skip": 0.0016827190442308664,
    "standing_water": 0.015964306228958156,
    "waterway": 0.012930148362618188,
    "weed_cluster": 0.1573061193628617
}

dontcare = 255


class Hdf5AgricultureDataset(Dataset):

    def __init__(
            self,
            hdf5_path: typing.AnyStr,
            group_name: typing.AnyStr,
            transforms: typing.Any = None,
            use_global_normalization: bool = True,
            keep_file_open: bool = False,
            load_meta_keys: bool = False,
            copy_to_slurm_tmpdir: bool = False,
    ):
        super().__init__(transforms, deepcopy=False)
        if copy_to_slurm_tmpdir:
            assert os.path.isfile(hdf5_path), f"invalid input hdf5 path: {hdf5_path}"
            slurm_tmpdir = thelper.utils.get_slurm_tmpdir()
            assert slurm_tmpdir is not None, "undefined SLURM_TMPDIR env variable"
            dest_hdf5_path = os.path.join(slurm_tmpdir, "agrivis.hdf5")
            if not os.path.isfile(dest_hdf5_path):
                shutil.copyfile(hdf5_path, dest_hdf5_path)
            hdf5_path = dest_hdf5_path
        logger.info(f"reading AgriVis challenge {group_name} data from: {hdf5_path}")
        self.hdf5_path = hdf5_path
        self.group_name = group_name
        self.load_meta_keys = load_meta_keys
        with h5py.File(self.hdf5_path, "r") as archive:
            assert group_name in archive, \
                "unexpected dataset name (should be train/val/test)"
            dataset = archive[group_name]
            expected_keys = ["boundaries", "features", "keys"]
            if group_name != "test":
                expected_keys += ["labels", "n_labelled_pixels"]
            assert all([k in dataset.keys() for k in expected_keys]), \
                "missing at least one of the expected dataset group keys"
            assert all([len(dataset[k]) == len(dataset["keys"]) for k in expected_keys]), \
                "dataset sample count mismatch across all subgroups"
            if group_name != "test":
                assert dataset["labels"].shape[-1] == len(class_names) - 1, \
                    "unexpected dataset label map count while accounting for background"
                meta_iter = zip(dataset["keys"], dataset["n_labelled_pixels"])
            else:
                meta_iter = zip(dataset["keys"], [None] * len(dataset["keys"]))
            self.samples = [{  # list pre-fill
                "image": None,
                "label_map": None,
                "key": key,
                "mask": None,
                "pxcounts": pxcounts,
            } for key, pxcounts in meta_iter]
        logger.info(f"loaded metadata for {len(self.samples)} patches")
        self.task = thelper.tasks.Segmentation(
            class_names=class_names, input_key="image", label_map_key="label_map",
            meta_keys=["key", "mask", "pxcounts"], dontcare=dontcare,
        )
        self.use_global_normalization = use_global_normalization
        self.image_mean = np.asarray([
            121.6028380635106,
            118.52572985557143,
            116.36513065674848,
            108.47336023815292,
        ], dtype=np.float32)
        self.image_stddev = np.asarray([
            41.47667301013803,
            41.782106439616534,
            45.04215840534553,
            44.53299631408866,
        ], dtype=np.float32)
        self.hdf5_handle = h5py.File(self.hdf5_path, "r") if keep_file_open else None
        # self.squished = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._getitems(idx)
        assert idx < len(self.samples), "sample index is out-of-range"
        if idx < 0:
            idx = len(self.samples) + idx
        label_map = None
        if self.hdf5_handle is not None:
            image = self.hdf5_handle[self.group_name]["features"][idx]
            mask = self.hdf5_handle[self.group_name]["boundaries"][idx]
            if self.group_name != "test":
                label_map = self.hdf5_handle[self.group_name]["labels"][idx]
        else:
            with h5py.File(self.hdf5_path, mode="r") as archive:
                image = archive[self.group_name]["features"][idx]
                mask = archive[self.group_name]["boundaries"][idx]
                if self.group_name != "test":
                    label_map = archive[self.group_name]["labels"][idx]
        if self.use_global_normalization:
            image = (image.astype(np.float32) - self.image_mean) / self.image_stddev
        mask = mask.astype(np.int16)
        if label_map is not None:
            # note: we might squish some overlapping labels, but these are very rare... (<0.07%)
            out_label_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.int16)
            for label_idx in range(1, len(class_names)):
                orig_label_map_idx = label_idx - 1
                curr_label_map = label_map[..., orig_label_map_idx]
                # overlap = np.logical_and(out_label_map != 0, curr_label_map)
                # self.squished += np.count_nonzero(overlap)
                out_label_map = np.where(curr_label_map, np.int16(label_idx), out_label_map)
            label_map = out_label_map
            label_map = np.where(mask, label_map, np.int16(dontcare))
        sample = {
            "image": image,
            "label_map": label_map,
            "mask": mask,
            # drop key if neccessary to fix batching w/ PyTorch default collate
            "key": self.samples[idx]["key"] if self.load_meta_keys else None,
            "pxcounts": copy.deepcopy(self.samples[idx]["pxcounts"]),
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


def _compute_statistics(dataset) -> typing.Dict:
    array_alloc_size = len(dataset)
    stat_arrays = [
        # alloc three vals per item: px-wise sum, px-wise sqsum, px count
        np.zeros((array_alloc_size, 3), dtype=np.float64) for band in range(4)
    ]
    for batch_idx, sample in enumerate(tqdm.tqdm(dataset)):
        image = sample["image"]
        assert image.ndim == 3 and image.shape[-1] == 4
        for ch_idx in range(4):
            image_ch = image[..., ch_idx]
            stat_arrays[ch_idx][batch_idx] = (
                np.sum(image_ch, dtype=np.float64),
                np.sum(np.square(image_ch, dtype=np.float64)),
                np.float64(image_ch.size),
            )
    stat_map = {}
    for band_idx, band_array in enumerate(stat_arrays):
        tot_size = np.sum(band_array[:, 2])
        mean = np.sum(band_array[:, 0]) / tot_size
        stddev = np.sqrt(np.sum(band_array[:, 1]) / tot_size - np.square(mean))
        stat_map[band_idx] = {"mean": mean, "stddev": stddev}
    return stat_map


def _compute_class_weights(dataset) -> typing.Dict:
    class_counts = {key: 0 for key in class_names}
    tot_samples = 0
    image_px_count = 512 * 512  # fixed size
    for sample in tqdm.tqdm(dataset):
        if sample["pxcounts"] is not None:
            tot_labeled_px = 0
            for class_idx, px_count in enumerate(sample["pxcounts"]):
                class_counts[class_names[class_idx + 1]] += px_count
                tot_labeled_px += px_count
            class_counts["background"] += image_px_count - tot_labeled_px
            tot_samples += 1
    tot_count = tot_samples * image_px_count
    class_weights = {
        key: count / tot_count for key, count in class_counts.items()
    }
    return class_weights


if __name__ == "__main__":
    # @@@@ TODO: CONVERT TO PROPER TEST
    logging.basicConfig()
    logging.getLogger().setLevel(logging.NOTSET)
    dataset = torch.utils.data.ConcatDataset([
        Hdf5AgricultureDataset(
            hdf5_path="/shared/data_ufast_ext4/datasets/agrivis/agri_v2.hdf5",
            group_name=group_name,
            use_global_normalization=False,
            keep_file_open=True,
        ) for group_name in ["train", "val", "test"]
    ])
    # out_map = _compute_statistics(dataset)
    out_map = _compute_class_weights(dataset)
    logging.info(f"out_map =\n{pprint.pformat(out_map, indent=4)}")
    print("all done")
