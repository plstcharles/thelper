import dataclasses
import datetime
import json
import logging
import os
import pickle
import pprint
import re
import typing

import cv2 as cv
import h5py
import numpy as np
import tqdm

import thelper.data
import thelper.tasks
import thelper.utils
from thelper.data.parsers import Dataset

logger = logging.getLogger(__name__)

# BigEarthNet Sentinel-2 Level 2A band mapping
# https://gisgeography.com/sentinel-2-bands-combinations/
# Blue = B2, Green = B3, Red = B4, NIR = B8
bgrnir_band_names = ["B02", "B03", "B04", "B08"]  # all should be 10m resolution (120x120)


@dataclasses.dataclass
class BigEarthNetPatch:

    mission_id: str
    coordinates: typing.Dict[str, float]
    tile_source: str
    tile_row: int
    tile_col: int
    acquisition_date: datetime.datetime
    projection: str
    root_path: str
    band_files: typing.List[str]
    labels: typing.List[str]

    @property
    def ulx(self) -> float:
        return self.coordinates["ulx"]

    @ulx.setter
    def ulx(self, x) -> None:
        self.coordinates["ulx"] = x

    @property
    def uly(self) -> float:
        return self.coordinates["uly"]

    @uly.setter
    def uly(self, y) -> None:
        self.coordinates["uly"] = y

    @property
    def lrx(self) -> float:
        return self.coordinates["lrx"]

    @lrx.setter
    def lrx(self, x) -> None:
        self.coordinates["lrx"] = x

    @property
    def lry(self) -> float:
        return self.coordinates["lry"]

    @lry.setter
    def lry(self, y) -> None:
        self.coordinates["lry"] = y

    def load_array(self,
                   target_size: int = 120,
                   target_bands: typing.Union[str, typing.List[str]] = "bgrnir",
                   target_dtype: np.dtype = np.dtype(np.uint16),
                   norm_meanstddev: typing.Optional[typing.Tuple[int, int]] = None
                   ):
        if isinstance(target_bands, str) and target_bands == "bgrnir":
            target_bands = bgrnir_band_names
        assert len(target_bands) > 0
        image = np.zeros((len(target_bands), target_size, target_size), dtype=target_dtype)
        for band_idx, band_suffix in enumerate(target_bands):
            for band_file in self.band_files:
                if band_file.endswith(band_suffix + ".tif"):
                    band_path = os.path.join(self.root_path, band_file)
                    assert os.path.isfile(band_path), f"could not locate band: {band_path}"
                    band = cv.imread(band_path, flags=cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
                    assert band.ndim == 2 and band.shape[0] == band.shape[1] and band.dtype == np.uint16
                    if band.shape[0] != target_size:
                        band = cv.resize(band, (target_size, target_size), interpolation=cv.INTER_CUBIC)
                    if norm_meanstddev:
                        band = (band.astype(np.float) - norm_meanstddev[0]) / norm_meanstddev[1]
                    if band.dtype != target_dtype:
                        band = band.astype(target_dtype)
                    image[band_idx] = band
        return image


class BigEarthNet(Dataset):

    def __init__(self, hdf5_path):
        # open HDF5 and read only metadata for now...
        logger.info(f"reading BigEarthNet metadata from: {hdf5_path}")
        self.hdf5_path = hdf5_path
        with h5py.File(self.hdf5_path, "r") as fd:
            self.target_size = fd.attrs["target_size"]
            self.target_bands = fd.attrs["target_bands"]
            self.target_dtype = np.dtype(fd.attrs["target_dtype"])
            self.norm_meanstddev = fd.attrs["norm_meanstddev"]
            patch_count = fd.attrs["patch_count"]
            metadata_dataset = fd["metadata"]
            self.patches = []
            for sample_idx in range(patch_count):
                self.patches.append(thelper.utils.fetch_hdf5_sample(metadata_dataset, sample_idx))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, sample_idx):
        # we should try to optimize the I/O... keep file handle open somehow, despite threading?
        with h5py.File(self.hdf5_path, "r") as fd:
            return thelper.utils.fetch_hdf5_sample(fd["imgdata"], sample_idx)


class HDF5Compactor:

    def __init__(self, root: typing.AnyStr):
        assert os.path.isdir(root), f"invalid big earth net root directory path ({root})"
        metadata_cache_path = os.path.join(root, "patches_metadata.pkl")
        if os.path.exists(metadata_cache_path):
            logger.info(f"loading patch metadata from cache: {metadata_cache_path}")
            with open(metadata_cache_path, "rb") as fd:
                self.patches = pickle.load(fd)
        else:
            logger.info(f"loading patch metadata from directory: {os.path.abspath(root)}")
            self.patches = self._load_patch_metadata(root)
            assert len(self.patches) > 0
            with open(metadata_cache_path, "wb") as fd:
                pickle.dump(self.patches, fd)
        tot_samples = len(self.patches)
        logger.info(f"loaded metadata for {tot_samples} patches")
        self.class_map = self._compute_class_map(self.patches)
        self.class_weights = {cname: len(cidxs) / tot_samples for cname, cidxs in self.class_map.items()}
        logger.debug(f"class weights:\n{pprint.PrettyPrinter(indent=2).pformat(self.class_weights)}")

    @staticmethod
    def _load_patch_metadata(root: typing.AnyStr, progress_bar: bool = True):
        assert os.path.isdir(root), f"invalid big earth net root directory path ({root})"
        name_pattern = re.compile(r"^([\w\d]+)_MSIL2A_(\d{8}T\d{6})_(\d+)_(\d+)$")
        patches = []
        patch_folders = os.listdir(root)
        patch_iter = tqdm.tqdm(patch_folders) if progress_bar else patch_folders
        for patch_folder in patch_iter:
            match_res = re.match(name_pattern, patch_folder)
            patch_folder_path = os.path.join(root, patch_folder)
            if match_res and os.path.isdir(patch_folder_path):
                patch_files = os.listdir(patch_folder_path)
                band_files = [p for p in patch_files if p.endswith(".tif")]
                metadata_files = [p for p in patch_files if p.endswith(".json")]
                assert len(band_files) == 12 and len(metadata_files) == 1
                metadata_path = os.path.join(patch_folder_path, metadata_files[0])
                with open(metadata_path, "r") as fd:
                    patch_metadata = json.load(fd)
                expected_meta_keys = ["labels", "coordinates", "projection", "tile_source", "acquisition_date"]
                assert all([key in patch_metadata for key in expected_meta_keys])
                acquisition_timestamp = datetime.datetime.strptime(patch_metadata["acquisition_date"],
                                                                   "%Y-%m-%d %H:%M:%S")
                file_timestamp = datetime.datetime.strptime(match_res.group(2), "%Y%m%dT%H%M%S")
                assert acquisition_timestamp == file_timestamp
                patches.append(BigEarthNetPatch(
                    root_path=os.path.abspath(patch_folder_path),
                    mission_id=match_res.group(1),
                    tile_col=int(match_res.group(3)),
                    tile_row=int(match_res.group(4)),
                    band_files=sorted(band_files),
                    **patch_metadata,
                ))
        return patches

    @staticmethod
    def _compute_class_map(patches: typing.List[BigEarthNetPatch]):
        assert len(patches) > 0
        class_map = {}
        for idx, patch in enumerate(patches):
            for class_name in patch.labels:
                if class_name not in class_map:
                    class_map[class_name] = []
                class_map[class_name].append(idx)
        return class_map

    def export(self,
               output_hdf5_path: typing.AnyStr,
               target_size: int = 120,
               target_bands: typing.Union[str, typing.List[str]] = "bgrnir",
               target_dtype: np.dtype = np.dtype(np.uint16),
               norm_meanstddev: typing.Optional[typing.Tuple[int, int]] = None,
               metadata_compression: typing.Optional[typing.Any] = None,
               image_compression: typing.Optional[typing.Any] = "chunk_lz4",
               progress_bar: bool = True,
               ):
        logger.info(f"exporting BigEarthNet to {output_hdf5_path}")
        if isinstance(target_bands, str) and target_bands == "bgrnir":
            target_bands = bgrnir_band_names
        pretty = pprint.PrettyPrinter(indent=2)
        with h5py.File(output_hdf5_path, "w") as fd:
            fd.attrs["source"] = thelper.utils.get_log_stamp()
            fd.attrs["git_sha1"] = thelper.utils.get_git_stamp()
            fd.attrs["version"] = thelper.__version__
            fd.attrs["target_size"] = target_size
            fd.attrs["target_bands"] = target_bands
            fd.attrs["target_dtype"] = target_dtype.str
            fd.attrs["norm_meanstddev"] = () if not norm_meanstddev else norm_meanstddev
            fd.attrs["metadata_compression"] = pretty.pformat(metadata_compression)
            fd.attrs["image_compression"] = pretty.pformat(image_compression)
            fd.attrs["patch_count"] = len(self.patches)
            logger.debug("dataset attributes: \n" +
                         pretty.pformat({key: val for key, val in fd.attrs.items()}))
            logger.debug("generating meta packets...")
            patch_meta_strs = np.asarray([repr(p) for p in self.patches])
            logger.debug("creating datasets...")
            assert metadata_compression not in thelper.utils.chunk_compression_flags
            metadata = thelper.utils.create_hdf5_dataset(
                fd=fd, name="metadata", max_len=len(patch_meta_strs),
                batch_like=patch_meta_strs, compression=metadata_compression)
            target_tensor_shape = (len(target_bands), target_size, target_size)
            fake_batch = np.zeros((1, *target_tensor_shape), dtype=target_dtype)
            if image_compression in thelper.utils.chunk_compression_flags or \
                    image_compression in thelper.utils.no_compression_flags:
                imgdata = thelper.utils.create_hdf5_dataset(
                    fd=fd,
                    name="imgdata",
                    max_len=len(self.patches),
                    batch_like=fake_batch,
                    compression=image_compression,
                    chunk_size=(1, *target_tensor_shape),
                    flatten=False
                )
            else:
                imgdata = thelper.utils.create_hdf5_dataset(
                    fd=fd,
                    name="imgdata",
                    max_len=len(self.patches),
                    batch_like=fake_batch,
                    compression=image_compression,
                    chunk_size=None,
                    flatten=True
                )
            logger.debug("exporting metadata...")
            if progress_bar:
                patch_meta_iter = tqdm.tqdm(patch_meta_strs, desc="exporting metadata")
            else:
                patch_meta_iter = patch_meta_strs
            for sample_idx, patch_meta_str in enumerate(patch_meta_iter):
                thelper.utils.fill_hdf5_sample(
                    metadata, sample_idx, sample_idx, patch_meta_strs, None)
            logger.debug("exporting image data...")
            if progress_bar:
                patch_iter = tqdm.tqdm(self.patches, desc="exporting image data")
            else:
                patch_iter = self.patches
            for sample_idx, patch in enumerate(patch_iter):
                patch_array = patch.load_array(
                    target_size=target_size,
                    target_bands=target_bands,
                    norm_meanstddev=norm_meanstddev
                )
                assert patch_array.shape == (len(target_bands), target_size, target_size)
                thelper.utils.fill_hdf5_sample(
                    imgdata, sample_idx, 0, patch_array.reshape((1, *target_tensor_shape)))

    def _test_close_vals(self, input_hdf5_path: typing.AnyStr):
        assert os.path.isfile(input_hdf5_path), f"invalid input hdf5 file path: {input_hdf5_path}"
        with h5py.File(input_hdf5_path, "r") as fd:
            target_size = fd.attrs["target_size"]
            target_bands = fd.attrs["target_bands"]
            target_dtype = np.dtype(fd.attrs["target_dtype"])
            norm_meanstddev = fd.attrs["norm_meanstddev"]
            patch_count = fd.attrs["patch_count"]
            assert patch_count == len(self.patches)
            patch_meta_strs = np.asarray([repr(p) for p in self.patches])
            metadata_dataset = fd["metadata"]
            for sample_idx in range(patch_count):
                loaded_meta_str = thelper.utils.fetch_hdf5_sample(metadata_dataset, sample_idx)
                assert patch_meta_strs[sample_idx] == loaded_meta_str
            random_sample_idxs = np.random.randint(low=0, high=patch_count, size=(100,))
            image_dataset = fd["imgdata"]
            for sample_idx in random_sample_idxs:
                generated_array = self.patches[sample_idx].load_array(
                    target_size=target_size,
                    target_bands=target_bands,
                    target_dtype=target_dtype,
                    norm_meanstddev=norm_meanstddev
                )
                loaded_array = thelper.utils.fetch_hdf5_sample(image_dataset, sample_idx)
                assert np.isclose(generated_array, loaded_array).all()


if __name__ == "__main__":
    # @@@@ TODO: CONVERT TO PROPER TEST
    logging.basicConfig()
    logging.getLogger().setLevel(logging.NOTSET)
    root_path = "/shared/data_ufast_ext4/datasets/bigearthnet/BigEarthNet-v1.0"
    dataset = HDF5Compactor(root_path)
    dataset.export("/shared/data_sfast/datasets/bigearthnet/bigearthnet-thelper.hdf5")
    dataset._test_close_vals("/shared/data_sfast/datasets/bigearthnet/bigearthnet-thelper.hdf5")
    print("all done")
