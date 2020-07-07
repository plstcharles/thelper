"""Geospatial data parser & utilities module."""

import functools
import json
import logging
import math
import os
import pickle

import cv2 as cv
import gdal
import numpy as np
import ogr
import osr
import shapely
import torch
import tqdm

import thelper.tasks
import thelper.utils
from thelper.data import Dataset, ImageFolderDataset
from thelper.data.geo.utils import parse_raster_metadata

logger = logging.getLogger(__name__)


class VectorCropDataset(Dataset):
    """Abstract dataset used to combine geojson vector data and rasters."""

    def __init__(self, raster_path, vector_path, px_size=None, skew=None,
                 allow_outlying_vectors=True, clip_outlying_vectors=True,
                 vector_area_min=0.0, vector_area_max=float("inf"),
                 vector_target_prop=None, feature_buffer=None, master_roi=None,
                 srs_target="3857", raster_key="raster", mask_key="mask",
                 cleaner=None, cropper=None, force_parse=False,
                 reproj_rasters=False, reproj_all_cpus=True,
                 keep_rasters_open=True, transforms=None):
        import thelper.data.geo as geo
        # before anything else, create a hash to cache parsed data
        cache_hash = thelper.utils.get_params_hash(
            {k: v for k, v in vars().items() if not k.startswith("_") and k != "self"}) if not force_parse else None
        assert isinstance(raster_path, str), "raster file/folder path should be given as string"
        assert isinstance(vector_path, str), "vector file/folder path should be given as string"
        self.raster_path = raster_path
        self.vector_path = vector_path
        assert px_size is None or \
            (isinstance(px_size, (list, tuple)) and all([isinstance(i, (float, int)) for i in px_size]) and len(px_size) == 2) or \
            isinstance(px_size, (float, int)), "pixel size (resolution) must be float/int or list/tuple"
        self.px_size = (1.0, 1.0) if px_size is None else (float(px_size[0]), float(px_size[1])) \
            if isinstance(px_size, (list, tuple)) else (float(px_size), float(px_size))
        assert skew is None or \
            (isinstance(skew, (list, tuple)) and all([isinstance(i, (float, int)) for i in skew]) and len(skew) == 2) or \
            isinstance(skew, (float, int)), "pixel skew must be float/int or list/tuple"
        self.skew = (0.0, 0.0) if skew is None else (float(skew[0]), float(skew[1])) \
            if isinstance(skew, (list, tuple)) else (float(skew), float(skew))
        assert isinstance(allow_outlying_vectors, bool), "unexpected flag type"
        assert isinstance(clip_outlying_vectors, bool), "unexpected flag type"
        assert isinstance(force_parse, bool), "unexpected flag type"
        assert isinstance(reproj_rasters, bool), "unexpected flag type"
        assert isinstance(reproj_all_cpus, bool), "unexpected flag type"
        assert isinstance(keep_rasters_open, bool), "unexpected flag type"
        self.allow_outlying = allow_outlying_vectors
        self.clip_outlying = clip_outlying_vectors
        self.force_parse = force_parse
        self.reproj_rasters = reproj_rasters
        self.reproj_all_cpus = reproj_all_cpus
        self.keep_rasters_open = keep_rasters_open
        assert isinstance(vector_area_min, (float, int)) and vector_area_min >= 0, \
            "min surface filter value must be > 0"
        assert isinstance(vector_area_max, (float, int)) and vector_area_max >= vector_area_min, \
            "max surface filter value must be greater than minimum surface value"
        self.area_min = float(vector_area_min)
        self.area_max = float(vector_area_max)
        assert vector_target_prop is None or isinstance(vector_target_prop, dict), \
            "feature target props should be specified as dictionary of property name-value pairs for search"
        self.target_prop = {} if vector_target_prop is None else vector_target_prop
        assert feature_buffer is None or (isinstance(feature_buffer, (int, float)) and feature_buffer > 0), \
            "feature roi 'buffer' value should be strictly positive int/float"
        self.feature_buffer = feature_buffer
        assert isinstance(master_roi, (str, shapely.geometry.polygon.Polygon,
                                       shapely.geometry.multipolygon.MultiPolygon)) or master_roi is None, \
            "invalid master roi (should be path to geojson/shapefile or polygon object)"
        assert isinstance(srs_target, (str, int, osr.SpatialReference)), \
            "target EPSG SRS must be given as int/str"
        self.srs_target = srs_target
        if isinstance(self.srs_target, (str, int)):
            if isinstance(self.srs_target, str):
                self.srs_target = int(self.srs_target.replace("EPSG:", ""))
            srs_target_obj = osr.SpatialReference()
            srs_target_obj.ImportFromEPSG(self.srs_target)
            self.srs_target = srs_target_obj
        self.master_roi = geo.utils.parse_roi(master_roi, srs_target=self.srs_target) \
            if isinstance(master_roi, str) else master_roi
        assert isinstance(raster_key, str), "raster key must be given as string"
        self.raster_key = raster_key
        assert isinstance(mask_key, str), "mask key must be given as string"
        self.mask_key = mask_key
        super().__init__(transforms=transforms)
        self.rasters_data, self.coverage = self._parse_rasters(self.raster_path, self.srs_target, reproj_rasters)
        if self.master_roi is not None:
            self.coverage = self.coverage.intersection(self.master_roi)
        if cleaner is None:
            cleaner = functools.partial(self._default_feature_cleaner, area_min=self.area_min,
                                        area_max=self.area_max, target_prop=self.target_prop)
        self.features = self._parse_features(self.vector_path, self.srs_target, self.coverage, cache_hash,
                                             self.allow_outlying, self.clip_outlying, cleaner)
        if cropper is None:
            cropper = functools.partial(self._default_feature_cropper, px_size=self.px_size,
                                        skew=self.skew, feature_buffer=self.feature_buffer)
        self.samples = self._parse_crops(cropper, self.vector_path, cache_hash)
        # all keys already in sample dicts should be 'meta'; mask & raster will be added later
        meta_keys = list(set([k for s in self.samples for k in s]))
        if self.mask_key not in meta_keys:
            meta_keys.append(self.mask_key)
        # create default task without gt specification (this is a pretty basic parser)
        self.task = thelper.tasks.Task(input_key=self.raster_key, meta_keys=meta_keys)
        self.display_debug = False  # for internal debugging purposes only

    @staticmethod
    def _default_feature_cleaner(features, area_min, area_max, target_prop=None):
        """Flags geometric features as 'clean' based on some criteria (may be modified in derived classes)."""
        # note: we use a flag here instead of removing bad features so that end-users can still use them if needed
        for feature in tqdm.tqdm(features, desc="cleaning up features"):
            assert isinstance(feature, dict) and "clean" not in feature
            feature["clean"] = True
            if target_prop is not None and "properties" in feature and isinstance(feature["properties"], dict):
                if not all([k in feature["properties"] and feature["properties"][k] == v for k, v in target_prop.items()]):
                    feature["clean"] = False
            if not (area_min <= feature["geometry"].area <= area_max):
                feature["clean"] = False
        return features

    @staticmethod
    def _default_feature_cropper(features, rasters_data, coverage, srs_target, px_size, skew, feature_buffer):
        """Returns the samples for a set of features (may be modified in derived classes)."""
        # note: default behavior = just center on the feature, and pad if required by user
        import thelper.data.geo as geo
        samples = []
        clean_feats = [f for f in features if f["clean"]]
        srs_target_wkt = srs_target.ExportToWkt()
        for feature in tqdm.tqdm(clean_feats, desc="validating crop candidates"):
            assert feature["clean"]  # should not get here with bad features
            roi, roi_tl, roi_br, crop_width, crop_height = \
                geo.utils.get_feature_roi(feature["geometry"], px_size, skew, feature_buffer)
            # test all raster regions that touch the selected feature
            raster_hits = []
            for raster_idx, raster_data in enumerate(rasters_data):
                if raster_data["target_roi"].intersects(roi):
                    raster_hits.append(raster_idx)
            # make list of all other features that may be included in the roi
            roi_radius = np.linalg.norm(np.asarray(roi_tl) - np.asarray(roi_br)) / 2
            roi_features = [f for f in features if feature["centroid"].distance(f["centroid"]) <= roi_radius and
                            f["geometry"].intersects(roi)]
            # prepare actual 'sample' for crop generation at runtime
            samples.append({
                "features": roi_features,
                "focal": feature,
                "roi": roi,
                "roi_tl": roi_tl,
                "roi_br": roi_br,
                "raster_hits": raster_hits,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "geotransform": np.asarray((roi_tl[0], px_size[0], skew[0],
                                            roi_tl[1], skew[1], px_size[1])),
                "srs": srs_target_wkt,
            })
        return samples

    @staticmethod
    def _parse_rasters(path, srs, reproj_rasters):
        """Parses rasters (geotiffs) and returns metadata/coverage information."""
        import thelper.data.geo as geo
        logger.info(f"parsing rasters from path '{path}'...")
        raster_paths = thelper.utils.get_file_paths(path, ".", allow_glob=True)
        rasters_data, coverage = geo.utils.parse_rasters(raster_paths, srs, reproj_rasters)
        assert rasters_data, f"could not find any usable rasters at '{raster_paths}'"
        logger.debug(f"rasters total coverage area = {coverage.area:.2f}")
        for idx, data in enumerate(rasters_data):
            logger.debug(f"raster #{idx + 1} area = {data['target_roi'].area:.2f}")
            # here, we enforce that raster datatypes/bandcounts match
            assert data["band_count"] == rasters_data[0]["band_count"], \
                "parser expects that all raster band counts match" + \
                f"(found {str(data['band_count'])} and {str(rasters_data[0]['band_count'])})"
            assert data["data_type"] == rasters_data[0]["data_type"], \
                "parser expects that all raster data types match" + \
                f"(found {str(data['data_type'])} and {str(rasters_data[0]['data_type'])})"
            data["to_target_transform"] = osr.CoordinateTransformation(data["srs"], srs)
            data["from_target_transform"] = osr.CoordinateTransformation(srs, data["srs"])
        return rasters_data, coverage

    @staticmethod
    def _parse_features(path, srs, roi, cache_hash, allow_outlying, clip_outlying, cleaner):
        """Parses vector files (geojsons) and returns geometry information."""
        import thelper.data.geo as geo
        logger.info(f"parsing vectors from path '{path}'...")
        assert os.path.isfile(path) and path.endswith("geojson"), \
            "vector file must be provided as geojson (shapefile support still incomplete)"
        cache_file_path = os.path.join(os.path.dirname(path), cache_hash + ".feats.pkl") \
            if cache_hash else None
        if cache_file_path is not None and os.path.exists(cache_file_path):
            logger.debug(f"parsing cached feature data from '{cache_file_path}'...")
            with open(cache_file_path, "rb") as fd:
                features = pickle.load(fd)
        else:
            with open(path) as vector_fd:
                vector_data = json.load(vector_fd)
            features = geo.utils.parse_geojson(vector_data, srs_target=srs, roi=roi,
                                               allow_outlying=allow_outlying, clip_outlying=clip_outlying)
            features = cleaner(features)
            if cache_file_path is not None:
                logger.debug(f"caching clean data to '{cache_file_path}'...")
                with open(cache_file_path, "wb") as fd:
                    pickle.dump(features, fd)
        logger.debug(f"cleanup resulted in {len([f for f in features if f['clean']])} features of interest")
        return features

    def _parse_crops(self, cropper, cache_file_path, cache_hash):
        """Parses crops based on prior feature/raster data.

        Each 'crop' corresponds to a sample that can be loaded at runtime.
        """
        logger.info("preparing crops...")
        cache_file_path = os.path.join(os.path.dirname(cache_file_path), cache_hash + ".crops.pkl") \
            if cache_hash else None
        if cache_file_path is not None and os.path.exists(cache_file_path):
            logger.debug(f"parsing cached crop data from '{cache_file_path}'...")
            with open(cache_file_path, "rb") as fd:
                samples = pickle.load(fd)
        else:
            samples = cropper(self.features, self.rasters_data, self.coverage, self.srs_target)
            if cache_file_path is not None:
                logger.debug(f"caching crop data to '{cache_file_path}'...")
                with open(cache_file_path, "wb") as fd:
                    pickle.dump(samples, fd)
        return samples

    def _process_crop(self, sample):
        """Returns a crop for a specific (internal) set of sampled features."""
        import thelper.data.geo as geo
        # remember: we assume that all rasters have the same intrinsic settings
        crop_datatype = geo.utils.GDAL2NUMPY_TYPE_CONV[self.rasters_data[0]["data_type"]]
        crop_size = (sample["crop_height"], sample["crop_width"], self.rasters_data[0]["band_count"])
        crop = np.ma.array(np.zeros(crop_size, dtype=crop_datatype), mask=np.ones(crop_size, dtype=np.uint8))
        mask = np.zeros(crop_size[0:2], dtype=np.uint8)
        crop_raster_gdal = gdal.GetDriverByName("MEM").Create("", crop_size[1], crop_size[0],
                                                              crop_size[2], self.rasters_data[0]["data_type"])
        crop_raster_gdal.SetGeoTransform(sample["geotransform"])
        crop_raster_gdal.SetProjection(self.srs_target.ExportToWkt())
        crop_mask_gdal = gdal.GetDriverByName("MEM").Create("", crop_size[1], crop_size[0], 1, gdal.GDT_Byte)
        crop_mask_gdal.SetGeoTransform(sample["geotransform"])
        crop_mask_gdal.SetProjection(self.srs_target.ExportToWkt())
        crop_mask_gdal.GetRasterBand(1).WriteArray(np.zeros(crop_size[0:2], dtype=np.uint8))
        ogr_dataset = ogr.GetDriverByName("Memory").CreateDataSource("mask")
        ogr_layer = ogr_dataset.CreateLayer("feature_mask", srs=self.srs_target)
        for feature in sample["features"]:
            ogr_feature = ogr.Feature(ogr_layer.GetLayerDefn())
            ogr_geometry = ogr.CreateGeometryFromWkt(feature["geometry"].wkt)
            ogr_feature.SetGeometry(ogr_geometry)
            ogr_layer.CreateFeature(ogr_feature)
        gdal.RasterizeLayer(crop_mask_gdal, [1], ogr_layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
        np.copyto(dst=mask, src=crop_mask_gdal.GetRasterBand(1).ReadAsArray())
        for raster_idx in sample["raster_hits"]:
            rasterfile = geo.utils.open_rasterfile(self.rasters_data[raster_idx],
                                                   keep_rasters_open=self.keep_rasters_open)
            assert rasterfile.RasterCount == crop_size[2], "unexpected raster count"
            # using all cpus should be ok since we probably cant parallelize this loader anyway (swig serialization issues)
            options = ["NUM_THREADS=ALL_CPUS"] if self.reproj_all_cpus else []
            geo.utils.reproject_crop(rasterfile, crop_raster_gdal, crop_size, crop_datatype, reproj_opt=options, fill_nodata=True)
            for raster_band_idx in range(crop_raster_gdal.RasterCount):
                curr_band = crop_raster_gdal.GetRasterBand(raster_band_idx + 1)
                curr_band_array = curr_band.ReadAsArray()
                flag_mask = curr_band_array != curr_band.GetNoDataValue()
                np.copyto(dst=crop.data[:, :, raster_band_idx], src=curr_band_array, where=flag_mask)
                np.bitwise_and(crop.mask[:, :, raster_band_idx], np.invert(flag_mask), out=crop.mask[:, :, raster_band_idx])
        # ogr_dataset = None  # noqa # close local fd
        # noinspection PyUnusedLocal
        crop_raster_gdal = None  # noqa # close local fd
        # noinspection PyUnusedLocal
        crop_mask_gdal = None  # noqa # close local fd
        # noinspection PyUnusedLocal
        rasterfile = None  # noqa # close input fd
        return crop, mask

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        assert idx < len(self.samples), "sample index is out-of-range"
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        crop, mask = self._process_crop(sample)
        if self.display_debug:
            crop = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)
            mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
            mask[:, :, 1:3] = 0
            crop = cv.normalize(crop, dst=crop, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            mask = cv.normalize(mask, dst=mask, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            crop = np.bitwise_or(crop, mask)
        sample = {
            self.raster_key: np.array(crop.data, copy=True),
            self.mask_key: mask,
            **sample
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class TileDataset(VectorCropDataset):
    """Abstract dataset used to systematically tile vector data and rasters."""

    def __init__(self, raster_path, vector_path, tile_size, tile_overlap=0,
                 skip_empty_tiles=False, skip_nodata_tiles=True, px_size=None,
                 allow_outlying_vectors=True, clip_outlying_vectors=True,
                 vector_area_min=0.0, vector_area_max=float("inf"),
                 vector_target_prop=None, master_roi=None, srs_target="3857",
                 raster_key="raster", mask_key="mask", cleaner=None,
                 force_parse=False, reproj_rasters=False,
                 reproj_all_cpus=True, keep_rasters_open=True, transforms=None):
        # note1: input 'tile_size' must be given in pixels
        # note2: input 'tile_overlap' must be given in pixels
        # note3: input 'px_size' must be given in meters/degrees
        if isinstance(tile_size, (float, int)):
            tile_size = (tile_size, tile_size)
        assert isinstance(tile_size, (tuple, list)) and len(tile_size) == 2, \
            "invalid tile size (should be scalar or two-elem tuple)"
        assert all([t > 0 for t in tile_size]), "unexpected tile size value (should be positive)"
        tile_size = [float(t) for t in tile_size]  # convert all vals if necessary
        assert isinstance(tile_overlap, (float, int)) and tile_overlap >= 0, \
            "unexpected tile overlap (should be non-negative scalar)"
        tile_overlap = float(tile_overlap)
        assert isinstance(skip_empty_tiles, bool), "unexpected flag type (should be bool)"
        assert isinstance(skip_nodata_tiles, bool), "unexpected flag type (should be bool)"
        cropper = functools.partial(self._tile_cropper, tile_size=tile_size, tile_overlap=tile_overlap,
                                    skip_empty_tiles=skip_empty_tiles, skip_nodata_tiles=skip_nodata_tiles,
                                    keep_rasters_open=keep_rasters_open, px_size=px_size)
        super().__init__(raster_path=raster_path, vector_path=vector_path, px_size=px_size, skew=None,
                         allow_outlying_vectors=allow_outlying_vectors, clip_outlying_vectors=clip_outlying_vectors,
                         vector_area_min=vector_area_min, vector_area_max=vector_area_max, vector_target_prop=vector_target_prop,
                         master_roi=master_roi, srs_target=srs_target, raster_key=raster_key, mask_key=mask_key,
                         cleaner=cleaner, cropper=cropper, force_parse=force_parse, reproj_rasters=reproj_rasters,
                         reproj_all_cpus=reproj_all_cpus, keep_rasters_open=keep_rasters_open, transforms=transforms)

    @staticmethod
    def _tile_cropper(features, rasters_data, coverage, srs_target, tile_size, tile_overlap,
                      skip_empty_tiles, skip_nodata_tiles, keep_rasters_open, px_size):
        """Returns the ROI information for a given feature (may be modified in derived classes)."""
        import thelper.data.geo as geo
        # instead of iterating over features to generate samples, we tile the raster(s)
        # note: the 'coverage' geometry should already be in the target srs
        roi_tl, roi_br = geo.utils.get_feature_bbox(coverage)
        roi_geotransform = (roi_tl[0], px_size[0], 0.0,
                            roi_tl[1], 0.0, px_size[1])
        srs_target_wkt = srs_target.ExportToWkt()
        # remember: we assume that all rasters have the same intrinsic settings
        crop_datatype = geo.utils.GDAL2NUMPY_TYPE_CONV[rasters_data[0]["data_type"]]
        crop_raster_gdal = gdal.GetDriverByName("MEM").Create("",
                                                              int(round(tile_size[1])),
                                                              int(round(tile_size[0])),
                                                              rasters_data[0]["band_count"],
                                                              rasters_data[0]["data_type"])
        crop_raster_gdal.SetProjection(srs_target_wkt)
        samples = []
        crop_id = 0
        roi_px_br = geo.utils.get_pxcoord(roi_geotransform, *roi_br)
        nb_iter_y = int(math.ceil((roi_px_br[1] + tile_overlap) / (tile_size[1] - tile_overlap)))
        nb_iter_x = int(math.ceil((roi_px_br[0] + tile_overlap) / (tile_size[0] - tile_overlap)))
        pbar = tqdm.tqdm(total=nb_iter_y * nb_iter_x, desc="validating crop candidates")
        roi_offset_px_y = -tile_overlap
        while roi_offset_px_y < roi_px_br[1]:
            roi_offset_px_x = -tile_overlap
            while roi_offset_px_x < roi_px_br[0]:
                pbar.update(1)
                crop_px_tl = (roi_offset_px_x, roi_offset_px_y)
                crop_px_br = (crop_px_tl[0] + tile_size[0], crop_px_tl[1] + tile_size[1])
                crop_tl = geo.utils.get_geocoord(roi_geotransform, *crop_px_tl)
                crop_br = geo.utils.get_geocoord(roi_geotransform, *crop_px_br)
                crop_geom = shapely.geometry.Polygon([crop_tl, (crop_br[0], crop_tl[1]),
                                                      crop_br, (crop_tl[0], crop_br[1])])
                crop_geotransform = (crop_tl[0], px_size[0], 0.0,
                                     crop_tl[1], 0.0, px_size[1])
                crop_raster_gdal.SetGeoTransform(crop_geotransform)
                raster_hits = []
                found_valid_intersection = False or not skip_nodata_tiles
                for raster_idx, raster_data in enumerate(rasters_data):
                    if raster_data["target_roi"].intersects(crop_geom):
                        if not found_valid_intersection:
                            rasterfile = geo.utils.open_rasterfile(raster_data, keep_rasters_open=keep_rasters_open)
                            # yeah, we reproject the crop, preprocessing is slow, deal with it
                            geo.utils.reproject_crop(rasterfile, crop_raster_gdal, tile_size, crop_datatype, fill_nodata=True)
                            for raster_band_idx in range(crop_raster_gdal.RasterCount):
                                curr_band = crop_raster_gdal.GetRasterBand(raster_band_idx + 1)
                                found_valid_intersection = found_valid_intersection or \
                                    np.count_nonzero(curr_band.ReadAsArray() != curr_band.GetNoDataValue()) > 0
                        raster_hits.append(raster_idx)
                if raster_hits and found_valid_intersection:
                    crop_centroid = crop_geom.centroid
                    crop_radius = np.linalg.norm(np.asarray(crop_tl) - np.asarray(crop_br)) / 2
                    crop_features = []
                    for f in features:
                        if f["geometry"].distance(crop_centroid) > crop_radius:
                            continue
                        inters = f["geometry"].intersection(crop_geom)
                        if inters.is_empty:
                            continue
                        crop_features.append(f)
                    if crop_features or not skip_empty_tiles:
                        # prepare actual 'sample' for crop generation at runtime
                        samples.append({
                            "features": crop_features,
                            "id": crop_id,
                            "roi": crop_geom,
                            "roi_tl": crop_tl,
                            "roi_br": crop_br,
                            "raster_hits": raster_hits,
                            "crop_width": int(round(tile_size[0])),
                            "crop_height": int(round(tile_size[1])),
                            "geotransform": np.asarray(crop_geotransform),
                        })
                crop_id += 1
                roi_offset_px_x += tile_size[0] - tile_overlap
            roi_offset_px_y += tile_size[1] - tile_overlap
        return samples


class ImageFolderGDataset(ImageFolderDataset):
    """Image folder dataset specialization interface for classification tasks on geospatial images.

    This specialization is used to parse simple image subfolders, and it essentially replaces the very
    basic ``torchvision.datasets.ImageFolder`` interface with similar functionalities. It it used to provide
    a proper task interface as well as path metadata in each loaded packet for metrics/logging output.

    The difference with the parent class ImageFolderDataset is the used of gdal to manage multi channels images found
    in remote sensing domain. The user can specify the channels to load. By default the first three channels are
    loaded [1,2,3].

    .. seealso::
        | :class:`thelper.data.parsers.ImageDataset`
        | :class:`thelper.data.parsers.ClassificationDataset`
        | :class:`thelper.data.parsers.ImageFolderDataset`
    """

    def __init__(self, root, transforms=None, image_key="image", label_key="label",
                 path_key="path", idx_key="idx", channels=None):
        """Image folder dataset parser constructor."""

        super(ImageFolderGDataset, self).__init__(root=root, transforms=transforms, image_key=image_key,
                                                  path_key=path_key, label_key=label_key, idx_key=idx_key)
        self.channels = channels if channels else [1, 2, 3]

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        if idx >= len(self.samples):
            raise AssertionError("sample index is out-of-range")
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        raster_path = sample[self.path_key]
        raster_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        if raster_ds is None:
            raise Exception(f"File not found: {raster_path}")

        image = []
        for channel in self.channels:
            image_arr = raster_ds.GetRasterBand(channel).ReadAsArray()
            if image_arr is None:
                logger.fatal(f"Band not found: {channel}")
            image.append(image_arr)
        image = np.dstack(image)
        raster_ds = None  # noqa # flush

        sample = {
            self.image_key: image,
            self.idx_key: idx,
            **sample
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class SlidingWindowDataset(Dataset):
    """Sliding window dataset specialization interface for classification tasks over a geospatial image.

    The dataset runs a sliding window over the whole geospatial image in order to return tile patches.
    The operation can be accomplished over multiple raster bands if they can be found in the provided raster container.
    """
    def __init__(self, raster_path, raster_bands, patch_size, transforms=None, image_key="image"):
        super().__init__(transforms=transforms)
        self.logger.debug("Creating %s with [%s]", type(self).__name__, raster_path)
        self.image_key = image_key
        self.center_key = "center"
        self.raster_dss = []

        # update raster metadata that can be used by other objects
        self.raster = {"path": raster_path, "bands": raster_bands}
        raster_ds = gdal.OpenShared(raster_path, gdal.GA_ReadOnly)
        parse_raster_metadata(self.raster, raster_ds)
        xsize = raster_ds.RasterXSize
        ysize = raster_ds.RasterYSize
        self.patch_size = patch_size
        self.raster["xsize"] = xsize
        self.raster["ysize"] = ysize
        self.raster["georef"] = raster_ds.GetProjectionRef()
        self.raster["affine"] = raster_ds.GetGeoTransform()
        raster_ds = None  # noqa # flush dataset

        # generate patch sample locations
        lines = ysize - self.patch_size
        cols = xsize - self.patch_size
        self.samples = []
        for y in range(lines):
            for x in range(cols):
                self.samples.append((x, y, self.patch_size, self.patch_size))
        self.n_samples = len(self.samples)
        self.logger.info(f"Number of samples: {self.n_samples}")
        self.done = False

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Get the number n of workers and the current worker's id
        info = torch.utils.data.get_worker_info()
        # Open the data with gdal n times in multithread shared mode
        # The operation is done once
        if not self.done:
            raster_path = self.raster.get('reader', self.raster['path'])
            self.logger.info(f"Single time load of raster: [{raster_path}]")
            for _ in range(info.num_workers):
                raster_ds = gdal.OpenShared(raster_path, gdal.GA_ReadOnly)
                self.raster_dss.append(raster_ds)
            self.done = True

        # Do your processing with the gdal dataset associated with the worker's id
        image = []
        patch = self.samples[idx]
        for raster_band in self.raster["bands"]:
            image.append(self.raster_dss[info.id].GetRasterBand(raster_band).ReadAsArray(*patch))
        image = np.dstack(image)
        offsets = patch[:2]
        half_size = self.patch_size // 2
        sample = {
            self.image_key: np.array(image.data, copy=True, dtype='float32'),
            self.center_key: (offsets[0] + half_size, offsets[1] + half_size),
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample
