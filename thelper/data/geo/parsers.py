"""Geospatial data parser & utilities module."""

import functools
import json
import logging
import os
import pickle

import cv2 as cv
import gdal
import numpy as np
import ogr
import osr
import tqdm

import thelper.data
import thelper.data.geo as geo

logger = logging.getLogger(__name__)


class VectorCropDataset(thelper.data.Dataset):
    """Abstract dataset used to combine geojson vector data and rasters."""

    def __init__(self, raster_path, vector_path, px_size=None, skew=None,
                 allow_outlying_vectors=True, clip_outlying_vectors=True,
                 vector_area_min=0.0, vector_area_max=float("inf"),
                 vector_target_prop=None, vector_roi_buffer=None,
                 srs_target="3857", raster_key="raster", mask_key="mask",
                 cleaner=None, cropper=None, force_parse=False,
                 reproj_rasters=False, reproj_all_cpus=True,
                 keep_rasters_open=True, transforms=None):
        # before anything else, create a hash to cache parsed data
        cache_hash = thelper.utils.get_params_hash(
            {k: v for k, v in vars().items() if not k.startswith("_") and k != "self"}) if not force_parse else None
        assert isinstance(raster_path, str), f"raster file/folder path should be given as string"
        assert isinstance(vector_path, str), f"vector file/folder path should be given as string"
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
        assert vector_roi_buffer is None or (isinstance(vector_roi_buffer, (int, float)) and vector_roi_buffer > 0), \
            "feature roi 'buffer' value should be strictly positive int/float"
        self.roi_buffer = vector_roi_buffer
        assert isinstance(srs_target, (str, int, osr.SpatialReference)), \
            "target EPSG SRS must be given as int/str"
        self.srs_target = srs_target
        if isinstance(self.srs_target, (str, int)):
            if isinstance(self.srs_target, str):
                self.srs_target = int(self.srs_target.replace("EPSG:", ""))
            srs_target_obj = osr.SpatialReference()
            srs_target_obj.ImportFromEPSG(self.srs_target)
            self.srs_target = srs_target_obj
        assert isinstance(raster_key, str), "raster key must be given as string"
        self.raster_key = raster_key
        assert isinstance(mask_key, str), "mask key must be given as string"
        self.mask_key = mask_key
        super().__init__(transforms=transforms)
        self.rasters_data, self.coverage = self._parse_rasters(self.raster_path, self.srs_target, reproj_rasters)
        if cleaner is None:
            cleaner = functools.partial(self._default_feature_cleaner, area_min=self.area_min,
                                        area_max=self.area_max, target_prop=self.target_prop)
        self.features = self._parse_features(self.vector_path, self.srs_target, self.coverage, cache_hash,
                                             self.allow_outlying, self.clip_outlying, cleaner)
        if cropper is None:
            cropper = functools.partial(self._default_feature_cropper, px_size=self.px_size,
                                        skew=self.skew, roi_buffer=self.roi_buffer)
        self.samples = self._parse_crops(self.features, self.rasters_data, cropper, self.vector_path, cache_hash)
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
    def _default_feature_cropper(features, rasters_data, px_size, skew, roi_buffer):
        """Returns the samples for a set of features (may be modified in derived classes)."""
        # note: default behavior = just center on the feature, and pad if required by user
        samples = []
        for feature in tqdm.tqdm(features, desc="preparing crop regions"):
            assert "clean" in feature, "cleaner should have added 'clean' flag to each feature"
            if not feature["clean"]:
                continue  # skip (will not use bad features as the origin of a 'sample')
            roi, roi_tl, roi_br, crop_width, crop_height = \
                geo.utils.get_feature_roi(feature["geometry"], px_size, skew, roi_buffer)
            # test all raster regions that touch the selected feature
            roi_hits = []
            for raster_idx, raster_data in enumerate(rasters_data):
                if raster_data["target_roi"].intersects(roi):
                    roi_hits.append(raster_idx)
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
                "roi_hits": roi_hits,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "geotransform": (roi_tl[0], px_size[0], skew[0],
                                 roi_tl[1], skew[1], px_size[1]),
            })
        return samples

    @staticmethod
    def _parse_rasters(path, srs, reproj_rasters):
        """Parses rasters (geotiffs) and returns metadata/coverage information."""
        logger.info(f"parsing rasters from path '{path}'...")
        raster_paths = thelper.utils.get_file_paths(path, ".", allow_glob=True)
        rasters_data, coverage = geo.utils.parse_rasters(raster_paths, srs, reproj_rasters)
        assert rasters_data, f"could not find any usable rasters at '{raster_paths}'"
        logger.debug(f"rasters total coverage area = {coverage.area:.2f}")
        for idx, data in enumerate(rasters_data):
            logger.debug(f"raster #{idx + 1} area = {data['target_roi'].area:.2f}")
            # here, we enforce that raster datatypes/bandcounts match
            assert data["band_count"] == rasters_data[0]["band_count"], \
                f"parser expects that all raster band counts match" + \
                f"(found {str(data['band_count'])} and {str(rasters_data[0]['band_count'])})"
            assert data["data_type"] == rasters_data[0]["data_type"], \
                f"parser expects that all raster data types match" + \
                f"(found {str(data['data_type'])} and {str(rasters_data[0]['data_type'])})"
            data["to_target_transform"] = osr.CoordinateTransformation(data["srs"], srs)
            data["from_target_transform"] = osr.CoordinateTransformation(srs, data["srs"])
        return rasters_data, coverage

    @staticmethod
    def _parse_features(path, srs, roi, cache_hash, allow_outlying, clip_outlying, cleaner):
        """Parses vector files (geojsons) and returns geometry information."""
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

    def _parse_crops(self, features, rasters_data, cropper, path, cache_hash):
        """Parses crops based on prior feature/raster data.

        Each 'crop' corresponds to a sample that can be loaded at runtime.
        """
        logger.info(f"preparing crops using {len(features)} features (total)...")
        cache_file_path = os.path.join(os.path.dirname(path), cache_hash + ".crops.pkl") \
            if cache_hash else None
        if cache_file_path is not None and os.path.exists(cache_file_path):
            logger.debug(f"parsing cached crop data from '{cache_file_path}'...")
            with open(cache_file_path, "rb") as fd:
                samples = pickle.load(fd)
        else:
            samples = cropper(features, rasters_data)
            srs_target_wkt = self.srs_target.ExportToWkt()
            for sample in samples:
                assert "srs" not in sample, "target srs must be added by _parse_crops"
                sample["srs"] = srs_target_wkt
            if cache_file_path is not None:
                logger.debug(f"caching crop data to '{cache_file_path}'...")
                with open(cache_file_path, "wb") as fd:
                    pickle.dump(samples, fd)
        return samples

    def _process_crop(self, sample):
        """Returns a crop for a specific (internal) set of sampled features."""
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
        for raster_idx in sample["roi_hits"]:
            raster_data = self.rasters_data[raster_idx]
            if "rasterfile" in raster_data:
                rasterfile = raster_data["rasterfile"]
            else:
                if raster_data["reproj_path"] is not None:
                    raster_path = raster_data["reproj_path"]
                else:
                    raster_path = raster_data["file_path"]
                rasterfile = gdal.Open(raster_path, gdal.GA_ReadOnly)
                assert rasterfile is not None, f"could not open raster data file at '{raster_path}'"
                if self.keep_rasters_open:
                    raster_data["rasterfile"] = rasterfile
            assert rasterfile.RasterCount == crop_size[2], "unexpected raster count"
            for raster_band_idx in range(rasterfile.RasterCount):
                curr_band = rasterfile.GetRasterBand(raster_band_idx + 1)
                nodataval = curr_band.GetNoDataValue()
                crop_raster_gdal.GetRasterBand(raster_band_idx + 1).WriteArray(
                    np.full(crop_size[0:2], fill_value=nodataval, dtype=crop_datatype))
            # using all cpus should be ok since we probably cant parallelize this loader anyway (swig serialization issues)
            options = ["NUM_THREADS=ALL_CPUS"] if self.reproj_all_cpus else []
            res = gdal.ReprojectImage(rasterfile, crop_raster_gdal, rasterfile.GetProjectionRef(),
                                      self.srs_target.ExportToWkt(), gdal.GRA_Bilinear, options=options)
            assert res == 0, "resampling failed"
            for raster_band_idx in range(crop_raster_gdal.RasterCount):
                curr_band = crop_raster_gdal.GetRasterBand(raster_band_idx + 1)
                curr_band_array = curr_band.ReadAsArray()
                flag_mask = curr_band_array != curr_band.GetNoDataValue()
                np.copyto(dst=crop.data[:, :, raster_band_idx], src=curr_band_array, where=flag_mask)
                np.bitwise_and(crop.mask[:, :, raster_band_idx], np.invert(flag_mask), out=crop.mask[:, :, raster_band_idx])
        # ogr_dataset = None # close local fd
        # noinspection PyUnusedLocal
        crop_raster_gdal = None  # close local fd
        # noinspection PyUnusedLocal
        crop_mask_gdal = None  # close local fd
        # noinspection PyUnusedLocal
        rasterfile = None  # close input fd
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
