"""Geospatial data parser & utilities module."""

import hashlib
import json
import logging
import os
import pickle

import cv2 as cv
import gdal
import numpy as np
import ogr
import osr

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
                 feature_key="feature", force_parse=False, transforms=None):
        # before anything else, create a hash to cache parsed data
        cache_hash = hashlib.sha1(str({k: v for k, v in vars().items() if not k.startswith("_") and k != "self"}).encode())
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
        self.skew = (0.0, 0.0) if px_size is None else (float(px_size[0]), float(px_size[1])) \
            if isinstance(px_size, (list, tuple)) else (float(px_size), float(px_size))
        assert isinstance(allow_outlying_vectors, bool), "unexpected flag type"
        assert isinstance(clip_outlying_vectors, bool), "unexpected flag type"
        assert isinstance(force_parse, bool), "unexpected flag type"
        self.allow_outlying = allow_outlying_vectors
        self.clip_outlying = clip_outlying_vectors
        self.force_parse =force_parse
        assert isinstance(vector_area_min, float) and vector_area_min >= 0, "min surface filter value must be > 0"
        assert isinstance(vector_area_max, float) and vector_area_max >= vector_area_min,\
            "max surface filter value must be greater than minimum surface value"
        self.area_min = vector_area_min
        self.area_max = vector_area_max
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
        assert isinstance(feature_key, str), "feature key must be given as string"
        self.feature_key = feature_key
        super().__init__(transforms=transforms)
        logger.info(f"parsing rasters from path '{self.raster_path}'...")
        raster_paths = thelper.utils.get_file_paths(self.raster_path, ".", allow_glob=True)
        self.rasters_data, self.coverage = geo.utils.parse_rasters(raster_paths, self.srs_target)
        assert self.rasters_data, f"could not find any usable rasters at '{raster_paths}'"
        logger.debug(f"rasters total coverage area = {self.coverage.area:.2f}")
        for idx, data in enumerate(self.rasters_data):
            logger.debug(f"raster #{idx + 1} area = {data['target_roi'].area:.2f}")
            # here, we enforce that raster datatypes/bandcounts match
            assert data["band_count"] == self.rasters_data[0]["band_count"], \
                f"parser expects that all raster band counts match" + \
                f"(found {str(data['band_count'])} and {str(self.rasters_data[0]['band_count'])})"
            assert data["data_type"] == self.rasters_data[0]["data_type"], \
                f"parser expects that all raster data types match" + \
                f"(found {str(data['data_type'])} and {str(self.rasters_data[0]['data_type'])})"
            data["to_target_transform"] = osr.CoordinateTransformation(data["srs"], self.srs_target)
            data["from_target_transform"] = osr.CoordinateTransformation(self.srs_target, data["srs"])
        logger.info(f"parsing vectors from path '{self.vector_path}'...")
        assert os.path.isfile(self.vector_path) and self.vector_path.endswith("geojson"), \
            "vector file must be provided as geojson (shapefile support still incomplete)"
        cache_file_path = os.path.join(os.path.dirname(self.vector_path), cache_hash.hexdigest() + ".pkl")
        if not force_parse and os.path.exists(cache_file_path):
            logger.debug(f"parsing cached feature data from '{cache_file_path}'...")
            with open(cache_file_path, "rb") as fd:
                features = pickle.load(fd)
        else:
            with open(self.vector_path) as vector_fd:
                vector_data = json.load(vector_fd)
            features = geo.utils.parse_geojson(vector_data, srs_target=self.srs_target, roi=self.coverage,
                                               allow_outlying=self.allow_outlying, clip_outlying=self.clip_outlying)
            if self.target_prop is not None:
                features = [feature for feature in features
                            if all([k in feature["properties"] and
                                    feature["properties"][k] == v for k, v in self.target_prop.items()])]
            features = [feature for feature in features
                        if self.area_min <= feature["geometry"].area <= self.area_max]
            logger.debug(f"parsed {len(features)} features of interest; preparing rois...")
            if not force_parse:
                logger.debug(f"caching feature data to '{cache_file_path}'...")
                with open(cache_file_path, "wb") as fd:
                    pickle.dump(features, fd)
        self.samples = []  # required attrib name for access via base class funcs
        for feature in features:
            # add ROIs and target crop sizes to pre-parsed features (note: this is in target_srs)
            roi, roi_tl, roi_br, crop_width, crop_height = \
                geo.utils.get_feature_roi(feature["geometry"], self.px_size, self.skew, self.roi_buffer)
            # test all regions that touch the selected feature
            roi_hits, roi_geoms = [], []
            for raster_idx, raster_data in enumerate(self.rasters_data):
                inters_geometry = raster_data["target_roi"].intersection(roi)
                if not inters_geometry.is_empty:
                    assert inters_geometry.geom_type in ["Polygon", "MultiPolygon"], "unexpected inters geom type"
                    roi_hits.append(raster_idx)
                    roi_geoms.append(inters_geometry)
            self.samples.append({
                **feature,
                "roi": roi,
                "roi_tl": roi_tl,
                "roi_br": roi_br,
                "roi_hits": roi_hits,
                "roi_geoms": roi_geoms,
                "crop_width": crop_width,
                "crop_height": crop_height
            })
        meta_keys = [self.feature_key, "srs", "geo_bbox"]
        # create default task without gt specification (this is a pretty basic parser)
        self.task = thelper.tasks.Task(input_key=self.raster_key, meta_keys=meta_keys)
        self.display_debug = True  # temporary @@@@

    def _process_feature(self, feature):
        """Returns a crop for a specific (internal) feature object."""
        crop_width, crop_height = feature["crop_width"], feature["crop_height"]
        roi_tl, roi_br = feature["roi_tl"], feature["roi_br"]
        # remember: we assume that all rasters have the same intrinsic settings
        crop_datatype = geo.utils.GDAL2NUMPY_TYPE_CONV[self.rasters_data[0]["data_type"]]
        crop_size = (crop_height, crop_width, self.rasters_data[0]["band_count"])

        crop = np.ma.array(np.zeros(crop_size, dtype=crop_datatype), mask=np.ones(crop_size, dtype=np.uint8))
        mask = np.zeros(crop_size[0:2], dtype=np.uint8)

        output_geotransform = (feature["roi_tl"][0], self.px_size[0], self.skew[0],
                               feature["roi_tl"][1], self.skew[1], self.px_size[1])
        crop_raster_gdal = gdal.GetDriverByName("MEM").Create("", crop_size[1], crop_size[0],
                                                              crop_size[2], self.rasters_data[0]["data_type"])
        crop_raster_gdal.SetGeoTransform(output_geotransform)
        crop_raster_gdal.SetProjection(self.srs_target.ExportToWkt())
        crop_mask_gdal = gdal.GetDriverByName("MEM").Create("", crop_size[1], crop_size[0], 1, gdal.GDT_Byte)
        crop_mask_gdal.SetGeoTransform(output_geotransform)
        crop_mask_gdal.SetProjection(self.srs_target.ExportToWkt())
        crop_mask_gdal.GetRasterBand(1).WriteArray(np.zeros(crop_size[0:2], dtype=np.uint8))
        ogr_dataset = ogr.GetDriverByName("Memory").CreateDataSource("mask")
        ogr_layer = ogr_dataset.CreateLayer("feature_mask", srs=self.srs_target)
        ogr_feature = ogr.Feature(ogr_layer.GetLayerDefn())
        ogr_geometry = ogr.CreateGeometryFromWkt(feature["geometry"].wkt)
        ogr_feature.SetGeometry(ogr_geometry)
        ogr_layer.CreateFeature(ogr_feature)
        gdal.RasterizeLayer(crop_mask_gdal, [1], ogr_layer, burn_values=[1], options=["ALL_TOUCHED=TRUE"])
        np.copyto(dst=mask, src=crop_mask_gdal.GetRasterBand(1).ReadAsArray())
        for raster_idx, inters_geom in zip(feature["roi_hits"], feature["roi_geoms"]):
            raster_data = self.rasters_data[raster_idx]
            rasterfile = gdal.Open(raster_data["file_path"], gdal.GA_ReadOnly)
            assert rasterfile is not None, f"could not open raster data file at '{raster_data['file_path']}'"
            assert rasterfile.RasterCount == crop_size[2], "unexpected raster count"
            for raster_band_idx in range(rasterfile.RasterCount):
                curr_band = rasterfile.GetRasterBand(raster_band_idx + 1)
                nodataval = curr_band.GetNoDataValue()
                crop_raster_gdal.GetRasterBand(raster_band_idx + 1).WriteArray(
                    np.full(crop_size[0:2], fill_value=nodataval, dtype=crop_datatype))
            res = gdal.ReprojectImage(rasterfile, crop_raster_gdal, raster_data["srs"].ExportToWkt(),
                                      self.srs_target.ExportToWkt(), gdal.GRA_Bilinear)
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
        return crop, mask, np.asarray(list(feature["roi_tl"]) + list(feature["roi_br"]))

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        assert idx < len(self.samples), "sample index is out-of-range"
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        crop, mask, bbox = self._process_feature(sample)
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
            self.feature_key: sample,
            "srs": self.srs_target,
            "geo_bbox": bbox
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample
