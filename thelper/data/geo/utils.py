import json
import logging
import math
import os

import affine
import gdal
import geojson
import numpy as np
import ogr
import osr
import shapely
import shapely.geometry
import shapely.ops
import shapely.wkt
import tqdm

logger = logging.getLogger(__name__)

NUMPY2GDAL_TYPE_CONV = {
    np.uint8: gdal.GDT_Byte,
    np.int8: gdal.GDT_Byte,
    np.uint16: gdal.GDT_UInt16,
    np.int16: gdal.GDT_Int16,
    np.uint32: gdal.GDT_UInt32,
    np.int32: gdal.GDT_Int32,
    np.float32: gdal.GDT_Float32,
    np.float64: gdal.GDT_Float64,
    np.complex64: gdal.GDT_CFloat32,
    np.complex128: gdal.GDT_CFloat64,
}

GDAL2NUMPY_TYPE_CONV = {
    gdal.GDT_Byte: np.uint8,
    gdal.GDT_UInt16: np.uint16,
    gdal.GDT_Int16: np.int16,
    gdal.GDT_UInt32: np.uint32,
    gdal.GDT_Int32: np.int32,
    gdal.GDT_Float32: np.float32,
    gdal.GDT_Float64: np.float64,
    gdal.GDT_CInt16: np.complex64,
    gdal.GDT_CInt32: np.complex64,
    gdal.GDT_CFloat32: np.complex64,
    gdal.GDT_CFloat64: np.complex128
}


def get_pxcoord(geotransform, x, y):
    inv_transform = ~affine.Affine.from_gdal(*geotransform)
    return inv_transform * (x, y)


def get_geocoord(geotransform, x, y):
    # orig_x,res_x,skew_x,orig_y,skew_y,res_y = geotransform
    # return (orig_x+x*res_x+y*skew_x,orig_y+x*skew_y+y*res_y)
    return affine.Affine.from_gdal(*geotransform) * (float(x), float(y))


def get_geoextent(geotransform, x, y, cols, rows):
    tl = get_geocoord(geotransform, x, y)
    bl = get_geocoord(geotransform, x, y + rows)
    br = get_geocoord(geotransform, x + cols, y + rows)
    tr = get_geocoord(geotransform, x + cols, y)
    return [tl, bl, br, tr]


def reproject_coords(coords, src_srs, tgt_srs):
    trans_coords = []
    transform = osr.CoordinateTransformation(src_srs, tgt_srs)
    for x, y in coords:
        x, y, z = transform.TransformPoint(x, y)
        trans_coords.append([x, y])
    return trans_coords


def parse_rasters(raster_paths, srs_target=None, reproj=False):
    # note: the rasters will not be projected in this function if an SRS is given
    assert isinstance(raster_paths, list) and all([isinstance(s, str) for s in raster_paths]), \
        "input raster paths must be provided as a list of strings"
    # remove reprojs from list (will be rediscovered later)
    raster_paths = [r for r in raster_paths if not r.endswith(".reproj.tif")]
    if srs_target is not None:
        assert isinstance(srs_target, (str, int, osr.SpatialReference)), \
            "target EPSG SRS must be given as int/str"
        if isinstance(srs_target, (str, int)):
            if isinstance(srs_target, str):
                srs_target = int(srs_target.replace("EPSG:", ""))
            srs_target_obj = osr.SpatialReference()
            srs_target_obj.ImportFromEPSG(srs_target)
            srs_target = srs_target_obj
    rasters_data = []
    target_rois = []
    for raster_path in raster_paths:
        rasterfile = gdal.Open(raster_path, gdal.GA_ReadOnly)
        assert rasterfile is not None, f"could not open raster data file at '{raster_path}'"
        logger.debug(f"Raster '{raster_path}' metadata printing below...")
        logger.debug(f"{str(rasterfile)}")
        logger.debug(f"{str(rasterfile.GetMetadata())}")
        logger.debug(f"band count: {str(rasterfile.RasterCount)}")
        raster_geotransform = rasterfile.GetGeoTransform()
        px_width, px_height = raster_geotransform[1], raster_geotransform[5]
        logger.debug(f"pixel WxH resolution: {px_width} x {px_height}")
        skew_x, skew_y = raster_geotransform[2], raster_geotransform[4]
        logger.debug(f"grid X/Y skew: {skew_x} / {skew_y}")
        raster_extent = get_geoextent(raster_geotransform, 0, 0, rasterfile.RasterXSize, rasterfile.RasterYSize)
        logger.debug(f"extent: {str(raster_extent)}")  # [tl, bl, br, tr]
        raster_curr_srs = osr.SpatialReference()
        raster_curr_srs_str = rasterfile.GetProjectionRef()
        if "unknown" not in raster_curr_srs_str:
            raster_curr_srs.ImportFromWkt(raster_curr_srs_str)
        else:
            assert srs_target is not None, "raster did not provide an srs, and no target EPSG srs provided"
            raster_curr_srs = srs_target
        logger.debug(f"spatial ref:\n{str(raster_curr_srs)}")
        raster_datatype = None
        for raster_band_idx in range(rasterfile.RasterCount):
            curr_band = rasterfile.GetRasterBand(raster_band_idx + 1)  # offset, starts at 1
            assert curr_band is not None, f"found invalid raster band in '{raster_path}'"
            if not raster_datatype:
                raster_datatype = curr_band.DataType
            assert raster_datatype == curr_band.DataType, "expected identical data types in all bands"
        local_roi = shapely.geometry.Polygon([list(pt) for pt in raster_extent])
        reproj_path = None
        if srs_target is not None and not raster_curr_srs.IsSame(srs_target):
            srs_transform = osr.CoordinateTransformation(raster_curr_srs, srs_target)
            ogr_geometry = ogr.CreateGeometryFromWkb(local_roi.wkb)
            ogr_geometry.Transform(srs_transform)
            target_roi = shapely.wkt.loads(ogr_geometry.ExportToWkt())
            if reproj:
                reproj_path = raster_path + ".reproj.tif"
                if not os.path.exists(reproj_path):
                    logger.info(f"reprojecting raster to '{reproj_path}'...")
                    gdal.Warp(reproj_path, rasterfile, dstSRS=srs_target.ExportToWkt(),
                              outputType=raster_datatype, xRes=px_width, yRes=px_height,
                              callback=lambda *args: logger.debug(f"reprojection @ {int(args[0] * 100)} %"),
                              options=["NUM_THREADS=ALL_CPUS"])
        else:
            target_roi = local_roi
        target_rois.append(target_roi)
        rasters_data.append({
            "srs": raster_curr_srs,
            "geotransform": np.asarray(raster_geotransform),
            "offset_geotransform": np.asarray((0, px_width, skew_x,
                                               0, skew_y, px_height)),
            "extent": raster_extent,
            "skew": (skew_x, skew_y),
            "resolution": (px_width, px_height),
            "band_count": rasterfile.RasterCount,
            "cols": rasterfile.RasterXSize,
            "rows": rasterfile.RasterYSize,
            "data_type": raster_datatype,
            "local_roi": local_roi,
            "target_roi": target_roi,
            "file_path": raster_path,
            "reproj_path": reproj_path
        })
        rasterfile = None  # noqa # close input fd
    target_coverage = shapely.ops.cascaded_union(target_rois)
    return rasters_data, target_coverage


def parse_geojson_crs(body):
    """Imports a coordinate reference system (CRS) from a GeoJSON tree."""
    crs_body = body.get("crs") or body.get("srs")
    crs_type = crs_body.get("type", "").upper()
    crs_opts = list(crs_body.get("properties").values())  # FIXME: no specific mapping of inputs, each is different
    crs = ogr.osr.SpatialReference()
    err = -1
    if crs_type == "EPSG":
        err = crs.ImportFromEPSG(*crs_opts)
    elif crs_type == "EPSGA":
        err = crs.ImportFromEPSGA(*crs_opts)
    elif crs_type == "ERM":
        err = crs.ImportFromERM(*crs_opts)
    elif crs_type == "ESRI":
        err = crs.ImportFromESRI(*crs_opts)
    elif crs_type == "USGS":
        err = crs.ImportFromUSGS(*crs_opts)
    elif crs_type == "PCI":
        err = crs.ImportFromPCI(*crs_opts)
    # add dirty hack for geojsons used in testbed15-d104
    elif crs_type == "NAME" and len(crs_opts) == 1 and ":EPSG:" in crs_opts[0]:
        err = crs.ImportFromEPSG(int(crs_opts[0].split(":")[-1]))
    assert not err, f"could not identify CRS/SRS type ({str(err)})"
    return crs


def parse_geojson(geojson, srs_target=None, roi=None, allow_outlying=False, clip_outlying=False):
    assert isinstance(geojson, dict), "unexpected geojson type (must be dict)"
    assert "features" in geojson and isinstance(geojson["features"], list), "unexpected geojson format"
    features = geojson["features"]
    logger.debug(f"parsing {len(features)} features from geojson...")
    srs_transform = None
    if srs_target is not None:
        assert isinstance(srs_target, (str, int, osr.SpatialReference)), \
            "target EPSG SRS must be given as int/str"
        if isinstance(srs_target, (str, int)):
            if isinstance(srs_target, str):
                srs_target = int(srs_target.replace("EPSG:", ""))
            srs_target_obj = osr.SpatialReference()
            srs_target_obj.ImportFromEPSG(srs_target)
            srs_target = srs_target_obj
        srs_origin = parse_geojson_crs(geojson)
        if not srs_origin.IsSame(srs_target):
            srs_transform = osr.CoordinateTransformation(srs_origin, srs_target)
    kept_features = []
    for feature in tqdm.tqdm(features, desc="parsing raw geojson features"):
        _postproc_feature(feature, kept_features, srs_transform, roi, allow_outlying, clip_outlying)
    logger.debug(f"kept {len(kept_features)} features after roi validation")
    return kept_features


def parse_shapefile(shapefile_path, srs_target=None, roi=None, allow_outlying=False, clip_outlying=False, layer_id=0):
    assert isinstance(shapefile_path, str), "unexpected shapefile path type (must be str)"
    shapefile = ogr.Open(shapefile_path, 0)  # 0 = read only?
    assert shapefile is not None, f"failed to open shapefile at '{shapefile_path}'"
    layer = shapefile.GetLayer(layer_id)
    assert layer.GetFeatureCount() > 0, "target shapefile layer did not possess any feature"
    logger.debug(f"parsing {layer.GetFeatureCount()} features from shapefile...")
    srs_transform = None
    if srs_target is not None:
        assert isinstance(srs_target, (str, int, osr.SpatialReference)), \
            "target EPSG SRS must be given as int/str"
        if isinstance(srs_target, (str, int)):
            if isinstance(srs_target, str):
                srs_target = int(srs_target.replace("EPSG:", ""))
            srs_target_obj = osr.SpatialReference()
            srs_target_obj.ImportFromEPSG(srs_target)
            srs_target = srs_target_obj
        srs_origin = layer.GetSpatialRef()
        if not srs_origin.IsSame(srs_target):
            srs_transform = osr.CoordinateTransformation(srs_origin, srs_target)
    kept_features = []
    attribs = [layer.GetLayerDefn().GetFieldDefn(i) for i in range(layer.GetLayerDefn().GetFieldCount())]
    for feature in tqdm.tqdm(layer, desc="parsing raw shapefile features"):
        feature = {
            "geometry": shapely.wkt.loads(feature.GetGeometryRef().ExportToWkt()),
            "properties": {attribs[i].GetNameRef(): feature.GetField(i) for i in range(len(attribs))},
        }
        _postproc_feature(feature, kept_features, srs_transform, roi, allow_outlying, clip_outlying)
    logger.debug(f"kept {len(kept_features)} features after roi validation")
    return kept_features


def _postproc_feature(feature, kept_features, srs_transform=None, roi=None, allow_outlying=False, clip_outlying=False):
    if isinstance(feature["geometry"], dict):
        assert feature["geometry"]["type"] in ["Polygon", "MultiPolygon"], \
            f"unhandled raw geometry type: {feature['geometry']['type']}"
        if feature["geometry"]["type"] == "Polygon":
            coords = feature["geometry"]["coordinates"]
            assert isinstance(coords, list), "unexpected poly coords type"
            assert len(coords) == 1, "unexpected coords embedding; should be list-of-list-of-points w/ unique ring"
            assert all([isinstance(c, list) and len(c) == 2 for c in coords[0]]) and len(coords[0]) >= 4, \
                "unexpected poly coord format"
            poly = shapely.geometry.Polygon(coords[0])
            if srs_transform is not None:
                ogr_geometry = ogr.CreateGeometryFromWkb(poly.wkb)
                ogr_geometry.Transform(srs_transform)
                poly = shapely.wkt.loads(ogr_geometry.ExportToWkt())
            feature["geometry"] = poly
            feature["type"] = "Polygon"
        elif feature["geometry"]["type"] == "MultiPolygon":
            multipoly = shapely.geometry.shape(feature["geometry"])
            assert multipoly.is_valid, "found invalid input multipolygon"
            if srs_transform is not None:
                ogr_geometry = ogr.CreateGeometryFromWkb(multipoly.wkb)
                ogr_geometry.Transform(srs_transform)
                multipoly = shapely.wkt.loads(ogr_geometry.ExportToWkt())
            feature["geometry"] = multipoly
            feature["type"] = "MultiPolygon"
    assert isinstance(feature["geometry"], (shapely.geometry.polygon.Polygon,
                                            shapely.geometry.multipolygon.MultiPolygon))
    bounds = feature["geometry"].bounds
    feature["tl"] = bounds[0:2]
    feature["br"] = bounds[2:4]
    feature["clipped"] = False
    feature["centroid"] = feature["geometry"].centroid
    if roi is None:
        kept_features.append(feature)
    else:
        if (allow_outlying and roi.intersects(feature["geometry"])) or \
                (not allow_outlying and roi.contains(feature["geometry"])):
            if clip_outlying:
                if not roi.contains(feature["geometry"]):
                    feature["clipped"] = True
                    feature["geometry"] = roi.intersection(feature["geometry"])
                assert feature["geometry"].type in ["Polygon", "MultiPolygon"], \
                    f"unhandled intersection geometry type: {feature['geometry'].type}"
            feature["type"] = feature["geometry"].type
            kept_features.append(feature)


def parse_roi(roi_path, srs_target=None):
    assert isinstance(roi_path, str), "input path type should be string"
    assert os.path.exists(roi_path), f"invalid roi path '{roi_path}'"
    if roi_path.lower().endswith("geojson"):
        with open(roi_path) as roi_fd:
            features = parse_geojson(json.load(roi_fd), srs_target=srs_target)
    elif roi_path.lower().endswith("shp"):
        features = parse_shapefile(roi_path, srs_target=srs_target)
    else:
        raise AssertionError("unexpected roi file type")
    return shapely.ops.cascaded_union([f["geometry"] for f in features])


def get_feature_bbox(geom, offsets=None):
    if offsets and len(offsets) != 2:
        raise AssertionError("offset param must be 2d")
    bounds = geom.bounds
    if offsets:
        centroid = geom.centroid
        roi_tl = (centroid.x - offsets[0], centroid.y + offsets[1])
        roi_br = (centroid.x + offsets[0], centroid.y - offsets[1])
    else:
        roi_tl = (bounds[0], bounds[3])
        roi_br = (bounds[2], bounds[1])
    return (min(roi_tl[0], roi_br[0]), min(roi_tl[1], roi_br[1])), \
           (max(roi_tl[0], roi_br[0]), max(roi_tl[1], roi_br[1]))


def get_feature_roi(geom, px_size, skew, roi_buffer=None, crop_img_size=None, crop_real_size=None):
    assert crop_img_size is None or crop_real_size is None, "should provide at most one type of crop resolution"
    assert isinstance(px_size, (list, tuple)) and len(px_size) == 2, "px size should be x/y tuple"
    assert isinstance(skew, (list, tuple)) and len(px_size) == 2, "skew should be x/y tuple"
    assert roi_buffer is None or isinstance(roi_buffer, (float, int)), "unexpected roi buffer value type"
    if roi_buffer is not None:
        geom = geom.buffer(roi_buffer)  # expand geometry context if required
    offset_geotransform = (0, px_size[0], skew[0], 0, skew[1], px_size[1])
    if crop_img_size or crop_real_size:
        if crop_img_size:
            crop_size = int(crop_img_size)
            x_offset, y_offset = get_geocoord(offset_geotransform, crop_size, crop_size)
            x_offset, y_offset = abs(x_offset / 2), abs(y_offset / 2)
        elif crop_real_size:
            x_offset = y_offset = float(crop_real_size) / 2
        else:
            raise ValueError()
        roi_tl, roi_br = get_feature_bbox(geom, (x_offset, y_offset))
    else:
        roi_tl, roi_br = get_feature_bbox(geom)
    roi_tl_offsetpx_real = get_pxcoord(offset_geotransform, roi_tl[0], roi_tl[1])
    roi_tl_offsetpx = (int(math.floor(roi_tl_offsetpx_real[0])), int(math.floor(roi_tl_offsetpx_real[1])))
    if crop_img_size:
        crop_width = crop_height = int(crop_img_size)
        roi_br_offsetpx = (roi_tl_offsetpx[0] + crop_width, roi_tl_offsetpx[1] + crop_height)
    else:
        roi_br_offsetpx_real = get_pxcoord(offset_geotransform, roi_br[0], roi_br[1])
        roi_br_offsetpx = (int(math.ceil(roi_br_offsetpx_real[0])), int(math.ceil(roi_br_offsetpx_real[1])))
        crop_width = max(roi_br_offsetpx[0] - roi_tl_offsetpx[0], 1)
        crop_height = max(roi_br_offsetpx[1] - roi_tl_offsetpx[1], 1)  # may sometimes be swapped if px res is negative
    roi_tl = get_geocoord(offset_geotransform, roi_tl_offsetpx[0], roi_tl_offsetpx[1])
    roi_br = get_geocoord(offset_geotransform, roi_br_offsetpx[0], roi_br_offsetpx[1])
    roi = shapely.geometry.Polygon([roi_tl, (roi_br[0], roi_tl[1]), roi_br, (roi_tl[0], roi_br[1])])
    return roi, roi_tl, roi_br, crop_width, crop_height


def open_rasterfile(raster_data, keep_rasters_open=False):
    assert isinstance(raster_data, dict), "unexpected raster data type (should be internal dict)"
    if "rasterfile" in raster_data:
        rasterfile = raster_data["rasterfile"]
    else:
        if raster_data["reproj_path"] is not None:
            raster_path = raster_data["reproj_path"]
        else:
            raster_path = raster_data["file_path"]
        rasterfile = gdal.Open(raster_path, gdal.GA_ReadOnly)
        assert rasterfile is not None, f"could not open raster data file at '{raster_path}'"
        if keep_rasters_open:
            raster_data["rasterfile"] = rasterfile
    return rasterfile


def reproject_crop(raster, crop_raster, crop_size, crop_datatype,
                   crop_nodataval=None, reproj_opt=None, fill_nodata=False):
    if fill_nodata:
        for raster_band_idx in range(raster.RasterCount):
            curr_band = raster.GetRasterBand(raster_band_idx + 1)
            if crop_nodataval is None:
                crop_nodataval = curr_band.GetNoDataValue()
            else:
                assert crop_nodataval == curr_band.GetNoDataValue()
            crop_raster.GetRasterBand(raster_band_idx + 1).WriteArray(
                np.full([int(round(c)) for c in crop_size[0:2]],
                        fill_value=crop_nodataval, dtype=crop_datatype))
    res = gdal.ReprojectImage(raster, crop_raster, raster.GetProjectionRef(),
                              crop_raster.GetProjectionRef(), gdal.GRA_Bilinear,
                              options=[] if not reproj_opt else reproj_opt)
    assert res == 0, "reprojection failed"


def export_geotiff(filepath, crop, srs, geotransform):
    assert isinstance(filepath, str), "filepath should be given as string"
    assert isinstance(crop, np.ndarray), "crop data should be given as numpy array"
    assert crop.ndim == 2 or crop.ndim == 3, "crop array should be 2D or 3D"
    assert isinstance(srs, (str, int, osr.SpatialReference)), "target EPSG SRS must be given as int/str"
    if isinstance(srs, (str, int)):
        if isinstance(srs, str):
            srs = int(srs.replace("EPSG:", ""))
        srs_obj = osr.SpatialReference()
        srs_obj.ImportFromEPSG(srs)
        srs = srs_obj
    assert isinstance(geotransform, (list, tuple, np.ndarray)) and len(geotransform) == 6, \
        "geotransform should be given as array of [x, px_w, sk_x, y, sk_y, px_h]"
    raster_size = crop.shape[1], crop.shape[0]
    raster_bands = crop.shape[2]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filepath, *raster_size, raster_bands, NUMPY2GDAL_TYPE_CONV[crop.dtype])
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection(srs.ExportToWkt())
    for b in range(raster_bands):
        dataset.GetRasterBand(b + 1).WriteArray(crop[:, :, b])
    dataset.FlushCache()
    dataset = None  # noqa # close fd


def export_geojson_with_crs(features, srs_target):
    """
    Exports a list of features along with their SRS into a GeoJSON-compat string.
    """

    class _FeatureCollection(geojson.FeatureCollection):
        def __init__(self, *args, srs="4326", **kwargs):
            assert isinstance(srs, (str, int, osr.SpatialReference)), \
                "target EPSG SRS must be given as int/str"
            if isinstance(srs, osr.SpatialReference):
                assert srs.GetAttrValue("AUTHORITY", 0) == "EPSG", \
                    "current implementation only supports EPSG spatial refs"
                srs = srs.GetAttrValue("AUTHORITY", 1)
            super().__init__(*args, **kwargs)
            self.srs = srs

        @property
        def __geo_interface__(self):
            res = super().__geo_interface__
            res.update({
                "type": "FeatureCollection",
                "crs": {"type": "EPSG", "properties": {"code": self.srs}}})
            return res

    assert isinstance(features, list), "unexpected feature list type"
    return geojson.dumps(_FeatureCollection(features, srs=srs_target), indent=2)


def parse_raster_metadata(raster_metadata, raster_dataset=None):
    # type: (dict, gdal.Dataset) -> None
    """Parses the provided raster metadata and updates it by adding extra details required for later use.

    The provided raster metadata is updated directly. Metadata is validated against the matching data storage.
    If any important, required or requested (bands) metadata is missing, the function raises the issue immediately.

    Args:
        raster_metadata (dict):
            raster metadata dictionary with minimally a file 'path' and list of 'bands' indices to process.
        raster_dataset (gdal.Dataset):
            (optional) preloaded dataset object corresponding to the raster metadata.

    Raises:
        ValueError: at least one input raster was missing a required metadata parameter or a parameter is erroneous.
        IOError: the raster path could not be found or reading it did not generate a valid raster using GDAL.
    """
    for key in ["path", "bands"]:
        if key not in raster_metadata:
            raise ValueError(f"Missing raster '{key}' in metadata")
    raster_path = raster_metadata['path']
    raster_bands = raster_metadata['bands']
    raster_metadata['reader'] = raster_path

    local_open = False
    if not raster_dataset:
        local_open = True
        raster_dataset = gdal.OpenShared(raster_path, gdal.GA_ReadOnly)
    if raster_dataset is None:
        logger.fatal(f"File not found: {raster_path}")
        raise IOError(f"Missing raster file could not be loaded by GDAL: [{raster_path}]")

    driver_shortname = raster_dataset.GetDriver().ShortName
    raster_metadata['format'] = 'DEFAULT'
    if driver_shortname == 'SENTINEL2':
        got_md = raster_dataset.GetMetadata('SUBDATASETS')
        if got_md is None:
            logger.fatal(f"Missing metadata: {raster_path}")
            raise ValueError("Missing raster metadata with expected Sentinel-2 format")
        raster_path = got_md["SUBDATASET_1_NAME"]
        # path has prefix and can have suffix such that path is formatted as:
        #   '<sensor-info>:<data-path>[:<resolution>:<csr>]'
        raster_metadata['path'] = os.path.dirname(raster_path.split(":", 1)[1])
        raster_dataset = gdal.OpenShared(raster_path, gdal.GA_ReadOnly)
        if raster_dataset is None:
            logger.fatal(f"File not found: [{raster_path}]")
            raise IOError(f"Invalid raster sub-dataset could not be loaded by GDAL: [{raster_path}]")
        raster_metadata['format'] = 'SENTINEL2'
        raster_metadata['reader'] = raster_path  # apply full sub-dataset reader path

    # in case explicit desired name provided, use it, otherwise find appropriate one from updated path
    raster_name = raster_metadata.get('name')
    if not raster_name:
        raster_metadata['name'] = os.path.split(raster_metadata['path'])[-1].split('.')[0]

    for b in raster_bands:
        raster_band = raster_dataset.GetRasterBand(b)
        if raster_band is None:
            logger.fatal(f"Raster band {b} not found: [{raster_path}]")
            raise ValueError("Invalid raster band missing")
        else:
            logger.debug(f"Using band {b} in [{raster_path}]")
    if local_open:
        raster_dataset = None  # noqa # flush dataset
