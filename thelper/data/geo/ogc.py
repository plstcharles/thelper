"""Data parsers & utilities module for OGC-related projects."""

import copy
import functools
import logging

import cv2 as cv
import numpy as np
import tqdm

import thelper.tasks
import thelper.utils
from thelper.data.geo.parsers import TileDataset, VectorCropDataset
from thelper.train.utils import DetectLogger

logger = logging.getLogger(__name__)


class TB15D104:
    """Wrapper class for OGC Testbed-15 (D104) identifiers."""

    TYPECE_RIVER = "10"
    TYPECE_LAKE = "21"

    BACKGROUND_ID = 0
    LAKE_ID = 1


class TB15D104Dataset(VectorCropDataset):
    """OGC Testbed-15 dataset parser for D104 (lake/river) segmentation task."""

    def __init__(self, raster_path, vector_path, px_size=None,
                 allow_outlying_vectors=True, clip_outlying_vectors=True,
                 lake_area_min=0.0, lake_area_max=float("inf"),
                 lake_river_max_dist=float("inf"), feature_buffer=1000,
                 master_roi=None, focus_lakes=True, srs_target="3857", force_parse=False,
                 reproj_rasters=False, reproj_all_cpus=True, display_debug=False,
                 keep_rasters_open=True, parallel=False, transforms=None):
        assert isinstance(lake_river_max_dist, (float, int)) and lake_river_max_dist >= 0, "unexpected dist type"
        self.lake_river_max_dist = float(lake_river_max_dist)
        assert isinstance(focus_lakes, bool), "unexpected flag type"
        self.focus_lakes = focus_lakes
        assert px_size is None or isinstance(px_size, (float, int)), "pixel size (resolution) must be float/int"
        px_size = (1.0, 1.0) if px_size is None else (float(px_size), float(px_size))
        # note: we wrap partial static functions for caching to see when internal parameters are changing
        cleaner = functools.partial(self.lake_cleaner, area_min=lake_area_min, area_max=lake_area_max,
                                    lake_river_max_dist=lake_river_max_dist, parallel=parallel)
        if self.focus_lakes:
            cropper = functools.partial(self.lake_cropper, px_size=px_size, skew=(0.0, 0.0),
                                        feature_buffer=feature_buffer, parallel=parallel)
        else:
            # TODO: implement river-focused cropper (i.e. river-length parsing?)
            raise NotImplementedError
        super().__init__(raster_path=raster_path, vector_path=vector_path, px_size=px_size, skew=None,
                         allow_outlying_vectors=allow_outlying_vectors, clip_outlying_vectors=clip_outlying_vectors,
                         vector_area_min=lake_area_min, vector_area_max=lake_area_max, vector_target_prop=None,
                         feature_buffer=feature_buffer, master_roi=master_roi, srs_target=srs_target,
                         raster_key="lidar", mask_key="hydro", cleaner=cleaner, cropper=cropper,
                         force_parse=force_parse, reproj_rasters=reproj_rasters, reproj_all_cpus=reproj_all_cpus,
                         keep_rasters_open=keep_rasters_open, transforms=transforms)
        meta_keys = self.task.meta_keys
        if "bboxes" in meta_keys:
            del meta_keys[meta_keys.index("bboxes")]  # placed in meta list by base class constr, moved to detect target below
        self.task = thelper.tasks.Detection(class_names={"background": TB15D104.BACKGROUND_ID, "lake": TB15D104.LAKE_ID},
                                            input_key="input", bboxes_key="bboxes",
                                            meta_keys=meta_keys, background=0, color_map={"lake": [255, 0, 0]})
        # update all already-created bboxes with new task ref
        for s in self.samples:
            for b in s["bboxes"]:
                b.task = self.task
        self.display_debug = display_debug
        self.parallel = parallel

    @staticmethod
    def lake_cleaner(features, area_min, area_max, lake_river_max_dist, parallel=False):
        """Flags geometric features as 'clean' based on type and distance to nearest river."""
        # note: we use a flag here instead of removing bad features so that end-users can still use them if needed
        for f in features:
            f["clean"] = False  # flag every as 'bad' by default, clear just the ones of interest below
        rivers = [f for f in features if f["properties"]["TYPECE"] == TB15D104.TYPECE_RIVER]
        lakes = [f for f in features if f["properties"]["TYPECE"] == TB15D104.TYPECE_LAKE]
        logger.info(f"labeling and cleaning {len(lakes)} lakes...")

        def clean_lake(lake):
            if area_min <= lake["geometry"].area <= area_max:
                if lake_river_max_dist == float("inf"):
                    return True
                else:
                    for river in rivers:
                        # note: distance check below seems to be "intelligent", i.e. it will
                        # first check bbox distance, then check chull distance, and finally use
                        # the full geometries (doing these steps here explicitly makes it slower)
                        if lake["geometry"].distance(river["geometry"]) < lake_river_max_dist:
                            return True
            return False

        if parallel:
            if not isinstance(parallel, int):
                import multiprocessing
                parallel = multiprocessing.cpu_count()
            assert parallel > 0, "unexpected min core count"
            import joblib
            flags = joblib.Parallel(n_jobs=parallel)(joblib.delayed(
                clean_lake)(lake) for lake in tqdm.tqdm(lakes, desc="labeling + cleaning lakes"))
            for flag, lake in zip(flags, lakes):
                lake["clean"] = flag
        else:
            for lake in tqdm.tqdm(lakes, desc="labeling + cleaning lakes"):
                lake["clean"] = clean_lake(lake)
        return features

    @staticmethod
    def lake_cropper(features, rasters_data, coverage, srs_target, px_size, skew, feature_buffer, parallel=False):
        """Returns the ROI information for a given feature (may be modified in derived classes)."""
        srs_target_wkt = srs_target.ExportToWkt()

        def crop_feature(feature):
            import thelper.data.geo as geo
            assert feature["clean"]  # should not get here with bad features
            roi, roi_tl, roi_br, crop_width, crop_height = \
                geo.utils.get_feature_roi(feature["geometry"], px_size, skew, feature_buffer)
            roi_geotransform = (roi_tl[0], px_size[0], skew[0],
                                roi_tl[1], skew[1], px_size[1])
            # test all raster regions that touch the selected feature
            raster_hits = []
            for raster_idx, raster_data in enumerate(rasters_data):
                if raster_data["target_roi"].intersects(roi):
                    raster_hits.append(raster_idx)
            # make list of all other features that may be included in the roi
            roi_centroid = feature["centroid"]
            roi_radius = np.linalg.norm(np.asarray(roi_tl) - np.asarray(roi_br)) / 2
            roi_features, bboxes = [], []
            # note: the 'image id' is in fact the id of the focal feature in the crop
            image_id = int(feature["properties"]["OBJECTID"])
            for f in features:
                # note: here, f may not be 'clean', test anyway
                if f["geometry"].distance(roi_centroid) > roi_radius:
                    continue
                inters = f["geometry"].intersection(roi)
                if inters.is_empty:
                    continue
                roi_features.append(f)
                if f["properties"]["TYPECE"] == TB15D104.TYPECE_RIVER:
                    continue
                # only lakes can generate bboxes; make sure to clip them to the roi bounds
                clip = f["clipped"] or not inters.equals(f["geometry"])
                if clip:
                    assert inters.geom_type in ["Polygon", "MultiPolygon"], "unexpected inters type"
                    corners = []
                    if inters.geom_type == "Polygon":
                        bounds = inters.bounds
                        corners.append((bounds[0:2], bounds[2:4]))
                    elif inters.geom_type == "MultiPolygon":
                        for poly in inters:
                            bounds = poly.bounds
                            corners.append((bounds[0:2], bounds[2:4]))
                else:
                    corners = [(f["tl"], f["br"])]
                for c in corners:
                    feat_tl_px = geo.utils.get_pxcoord(roi_geotransform, *c[0])
                    feat_br_px = geo.utils.get_pxcoord(roi_geotransform, *c[1])
                    bbox = [max(0, feat_tl_px[0]), max(0, feat_tl_px[1]),
                            min(crop_width - 1, feat_br_px[0]),
                            min(crop_height - 1, feat_br_px[1])]
                    if bbox[2] - bbox[0] <= 1 or bbox[3] - bbox[1] <= 1:
                        continue  # skip all bboxes smaller than 1 px (c'mon...)
                    # note: lake class id is 1 by definition
                    bboxes.append(thelper.tasks.detect.BoundingBox(TB15D104.LAKE_ID,
                                                                   bbox=bbox,
                                                                   include_margin=False,
                                                                   truncated=clip,
                                                                   image_id=image_id))
            # prepare actual 'sample' for crop generation at runtime
            return {
                "features": roi_features,
                "bboxes": bboxes,
                "focal": feature,
                "id": image_id,
                "roi": roi,
                "roi_tl": roi_tl,
                "roi_br": roi_br,
                "raster_hits": raster_hits,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "geotransform": np.asarray(roi_geotransform),
                "srs": srs_target_wkt,
            }

        clean_feats = [f for f in features if f["clean"]]
        if parallel:
            if not isinstance(parallel, int):
                import multiprocessing
                parallel = multiprocessing.cpu_count()
            assert parallel > 0, "unexpected min core count"
            import joblib
            samples = joblib.Parallel(n_jobs=parallel)(joblib.delayed(
                crop_feature)(feat) for feat in tqdm.tqdm(clean_feats, desc="preparing crop regions"))
        else:
            samples = []
            for feature in tqdm.tqdm(clean_feats, desc="validating crop candidates"):
                samples.append(crop_feature(feature))
        return [s for s in samples if s is not None]

    def _show_stats_plots(self, show=False, block=False):
        """Draws and returns feature stats histograms using pyplot."""
        import matplotlib.pyplot as plt
        feature_categories = {}
        for feat in self.features:
            curr_cat = feat["properties"]["TYPECE"]
            if curr_cat not in feature_categories:
                feature_categories[curr_cat] = []
            feature_categories[curr_cat].append(feat)
        fig, axes = plt.subplots(len(feature_categories))
        for idx, (cat, features) in enumerate(feature_categories.items()):
            areas = [f["geometry"].area for f in features]
            axes[idx].hist(areas, density=True, bins=30,
                           range=(max(self.area_min, min(areas)), min(self.area_max, max(areas))))
            axes[idx].set_xlabel("Surface (m^2)")
            axes[idx].set_title(f"TYPECE = {cat}")
            axes[idx].set_xlim(xmin=0)
        if show:
            fig.show()
            if block:
                plt.show(block=block)
                return fig
            plt.pause(0.5)
        return fig, axes

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        assert idx < len(self.samples), "sample index is out-of-range"
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        crop, mask = self._process_crop(sample)
        assert crop.shape[2] == 1, "unexpected lidar raster band count"
        crop = crop[:, :, 0]
        dmap = cv.distanceTransform(np.where(mask, np.uint8(0), np.uint8(255)), cv.DIST_L2, cv.DIST_MASK_PRECISE)
        dmap_inv = cv.distanceTransform(np.where(mask, np.uint8(255), np.uint8(0)), cv.DIST_L2, cv.DIST_MASK_PRECISE)
        dmap = np.where(mask, -dmap_inv, dmap)
        dmap *= self.px_size[0]  # constructor enforces same px width/height size
        # dc mask is crop.mask, but most likely lost below
        if self.display_debug:
            crop = cv.normalize(crop, dst=crop, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                dtype=cv.CV_8U, mask=(~crop.mask).astype(np.uint8))
            mask = cv.normalize(mask, dst=mask, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            dmap = cv.normalize(dmap, dst=dmap, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        sample = {
            "input": np.stack([crop, mask, dmap], axis=-1),
            # note: bboxes are automatically added in the "cropper" preprocessing function
            "hydro": mask,
            **sample
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class TB15D104TileDataset(TileDataset):
    """OGC Testbed-15 dataset parser for D104 (lake/river) segmentation task."""

    def __init__(self, raster_path, vector_path, tile_size, tile_overlap,
                 px_size=None, allow_outlying_vectors=True, clip_outlying_vectors=True,
                 lake_area_min=0.0, lake_area_max=float("inf"), master_roi=None, srs_target="3857",
                 force_parse=False, reproj_rasters=False, reproj_all_cpus=True, display_debug=False,
                 keep_rasters_open=True, parallel=False, transforms=None):
        assert px_size is None or isinstance(px_size, (float, int)), "pixel size (resolution) must be float/int"
        px_size = (1.0, 1.0) if px_size is None else (float(px_size), float(px_size))
        # note: we wrap partial static functions for caching to see when internal parameters are changing
        cleaner = functools.partial(TB15D104Dataset.lake_cleaner, area_min=lake_area_min, area_max=lake_area_max,
                                    lake_river_max_dist=float("inf"), parallel=parallel)
        super().__init__(raster_path=raster_path, vector_path=vector_path, tile_size=tile_size,
                         tile_overlap=tile_overlap, skip_empty_tiles=True, skip_nodata_tiles=False,
                         px_size=px_size, allow_outlying_vectors=allow_outlying_vectors,
                         clip_outlying_vectors=clip_outlying_vectors, vector_area_min=lake_area_min,
                         vector_area_max=lake_area_max, vector_target_prop=None, master_roi=master_roi,
                         srs_target=srs_target, raster_key="lidar", mask_key="hydro", cleaner=cleaner,
                         force_parse=force_parse, reproj_rasters=reproj_rasters, reproj_all_cpus=reproj_all_cpus,
                         keep_rasters_open=keep_rasters_open, transforms=transforms)
        meta_keys = self.task.meta_keys
        self.task = thelper.tasks.Detection(class_names={"background": TB15D104.BACKGROUND_ID, "lake": TB15D104.LAKE_ID},
                                            input_key="input", bboxes_key="bboxes",
                                            meta_keys=meta_keys, background=0, color_map={"lake": [255, 0, 0]})
        self.display_debug = display_debug
        self.parallel = parallel

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        if isinstance(idx, slice):
            return self._getitems(idx)
        import thelper.data.geo as geo
        assert idx < len(self.samples), "sample index is out-of-range"
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        crop, mask = self._process_crop(sample)
        assert crop.shape[2] == 1, "unexpected lidar raster band count"
        crop = crop[:, :, 0]
        dmap = cv.distanceTransform(np.where(mask, np.uint8(0), np.uint8(255)), cv.DIST_L2, cv.DIST_MASK_PRECISE)
        dmap_inv = cv.distanceTransform(np.where(mask, np.uint8(255), np.uint8(0)), cv.DIST_L2, cv.DIST_MASK_PRECISE)
        dmap = np.where(mask, -dmap_inv, dmap)
        dmap *= self.px_size[0]  # constructor enforces same px width/height size
        # dc mask is crop.mask, but most likely lost below
        if self.display_debug:
            crop = cv.normalize(crop, dst=crop, alpha=0, beta=255, norm_type=cv.NORM_MINMAX,
                                dtype=cv.CV_8U, mask=(~crop.mask).astype(np.uint8))
            mask = cv.normalize(mask, dst=mask, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
            dmap = cv.normalize(dmap, dst=dmap, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        # note1: contrarily to the 'smart' dataset parser above, we need to add bboxes here
        # note2: bboxes should only be added over areas that are not 'nodata'
        bboxes = []
        for f in sample["features"]:
            if f["properties"]["TYPECE"] == TB15D104.TYPECE_RIVER:
                continue
            # only lakes can generate bboxes; make sure to clip them to the crop bounds
            inters = f["geometry"].intersection(sample["roi"])
            clip = f["clipped"] or not inters.equals(f["geometry"])
            if clip:
                assert inters.geom_type in ["Polygon", "MultiPolygon"], "unexpected inters type"
                corners = []
                if inters.geom_type == "Polygon":
                    bounds = inters.bounds
                    corners.append((bounds[0:2], bounds[2:4]))
                elif inters.geom_type == "MultiPolygon":
                    for poly in inters:
                        bounds = poly.bounds
                        corners.append((bounds[0:2], bounds[2:4]))
            else:
                corners = [(f["tl"], f["br"])]
            for c in corners:
                feat_tl_px = geo.utils.get_pxcoord(sample["geotransform"], *c[0])
                feat_br_px = geo.utils.get_pxcoord(sample["geotransform"], *c[1])
                bbox = [max(0, feat_tl_px[0]), max(0, feat_tl_px[1]),
                        min(sample["crop_width"] - 1, feat_br_px[0]),
                        min(sample["crop_height"] - 1, feat_br_px[1])]
                if bbox[2] - bbox[0] <= 1 or bbox[3] - bbox[1] <= 1:
                    continue  # skip all bboxes smaller than 1 px (c'mon...)
                # note: lake class id is 1 by definition
                bboxes.append(thelper.tasks.detect.BoundingBox(TB15D104.LAKE_ID,
                                                               bbox=bbox,
                                                               include_margin=False,
                                                               truncated=clip,
                                                               image_id=sample["id"],
                                                               task=self.task))
        sample = {
            "input": np.stack([crop, mask, dmap], axis=-1),
            "hydro": mask,
            "bboxes": bboxes,
            **sample
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample


class TB15D104DetectLogger(DetectLogger):

    def __init__(self, conf_threshold=0.5):
        super().__init__(conf_threshold=conf_threshold, target_name="lake",
                         log_keys=["id", "geotransform"], format="geojson")

    def report_geojson(self):
        # here, we only care about reporting predictions, we ignore the (possibly missing) gt bboxes
        import shapely
        import geojson
        import thelper.data.geo as geo
        batch_size = len(self.bbox[0])
        bbox_lists = [bboxes for batch in self.bbox for bboxes in batch]  # one list per crop
        if batch_size > 1:
            geotransforms = [np.asarray(geot) for batch in self.meta["geotransform"] for geot in batch]
        else:
            # special check to avoid issues when unpacking (1,6)-dim tensor
            geotransforms = [np.asarray(geot) for geot in self.meta["geotransform"]]
        crop_ids = [id for batch in self.meta["id"] for id in batch]
        output_features = []
        for bboxes, geotransform, id in zip(bbox_lists, geotransforms, crop_ids):
            for bbox in bboxes:
                if geotransform.shape[0] == 1:
                    geotransform = geotransform[0]
                bbox_tl = geo.utils.get_geocoord(geotransform, *bbox.top_left)
                bbox_br = geo.utils.get_geocoord(geotransform, *bbox.bottom_right)
                bbox_geom = shapely.geometry.Polygon([bbox_tl, (bbox_br[0], bbox_tl[1]),
                                                      bbox_br, (bbox_tl[0], bbox_br[1])])
                output_features.append(geojson.Feature(geometry=bbox_geom, properties={
                    "image_id": id, "confidence": bbox.confidence}))
        return geojson.dumps(geojson.FeatureCollection(output_features), indent=2)


def postproc_features(input_file, bboxes_srs, orig_geoms_path, output_file,
                      final_srs=None, write_shapefile_copy=False):
    """Post-processes bounding box detections produced during an evaluation session into a GeoJSON file."""
    import ogr
    import osr
    import json
    import geojson
    import shapely
    import thelper.data.geo as geo
    logger.debug("importing bboxes SRS...")
    assert isinstance(bboxes_srs, (str, int, osr.SpatialReference)), \
        "target EPSG SRS must be given as int/str"
    if isinstance(bboxes_srs, (str, int)):
        if isinstance(bboxes_srs, str):
            bboxes_srs = int(bboxes_srs.replace("EPSG:", ""))
        bboxes_srs_obj = osr.SpatialReference()
        bboxes_srs_obj.ImportFromEPSG(bboxes_srs)
        bboxes_srs = bboxes_srs_obj
    logger.debug("importing lake bboxes geojson...")
    with open(input_file) as bboxes_fd:
        bboxes_geoms = geo.utils.parse_geojson(json.load(bboxes_fd))
    logger.debug("importing hydro features geojson...")
    with open(orig_geoms_path) as hydro_fd:
        hydro_geoms = geo.utils.parse_geojson(json.load(hydro_fd), srs_target=bboxes_srs)
    logger.debug("computing global cascade of lake bboxes...")
    detect_roi = shapely.ops.cascaded_union([bbox["geometry"] for bbox in bboxes_geoms])
    output_features = []

    def append_poly(feat, props, srs_transform=None):
        if feat.is_empty:
            return
        elif feat.type == "Polygon":
            if srs_transform is not None:
                ogr_geometry = ogr.CreateGeometryFromWkb(feat.wkb)
                ogr_geometry.Transform(srs_transform)
                feat = shapely.wkt.loads(ogr_geometry.ExportToWkt())
            output_features.append((geojson.Feature(geometry=feat, properties=props), feat))
        elif feat.type == "MultiPolygon" or feat.type == "GeometryCollection":
            for f in feat:
                append_poly(f, props)

    srs_transform = None
    if final_srs is not None:
        import osr
        logger.debug("importing output SRS...")
        assert isinstance(final_srs, (str, int, osr.SpatialReference)), \
            "target EPSG SRS must be given as int/str"
        if isinstance(final_srs, (str, int)):
            if isinstance(final_srs, str):
                final_srs = int(final_srs.replace("EPSG:", ""))
            final_srs_obj = osr.SpatialReference()
            final_srs_obj.ImportFromEPSG(final_srs)
            final_srs = final_srs_obj
        if not bboxes_srs.IsSame(final_srs):
            srs_transform = osr.CoordinateTransformation(bboxes_srs, final_srs)
    logger.debug("running hydro feature and lake bboxes intersection loop...")
    for hydro_feat in tqdm.tqdm(hydro_geoms, desc="computing bbox intersections"):
        # find intersection and append to list of 'lakes'
        intersection = hydro_feat["geometry"].intersection(detect_roi)
        hydro_feat["properties"]["TYPECE"] = TB15D104.TYPECE_LAKE
        append_poly(intersection, copy.deepcopy(hydro_feat["properties"]), srs_transform)
        if not intersection.is_empty:
            # subtract bbox region from feature if intersection found (leftovers at end will be 'rivers')
            hydro_feat["geometry"] = hydro_feat["geometry"].difference(detect_roi)
    logger.debug("running river cleanup loop...")
    for hydro_feat in tqdm.tqdm(hydro_geoms, desc="appending leftover geometries as rivers"):
        if not hydro_feat["geometry"].is_empty:
            # remark: hydro features outside the original ROI will appear as rivers despite never being processed
            hydro_feat["properties"]["TYPECE"] = TB15D104.TYPECE_RIVER
            append_poly(hydro_feat["geometry"], copy.deepcopy(hydro_feat["properties"]), srs_transform)
    logger.debug("exporting final geojson...")
    with open(output_file, "w") as fd:
        out_srs = final_srs if final_srs is not None else bboxes_srs
        fd.write(geo.utils.export_geojson_with_crs([o[0] for o in output_features], srs_target=out_srs))
    if write_shapefile_copy:
        logger.debug("exporting final shapefile...")
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(output_file + ".shp")
        layer = data_source.CreateLayer("lakes", final_srs if final_srs is not None else bboxes_srs, ogr.wkbPolygon)
        for feat_tuple in output_features:
            feature = ogr.Feature(layer.GetLayerDefn())
            point = ogr.CreateGeometryFromWkt(feat_tuple[1].wkt)
            feature.SetGeometry(point)
            layer.CreateFeature(feature)
            feature = None  # noqa # flush
        data_source = None  # noqa # flush
