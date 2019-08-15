"""Data parsers & utilities module for OGC-related projects."""

import logging

import cv2 as cv
import numpy as np

import thelper.data
import thelper.data.geo as geo

logger = logging.getLogger(__name__)


class TB15D104Dataset(geo.parsers.VectorCropDataset):
    """OGC-Testbed 15 dataset parser for D104 (lake/river) segmentation task."""

    TYPECE_RIVER = "10"
    TYPECE_LAKE = "21"

    def __init__(self, raster_path, vector_path, px_size=None, skew=None,
                 allow_outlying_vectors=True, clip_outlying_vectors=True,
                 lake_area_min=0.0, lake_area_max=float("inf"),
                 lake_river_max_dist=float("inf"), focus_lakes=True,
                 srs_target="3857", force_parse=False,
                 reproj_rasters=False, reproj_all_cpus=True,
                 keep_rasters_open=True, transforms=None):
        assert isinstance(lake_river_max_dist, (float, int)) and lake_river_max_dist >= 0, "unexpected dist type"
        self.lake_river_max_dist = float(lake_river_max_dist)
        assert isinstance(focus_lakes, bool), "unexpected flag type"
        self.focus_lakes = focus_lakes
        super().__init__(raster_path=raster_path, vector_path=vector_path, px_size=px_size, skew=skew,
                         allow_outlying_vectors=allow_outlying_vectors, clip_outlying_vectors=clip_outlying_vectors,
                         vector_area_min=lake_area_min, vector_area_max=lake_area_max,
                         vector_target_prop=None, vector_roi_buffer=None, srs_target=srs_target,
                         raster_key="lidar", mask_key="hydro",  force_parse=force_parse, reproj_rasters=reproj_rasters,
                         reproj_all_cpus=reproj_all_cpus, keep_rasters_open=keep_rasters_open, transforms=transforms)
        self.task = thelper.tasks.Detection(class_names=["background", "lake"],
                                            input_key="input", bboxes_key="bboxes",
                                            meta_keys=self.task.meta_keys, background=0)
        #self._show_stats_plots(True, True)

    def _default_feature_cleaner(self, features):
        """Flags geometric features as 'clean' based on type and distance to nearest river."""
        # note: we use a flag here instead of removing bad features so that end-users can still use them if needed
        for f in features:
            f["clean"] = False  # flag every as 'bad' by default, clear just the ones of interest below
        rivers = [f for f in features if f["properties"]["TYPECE"] == self.TYPECE_RIVER]
        lakes = [f for f in features if f["properties"]["TYPECE"] == self.TYPECE_LAKE]
        for lake in lakes:
            if self.area_min <= lake["geometry"].area <= self.area_max:
                if self.lake_river_max_dist == float("inf"):
                    lake["clean"] = True
                else:
                    for river in rivers:
                        if lake["geometry"].distance(river["geometry"]) < self.lake_river_max_dist:
                            lake["clean"] = True
                            break
        return features

    def _default_feature_cropper(self, feature):
        """Returns the ROI information for a given feature (may be modified in derived classes)."""
        # note: default behavior = just center on the feature, and pad if required by user
        if self.focus_lakes:
            return geo.utils.get_feature_roi(feature["geometry"], self.px_size, self.skew, self.roi_buffer)
        else:
            raise NotImplementedError  # TODO @@@@

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

    def _get_bboxes(self, sample):
        bboxes = []
        for feat, clip in zip(sample["features"], sample["clipped_flags"]):
            feat_tl_px = geo.utils.get_pxcoord(sample["geotransform"], *feat["tl"])
            feat_br_px = geo.utils.get_pxcoord(sample["geotransform"], *feat["br"])
            bbox = [max(0, feat_tl_px[0]), max(0, feat_tl_px[1]),
                    min(sample["crop_width"] - 1, feat_br_px[0]),
                    min(sample["crop_height"] - 1, feat_br_px[1])]
            bboxes.append(thelper.tasks.detect.BoundingBox(self.task.class_names.index("lake"), bbox=bbox,
                                                           include_margin=False, truncated=clip,
                                                           task=self.task))
        return bboxes

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
        else:
            pass  # TODO: MERGE CROP + MASK(DTRANSF) @@@@
        sample = {
            "input": np.array(crop.data, copy=True),
            "bboxes": self._get_bboxes(sample),
            self.mask_key: mask,
            **sample
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample
