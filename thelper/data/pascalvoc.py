"""PASCAL VOC dataset parser module.

This module contains a dataset parser used to load the PASCAL Visual Object Classes (VOC) dataset for
semantic segmentation or object detection. See http://host.robots.ox.ac.uk/pascal/VOC/ for more info.
"""

import logging
import os
import xml.etree.ElementTree

import cv2 as cv
import numpy as np

from thelper.data.parsers import Dataset
import thelper.tasks
import thelper.utils

logger = logging.getLogger(__name__)


class PASCALVOC(Dataset):
    """PASCAL VOC dataset parser.

    This class can be used to parse the PASCAL VOC dataset for either semantic segmentation or object
    detection. The task object it exposes will be changed accordingly. In all cases, the 2012 version
    of the dataset will be used.

    TODO: Finish implementation of object detection task setup.
    TODO: Add support for semantic instance segmentation.

    .. seealso::
        | :class:`thelper.data.parsers.Dataset`
    """

    _label_idx_map = {
        0: "background",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor",
        255: "dontcare"
    }

    _label_name_map = {
        name: idx for idx, name in _label_idx_map.items()
    }

    _label_colors = {
        idx: thelper.utils.get_label_color_mapping(idx)[::-1] for idx in _label_idx_map
    }

    _supported_tasks = ["detect", "segm"]
    _supported_subsets = ["train", "trainval", "val", "test"]
    _train_archive_name = "VOCtrainval_11-May-2012.tar"
    _test_archive_name = "VOC2012test.tar"
    _train_archive_url = "http://pjreddie.com/media/files/" + _train_archive_name
    _test_archive_url = "http://pjreddie.com/media/files/" + _test_archive_name
    _train_archive_md5 = "6cd6e144f989b92b3379bac3b3de84fd"
    _test_archive_md5 = "9065beb292b6c291fad82b2725749fda"

    def __init__(self, config, transforms=None):
        self.task_name = str(thelper.utils.get_key_def("task", config, "segm")).lower()
        if self.task_name not in self._supported_tasks:
            raise AssertionError("unrecognized task type '%s'" % self.task_name)
        subset = str(thelper.utils.get_key_def("subset", config, "trainval")).lower()
        if subset not in self._supported_subsets:
            raise AssertionError("unrecognized data subset '%s'" % subset)
        download = thelper.utils.str2bool(thelper.utils.get_key_def("download", config, False))
        root = os.path.abspath(thelper.utils.get_key("root", config))
        devkit_path = os.path.join(root, "VOCdevkit")
        if not os.path.isdir(devkit_path):
            if not download:
                raise AssertionError("invalid devkit path '%s'" % devkit_path)
            else:
                logger.info("downloading training data archive...")
                train_archive_path = thelper.utils.download_file(self._train_archive_url,
                                                                 root, self._train_archive_name,
                                                                 self._train_archive_md5)
                logger.info("extracting training data archive...")
                thelper.utils.extract_tar(train_archive_path, root, flags="r:")
                logger.info("downloading test data archive...")
                test_archive_path = thelper.utils.download_file(self._test_archive_url,
                                                                root, self._test_archive_name,
                                                                self._test_archive_md5)
                logger.info("extracting test data archive...")
                thelper.utils.extract_tar(test_archive_path, root, flags="r:")
        if not os.path.isdir(devkit_path):
            raise AssertionError("messed up tar extraction")
        dataset_path = os.path.join(devkit_path, "VOC2012")
        if not os.path.isdir(dataset_path):
            raise AssertionError("could not locate dataset folder 'VOC2012' at '%s'" % devkit_path)
        imagesets_path = os.path.join(dataset_path, "ImageSets")
        if not os.path.isdir(dataset_path):
            raise AssertionError("could not locate image sets folder at '%s'" % imagesets_path)
        super().__init__(config=config, transforms=transforms, bypass_deepcopy=True)
        self.preload = thelper.utils.str2bool(thelper.utils.get_key_def("preload", config, True))  # @@@ CHANGE TODO
        # should use_difficult be true for training, but false for validation?
        use_difficult = thelper.utils.str2bool(thelper.utils.get_key_def("use_difficult", config, False))
        use_occluded = thelper.utils.str2bool(thelper.utils.get_key_def("use_occluded", config, True))
        use_truncated = thelper.utils.str2bool(thelper.utils.get_key_def("use_truncated", config, True))
        self.image_key = thelper.utils.get_key_def("image_key", config, "image")
        self.sample_name_key = thelper.utils.get_key_def("name_key", config, "name")
        self.image_path_key = thelper.utils.get_key_def("image_path_key", config, "image_path")
        self.gt_path_key = thelper.utils.get_key_def("gt_path_key", config, "gt_path")
        meta_keys = [self.sample_name_key, self.image_path_key, self.gt_path_key]
        self.gt_key = None
        self.task = None
        imageset_name = None
        if self.task_name == "detect":
            self.gt_key = thelper.utils.get_key_def("bboxes_key", config, "bboxes")
            # self.task = thelper.tasks.Detection(...)
            imageset_name = "Main"
            raise NotImplementedError
        elif self.task_name == "segm":
            self.gt_key = thelper.utils.get_key_def("label_map_key", config, "label_map")
            color_map = {name: self._label_colors[idx][::-1] for name, idx in self._label_name_map.items()}
            self.task = thelper.tasks.Segmentation(self._label_name_map, input_key=self.image_key,
                                                   label_map_key=self.gt_key, meta_keys=meta_keys,
                                                   dontcare=self._label_name_map["dontcare"],
                                                   color_map=color_map)
            imageset_name = "Segmentation"
        imageset_path = os.path.join(imagesets_path, imageset_name, subset + ".txt")
        if not os.path.isfile(imageset_path):
            raise AssertionError("cannot locate sample set file at '%s'" % imageset_path)
        image_folder_path = os.path.join(dataset_path, "JPEGImages")
        if not os.path.isdir(image_folder_path):
            raise AssertionError("cannot locate image folder at '%s'" % image_folder_path)
        with open(imageset_path) as fd:
            sample_names = fd.read().splitlines()
        action = "preloading" if self.preload else "initializing"
        logger.info("%s pascal voc dataset for task='%s' and set='%s'..." % (action, self.task_name, subset))
        self.samples = []
        if self.preload:
            try:
                from tqdm import tqdm
            except ImportError:
                def tqdm(x): return x
        else:
            def tqdm(x): return x
        for sample_name in tqdm(sample_names):
            annotation_file_path = os.path.join(dataset_path, "Annotations", sample_name + ".xml")
            if not os.path.isfile(annotation_file_path):
                raise AssertionError("cannot load annotation file for sample '%s'" % sample_name)
            annotation = xml.etree.ElementTree.parse(annotation_file_path).getroot()
            if annotation.tag != "annotation":
                raise AssertionError("unexpected xml content")
            image_path = os.path.join(image_folder_path, annotation.find("filename").text)
            if not os.path.isfile(image_path):
                raise AssertionError("cannot locate image for sample '%s'" % sample_name)
            image = None
            if self.preload:
                image = cv.imread(image_path)
                if image is None:
                    raise AssertionError("could not load image '%s' via opencv" % image_path)
            gt, gt_path = None, None
            if self.task_name == "segm":
                if int(annotation.find("segmented").text) != 1:
                    raise AssertionError("unexpected segmented flag for sample '%s'" % sample_name)
                gt_path = os.path.join(dataset_path, "SegmentationClass", sample_name + ".png")
                if self.preload:
                    gt = cv.imread(gt_path)
                    if gt is None or gt.shape != image.shape:
                        raise AssertionError("unexpected gt shape for sample '%s'" % sample_name)
                    gt = self.encode_label_map(gt)
                    #gt_decoded = self.decode_label_map(gt)
                    #if not np.array_equal(cv.imread(gt_path), gt_decoded):
                    #    raise AssertionError("messed up encoding/decoding functions")
            elif self.task_name == "detect":
                gt_path = annotation_file_path
                gt = []
                for obj in annotation.iter("object"):
                    # TODO: update w/ task-compat structure
                    if not use_difficult and obj.find("difficult").text == "1":
                        continue
                    if not use_occluded and obj.find("occluded").text == "1":
                        continue
                    if not use_truncated and obj.find("truncated").text == "1":
                        continue
                    bbox = obj.find("bndbox")
                    gt.append({
                        "label": obj.find("name").text,
                        "id": self._label_name_map[obj.find("name").text],
                        "bbox": {
                            "xmax": bbox.find("xmax").text,
                            "xmin": bbox.find("xmin").text,
                            "ymax": bbox.find("ymax").text,
                            "ymin": bbox.find("ymin").text,
                        },
                        "difficult": thelper.utils.str2bool(obj.find("difficult").text),
                        "occluded": thelper.utils.str2bool(obj.find("occluded").text),
                        "truncated": thelper.utils.str2bool(obj.find("truncated").text),
                    })
                if not gt:
                    continue
            self.samples.append({
                self.sample_name_key: sample_name,
                self.image_path_key: image_path,
                self.gt_path_key: gt_path,
                self.image_key: image,
                self.gt_key: gt,
            })
        logger.info("initialized %d samples" % len(self.samples))

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index."""
        sample = self.samples[idx]
        if not self.preload:
            image = cv.imread(sample[self.image_path_key])
            if image is None:
                raise AssertionError("could not load image '%s' via opencv" % sample[self.image_path_key])
            image = image[..., ::-1]  # BGR to RGB
            gt = None
            if self.task_name == "segm":
                gt = cv.imread(sample[self.gt_path_key])
                if gt is None or gt.shape != image.shape:
                    raise AssertionError("unexpected gt shape for sample '%s'" % sample[self.sample_name_key])
                gt = self.encode_label_map(gt)
        else:
            image = sample[self.image_key]
            gt = sample[self.gt_key]
        if self.transforms:
            image = self.transforms(image)
            # TODO : gt maps are not currently transformed! (need refact w/ dict keys)
        return {
            self.sample_name_key: sample[self.sample_name_key],
            self.image_path_key: sample[self.image_path_key],
            self.gt_path_key: sample[self.gt_path_key],
            self.image_key: image,
            self.gt_key: gt,
        }

    def get_task(self):
        """Returns the dataset task object that provides the i/o keys for parsing sample dicts."""
        return self.task

    def decode_label_map(self, label_map):
        """Returns a color image from a label indices map."""
        if not isinstance(label_map, np.ndarray) or label_map.ndim != 2:
            raise AssertionError("unexpected label map type/shape, should be 2D np.ndarray")
        dontcare_val = self._label_colors[self._label_name_map["dontcare"]]
        output = np.full(list(label_map.shape) + [3], fill_value=dontcare_val, dtype=np.uint8)
        for label_idx, label_color in self._label_colors.items():
            output[np.where(label_map == label_idx)] = label_color
        return output

    def encode_label_map(self, label_map):
        """Returns a map of label indices from a color image."""
        if not isinstance(label_map, np.ndarray) or label_map.ndim != 3 or label_map.dtype != np.uint8:
            raise AssertionError("unexpected label map type/shape, should be 3D np.ndarray")
        dontcare_val = self._label_name_map["dontcare"]
        output = np.full(label_map.shape[:2], fill_value=dontcare_val, dtype=np.uint8)
        # TODO: loss might not like uint8, check for useless convs later
        for label_idx, label_color in self._label_colors.items():
            output = np.where(np.all(label_map == label_color, axis=2), label_idx, output)
        return output
