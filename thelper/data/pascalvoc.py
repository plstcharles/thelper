"""PASCAL VOC dataset parser module.

This module contains a dataset parser used to load the PASCAL Visual Object Classes (VOC) dataset for
semantic segmentation or object detection. See http://host.robots.ox.ac.uk/pascal/VOC/ for more info.
"""

import copy
import logging
import os
import xml.etree.ElementTree

import cv2 as cv
import numpy as np

import thelper.data
import thelper.tasks
import thelper.utils
from thelper.data.parsers import Dataset

logger = logging.getLogger(__name__)


class PASCALVOC(Dataset):
    """PASCAL VOC dataset parser.

    This class can be used to parse the PASCAL VOC dataset for either semantic segmentation or object
    detection. The task object it exposes will be changed accordingly. In all cases, the 2012 version
    of the dataset will be used.

    TODO: Add support for semantic instance segmentation.

    .. seealso::
        | :class:`thelper.data.parsers.Dataset`
    """

    _label_name_map = {
        "background": 0,
        "aeroplane": 1,
        "bicycle": 2,
        "bird": 3,
        "boat": 4,
        "bottle": 5,
        "bus": 6,
        "car": 7,
        "cat": 8,
        "chair": 9,
        "cow": 10,
        "diningtable": 11,
        "dog": 12,
        "horse": 13,
        "motorbike": 14,
        "person": 15,
        "pottedplant": 16,
        "sheep": 17,
        "sofa": 18,
        "train": 19,
        "tvmonitor": 20,
        "dontcare": 255,
    }

    _dontcare_val = 255
    _supported_tasks = ["detect", "segm"]
    _supported_subsets = ["train", "trainval", "val", "test"]
    _train_archive_name = "VOCtrainval_11-May-2012.tar"
    _test_archive_name = "VOC2012test.tar"
    _train_archive_url = "http://pjreddie.com/media/files/" + _train_archive_name
    _test_archive_url = "http://pjreddie.com/media/files/" + _test_archive_name
    _train_archive_md5 = "6cd6e144f989b92b3379bac3b3de84fd"
    _test_archive_md5 = "9065beb292b6c291fad82b2725749fda"

    def __init__(self, root, task="segm", subset="trainval", target_labels=None, download=False, preload=True, use_difficult=False,
                 use_occluded=True, use_truncated=True, transforms=None, image_key="image", sample_name_key="name", idx_key="idx",
                 image_path_key="image_path", gt_path_key="gt_path", bboxes_key="bboxes", label_map_key="label_map"):
        self.task_name = task
        assert self.task_name in self._supported_tasks, f"unrecognized task type '{self.task_name}'"
        assert subset in self._supported_subsets, f"unrecognized data subset '{subset}'"
        root = os.path.abspath(root)
        devkit_path = os.path.join(root, "VOCdevkit")
        if not os.path.isdir(devkit_path):
            assert download, f"invalid PASCALVOC devkit path '{devkit_path}'"
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
        assert os.path.isdir(devkit_path), "messed up PASCALVOC devkit tar extraction"
        dataset_path = os.path.join(devkit_path, "VOC2012")
        assert os.path.isdir(dataset_path), f"could not locate dataset folder 'VOC2012' at '{devkit_path}'"
        imagesets_path = os.path.join(dataset_path, "ImageSets")
        assert os.path.isdir(dataset_path), f"could not locate image sets folder at '{imagesets_path}'"
        super().__init__(transforms=transforms)
        self.preload = preload
        # should use_difficult be true for training, but false for validation?
        self.image_key = image_key
        self.idx_key = idx_key
        self.sample_name_key = sample_name_key
        self.image_path_key = image_path_key
        self.gt_path_key = gt_path_key
        meta_keys = [self.sample_name_key, self.image_path_key, self.gt_path_key, self.idx_key]
        self.gt_key = None
        self.task = None
        image_set_name = None
        valid_sample_names = None
        if target_labels is not None:
            if not isinstance(target_labels, list):
                target_labels = [target_labels]
            assert target_labels or all([label in self._label_name_map for label in target_labels]), \
                "target labels should be given as list of names (strings) that already exist"
            self.label_name_map = {}
            for name in self._label_name_map:
                if name in target_labels or name == "background" or name == "dontcare":
                    self.label_name_map[name] = len(self.label_name_map) if name != "dontcare" else self._dontcare_val
        else:
            self.label_name_map = copy.deepcopy(self._label_name_map)
        # if using target labels, must rely on image set luts to confirm content
        if target_labels is not None:
            valid_sample_names = set()
            for label in self.label_name_map:
                if label in ["background", "dontcare"]:
                    continue
                with open(os.path.join(imagesets_path, "Main", label + "_" + subset + ".txt")) as image_subset_fd:
                    for line in image_subset_fd:
                        sample_name, val = line.split()
                        if int(val) > 0:
                            valid_sample_names.add(sample_name)
        self.label_colors = {idx: thelper.utils.get_label_color_mapping(self._label_name_map[name])[::-1]
                             for name, idx in self.label_name_map.items()}
        color_map = {idx: self.label_colors[idx][::-1] for idx in self.label_name_map.values()}
        if self.task_name == "detect":
            if "dontcare" in self.label_name_map:
                del color_map[self.label_name_map["dontcare"]]
                del self.label_name_map["dontcare"]  # no need for obj detection
            self.gt_key = bboxes_key
            self.task = thelper.tasks.Detection(self.label_name_map, input_key=self.image_key,
                                                bboxes_key=self.gt_key, meta_keys=meta_keys,
                                                background=self.label_name_map["background"],
                                                color_map=color_map)
            image_set_name = "Main"
        elif self.task_name == "segm":
            self.gt_key = label_map_key
            self.task = thelper.tasks.Segmentation(self.label_name_map, input_key=self.image_key,
                                                   label_map_key=self.gt_key, meta_keys=meta_keys,
                                                   dontcare=self._dontcare_val, color_map=color_map)
            image_set_name = "Segmentation"
        imageset_path = os.path.join(imagesets_path, image_set_name, subset + ".txt")
        assert os.path.isfile(imageset_path), "cannot locate sample set file at '%s'" % imageset_path
        image_folder_path = os.path.join(dataset_path, "JPEGImages")
        assert os.path.isdir(image_folder_path), "cannot locate image folder at '%s'" % image_folder_path
        with open(imageset_path) as image_subset_fd:
            if valid_sample_names is None:
                sample_names = image_subset_fd.read().splitlines()
            else:
                sample_names = set()
                for sample_name in image_subset_fd:
                    sample_name = sample_name.strip()
                    if sample_name in valid_sample_names:
                        sample_names.add(sample_name)
                sample_names = list(sample_names)
        action = "preloading" if self.preload else "initializing"
        logger.info("%s pascal voc dataset for task='%s' and set='%s'..." % (action, self.task_name, subset))
        self.samples = []
        if self.preload:
            from tqdm import tqdm
        else:
            def tqdm(x):
                return x
        for sample_name in tqdm(sample_names):
            annotation_file_path = os.path.join(dataset_path, "Annotations", sample_name + ".xml")
            assert os.path.isfile(annotation_file_path), "cannot load annotation file for sample '%s'" % sample_name
            annotation = xml.etree.ElementTree.parse(annotation_file_path).getroot()
            assert annotation.tag == "annotation", "unexpected xml content"
            filename = annotation.find("filename").text
            image_path = os.path.join(image_folder_path, filename)
            assert os.path.isfile(image_path), "cannot locate image for sample '%s'" % sample_name
            image = None
            if self.preload:
                image = cv.imread(image_path)
                assert image is not None, "could not load image '%s' via opencv" % image_path
            gt, gt_path = None, None
            if self.task_name == "segm":
                assert int(annotation.find("segmented").text) == 1, "unexpected segmented flag for sample '%s'" % sample_name
                gt_path = os.path.join(dataset_path, "SegmentationClass", sample_name + ".png")
                if self.preload:
                    gt = cv.imread(gt_path)
                    assert gt is not None and gt.shape != image.shape, "unexpected gt shape for sample '%s'" % sample_name
                    gt = self.encode_label_map(gt)
                    #gt_decoded = self.decode_label_map(gt)
                    #assert np.array_equal(cv.imread(gt_path), gt_decoded), "messed up encoding/decoding functions"
            elif self.task_name == "detect":
                gt_path = annotation_file_path
                gt = []
                for obj in annotation.iter("object"):
                    if not use_difficult and obj.find("difficult").text == "1":
                        continue
                    if not use_occluded and obj.find("occluded").text == "1":
                        continue
                    if not use_truncated and obj.find("truncated").text == "1":
                        continue
                    bbox = obj.find("bndbox")
                    label = obj.find("name").text
                    if label not in self.label_name_map:
                        continue  # user is skipping some labels from the complete set
                    image_id = int(os.path.splitext(filename)[0])
                    gt.append(thelper.data.BoundingBox(class_id=self.label_name_map[label],
                                                       bbox=(int(bbox.find("xmin").text), int(bbox.find("ymin").text),
                                                             int(bbox.find("xmax").text), int(bbox.find("ymax").text)),
                                                       difficult=thelper.utils.str2bool(obj.find("difficult").text),
                                                       occluded=thelper.utils.str2bool(obj.find("occluded").text),
                                                       truncated=thelper.utils.str2bool(obj.find("truncated").text),
                                                       confidence=None, image_id=image_id, task=self.task))
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
        if isinstance(idx, slice):
            return self._getitems(idx)
        assert idx < len(self.samples), "sample index is out-of-range"
        if idx < 0:
            idx = len(self.samples) + idx
        sample = self.samples[idx]
        if not self.preload:
            image = cv.imread(sample[self.image_path_key])
            assert image is not None, "could not load image '%s' via opencv" % sample[self.image_path_key]
            image = image[..., ::-1]  # BGR to RGB
            gt = None
            if self.task_name == "segm":
                gt = cv.imread(sample[self.gt_path_key])
                assert gt is not None and gt.shape == image.shape, "unexpected gt shape for sample '%s'" % sample[self.sample_name_key]
                gt = self.encode_label_map(gt)
            elif self.task_name == "detect":
                gt = sample[self.gt_key]
        else:
            image = sample[self.image_key]
            gt = sample[self.gt_key]
        sample = {
            self.sample_name_key: sample[self.sample_name_key],
            self.image_path_key: sample[self.image_path_key],
            self.gt_path_key: sample[self.gt_path_key],
            self.image_key: image,
            self.gt_key: gt,
            self.idx_key: idx
        }
        if isinstance(sample[self.image_key], np.ndarray) and any([s < 0 for s in sample[self.image_key].strides]):
            # fix unsupported negative strides in PyTorch <= 1.1.0
            sample[self.image_key] = sample[self.image_key].copy()
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def decode_label_map(self, label_map):
        """Returns a color image from a label indices map."""
        assert isinstance(label_map, np.ndarray) and label_map.ndim == 2, "unexpected label map type/shape, should be 2D np.ndarray"
        output = np.full(list(label_map.shape) + [3], fill_value=self._dontcare_val, dtype=np.uint8)
        for label_idx, label_color in self.label_colors.items():
            output[np.where(label_map == label_idx)] = label_color
        return output

    def encode_label_map(self, label_map):
        """Returns a map of label indices from a color image."""
        assert isinstance(label_map, np.ndarray) and label_map.ndim == 3 or label_map.dtype != np.uint8, \
            "unexpected label map type/shape, should be 3D np.ndarray"
        output = np.full(label_map.shape[:2], fill_value=self._dontcare_val, dtype=np.uint8)
        # TODO: loss might not like uint8, check for useless convs later
        for label_idx, label_color in self.label_colors.items():
            output = np.where(np.all(label_map == label_color, axis=2), label_idx, output)
        return output
