import glob
import os

import numpy as np

import thelper

# note: the bbox data used here is taken from the 'Object Detection Metrics' repository of Rafael Padilla.
# See https://github.com/rafaelpadilla/Object-Detection-Metrics for more information.
# The original code is distributed under the MIT License, Copyright (c) 2018 Rafael Padilla.


def get_bboxes(dirpath, gt=False):
    bboxes = []
    files = glob.glob(os.path.join(dirpath, "*.txt"))
    files.sort()
    for filename in files:
        with open(filename, "r") as fd:
            for line in fd:
                line = line.replace("\n", "")
                if line.replace(" ", "") == "":  # pragma: no cover
                    continue
                split = line.split(" ")
                if gt:
                    xmin = int(split[1])
                    ymin = int(split[2])
                    xmax = xmin + int(split[3])
                    ymax = ymin + int(split[4])
                    bboxes.append(thelper.tasks.detect.BoundingBox(split[0],
                                                                   [xmin, ymin, xmax, ymax],
                                                                   image_id=os.path.basename(filename)))
                else:
                    xmin = int(split[2])
                    ymin = int(split[3])
                    xmax = xmin + int(split[4])
                    ymax = ymin + int(split[5])
                    bboxes.append(thelper.tasks.detect.BoundingBox(split[0],
                                                                   [xmin, ymin, xmax, ymax],
                                                                   confidence=float(split[1]),
                                                                   image_id=os.path.basename(filename)))
    return bboxes


def test_bbox_map():
    curr_path = os.path.dirname(os.path.abspath(__file__))
    preds_path = os.path.join(curr_path, "detections")
    assert os.path.isdir(preds_path)
    targets_path = os.path.join(curr_path, "groundtruths")
    assert os.path.isdir(targets_path)
    preds = get_bboxes(preds_path, False)
    targets = get_bboxes(targets_path, True)
    assert targets and preds
    task = thelper.tasks.Detection(["person"], "in", "gt")
    res = thelper.optim.compute_pascalvoc_metrics(preds, targets, task, iou_threshold=0.3)
    ap = res["person"]["AP"]
    assert np.isclose(ap, 0.24568668046928915)  # obtained via the original example
