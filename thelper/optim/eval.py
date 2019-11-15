"""Evaluation classes/funcs module.

This module contains procedures used to evaluate models and prediction results on specific
tasks or datasets. These procedures may be used as part of metric classes (defined in
:mod:`thelper.optim.metrics`) or high-level debug/drawing utilities.
"""
from typing import Dict, List, Optional, Union  # noqa: F401

import numpy as np
import torch

import thelper


@thelper.concepts.detection
def compute_bbox_iou(bbox1, bbox2):
    # type: (thelper.tasks.detect.BoundingBox, thelper.tasks.detect.BoundingBox) -> float
    """Computes and returns the Intersection over Union (IoU) of two bounding boxes."""
    assert isinstance(bbox1, thelper.data.BoundingBox) and isinstance(bbox2, thelper.data.BoundingBox), \
        "unexpected input bounding box types"
    bbox1_marg = 1 if bbox1.include_margin else 0
    bbox2_marg = 1 if bbox2.include_margin else 0
    intersection_width = min(bbox1.right + bbox1_marg, bbox2.right + bbox2_marg) - max(bbox1.left, bbox2.left)
    intersection_height = min(bbox1.bottom + bbox1_marg, bbox2.bottom + bbox2_marg) - max(bbox1.top, bbox2.top)
    intersection_area = max(0, intersection_width) * max(0, intersection_height)
    return float(intersection_area / float(bbox1.area + bbox2.area - intersection_area))


@thelper.concepts.segmentation
def compute_mask_iou(mask1, mask2, class_indices=None, dontcare=None):
    # type: (np.ndarray, np.ndarray, Union[List[int], np.ndarray, torch.Tensor], Optional[int]) -> Dict[int, float]
    """Computes and returns a map of Intersection over Union (IoU) scores for two segmentation masks."""
    # untested as of 11/2019; needs utest!
    assert isinstance(mask1, np.ndarray) and isinstance(mask2, np.ndarray), "invalid mask type"
    assert mask1.shape == mask2.shape and mask1.dtype == mask2.dtype, "mismatched mask shape/types"
    if not class_indices:
        class_indices = np.unique(np.stack([mask1, mask2]))
    assert isinstance(class_indices, (list, np.ndarray, torch.Tensor)), "invalid class indices array type"
    iou_dict = {}
    for class_idx in class_indices:
        if dontcare is not None:
            target_c = np.logical_and(mask2 == class_idx, mask1 != dontcare)
            pred_c = np.logical_and(mask1 == class_idx, mask2 != dontcare)
        else:
            target_c = mask2 == class_idx
            pred_c = mask1 == class_idx
        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()
        if float(union) != 0.0:
            iou_dict[class_idx] = (float(intersection) / float(union))
        else:
            iou_dict[class_idx] = 0.0
    return iou_dict


@thelper.concepts.detection
def compute_pascalvoc_metrics(pred_bboxes, gt_bboxes, task, iou_threshold=0.5, method="all-points"):
    """Computes the metrics used by the VOC Pascal 2012 challenge.

    This function is inspired from the 'Object Detection Metrics' repository of Rafael Padilla.
    See https://github.com/rafaelpadilla/Object-Detection-Metrics for more information.
    The original code is distributed under the MIT License, Copyright (c) 2018 Rafael Padilla.

    Args:
        pred_bboxes: list of bbox predictions generated by the model under evaluation.
        gt_bboxes: list of groundtruth bounding boxes defined by the dataset.
        task: task definition object that holds a vector of all class names.
        iou_threshold: Intersection Over Union (IOU) threshold for true/false positive classification.
        method: the evaluation method to use; can be the the latest & official PASCAL VOC toolkit
            approach ("all-points"), or the 11-point approach ("11-points") described in the original
            paper ("The PASCAL Visual Object Classes(VOC) Challenge").

    Returns:
        A dictionary containing evaluation information and metrics for each class. Each entry contains:
        - ``precision``: array with the precision values;
        - ``recall``: array with the recall values;
        - ``AP``: average precision;
        - ``interpolated precision``: interpolated precision values;
        - ``interpolated recall``: interpolated recall values;
        - ``total positives``: total number of ground truth positives;
        - ``total TP``: total number of True Positive detections;
        - ``total FP``: total number of False Negative detections.
    """
    assert isinstance(pred_bboxes, (list, np.ndarray)) and all([isinstance(b, thelper.data.BoundingBox) for b in pred_bboxes]), \
        "invalid predictions format (expected list of bounding box objects)"
    assert all([isinstance(bbox.confidence, float) and 0 <= bbox.confidence <= 1 for bbox in pred_bboxes]), \
        "predicted bounding boxes must be provided with confidence values in [0,1]"
    assert all([bbox.image_id is not None for bbox in pred_bboxes]), "predicted bbox image id must be defined"
    assert isinstance(gt_bboxes, (list, np.ndarray)) and all([isinstance(b, thelper.data.BoundingBox) for b in gt_bboxes]), \
        "invalid input groundtruth format (expected list of bounding box objects)"
    assert all([bbox.image_id is not None for bbox in gt_bboxes]), "gt bbox image id must be defined"
    assert isinstance(task, thelper.tasks.Detection) and task.class_names, "invalid task object (should be detection)"
    assert 0 < iou_threshold <= 1, "invalid intersection over union value (should be in ]0,1])"
    assert method in ["all-points", "11-points"], "invalid method (should be 'all-points' or '11-points')"
    image_ids = list(set([bbox.image_id for bbox in pred_bboxes]) | set([bbox.image_id for bbox in gt_bboxes]))
    image_ids = {k: idx for idx, k in enumerate(image_ids)}
    gt_used_flags = [[[[bbox, False] for bbox in gt_bboxes
                       if (((isinstance(bbox.class_id, int) and bbox.class_id == ci) or bbox.class_id == cn) and
                           bbox.image_id == iid)] for iid in image_ids] for ci, cn in enumerate(task.class_names)]
    ret = {}
    for class_idx, class_name in enumerate(task.class_names):
        if task.background is not None and class_name == "background":
            continue
        curr_pred_bboxes = [bbox for bbox in pred_bboxes if (isinstance(bbox.class_id, int) and bbox.class_id == class_idx) or
                            bbox.class_id == class_name]
        curr_pred_bboxes = sorted(curr_pred_bboxes, key=lambda bbox: bbox.confidence, reverse=True)
        true_positives = np.zeros(len(curr_pred_bboxes))
        false_positives = np.zeros(len(curr_pred_bboxes))
        for pred_bbox_idx, pred_bbox in enumerate(curr_pred_bboxes):
            curr_gt_bboxes = gt_used_flags[class_idx][image_ids[pred_bbox.image_id]]
            best_gt_bbox_idx, best_gt_bbox_iou = -1, float("-inf")
            for gt_bbox_idx, (gt_bbox, gt_bbox_flag) in enumerate(curr_gt_bboxes):
                iou = compute_bbox_iou(pred_bbox, gt_bbox)
                if iou > best_gt_bbox_iou:
                    best_gt_bbox_iou = iou
                    best_gt_bbox_idx = gt_bbox_idx
            if best_gt_bbox_iou >= iou_threshold:
                curr_best_gt_bbox_used_flag = curr_gt_bboxes[best_gt_bbox_idx][1]
                if not curr_best_gt_bbox_used_flag:
                    true_positives[pred_bbox_idx] = 1
                    # we can only use GT bboxes once, flag them as 'seen' after that
                    curr_gt_bboxes[best_gt_bbox_idx][1] = True
                else:
                    # if best GT bbox was already used, we discard this detection
                    # (note: we could do some combinatorial optim w/ hungarian method to solve ideally instead)
                    false_positives[pred_bbox_idx] = 1
            else:
                # if we fail to meet the minimum iou threshold, discard this detection
                false_positives[pred_bbox_idx] = 1
        true_positive_cumsum = np.cumsum(true_positives)
        npos = sum([len(bboxes) for bboxes in gt_used_flags[class_idx]])
        recall = true_positive_cumsum / npos
        precision = np.divide(true_positive_cumsum, (np.cumsum(false_positives) + true_positive_cumsum))
        avg_prec, mpre, mrec, _ = compute_average_precision(precision.tolist(), recall.tolist(), method)
        ret[class_name] = {
            "class_name": class_name,
            "iou_threshold": iou_threshold,
            "eval_method": method,
            "precision": precision,
            "recall": recall,
            "AP": avg_prec,
            "interpolated precision": mpre,
            "interpolated recall": mrec,
            "total positives": npos,
            "total TP": np.sum(true_positives),
            "total FP": np.sum(false_positives)
        }
    return ret


@thelper.concepts.detection
def compute_average_precision(precision, recall, method="all-points"):
    """Computes the average precision given an array of precision and recall values.

    This function is inspired from the 'Object Detection Metrics' repository of Rafael Padilla.
    See https://github.com/rafaelpadilla/Object-Detection-Metrics for more information.
    The original code is distributed under the MIT License, Copyright (c) 2018 Rafael Padilla.

    Args:
        precision: list of precision values for the evaluated predictions of a class.
        recall: list of recall values for the evaluated predictions of a class.
        method: the evaluation method to use; can be the the latest & official PASCAL VOC toolkit
            approach ("all-points"), or the 11-point approach ("11-points") described in the original
            paper ("The PASCAL Visual Object Classes(VOC) Challenge").

    Returns:
        A 4-element tuple containing the average precision, rectified precision/recall arrays, and
        the indices used for the integral.
    """
    assert isinstance(precision, list) and all([0 <= p <= 1 for p in precision])
    assert isinstance(recall, list) and all([0 <= r <= 1 for r in recall])
    assert method in ["all-points", "11-points"]
    if method == "all-points":
        mprecision = [0, *precision, 0]  # pad with extrema
        # run backwards through precision values, eliminate ridges
        for idx in range(len(mprecision) - 1, 0, -1):
            mprecision[idx - 1] = max(mprecision[idx - 1], mprecision[idx])
        mrecall = [0, *recall, 1]  # pad with extrema
        # eliminate duplicates
        idxs = [idx + 1 for idx in range(len(mrecall) - 1) if mrecall[1:][idx] != mrecall[0:-1][idx]]
        avg_prec = 0
        # compute integral (AUC)
        for idx in idxs:
            avg_prec = avg_prec + np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
        return avg_prec, mprecision[0:len(mprecision) - 1], mrecall[0:len(mprecision) - 1], idxs
    else:
        mprecision = [*precision]
        rho_interp, recall_val_id = [], []
        for r in np.linspace(0, 1, 11)[::-1]:
            ridxs = np.argwhere(np.asarray(recall) >= r)
            rho_interp.append(max(mprecision[ridxs.min():]) if ridxs.size != 0 else 0)
            recall_val_id.append(r)
        avg_prec = sum(rho_interp) / 11
        rvals = [recall_val_id[0], *recall_val_id, 0]
        pvals = [0, *rho_interp, 0]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        return [avg_prec, [i[1] for i in cc], [i[0] for i in cc], None]
