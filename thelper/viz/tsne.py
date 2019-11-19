"""Tools related to the t-Distributed Stochastic Neighbor Embedding (t-SNE, or TSNE).

For more information on t-SNE, see https://lvdmaaten.github.io/tsne/ for the original author's
repository, or https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html for
the ``scikit-learn`` tools.
"""

from typing import Any, AnyStr, Dict, List, Optional, Union  # noqa: F401

import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.manifold
import torch
import tqdm

import thelper.utils


def plot(projs,                # type: np.ndarray
         targets,              # type: np.ndarray
         preds,                # type: np.ndarray
         color_map=None,       # type: Optional[Dict[int, str]]
         task=None,            # type: Optional[thelper.typedefs.TaskType]
         **kwargs,
         ):                    # type: (...) -> matplotlib.figure.Figure
    """Returns a matplotlib figure of a set of projected embeddings with optional target/predicted labels."""
    proj_min, proj_max = np.min(projs, 0), np.max(projs, 0)
    projs = (projs - proj_min) / (proj_max - proj_min)
    figsize = thelper.utils.get_key_def("figsize", kwargs, (6, 6))
    dpi = thelper.utils.get_key_def("dpi", kwargs, 160)
    fig = plt.figure(num="thelper.viz", figsize=figsize, dpi=dpi,
                     facecolor="w", edgecolor="k", clear=True)
    ax = fig.add_subplot(1, 1, 1)
    default_color = thelper.utils.get_key_def("default_color", kwargs, "#666666")
    for i in range(projs.shape[0]):
        pred_color = color_map[preds[i]] if color_map else default_color
        plt.plot(projs[i, 0], projs[i, 1], "o", color=pred_color, MarkerSize=8)
        if preds[i] != targets[i]:
            plt.plot(projs[i, 0], projs[i, 1], "o", color="#FFFFFF", MarkerSize=5)
        target_color = color_map[targets[i]] if color_map else default_color
        plt.plot(projs[i, 0], projs[i, 1], "o", color=target_color, MarkerSize=4)
    fig.set_tight_layout(True)
    if task is not None and isinstance(task, thelper.tasks.Classification):
        ax.set_xlabel("Center: Label, Border: Prediction")
        assert color_map is not None, "should provide color map is classif task"
        legend_handles = [matplotlib.patches.Patch(facecolor=color_map[idx], label=lbl)
                          for lbl, idx in task.class_indices.items()]
        fig.legend(handles=legend_handles)
    title = thelper.utils.get_key_def("title", kwargs, None)
    if title is not None:
        fig.title(title)
    return fig


def visualize(model,              # type: thelper.typedefs.ModelType
              task,               # type: thelper.typedefs.TaskType
              loader,             # type: thelper.typedefs.LoaderType
              draw=False,         # type: bool
              color_map=None,     # type: Optional[Dict[int, np.ndarray]]
              max_samples=None,   # type: Optional[int]
              return_meta=False,  # type: Union[bool, List[AnyStr]]
              **kwargs
              ):                  # type: (...) -> Dict[AnyStr, Any]
    """
    Creates (and optionally displays) a 2D t-SNE visualization of sample embeddings.

    By default, all samples from the data loader will be projected using the model and used
    for the visualization. If the task is related to classification, the prediction and groundtruth
    labels will be highlighting using various colors.

    If the model does not possess a ``get_embedding`` attribute, its raw output will be
    used for projections. Otherwise, ``get_embedding`` will be called.

    Args:
        model: the model which will be used to produce embeddings.
        task: the task object used to decode predictions and color samples (if possible).
        loader: the data loader used to get data samples to project.
        draw: boolean flag used to toggle internal display call on or off.
        color_map: map of RGB triplets used to color predictions (for classification only).
        max_samples: maximum number of samples to draw from the data loader.
        return_meta: toggles whether sample metadata should be provided as output or not.

    Returns:
        A dictionary of the visualization result (an RGB image in numpy format), a list of projected
        embedding coordinates, the labels of the samples, and the predictions of the samples.
    """
    assert loader is not None and len(loader) > 0, "no available data to load"
    assert model is not None and isinstance(model, torch.nn.Module), "invalid model"
    assert task is not None and isinstance(task, thelper.tasks.Task), "invalid task"
    assert max_samples is None or max_samples > 0, "invalid maximum loader sample count"
    thelper.viz.logger.debug("fetching data loader samples for t-SNE visualization...")
    embeddings, labels, preds, idxs = [], [], [], []
    if isinstance(task, thelper.tasks.Classification):
        assert all([isinstance(n, str) for n in task.class_names]), "unexpected class name types"
        if not color_map:
            if hasattr(task, "color_map"):
                color_map = task.color_map
            else:
                color_map = {idx: thelper.draw.get_label_color_mapping(idx + 1) for idx in task.class_indices.values()}
        color_map = {idx: f"#{c[0]:02X}{c[1]:02X}{c[2]:02X}" for idx, c in color_map.items()}
    if isinstance(return_meta, bool):
        return_meta = task.meta_keys if return_meta else []
    assert isinstance(return_meta, list) and all([isinstance(key, str) for key in return_meta]), \
        "sample metadata keys must be provided as a list of strings"
    meta = {key: [] for key in return_meta}
    for sample_idx, sample in tqdm.tqdm(enumerate(loader), desc="extracting embeddings"):
        if max_samples is not None and sample_idx > max_samples:
            break
        with torch.no_grad():
            input_tensor = sample[task.input_key]
            if task is not None and isinstance(task, thelper.tasks.Classification) and task.gt_key in sample:
                label = sample[task.gt_key]
                if isinstance(label, torch.Tensor):
                    label = label.cpu().numpy()
                if all([isinstance(lbl, str) for lbl in label]):
                    label = [task.class_indices[lbl] for lbl in label]
                pred = model(input_tensor).topk(k=1, dim=1)[1].view(input_tensor.size(0)).cpu().numpy()
                labels.append(label)
                preds.append(pred)
            if hasattr(model, "get_embedding"):
                embedding = model.get_embedding(input_tensor)
            else:
                if not thelper.viz.warned_missing_get_embedding:
                    thelper.viz.logger.warning("missing 'get_embedding' function in model object; will use output instead")
                    thelper.viz.warned_missing_get_embedding = True
                embedding = model(input_tensor)
            if embedding.dim() > 2:  # reshape to BxC
                embedding = embedding.view(embedding.size(0), -1)
        embeddings.append(embedding.cpu().numpy())
        idxs.append(sample_idx)
        for key in return_meta:
            for v in sample[key]:
                meta[key].append(v)
    embeddings = np.concatenate(embeddings)
    if labels and preds:
        labels, preds = np.concatenate(labels), np.concatenate(preds)
    else:
        labels, preds = [0] * len(embeddings), [0] * len(embeddings)
    default_tsne_args = {"n_components": 2, "init": "pca", "random_state": 0}
    tsne_args = thelper.utils.get_key_def("tsne_args", kwargs, default_tsne_args)
    tsne = sklearn.manifold.TSNE(**tsne_args)
    thelper.viz.logger.debug("computing t-SNE projection...")
    embeddings = tsne.fit_transform(embeddings)
    fig = plot(embeddings, labels, preds, color_map=color_map, task=task, **kwargs)
    img = thelper.draw.fig2array(fig).copy()
    if draw:
        thelper.viz.logger.debug("displaying t-SNE projection...")
        cv.imshow("thelper.viz.tsne", img[..., ::-1])  # RGB to BGR for opencv display
        cv.waitKey(1)
    return {
        # key formatting should be compatible with _write_data in thelper/train/base.py
        "tsne-projs/pickle": embeddings,
        "tsne-labels/json": labels.tolist(),
        "tsne-preds/json": preds.tolist(),
        "tsne-idxs/json": idxs,
        "tsne-meta/json": meta,
        "tsne/image": img
    }
