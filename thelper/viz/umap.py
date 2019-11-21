"""Tools related to the Uniform Manifold Approximation and Projection (UMAP).

For more information on UMAP, see https://github.com/lmcinnes/umap for the original author's
repository.
"""

from typing import Any, AnyStr, Dict, List, Optional, Union  # noqa: F401

import cv2 as cv
import numpy as np
import torch
import tqdm

import thelper.utils
from thelper.viz.tsne import plot


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
    Creates (and optionally displays) a 2D UMAP visualization of sample embeddings.

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
    assert thelper.utils.check_installed("umap"), \
        "could not import optional 3rd-party dependency 'umap-learn'; make sure you install it first!"
    import umap
    assert loader is not None and len(loader) > 0, "no available data to load"
    assert model is not None and isinstance(model, torch.nn.Module), "invalid model"
    assert task is not None and isinstance(task, thelper.tasks.Task), "invalid task"
    assert max_samples is None or max_samples > 0, "invalid maximum loader sample count"
    thelper.viz.logger.debug("fetching data loader samples for UMAP visualization...")
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
    seed = thelper.utils.get_key_def("seed", kwargs, 0)
    if seed is None:
        seed = np.random.randint(np.iinfo(np.int32).max)
    prev_state = np.random.get_state()
    np.random.seed(seed)
    default_umap_args = {"n_components": 2}
    umap_args = thelper.utils.get_key_def("umap_args", kwargs, default_umap_args)
    umap_engine = umap.UMAP(**umap_args)
    thelper.viz.logger.debug("computing UMAP projection...")
    embeddings = umap_engine.fit_transform(embeddings)
    np.random.set_state(prev_state)
    fig = plot(embeddings, labels, preds, color_map=color_map, task=task, **kwargs)
    img = thelper.draw.fig2array(fig).copy()
    if draw:
        thelper.viz.logger.debug("displaying UMAP projection...")
        cv.imshow("thelper.viz.umap", img[..., ::-1])  # RGB to BGR for opencv display
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
