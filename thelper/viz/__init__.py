"""Visualization package.

In contrast with :mod:`thelper.draw`, this package regroups utilities and tools used to create
visualizations that can be used to debug and understand the behavior of models. For example, it
contains a t-SNE module that can create a projection of high-dimensional embeddings created by
a model, and different modules to visualize the image regions that cause certain activations
inside a model. All these techniques are used to create logs and/or images which in turn can be
displayed using the :mod:`thelper.draw` module.
"""

import logging
from typing import Any, AnyStr  # noqa: F401

import thelper.typedefs  # noqa: F401
import thelper.viz.tsne  # noqa: F401
import thelper.viz.umap  # noqa: F401

logger = logging.getLogger("thelper.viz")

supported_types = ["tsne", "umap"]  # more will be added once properly tested

warned_missing_get_embedding = False


def visualize(model,         # type: thelper.typedefs.ModelType
              task,          # type: thelper.typedefs.TaskType
              loader,        # type: thelper.typedefs.LoaderType
              viz_type,      # type: AnyStr
              **kwargs
              ):             # type: (...) -> Any
    """Dispatches a visualization call to the proper package module."""
    assert viz_type in supported_types, f"unsupported visualization type '{viz_type}'"
    if viz_type == "tsne":
        return thelper.viz.tsne.visualize(model, task, loader, **kwargs)
    elif viz_type == "umap":
        return thelper.viz.umap.visualize(model, task, loader, **kwargs)
