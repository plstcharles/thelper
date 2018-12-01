"""Optimization/metrics utility module.

This module contains utility functions and tools used by the other modules of this package.
"""

import logging

import thelper.optim.metrics
import thelper.utils

logger = logging.getLogger(__name__)


def create_metrics(config):
    """Instantiates and returns the metrics defined in the configuration dictionary.

    All arguments are expected to be handed in through the configuration via a dictionary named 'params'.
    """
    if not isinstance(config, dict):
        raise AssertionError("config should be provided as a dictionary")
    metrics = {}
    for name, metric_config in config.items():
        if not isinstance(metric_config, dict):
            raise AssertionError("metric config should be provided as a dictionary")
        if "type" not in metric_config or not metric_config["type"]:
            raise AssertionError("metric config missing 'type' field")
        metric_type = thelper.utils.import_class(metric_config["type"])
        metric_params = thelper.utils.get_key_def("params", metric_config, {})
        metric = metric_type(**metric_params)
        if not isinstance(metric, thelper.optim.metrics.Metric):
            raise AssertionError("invalid metric type, must derive from 'thelper.optim.metrics.Metric'")
        goal = getattr(metric, "goal", None)
        if not callable(goal):
            raise AssertionError("expected metric to define 'goal' based on parent interface")
        metrics[name] = metric
    return metrics
