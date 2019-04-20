"""Graphical User Interface (GUI) utility module.

This module contains various tools and utilities used to instantiate annotators and GUI elements.
"""
import logging

import thelper.utils

logger = logging.getLogger(__name__)


def create_key_listener(callback):
    """Returns a key press listener based on pynput.keyboard (used for mocking)."""
    import pynput.keyboard
    return pynput.keyboard.Listener(on_press=callback)


def create_annotator(session_name, save_dir, config, datasets):
    """Instantiates a GUI annotation tool based on the type contained in the config dictionary.

    The tool type is expected to be in the configuration dictionary's `annotator` field, under the `type` key. For more
    information on the configuration, refer to :class:`thelper.gui.annotators.Annotator`. The instantiated type must be
    compatible with the constructor signature of :class:`thelper.gui.annotators.Annotator`. The object's constructor will
    be given the full config dictionary.

    Args:
        session_name: name of the annotation session used for printing and to create output directories.
        save_dir: path to the session directory where annotations and other outputs will be saved.
        config: full configuration dictionary that will be parsed for annotator parameters.
        datasets: map of named dataset parsers that will provide the data to annotate.

    Returns:
        The fully-constructed annotator object, ready to begin annotation via its ``run()`` function.

    .. seealso::
        | :class:`thelper.gui.annotators.Annotator`

    """
    if "annotator" not in config or not config["annotator"]:
        raise AssertionError("config missing 'annotator' field")
    annotator_config = config["annotator"]
    if "type" not in annotator_config or not annotator_config["type"]:
        raise AssertionError("annotator config missing 'type' field")
    annotator_type = thelper.utils.import_class(annotator_config["type"])
    return annotator_type(session_name, config, save_dir, datasets)
