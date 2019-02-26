"""Training/evaluation utilities module.

This module contains utilities and tools used to instantiate training sessions.
"""

import logging
from typing import AnyStr, Optional  # noqa: F401

import thelper.utils

logger = logging.getLogger(__name__)


def create_trainer(session_name,    # type: AnyStr
                   save_dir,        # type: AnyStr
                   config,          # type: thelper.typedefs.ConfigDict
                   model,           # type: thelper.typedefs.ModelType
                   loaders,         # type: thelper.typedefs.MultiLoaderType
                   ckptdata=None    # type: Optional[thelper.typedefs.CheckpointContentType]
                   ):               # type: (...) -> thelper.train.Trainer
    """Instantiates the trainer object based on the type contained in the config dictionary.

    The trainer type is expected to be in the configuration dictionary's `trainer` field, under the `type` key. For more
    information on the configuration, refer to :class:`thelper.train.trainers.Trainer`. The instantiated type must be
    compatible with the constructor signature of :class:`thelper.train.trainers.Trainer`. The object's constructor will
    be given the full config dictionary and the checkpoint data for resuming the session (if available).

    Args:
        session_name: name of the training session used for printing and to create internal tensorboardX directories.
        save_dir: path to the session directory where logs and checkpoints will be saved.
        config: full configuration dictionary that will be parsed for trainer parameters and saved in checkpoints.
        model: model to train/evaluate; should be compatible with :class:`thelper.nn.utils.Module`.
        loaders: a tuple containing the training/validation/test data loaders (a loader can be ``None`` if empty).
        ckptdata: raw checkpoint to parse data from when resuming a session (if ``None``, will start from scratch).

    Returns:
        The fully-constructed trainer object, ready to begin model training/evaluation.

    .. seealso::
        | :class:`thelper.train.trainers.Trainer`

    """
    if "trainer" not in config or not config["trainer"]:
        raise AssertionError("config missing 'trainer' field")
    trainer_config = config["trainer"]
    if "type" not in trainer_config or not trainer_config["type"]:
        raise AssertionError("trainer config missing 'type' field")
    trainer_type = thelper.utils.import_class(trainer_config["type"])
    return trainer_type(session_name, save_dir, model, loaders, config, ckptdata=ckptdata)


# noinspection PyUnusedLocal
def _draw_minibatch_wrapper(sample,         # type: thelper.typedefs.SampleType
                            task,           # type: thelper.tasks.utils.Task
                            pred,           # type: thelper.typedefs.PredictionType
                            iter_idx,       # type: int
                            max_iters,      # type: int
                            epoch_idx,      # type: int
                            max_epochs,     # type: int
                            ):              # type: (...) -> None

    """Wrapper to :func:`thelper.utils.draw_minibatch` used as a callback entrypoint for trainers."""
    thelper.utils.draw_minibatch(sample, task, preds=pred, ch_transpose=True, flip_bgr=False, block=False)
