from typing import AnyStr, Optional

import thelper.infer.base
import thelper.tasks


def create_tester(session_name,    # type: AnyStr
                  save_dir,        # type: AnyStr
                  config,          # type: thelper.typedefs.ConfigDict
                  model,           # type: thelper.typedefs.ModelType
                  task,            # type: thelper.tasks.Task
                  loaders,         # type: thelper.typedefs.MultiLoaderType
                  ckptdata=None    # type: Optional[thelper.typedefs.CheckpointContentType]
                  ):               # type: (...) -> thelper.infer.base.Tester
    """Instantiates the tester object based on the type contained in the config dictionary.

    The tester type is expected to be in the configuration dictionary's `tester` field, under the `type` key.
    For backward compatibility, the fields `runner` and `trainer` will also be looked for.
    For more
    information on the configuration, refer to :class:`thelper.train.base.Trainer`. The instantiated type must be
    compatible with the constructor signature of :class:`thelper.train.base.Trainer`. The object's constructor will
    be given the full config dictionary and the checkpoint data for resuming the session (if available).

    If the trainer type is missing, it will be automatically deduced based on the task object.

    Args:
        session_name: name of the training session used for printing and to create internal tensorboardX directories.
        save_dir: path to the session directory where logs and checkpoints will be saved.
        config: full configuration dictionary that will be parsed for trainer parameters and saved in checkpoints.
        model: model to train/evaluate; should be compatible with :class:`thelper.nn.utils.Module`.
        task: global task interface defining the type of model and training goal for the session.
        loaders: a tuple containing the training/validation/test data loaders (a loader can be ``None`` if empty).
        ckptdata: raw checkpoint to parse data from when resuming a session (if ``None``, will start from scratch).

    Returns:
        The fully-constructed trainer object, ready to begin model training/evaluation.

    .. seealso::
        | :class:`thelper.infer.base.Tester`

    """

    # NOTE:
    #   counter intuitive name 'trainer', but nothing will actually be trained, only to match other thelper modes
    runner_config = config.get("tester", config.get("runner", config.get("trainer")))
    if not runner_config or not isinstance(runner_config, dict):
        raise AssertionError("Could not retrieve any session runner definition from configuration")

    if "type" not in runner_config:
        if isinstance(task, thelper.tasks.Classification):
            runner_type = thelper.infer.ImageClassifTester
        elif isinstance(task, thelper.tasks.Detection):
            runner_type = thelper.infer.ObjDetectTester
        elif isinstance(task, thelper.tasks.Regression):
            runner_type = thelper.infer.RegressionTester
        elif isinstance(task, thelper.tasks.Segmentation):
            runner_type = thelper.infer.ImageSegmTester
        else:
            raise AssertionError(f"unknown trainer type required for task '{str(task)}'")
    else:
        runner_type = thelper.utils.import_class(runner_config["type"])
    return runner_type(session_name, save_dir, model, task, loaders, config, ckptdata=ckptdata)
