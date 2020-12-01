from abc import abstractmethod
from typing import TYPE_CHECKING

import thelper.utils
from thelper.train.base import Trainer

if TYPE_CHECKING:
    from typing import AnyStr, Callable, Optional, Type  # noqa: F401
    import thelper.typedefs  # noqa: F401


class Tester(Trainer):
    """Base interface of a session runner for testing.

    This call mostly delegates calls to existing Trainer implementation, but limiting their use to 'eval' methods to
    make sure that 'train' operations are not called by mistake.

    .. seealso::
        | :class:`thelper.train.base.Trainer`
    """

    def __init__(self,
                 session_name,    # type: AnyStr
                 session_dir,     # type: AnyStr
                 model,           # type: thelper.typedefs.ModelType
                 task,            # type: thelper.tasks.Task
                 loaders,         # type: thelper.typedefs.MultiLoaderType
                 config,          # type: thelper.typedefs.ConfigDict
                 ckptdata=None    # type: Optional[thelper.typedefs.CheckpointContentType]
                 ):
        runner_config = thelper.utils.get_key_def(["runner", "tester"], config) or {}
        # default epoch 0 if omitted as they are not actually needed for single pass inference
        if "epochs" not in runner_config:
            runner_config["epochs"] = 1
        config["trainer"] = runner_config
        if "tester" not in config:
            config["tester"] = runner_config
        super().__init__(session_name, session_dir, model, task, loaders, config, ckptdata=ckptdata)

    def train(self):
        raise RuntimeError(f"Invalid call to 'train' using '{type(self).__name__}' (Tester)")

    def train_epoch(self, model, epoch, dev, loss, optimizer, loader, metrics, output_path):
        raise RuntimeError(f"Invalid call to 'train_epoch' using '{type(self).__name__}' (Tester)")

    def test(self):
        return self.eval()

    def test_epoch(self, *args, **kwargs):
        return self.eval_epoch(*args, **kwargs)

    @abstractmethod
    def eval_epoch(self, model, epoch, dev, loader, metrics, output_path):
        """Evaluates the model using the provided objects.

        Args:
            model: the model with which to run inference that is already uploaded to the target device(s).
            epoch: the epoch index we are training for (0-based, and should normally only be 0 for single test pass).
            dev: the target device that tensors should be uploaded to (corresponding to model's device(s)).
            loader: the data loader used to get transformed test samples.
            metrics: the dictionary of metrics/consumers to report inference results (mostly loggers and basic report
                generator in this case since there shouldn't be ground truth labels to validate against).
            output_path: directory where output files should be written, if necessary.
        """
        raise NotImplementedError
