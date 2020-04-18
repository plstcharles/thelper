from abc import abstractmethod
from typing import TYPE_CHECKING

from thelper.train.base import Trainer

if TYPE_CHECKING:
    from typing import AnyStr, Callable, Optional, Type  # noqa: F401
    import thelper.typedefs  # noqa: F401


def make_tester_from_trainer(trainer):
    # type: (Type[Trainer]) -> Callable
    def make_tester(tester):
        # type: (Type[Tester]) -> Callable
        """
        Decorator that wraps a Tester session runner by replacing any training-related methods with RuntimeError
        to make sure they cannot be erroneously called.
        It also adds any missing testing-related method from the base tester in order to support redirection
        to evaluation methods of the specified trainer.
        """
        class TesterWrapper(object):
            def __new__(cls, *args, **kwargs):
                cls.__wrapped__ = tester
                setattr(cls, "eval", lambda *a, **kw: trainer.eval(*a, **kw))
                setattr(cls, "eval_epoch", lambda *a, **kw: trainer.eval_epoch(*a, **kw))
                # if item correctly inherits from Tester, redirects should already be there
                # but make sure that a direct reference to a Trainer class for inference will still work
                if not hasattr(cls, "test"):
                    setattr(cls, "test", getattr(tester, "test"))
                if not hasattr(cls, "test_epoch"):
                    setattr(cls, "test_epoch", getattr(tester, "test_epoch"))
                return cls

            def train(self):
                raise RuntimeError(f"Invalid call to 'train' using '{tester.__name__}' (Tester)")

            def train_epoch(self, model, epoch, dev, loss, optimizer, loader, metrics, output_path):
                raise RuntimeError(f"Invalid call to 'train_epoch' using '{tester.__name__}' (Tester)")

        return TesterWrapper
    return make_tester


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
        super(Trainer, self).__init__(session_name, session_dir, model, task, loaders, config, ckptdata=ckptdata)

    def train(self):
        raise RuntimeError(f"Invalid call to 'train' using '{type(self).__name__}' (Tester)")

    def train_epoch(self, model, epoch, dev, loss, optimizer, loader, metrics, output_path):
        raise RuntimeError(f"Invalid call to 'train_epoch' using '{type(self).__name__}' (Tester)")

    def test(self):
        __doc__ = self.eval.__doc__  # noqa:F841
        return self.eval()

    def test_epoch(self, *args, **kwargs):
        __doc__ = self.eval_epoch.__doc__  # noqa:F841
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
