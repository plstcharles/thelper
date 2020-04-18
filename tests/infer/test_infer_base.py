import thelper.utils
from thelper.infer.base import Tester

from tests.train.train_utils import (  # noqa: F401 isort:skip
    mnist_config, test_classif_mnist_ft_path, test_classif_mnist_name,
    test_classif_mnist_path, test_save_path
)

config = mnist_config


def test_infer_base_tester_cannot_run_train(config):
    session_name = thelper.utils.get_config_session_name(config)
    save_dir = thelper.utils.get_save_dir(test_save_path, session_name, config)
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(config, save_dir)
    model = thelper.nn.create_model(config, task, save_dir=save_dir)
    loaders = (train_loader, valid_loader, test_loader)
    t = Tester(session_name, save_dir, model, task, loaders, config)
    try:
        t.train()
    except RuntimeError as ex:
        assert "Invalid call to 'train'" in str(ex) and "(Tester)" in str(ex), \
            "RuntimeError was raised, but not the expected one"
    else:
        raise AssertionError("Tester.train should not be allowed to be called")
