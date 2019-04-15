import os
import shutil

import pytest

import thelper

test_save_path = ".pytest_cache"

test_create_simple_path = os.path.join(test_save_path, "simple")
test_create_simple_images_path = os.path.join(test_save_path, "simple_images")


@pytest.fixture
def simple_config(request):
    def fin():
        shutil.rmtree(test_create_simple_path, ignore_errors=True)
        shutil.rmtree(test_create_simple_images_path, ignore_errors=True)
    fin()
    request.addfinalizer(fin)
    os.makedirs(test_create_simple_images_path, exist_ok=True)
    for cls in range(10):
        os.makedirs(os.path.join(test_create_simple_images_path, str(cls)))
        for idx in range(10):
            open(os.path.join(test_create_simple_images_path, str(cls), str(idx) + ".jpg"), "a").close()
    return {
        "name": "session",
        "bypass_queries": True,
        "datasets": {
            "dset": {
                "type": "thelper.data.ImageFolderDataset",
                "params": {
                    "root": test_create_simple_images_path
                }
            }
        },
        "loaders": {
            "shuffle": True,
            "batch_size": 32,
            "skip_class_balancing": True,
            "train_split": {
                "dset": 0.8
            },
            "valid_split": {
                "dset": 0.1
            },
            "test_split": {
                "dset": 0.1
            }
        },
        "model": {
            "type": "thelper.nn.resnet.ResNet"
        },
        "trainer": {
            "type": "thelper.train.ImageClassifTrainer",
            "epochs": 2,
            "monitor": "accuracy",
            "optimization": {
                "loss": {
                    "type": "torch.nn.CrossEntropyLoss"
                },
                "optimizer": {
                    "type": "torch.optim.Adam",
                    "params": {
                        "lr": 0.001
                    }
                }
            },
            "metrics": {
                "accuracy": {
                    "type": "thelper.optim.CategoryAccuracy",
                    "params": {
                        "top_k": 1
                    }
                }
            }
        }
    }


def test_create_session_nameless(simple_config, mocker):
    fake_train = mocker.patch.object(thelper.train.base.Trainer, "train")
    fake_eval = mocker.patch.object(thelper.train.base.Trainer, "eval")
    del simple_config["name"]
    with pytest.raises(AssertionError):
        thelper.cli.create_session(simple_config, test_save_path)
    assert fake_train.call_count == 0
    assert fake_eval.call_count == 0


def test_create_session_train(simple_config, mocker):
    fake_train = mocker.patch.object(thelper.train.base.Trainer, "train")
    fake_eval = mocker.patch.object(thelper.train.base.Trainer, "eval")
    thelper.cli.create_session(simple_config, test_save_path)
    assert fake_train.call_count == 1
    assert fake_eval.call_count == 0


def test_create_session_eval(simple_config, mocker):
    fake_train = mocker.patch.object(thelper.train.base.Trainer, "train")
    fake_eval = mocker.patch.object(thelper.train.base.Trainer, "eval")
    del simple_config["loaders"]["train_split"]
    thelper.cli.create_session(simple_config, test_save_path)
    assert fake_train.call_count == 0
    assert fake_eval.call_count == 1
