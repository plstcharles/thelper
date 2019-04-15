import copy
import os
import shutil

import pytest

import thelper

test_save_path = ".pytest_cache"

test_classif_mnist_name = "test-classif-mnist"
test_classif_mnist_path = os.path.join(test_save_path, test_classif_mnist_name)
test_classif_mnist_ft_path = os.path.join(test_save_path, test_classif_mnist_name + "-finetune")


@pytest.fixture
def config(request):
    def fin():
        shutil.rmtree(test_classif_mnist_path, ignore_errors=True)
        shutil.rmtree(test_classif_mnist_ft_path, ignore_errors=True)
    fin()
    request.addfinalizer(fin)
    return {
        "name": test_classif_mnist_name,
        "bypass_queries": True,
        "datasets": {
            "mnist": {
                "type": "torchvision.datasets.MNIST",
                "params": {
                    "root": os.path.join(test_classif_mnist_path, "mnist"),
                    "train": False,  # use test set, its smaller (quicker test)
                    "download": True
                },
                "task": {
                    "type": "thelper.tasks.Classification",
                    "params": {
                        "class_names": [
                            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
                        ],
                        "input_key": "0",
                        "label_key": "1"
                    }
                }
            }
        },
        "loaders": {
            "shuffle": True,
            "batch_size": 32,
            "workers": 0,
            "skip_class_balancing": True,
            "base_transforms": [
                {
                    "operation": "thelper.transforms.NormalizeMinMax",
                    "params": {
                        "min": [127],
                        "max": [255]
                    }
                },
                {
                    "operation": "thelper.transforms.Unsqueeze",
                    "params": {
                        "axis": 0
                    }
                }

            ],
            "train_scale": 0.1,
            "train_split": {
                "mnist": 0.8
            },
            "valid_scale": 0.1,
            "valid_split": {
                "mnist": 0.1
            },
            "test_scale": 0.1,
            "test_split": {
                "mnist": 0.1
            }
        },
        "model": {
            "type": "thelper.nn.lenet.LeNet"
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


def test_classif_mnist(config):
    train_outputs = thelper.cli.create_session(config, test_save_path)
    assert len(train_outputs) == 2
    assert train_outputs[0]["train/metrics"]["accuracy"] < train_outputs[1]["train/metrics"]["accuracy"]
    ckptdata = thelper.utils.load_checkpoint(test_classif_mnist_path, always_load_latest=True)
    override_config = copy.deepcopy(config)
    override_config["trainer"]["epochs"] = 3
    resume_outputs = thelper.cli.resume_session(ckptdata, save_dir=test_save_path, config=override_config)
    assert len(resume_outputs) == 3
    assert train_outputs[1]["train/metrics"]["accuracy"] < resume_outputs[2]["train/metrics"]["accuracy"]
    ckptdata = thelper.utils.load_checkpoint(test_classif_mnist_path)
    eval_outputs = thelper.cli.resume_session(ckptdata, save_dir=test_save_path, eval_only=True)
    assert any(["test/metrics" in v for v in eval_outputs.values()])
    override_config["trainer"]["epochs"] = 1
    override_config["model"] = {"ckptdata": test_classif_mnist_path}
    override_config["name"] += "-finetune"
    finetune_outputs = thelper.cli.create_session(override_config, test_save_path)
    assert len(finetune_outputs) == 1
    assert finetune_outputs[0]["train/metrics"]["accuracy"] > train_outputs[1]["train/metrics"]["accuracy"]
