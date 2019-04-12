import copy
import shutil

import pytest

import thelper


@pytest.fixture
def config():
    return {
        "name": "test-classif-mnist",
        "bypass_queries": True,
        "overwrite": True,
        "datasets": {
            "mnist": {
                "type": "torchvision.datasets.MNIST",
                "params": {
                    "root": "data/mnist/test",
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
    shutil.rmtree("data/saved/test-classif-mnist", ignore_errors=True)
    train_outputs = thelper.cli.create_session(config, "data/saved")
    assert len(train_outputs) == 2
    assert train_outputs[0]["train/metrics"]["accuracy"] < train_outputs[1]["train/metrics"]["accuracy"]
    ckptdata = thelper.utils.load_checkpoint("data/saved/test-classif-mnist", always_load_latest=True)
    override_config = copy.deepcopy(config)
    override_config["trainer"]["epochs"] = 3
    resume_outputs = thelper.cli.resume_session(ckptdata, save_dir="data/saved", config=override_config)
    assert len(resume_outputs) == 3
    assert train_outputs[1]["train/metrics"]["accuracy"] < resume_outputs[2]["train/metrics"]["accuracy"]
    ckptdata = thelper.utils.load_checkpoint("data/saved/test-classif-mnist")
    eval_outputs = thelper.cli.resume_session(ckptdata, save_dir="data/saved", eval_only=True)
    assert any(["test/metrics" in v for v in eval_outputs.values()])
    override_config["trainer"]["epochs"] = 1
    override_config["model"] = {"ckptdata": "data/saved/test-classif-mnist"}
    override_config["name"] = "test-classif-mnist-finetune"
    shutil.rmtree("data/saved/test-classif-mnist-finetune", ignore_errors=True)
    finetune_outputs = thelper.cli.create_session(override_config, "data/saved")
    assert len(finetune_outputs) == 1
    assert finetune_outputs[0]["train/metrics"]["accuracy"] > train_outputs[1]["train/metrics"]["accuracy"]
    shutil.rmtree("data/saved/test-classif-mnist", ignore_errors=True)
    shutil.rmtree("data/saved/test-classif-mnist-finetune", ignore_errors=True)
