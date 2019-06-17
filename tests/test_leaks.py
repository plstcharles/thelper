import os
import shutil

import numpy as np
import pytest
import torch

import thelper
import thelper.nn.resnet

test_save_path = ".pytest_cache"
test_classif_leak_name = "test-classif-leak"

test_classif_leak_path = os.path.join(test_save_path, test_classif_leak_name)


def check_gpu_compat():
    return torch.cuda.is_available() and \
        torch.cuda.get_device_properties(torch.device("cuda")).total_memory >= 4 * 1024 * 1024 * 1024


@pytest.mark.skipif(not check_gpu_compat(), reason="test requires GPU w/ >4GB RAM")
def test_model_leak():
    device = torch.device("cuda")
    for test_idx in range(25):
        model = thelper.nn.resnet.ResNet(thelper.tasks.Classification([str(idx) for idx in range(1000)], "0", "1")).to(device)
        model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        batch_size = 40
        for iter_idx in range(5):
            optimizer.zero_grad()
            pred = model(torch.randn(batch_size, 3, 224, 224).to(device))
            loss_fn(pred, torch.randint(1000, size=(batch_size,)).to(device)).backward()
            optimizer.step()
        del pred
        del optimizer
        del model


class DummyDataset(thelper.data.Dataset):
    def __init__(self, nb_samples, transforms=None, deepcopy=None):
        super().__init__(transforms=transforms, deepcopy=deepcopy)
        self.samples = np.arange(nb_samples)
        self.task = thelper.tasks.Classification([str(idx) for idx in range(1000)], "0", "1")

    def __getitem__(self, idx):  # pragma: no cover
        return {"0": torch.randn(3, 224, 224), "1": torch.randint(1000, size=())}


@pytest.fixture
def config(request):
    def fin():
        shutil.rmtree(test_classif_leak_path, ignore_errors=True)
    fin()
    request.addfinalizer(fin)
    return {
        "name": test_classif_leak_name,
        "bypass_queries": True,
        "datasets": {
            "dummy": DummyDataset(100)
        },
        "loaders": {
            "shuffle": True,
            "batch_size": 35,
            "skip_class_balancing": True,
            "train_split": {
                "dummy": 0.9
            },
            "valid_split": {
                "dummy": 0.1
            }
        },
        "model": {
            "type": "thelper.nn.resnet.ResNet"
        },
        "trainer": {
            "type": "thelper.train.ImageClassifTrainer",
            "epochs": 1,
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


@pytest.mark.skipif(not check_gpu_compat(), reason="test requires GPU w/ >4GB RAM")
def test_train_decomposed_leak(config, mocker):
    mocker.patch("json.dump")
    mocker.patch("torch.save")
    session_name = config["name"]
    save_dir = thelper.utils.get_save_dir(test_save_path, session_name, config)
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(config, save_dir)
    loaders = (train_loader, valid_loader, test_loader)
    outputs = []
    for idx in range(5):
        model = thelper.nn.create_model(config, task, save_dir=save_dir)
        trainer = thelper.train.create_trainer(session_name, save_dir, config, model, task, loaders)
        trainer.train()
        outputs.append(trainer.outputs)


@pytest.mark.skipif(not check_gpu_compat(), reason="test requires GPU w/ >4GB RAM")
def test_train_leak(config, mocker):
    mocker.patch("json.dump")
    mocker.patch("torch.save")
    for idx in range(5):
        train_outputs = thelper.cli.create_session(config, test_save_path)
        del train_outputs
