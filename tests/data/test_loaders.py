import random

import numpy as np
import pytest
import torch

import thelper


class CustomTensorDataset(torch.utils.data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        rand_max = 2 ** 16 - 1
        return (self.tensor[index],
                np.random.randint(rand_max),
                torch.randint(0, rand_max, size=(1,)).item(),
                random.randint(0, rand_max))

    def __len__(self):
        return self.tensor.size(0)


@pytest.fixture
def tensor_dataset():
    return CustomTensorDataset(torch.Tensor([torch.Tensor([v]) for v in range(100)]))


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_tensor_loader_interface(tensor_dataset, num_workers):
    loader = thelper.data.DataLoader(tensor_dataset, num_workers=num_workers, batch_size=2, seeds={
        "torch": 0, "numpy": 0, "random": 0
    })
    assert len(loader) == len(tensor_dataset) // 2
    assert loader.sample_count == len(tensor_dataset)
    with pytest.raises(AssertionError):
        thelper.data.DataLoader(tensor_dataset, seeds=0)
    with pytest.raises(AssertionError):
        thelper.data.DataLoader(tensor_dataset, epoch=None)
    rand_vals = []
    assert loader.epoch == 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            rand_vals = batch[1:4]
        else:
            # pretty unlikely to get triple collision...
            assert not all([torch.all(torch.eq(rand_vals[idx], batch[1:4][idx])) for idx in range(3)])
        assert batch[0].shape == (2,)
    assert loader.epoch == 1
    with pytest.raises(AssertionError):
        loader.set_epoch(None)
    loader.set_epoch(0)
    assert loader.epoch == 0
    for batch in loader:
        assert all([torch.all(torch.eq(rand_vals[idx], batch[1:4][idx])) for idx in range(3)])
        break
    assert loader.epoch == 1


class DummyClassifDataset(thelper.data.Dataset):
    def __init__(self, nb_samples, nb_classes, subset, transforms=None, deepcopy=False):
        super().__init__(transforms=transforms, deepcopy=deepcopy)
        inputs = torch.randint(0, 2 ** 16 - 1, size=(nb_samples, 1))
        labels = torch.remainder(torch.randperm(nb_samples), nb_classes)
        self.samples = [{"input": inputs[idx], "label": labels[idx], "idx": idx, "subset": subset} for idx in range(nb_samples)]
        self.task = thelper.tasks.Classification([str(idx) for idx in range(nb_classes)], "input", "label", meta_keys=["idx", "subset"])

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_task(self):
        return self.task


@pytest.fixture
def class_split_config():
    return {
        "datasets": {
            "dataset_A": DummyClassifDataset(1000, 10, "A"),
            "dataset_B": DummyClassifDataset(1000, 10, "B"),
            "dataset_C": DummyClassifDataset(1000, 10, "C")
        },
        "loaders": {
            "batch_size": 32,
            "train_split": {
                "dataset_A": 0.5,
                "dataset_B": 0.7
            },
            "valid_split": {
                "dataset_A": 0.4,
                "dataset_B": 0.3
            },
            "test_split": {
                "dataset_A": 0.1,
                "dataset_C": 1.0
            }
        }
    }


def test_classif_split(class_split_config):
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(class_split_config)
    assert task.check_compat(class_split_config["datasets"]["dataset_A"].get_task(), exact=True)
    assert task.check_compat(class_split_config["datasets"]["dataset_B"].get_task(), exact=True)
    assert task.check_compat(class_split_config["datasets"]["dataset_C"].get_task(), exact=True)
    train_samples, valid_samples, test_samples = {}, {}, {}
    train_class_counts, valid_class_counts, test_class_counts = [0] * 10, [0] * 10, [0] * 10
    for loader, samples, class_counts in [(train_loader, train_samples, train_class_counts),
                                          (valid_loader, valid_samples, valid_class_counts),
                                          (test_loader, test_samples, test_class_counts)]:
        for batch in loader:
            for idx in range(batch["input"].size(0)):
                name = batch["subset"][idx] + str(batch["idx"][idx].item())
                assert name not in samples
                label = batch["label"][idx].item()
                class_counts[label] += 1
                samples[name] = label
    assert not bool(set(train_samples) & set(valid_samples))
    assert not bool(set(train_samples) & set(test_samples))
    assert not bool(set(valid_samples) & set(test_samples))
    assert all([count == train_class_counts[0] for count in train_class_counts])
    assert all([count == valid_class_counts[0] for count in valid_class_counts])
    assert all([count == test_class_counts[0] for count in test_class_counts])


def test_classif_split_no_balancing(class_split_config):
    class_split_config["loaders"]["skip_class_balancing"] = True
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(class_split_config)
    assert task.check_compat(class_split_config["datasets"]["dataset_A"].get_task(), exact=True)
    assert task.check_compat(class_split_config["datasets"]["dataset_B"].get_task(), exact=True)
    assert task.check_compat(class_split_config["datasets"]["dataset_C"].get_task(), exact=True)
    train_samples, valid_samples, test_samples = {}, {}, {}
    for loader, samples in [(train_loader, train_samples), (valid_loader, valid_samples), (test_loader, test_samples)]:
        for batch in loader:
            for idx in range(batch["input"].size(0)):
                name = batch["subset"][idx] + str(batch["idx"][idx].item())
                assert name not in samples
                label = batch["label"][idx].item()
                samples[name] = label
    assert not bool(set(train_samples) & set(valid_samples))
    assert not bool(set(train_samples) & set(test_samples))
    assert not bool(set(valid_samples) & set(test_samples))
