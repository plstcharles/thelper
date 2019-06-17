import copy
import math
import random

import numpy as np
import pytest
import torch

import thelper


class CustomSampler(torch.utils.data.sampler.RandomSampler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = None

    def set_epoch(self, epoch=0):
        self.epoch = epoch


class CustomTransformer:

    def __init__(self):
        self.epoch = None

    def __call__(self, x):
        return x

    def set_epoch(self, epoch=0):
        self.epoch = epoch


class CustomTensorDataset(torch.utils.data.Dataset):

    def __init__(self, tensor):
        self.tensor = tensor
        self.epoch = None
        self.transforms = CustomTransformer()

    def __getitem__(self, index):
        rand_max = 2 ** 16 - 1
        return self.transforms((self.tensor[index],
                                np.random.randint(rand_max),
                                torch.randint(0, rand_max, size=(1,)).item(),
                                random.randint(0, rand_max)))

    def __len__(self):
        return self.tensor.size(0)

    def set_epoch(self, epoch=0):
        self.epoch = epoch


@pytest.fixture
def tensor_dataset():
    return CustomTensorDataset(torch.Tensor([torch.Tensor([v]) for v in range(100)]))


@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_tensor_loader_interface(tensor_dataset, num_workers):
    loader = thelper.data.DataLoader(tensor_dataset, num_workers=num_workers, batch_size=2, seeds={
        "torch": 0, "numpy": 0, "random": 0
    })
    assert len(loader) == len(tensor_dataset) // 2
    loader = thelper.data.DataLoader(tensor_dataset, sampler=CustomSampler(tensor_dataset), num_workers=num_workers, batch_size=2, seeds={
        "torch": 0, "numpy": 0, "random": 0
    })
    assert len(loader) == len(tensor_dataset) // 2
    assert loader.sample_count == len(tensor_dataset)
    with pytest.raises(AssertionError):
        _ = thelper.data.DataLoader(tensor_dataset, seeds=0)
    with pytest.raises(AssertionError):
        _ = thelper.data.DataLoader(tensor_dataset, epoch=None)
    rand_vals = []
    assert loader.epoch == 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            rand_vals = batch[1:4]
        else:
            # pretty unlikely to get triple collision...
            assert not all([torch.all(torch.eq(rand_vals[idx], batch[1:4][idx])) for idx in range(3)])
        assert loader.sampler.epoch == 0
        assert loader.dataset.epoch == 0
        assert loader.dataset.transforms.epoch == 0
        assert batch[0].shape == (2,)
    assert loader.epoch == 1
    with pytest.raises(AssertionError):
        loader.set_epoch(None)
    loader.set_epoch(0)
    assert loader.epoch == 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            assert all([torch.all(torch.eq(rand_vals[idx], batch[1:4][idx])) for idx in range(3)])
        assert batch[0].shape == (2,)
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
    assert task.check_compat(class_split_config["datasets"]["dataset_A"].task, exact=True)
    assert task.check_compat(class_split_config["datasets"]["dataset_B"].task, exact=True)
    assert task.check_compat(class_split_config["datasets"]["dataset_C"].task, exact=True)
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
    assert task.check_compat(class_split_config["datasets"]["dataset_A"].task, exact=True)
    assert task.check_compat(class_split_config["datasets"]["dataset_B"].task, exact=True)
    assert task.check_compat(class_split_config["datasets"]["dataset_C"].task, exact=True)
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


def collate_fn(*args, **kwargs):  # pragma: no cover
    return torch.utils.data.dataloader.default_collate(*args, **kwargs)


def test_loader_factory(class_split_config, mocker):
    fake_loaders_config = copy.deepcopy(class_split_config["loaders"])
    fake_loaders_config["train_batch_size"] = 16
    with pytest.raises(AssertionError):
        _ = thelper.data.loaders.LoaderFactory(fake_loaders_config)
    factory = thelper.data.loaders.LoaderFactory(class_split_config["loaders"])
    assert factory.train_split["dataset_A"] == 0.5 and factory.valid_split["dataset_A"] == 0.4 and factory.test_split["dataset_A"] == 0.1
    assert factory.train_split["dataset_B"] == 0.7 and factory.valid_split["dataset_B"] == 0.3
    assert factory.test_split["dataset_C"] == 1.0
    loaders_config_drop = copy.deepcopy(class_split_config["loaders"])
    loaders_config_drop["drop_last"] = True
    factory = thelper.data.loaders.LoaderFactory(loaders_config_drop)
    assert factory.drop_last
    loaders_config_sampler = copy.deepcopy(class_split_config["loaders"])
    loaders_config_sampler["sampler"] = {
        "type": "torch.utils.data.sampler.RandomSampler"
    }
    factory = thelper.data.loaders.LoaderFactory(loaders_config_sampler)
    assert factory.sampler_type == torch.utils.data.sampler.RandomSampler
    assert factory.train_sampler
    _ = mocker.patch("thelper.transforms.load_transforms", return_value="dummy")
    loaders_config_transfs = copy.deepcopy(class_split_config["loaders"])
    loaders_config_transfs["base_transforms"] = [
        {
            "operation": "thelper.transforms.RandomResizedCrop",
            "params": {
                "output_size": [100, 100]
            }
        },
        {
            "operation": "thelper.transforms.CenterCrop",
            "params": {
                "size": [200, 200]
            }
        }
    ]
    factory = thelper.data.loaders.LoaderFactory(loaders_config_transfs)
    assert factory.base_transforms is not None
    assert factory.get_base_transforms() == factory.base_transforms
    loaders_config_transfs["train_augments"] = {"append": True, "transforms": [
        {
            "operation": "thelper.transforms.RandomResizedCrop",
            "params": {
                "output_size": [100, 100]
            }
        },
        {
            "operation": "thelper.transforms.CenterCrop",
            "params": {
                "size": [200, 200]
            }
        }
    ]}
    factory = thelper.data.loaders.LoaderFactory(loaders_config_transfs)
    assert factory.train_augments is not None and factory.train_augments_append
    fake_loaders_split_config = copy.deepcopy(class_split_config["loaders"])
    del fake_loaders_split_config["train_split"]
    del fake_loaders_split_config["valid_split"]
    del fake_loaders_split_config["test_split"]
    with pytest.raises(AssertionError):
        _ = thelper.data.loaders.LoaderFactory(fake_loaders_split_config)
    normalize_query = mocker.patch("thelper.utils.query_yes_no", return_value=True)
    denormalized_loaders_config = copy.deepcopy(class_split_config["loaders"])
    denormalized_loaders_config["train_split"]["dataset_A"] = 1.5
    factory = thelper.data.loaders.LoaderFactory(denormalized_loaders_config)
    assert math.isclose(factory.train_split["dataset_A"] + factory.valid_split["dataset_A"] + factory.test_split["dataset_A"], 1)
    denormalized_loaders_config["train_split"]["dataset_A"] = 0.1
    factory = thelper.data.loaders.LoaderFactory(denormalized_loaders_config)
    assert math.isclose(factory.train_split["dataset_A"] + factory.valid_split["dataset_A"] + factory.test_split["dataset_A"], 1)
    assert normalize_query.call_count == 1
    loaders_collate_config = copy.deepcopy(class_split_config["loaders"])
    loaders_collate_config["collate_fn"] = 0
    with pytest.raises(AssertionError):
        _ = thelper.data.loaders.LoaderFactory(loaders_collate_config)
    loaders_collate_config["collate_fn"] = collate_fn
    factory = thelper.data.loaders.LoaderFactory(loaders_collate_config)
    assert factory.train_collate_fn == collate_fn
    loaders_collate_config["collate_fn"] = {
        "type": "torch.utils.data.dataloader.default_collate"
    }
    factory = thelper.data.loaders.LoaderFactory(loaders_collate_config)
    assert factory.train_collate_fn == torch.utils.data.dataloader.default_collate
    loaders_collate_config["collate_fn"] = "torch.utils.data.dataloader.default_collate"
    factory = thelper.data.loaders.LoaderFactory(loaders_collate_config)
    assert factory.train_collate_fn == torch.utils.data.dataloader.default_collate
    loaders_collate_config["train_collate_fn"] = "torch.utils.data.dataloader.default_collate"
    with pytest.raises(AssertionError):
        _ = thelper.data.loaders.LoaderFactory(loaders_collate_config)
    loaders_seed_config = copy.deepcopy(class_split_config["loaders"])
    loaders_seed_config["test_seed"] = 0
    loaders_seed_config["valid_seed"] = 1
    loaders_seed_config["torch_seed"] = 2
    loaders_seed_config["numpy_seed"] = 3
    loaders_seed_config["random_seed"] = "4"
    with pytest.raises(AssertionError):
        _ = thelper.data.loaders.LoaderFactory(loaders_seed_config)
    loaders_seed_config["random_seed"] = 4
    factory = thelper.data.loaders.LoaderFactory(loaders_seed_config)
    assert factory.seeds["test"] == 0
    assert factory.seeds["valid"] == 1
    assert factory.seeds["torch"] == 2
    assert factory.seeds["numpy"] == 3
    assert factory.seeds["random"] == 4
