import collections
import copy
import math
import os
import random
import shutil

import numpy as np
import pytest
import torch

import thelper

test_save_path = ".pytest_cache"


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


def fake_sample_wrapper(sample):
    assert isinstance(sample, (tuple, list)) and len(sample) == 4
    return [*sample, "fake"]


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
    rand_vals = None
    assert loader.epoch == 0
    for batch_idx, batch in enumerate(loader):
        if rand_vals is None:
            rand_vals = batch[1:4]
        else:
            # pretty unlikely to get triple collision...
            assert not all([torch.all(torch.eq(rand_vals[idx], batch[1:4][idx])) for idx in range(3)])
        assert loader.sampler.epoch == 0
        assert loader.dataset.epoch == 0
        assert loader.dataset.transforms.epoch == 0
        assert batch[0].shape == (2,)
    assert loader.epoch == 1
    batch = next(iter(loader))
    assert not all([torch.all(torch.eq(rand_vals[idx], batch[1:4][idx])) for idx in range(3)])
    assert loader.epoch == 2
    loader.set_epoch(0)
    batch = next(iter(loader))
    assert all([torch.all(torch.eq(rand_vals[idx], batch[1:4][idx])) for idx in range(3)])
    with pytest.raises(AssertionError):
        loader.set_epoch(None)
    loader.set_epoch(0)
    assert loader.epoch == 0
    for batch_idx, batch in enumerate(loader):
        if batch_idx == 0:
            assert all([torch.all(torch.eq(rand_vals[idx], batch[1:4][idx])) for idx in range(3)])
        assert batch[0].shape == (2,)
    assert loader.epoch == 1
    wrapped_loader = thelper.data.DataLoaderWrapper(loader, fake_sample_wrapper)
    assert len(wrapped_loader) == len(loader)
    assert wrapped_loader.dataset == loader.dataset
    for batch_idx, batch in enumerate(wrapped_loader):
        assert len(batch) == 5 and batch[4] == "fake"
    loader = thelper.data.DataLoader(tensor_dataset, num_workers=num_workers)  # without fixed seed
    rand_vals = None
    assert loader.epoch == 0
    for loop1 in range(3):  # pragma: no cover
        for loop2 in range(3):
            for batch_idx, batch in enumerate(loader):
                if rand_vals is None:
                    rand_vals = batch[1:4]
                else:
                    # pretty unlikely to get triple collision...
                    assert not all([torch.all(torch.eq(rand_vals[idx], batch[1:4][idx])) for idx in range(3)])
                    break
            if loader.epoch > 1:
                break
        loader.set_epoch(0)


class DummyClassifDataset(thelper.data.Dataset):
    def __init__(self, nb_samples, nb_classes, subset, transforms=None, deepcopy=False, seed=None):
        super().__init__(transforms=transforms, deepcopy=deepcopy)
        if seed is not None:
            torch.manual_seed(seed)
        inputs = torch.randint(0, 2 ** 16 - 1, size=(nb_samples, 1))
        labels = torch.remainder(torch.randperm(nb_samples), nb_classes)
        self.samples = [{"input": inputs[idx], "label": labels[idx], "transf": f"{idx}",
                         "idx": idx, "subset": subset} for idx in range(nb_samples)]
        self.task = thelper.tasks.Classification([str(idx) for idx in range(nb_classes)],
                                                 "input", "label", meta_keys=["idx", "subset", "transf"])

    def __getitem__(self, idx):
        if self.transforms:
            return self.transforms(self.samples[idx])
        return self.samples[idx]


@pytest.fixture
def class_split_config():
    return {
        "datasets": {
            "dataset_A": DummyClassifDataset(1000, 10, "A", deepcopy=True),
            "dataset_B": DummyClassifDataset(1000, 10, "B", deepcopy=False),
            "dataset_C": DummyClassifDataset(1000, 10, "C", transforms=lambda x: x)
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


@pytest.fixture
def augm_config():

    def transfA(sample):
        assert isinstance(sample, dict) and "transf" in sample
        sample["transf"] += "A"
        return sample

    def transfB(sample):
        assert isinstance(sample, dict) and "transf" in sample
        sample["transf"] += "B"
        return sample

    return {
        "datasets": {
            "dataset_A": DummyClassifDataset(100, 10, "A", transforms=transfA),
        },
        "loaders": {
            "train_split": {
                "dataset_A": 0.4
            },
            "valid_split": {
                "dataset_A": 0.5
            },
            "test_split": {
                "dataset_A": 0.1
            },
            "train_augments": {
                "append": False,
                "transforms": [
                    transfB
                ]
            },
            "valid_augments": {
                "append": True,
                "transforms": [
                    transfB
                ]
            }
        }
    }


def test_augm(augm_config):
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(augm_config)
    assert task.check_compat(augm_config["datasets"]["dataset_A"].task, exact=True)
    for batch in train_loader:
        assert len(batch["transf"]) == 1 and batch["transf"][0].endswith("BA")
    for batch in valid_loader:
        assert len(batch["transf"]) == 1 and batch["transf"][0].endswith("AB")
    for batch in test_loader:
        assert len(batch["transf"]) == 1 and batch["transf"][0].endswith("A")
    augm_config["datasets"]["dataset_A"] = DummyClassifDataset(100, 10, "A")
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(augm_config)
    assert task.check_compat(augm_config["datasets"]["dataset_A"].task, exact=True)
    for batch in train_loader:
        assert len(batch["transf"]) == 1 and batch["transf"][0].endswith("B")
    for batch in valid_loader:
        assert len(batch["transf"]) == 1 and batch["transf"][0].endswith("B")
    for batch in test_loader:
        assert len(batch["transf"]) == 1 and "B" not in batch["transf"][0]
    test_loader.set_epoch(0)


@pytest.fixture
def sampler_config():

    class FakeSamplerA(torch.utils.data.sampler.Sampler):

        def __init__(self, indices, labels, scale=1.0, seeds=None, check=True):
            super().__init__(None)
            if check:
                assert scale == 0.5
                assert seeds is not None
                assert len(indices) == len(labels)
            self.nb_samples = len(indices)

        def __iter__(self):
            return iter([42] * self.nb_samples)

        def __len__(self):
            return self.nb_samples

    class FakeSamplerB(torch.utils.data.sampler.Sampler):

        def __init__(self, indices):
            super().__init__(None)
            self.nb_samples = len(indices)
            self.epoch = None

        def __iter__(self):
            return iter([13] * self.nb_samples)

        def set_epoch(self, idx):
            self.epoch = idx

        def __len__(self):
            return self.nb_samples

    return {
        "datasets": {
            "dataset_A": DummyClassifDataset(100, 10, "A"),
        },
        "loaders": {
            "train_split": {
                "dataset_A": 0.4
            },
            "train_sampler": {
                "pass_labels": True,
                "type": FakeSamplerA,
            },
            "train_scale": 0.5,
            "valid_split": {
                "dataset_A": 0.15
            },
            "valid_sampler": {
                "type": FakeSamplerB
            },
            "valid_scale": 1.0,
            "test_split": {
                "dataset_A": 0.1
            },
            "skip_norm": True
        }
    }


def test_custom_sampler(sampler_config):
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(sampler_config)
    assert task.check_compat(sampler_config["datasets"]["dataset_A"].task, exact=True)
    for batch in train_loader:
        assert len(batch["transf"]) == 1 and batch["idx"][0].item() == 42
    train_loader.set_epoch(42)  # sampler does not support it, but it should not crash
    for batch in valid_loader:
        assert len(batch["transf"]) == 1 and batch["idx"][0].item() == 13
    valid_loader.set_epoch(13)
    assert valid_loader.sampler.epoch == 13
    test_loader.set_epoch(0)  # does not have a sampler, but should not forward the call
    bad_sampler_config = copy.deepcopy(sampler_config)
    bad_sampler_config["loaders"]["sampler"] = {"type": sampler_config["loaders"]["train_sampler"]["type"]}
    with pytest.raises(AssertionError):
        _ = thelper.data.create_loaders(bad_sampler_config)
    bad_sampler_config = copy.deepcopy(sampler_config)
    bad_sampler_config["loaders"]["train_sampler"] = \
        sampler_config["loaders"]["train_sampler"]["type"]([], [], check=False)
    with pytest.raises(AssertionError):
        _ = thelper.data.create_loaders(bad_sampler_config)


def test_default_collate():
    with pytest.raises(AssertionError):
        _ = thelper.data.loaders.default_collate([{"a": 1}, None, {"a": 3}])
    batch = thelper.data.loaders.default_collate([None, None, None])
    assert batch is None
    batch = thelper.data.loaders.default_collate([np.uint8(1), np.uint8(2), np.uint8(3)])
    assert isinstance(batch, torch.Tensor) and batch.dtype == torch.uint8 and len(batch) == 3
    batch = thelper.data.loaders.default_collate([0.1, 0.2, 0.3])
    assert isinstance(batch, torch.Tensor) and (batch.dtype == torch.float32 or batch.dtype == torch.float64)
    ntupl = collections.namedtuple("FIZZ", "buzz bizz bozz")
    batch = thelper.data.loaders.default_collate([
        ntupl(buzz=1, bizz=2.0, bozz="3"),
        ntupl(buzz=4, bizz=5.0, bozz="6")
    ])
    assert isinstance(batch, ntupl) and isinstance(batch.buzz, torch.Tensor) and isinstance(batch.bizz, torch.Tensor)
    BBox = thelper.data.BoundingBox
    bboxes = [[BBox(0, [0, 0, 1, 1])], [], [BBox(0, [0, 0, 1, 1]), BBox(0, [0, 0, 1, 1])]]
    batch = thelper.data.loaders.default_collate(bboxes)
    assert batch == bboxes

    class Potato:
        def __init__(self):
            pass

    with pytest.raises(AssertionError):
        _ = thelper.data.loaders.default_collate([Potato(), Potato(), Potato()])
    batch = thelper.data.loaders.default_collate([Potato(), Potato(), Potato()], force_tensor=False)
    assert len(batch) == 3 and all([isinstance(p, Potato) for p in batch])


class ExtDataSamples:

    def __init__(self, n=1000, m=10, subset="X", use_samples_attrib=True):
        self.dataset = [{"in": np.random.rand(), "out": np.random.randint(m),
                         "subset": subset, "idx": idx} for idx in range(n)]
        if use_samples_attrib:
            self.samples = self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


@pytest.fixture
def ext_split_config():
    return {
        "datasets": {
            "dataset_A": {
                "type": ExtDataSamples,
                "params": {"n": 100, "m": 5, "subset": "A", "use_samples_attrib": True},
                "task": {
                    "type": "thelper.tasks.Classification",
                    "params": {
                        "class_names": ["0", "1", "2", "3", "4"],
                        "input_key": "in",
                        "label_key": "out",
                        "meta_keys": ["idx"]
                    }
                }
            },
            "dataset_B": {
                "type": ExtDataSamples,
                "params": {"n": 100, "m": 5, "subset": "B", "use_samples_attrib": False},
                "task": {
                    "type": "thelper.tasks.Classification",
                    "params": {
                        "class_names": ["0", "1", "2", "3", "4"],
                        "input_key": "in",
                        "label_key": "out",
                        "meta_keys": ["idx"]
                    }
                }
            },
            "dataset_C": {
                "type": ExtDataSamples,
                "params": {"n": 100, "m": 5, "subset": "C", "use_samples_attrib": True},
                "task": {
                    "type": "thelper.tasks.Classification",
                    "params": {
                        "class_names": ["0", "1", "2", "3", "4"],
                        "input_key": "in",
                        "label_key": "out",
                        "meta_keys": ["idx"]
                    }
                }
            },
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


def test_external_split(ext_split_config, mocker):
    logger_patch = mocker.patch.object(thelper.data.loaders.logger, "warning")
    logger_patch.start()
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(ext_split_config)
    train_samples, valid_samples, test_samples = {}, {}, {}
    for loader, samples in [(train_loader, train_samples), (valid_loader, valid_samples), (test_loader, test_samples)]:
        for batch in loader:
            for idx in range(batch["in"].shape[0]):
                name = batch["subset"][idx] + str(batch["idx"][idx].item())
                assert name not in samples
                samples[name] = batch["out"][idx].item()
    assert logger_patch.call_count == 1
    assert logger_patch.call_args[0][0].startswith("must fully parse")
    logger_patch.stop()
    assert not bool(set(train_samples) & set(valid_samples))
    assert not bool(set(train_samples) & set(test_samples))
    assert not bool(set(valid_samples) & set(test_samples))


@pytest.fixture
def verif_dir_path(request):
    test_verif_path = os.path.join(test_save_path, "verif_data")

    def fin():
        shutil.rmtree(test_verif_path, ignore_errors=True)

    fin()
    request.addfinalizer(fin)
    os.makedirs(test_verif_path, exist_ok=False)
    return test_verif_path


@pytest.fixture
def verif_config():
    return {
        "datasets": {
            "dataset_A": DummyClassifDataset(100, 3, "A", seed=1),
            "dataset_B": DummyClassifDataset(100, 3, "B", seed=2),
            "dataset_C": DummyClassifDataset(100, 3, "C", seed=3)
        },
        "loaders": {
            "test_seed": 0,
            "valid_seed": 0,
            "torch_seed": 0,
            "numpy_seed": 0,
            "random_seed": 0,
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


def test_deprecated_fields(verif_config, mocker):
    deprecated_config = copy.deepcopy(verif_config)
    deprecated_config["data_config"] = deprecated_config["loaders"]
    del deprecated_config["loaders"]
    logger_patch = mocker.patch.object(thelper.data.utils.logger, "warning")
    logger_patch.start()
    _ = thelper.data.create_loaders(deprecated_config)
    assert logger_patch.call_count == 1
    assert logger_patch.call_args[0][0].startswith("using 'data_config")
    logger_patch.stop()


def test_sample_data_verif(verif_config, verif_dir_path, mocker):
    thelper.utils.init_logger(filename=os.path.join(verif_dir_path, "thelper.log"))
    _ = thelper.data.create_loaders(verif_config, save_dir=verif_dir_path)
    # previous call created dataset logs, next call(s) will verify them for matches
    verif_config["loaders"]["skip_verif"] = False  # true by default, even when missing...
    _ = thelper.data.create_loaders(verif_config, save_dir=verif_dir_path)
    # if we get here, the checks worked; now we will make them fail on purpose
    bad_config = copy.deepcopy(verif_config)
    bad_config["datasets"]["dataset_C"] = DummyClassifDataset(50, 3, "C", seed=3)
    proceed_query = mocker.patch("thelper.utils.query_yes_no", return_value=False)
    with pytest.raises(SystemExit):
        _ = thelper.data.create_loaders(bad_config, save_dir=verif_dir_path)
    assert proceed_query.call_count == 1
    proceed_query = mocker.patch("thelper.utils.query_yes_no", return_value=True)
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(bad_config, save_dir=verif_dir_path)
    assert len(test_loader) == 2  # should be 2 batches of 32 samples instead of 4
    assert proceed_query.call_count == 1

    shutil.rmtree(verif_dir_path, ignore_errors=True)  # rebuild correct logs with next call for new test
    _ = thelper.data.create_loaders(verif_config, save_dir=verif_dir_path)
    bad_config["datasets"]["dataset_C"] = DummyClassifDataset(100, 3, "C", seed=3)
    bad_config["loaders"]["test_split"]["dataset_C"] = 0.9
    bad_config["loaders"]["skip_split_norm"] = True
    proceed_query = mocker.patch("thelper.utils.query_yes_no", return_value=False)
    with pytest.raises(SystemExit):
        _ = thelper.data.create_loaders(bad_config, save_dir=verif_dir_path)
    assert proceed_query.call_count == 1
    proceed_query = mocker.patch("thelper.utils.query_yes_no", return_value=True)
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(bad_config, save_dir=verif_dir_path)
    assert proceed_query.call_count == 1
    assert sum([subset == "C" for b in test_loader for subset in b["subset"]]) < 100

    shutil.rmtree(verif_dir_path, ignore_errors=True)  # rebuild correct logs with next call for new test
    _ = thelper.data.create_loaders(verif_config, save_dir=verif_dir_path)
    bad_config["datasets"]["dataset_C"] = DummyClassifDataset(100, 3, "D", seed=3)
    bad_config["loaders"]["test_split"]["dataset_C"] = 1.0
    proceed_query = mocker.patch("thelper.utils.query_yes_no", return_value=False)
    with pytest.raises(SystemExit):
        _ = thelper.data.create_loaders(bad_config, save_dir=verif_dir_path)
    assert proceed_query.call_count == 1
    proceed_query = mocker.patch("thelper.utils.query_yes_no", return_value=True)
    task, train_loader, valid_loader, test_loader = thelper.data.create_loaders(bad_config, save_dir=verif_dir_path)
    assert proceed_query.call_count == 1
    assert sum([subset == "D" for b in test_loader for subset in b["subset"]]) == 100


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
