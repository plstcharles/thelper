import os
import shutil

import mock
import numpy as np
import pytest
import torchvision

import thelper

test_save_path = ".pytest_cache"

test_mnist_path = os.path.join(test_save_path, "mnist")
test_hdf5_path = os.path.join(test_save_path, "test.hdf5")


class DummyIntegerDataset(thelper.data.Dataset):
    def __init__(self, nb_samples, transforms=None, deepcopy=None):
        super().__init__(transforms=transforms, deepcopy=deepcopy)
        self.samples = np.arange(nb_samples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._getitems(idx)
        return {"0": self.samples[idx]}

    def get_task(self):
        return thelper.tasks.Task("0")


@pytest.fixture
def mnist(request):
    def fin():
        shutil.rmtree(test_mnist_path, ignore_errors=True)
    fin()
    request.addfinalizer(fin)
    return torchvision.datasets.MNIST(root=test_mnist_path, train=False, download=True)


@pytest.fixture
def dummy_int():
    return DummyIntegerDataset(1000)


def test_dataset_inteface():
    transf = thelper.transforms.operations.NoTransform()
    dataset = thelper.data.Dataset(transforms=transf, deepcopy=False)
    assert str(dataset.transforms) == str(thelper.transforms.operations.NoTransform())
    assert not dataset.deepcopy
    with pytest.raises(NotImplementedError):
        _ = dataset[0]
    with pytest.raises(NotImplementedError):
        _ = dataset.get_task()


def test_dummy_dataset_getters(dummy_int):
    assert len(dummy_int) == 1000
    assert str(dummy_int.get_task()) == str(thelper.tasks.Task("0"))
    tot_count = 0
    for idx, sample in enumerate(dummy_int):
        assert sample["0"] == idx
        tot_count += 1
    assert tot_count == 1000
    with pytest.raises(AssertionError):
        _ = dummy_int._getitems([0, 1])
    sliced = dummy_int[10:20]
    assert len(sliced) == 10
    for idx, sample in enumerate(sliced):
        assert sample["0"] == 10 + idx
    assert "DummyIntegerDataset" in dummy_int._get_derived_name()
    assert "DummyIntegerDataset" in str(dummy_int)
    assert "size: 1000" in str(dummy_int)


@mock.patch.object(thelper.transforms.CenterCrop, "__call__")
def test_external_dataset(fake_op, mnist):
    fake_op.side_effect = lambda x: x
    with pytest.raises(AssertionError):
        _ = thelper.data.ExternalDataset(dataset=None, task=thelper.tasks.Task("0"))
    with pytest.raises(ImportError):
        _ = thelper.data.ExternalDataset(dataset="unknown.type.name", task=thelper.tasks.Task("0"))
    with pytest.raises(AssertionError):
        _ = thelper.data.ExternalDataset(dataset="thelper.transforms.operations.NoTransform", task=thelper.tasks.Task("0"))
    dataset = thelper.data.ExternalDataset(task=thelper.tasks.Task("0"), dataset=[None, [0, 1], np.arange(5), {"0": "nice"}])
    with pytest.raises(AssertionError):
        _ = dataset[0]
    sample = dataset[1]
    assert dataset.warned_dictionary
    assert sample["0"] == 0 and sample["1"] == 1
    assert np.array_equal(dataset[2]["0"], np.arange(5))
    assert dataset[3]["0"] == "nice"
    with pytest.raises(TypeError):
        _ = thelper.data.ExternalDataset(dataset=torchvision.datasets.MNIST, task=thelper.tasks.Task("0"))
    with pytest.raises(AssertionError):
        _ = thelper.data.ExternalDataset(dataset=torchvision.datasets.MNIST, task=None, root=test_mnist_path, train=False)
    dataset = thelper.data.ExternalDataset(dataset=torchvision.datasets.MNIST, task=thelper.tasks.Task("0", "1"), root=test_mnist_path, train=False)
    assert dataset._get_derived_name() == "torchvision.datasets.MNIST" or dataset._get_derived_name() == "torchvision.datasets.mnist.MNIST"
    dataset = thelper.data.ExternalDataset(dataset=mnist, task=thelper.tasks.Task("0", "1"), root=test_mnist_path, train=False)
    assert dataset._get_derived_name() == "torchvision.datasets.MNIST" or dataset._get_derived_name() == "torchvision.datasets.mnist.MNIST"
    assert str(dataset.get_task()) == str(thelper.tasks.Task("0", "1"))
    with pytest.raises(AssertionError):
        _ = dataset[len(dataset)]
    sliced = dataset[5:8]
    assert len(sliced) == 3
    for sample in sliced:
        assert isinstance(sample, dict) and "0" in sample and "1" in sample
        assert sample["0"].shape == (28, 28)
        assert thelper.utils.is_scalar(sample["1"])
    dataset.transforms = thelper.transforms.Compose([thelper.transforms.CenterCrop(size=5)])
    sample = dataset[len(dataset) - 1]
    assert sample["0"].shape == (28, 28)
    assert thelper.utils.is_scalar(sample["1"])
    fake_op.assert_called_with(sample)


@pytest.fixture
def dummy_hdf5(request):

    def fin():
        if os.path.exists(test_hdf5_path):
            os.remove(test_hdf5_path)

    fin()
    request.addfinalizer(fin)

    class DummyDataset(thelper.data.Dataset):
        def __init__(self, nb_samples, transforms=None, deepcopy=None):
            super().__init__(transforms=transforms, deepcopy=deepcopy)
            self.samples = []
            for idx in range(nb_samples):
                self.samples.append({"0": np.random.randint(1000), "1": np.random.rand(2, 3, 4), "2": str(idx)})

        def __getitem__(self, idx):
            return self.samples[idx]

        def get_task(self):
            return thelper.tasks.Task(input_key="0", gt_key="1", meta_keys=["2"])

    dataset = DummyDataset(1000)
    data_loader = thelper.data.DataLoader(dataset, num_workers=0, batch_size=3)
    thelper.data.create_hdf5(test_hdf5_path, dataset.get_task(), data_loader, None, None)
    return dataset


def test_hdf5_dataset(dummy_hdf5):
    with pytest.raises(AssertionError):
        _ = thelper.data.HDF5Dataset(test_hdf5_path, subset="valid")
    with pytest.raises(AssertionError):
        _ = thelper.data.HDF5Dataset(test_hdf5_path, subset="test")
    with pytest.raises(AssertionError):
        _ = thelper.data.HDF5Dataset(test_hdf5_path, subset="potato")
    with pytest.raises(OSError):
        _ = thelper.data.HDF5Dataset("something")
    hdf5_dataset = thelper.data.HDF5Dataset(test_hdf5_path, subset="train")
    assert len(dummy_hdf5) == len(hdf5_dataset)
    assert dummy_hdf5.get_task().check_compat(hdf5_dataset.get_task(), exact=True)
    keys = dummy_hdf5.get_task().get_keys()
    for idx in range(len(dummy_hdf5)):
        for key in keys:
            assert np.array_equal(dummy_hdf5[idx][key], hdf5_dataset[idx][key])
