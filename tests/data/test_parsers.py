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
test_images_path = os.path.join(test_save_path, "images")
test_folders_path = os.path.join(test_save_path, "folders")


class DummyIntegerDataset(thelper.data.Dataset):
    def __init__(self, nb_samples, transforms=None, deepcopy=None):
        super().__init__(transforms=transforms, deepcopy=deepcopy)
        self.samples = np.arange(nb_samples)
        self.task = thelper.tasks.Task("0")

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._getitems(idx)
        return {"0": self.samples[idx]}


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


def test_dummy_dataset_getters(dummy_int):
    assert len(dummy_int) == 1000
    assert str(dummy_int.task) == str(thelper.tasks.Task("0"))
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
    assert "data.test_parsers.DummyIntegerDataset" in str(dummy_int)


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
    dataset = thelper.data.ExternalDataset(dataset=torchvision.datasets.MNIST, task=thelper.tasks.Task("0", "1"),
                                           root=test_mnist_path, train=False)
    assert dataset._get_derived_name() == "torchvision.datasets.MNIST" or dataset._get_derived_name() == "torchvision.datasets.mnist.MNIST"
    dataset = thelper.data.ExternalDataset(dataset=mnist, task=thelper.tasks.Task("0", "1"), root=test_mnist_path, train=False)
    assert dataset._get_derived_name() == "torchvision.datasets.MNIST" or dataset._get_derived_name() == "torchvision.datasets.mnist.MNIST"
    assert str(dataset.task) == str(thelper.tasks.Task("0", "1"))
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
            self.task = thelper.tasks.Task(input_key="0", gt_key="1", meta_keys=["2"])

        def __getitem__(self, idx):
            return self.samples[idx]

    dataset = DummyDataset(1000)
    data_loader = thelper.data.DataLoader(dataset, num_workers=0, batch_size=3)
    thelper.data.create_hdf5(test_hdf5_path, dataset.task, data_loader, None, None)
    return dataset


@mock.patch.object(thelper.transforms.CenterCrop, "__call__")
def test_hdf5_dataset(fake_op, dummy_hdf5):
    fake_op.side_effect = lambda x: x
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
    assert dummy_hdf5.task.check_compat(hdf5_dataset.task, exact=True)
    keys = dummy_hdf5.task.keys
    for idx in range(len(dummy_hdf5)):
        for key in keys:
            assert np.array_equal(dummy_hdf5[idx][key], hdf5_dataset[idx][key])
    with pytest.raises(AssertionError):
        _ = hdf5_dataset[len(hdf5_dataset)]
    sliced = hdf5_dataset[5:8]
    assert len(sliced) == 3
    hdf5_dataset.transforms = thelper.transforms.Compose([thelper.transforms.CenterCrop(size=5)])
    sample = hdf5_dataset[len(hdf5_dataset) - 1]
    fake_op.assert_called_with(sample)
    hdf5_dataset.close()


def test_classif_dataset():
    with pytest.raises(AssertionError):
        _ = thelper.data.ClassificationDataset(["0", "1"], None, "label")
    with pytest.raises(AssertionError):
        _ = thelper.data.ClassificationDataset(["0", "1"], "input", "label", "meta")
    with pytest.raises(AssertionError):
        _ = thelper.data.ClassificationDataset([], "input", "label")
    dataset = thelper.data.ClassificationDataset(["0", "1"], "input", "label")
    assert dataset.task.check_compat(thelper.tasks.Classification(["0", "1"], "input", "label"), exact=True)
    with pytest.raises(NotImplementedError):
        _ = dataset[0]


def test_segm_dataset():
    with pytest.raises(AssertionError):
        _ = thelper.data.SegmentationDataset(["0", "1"], None, "label")
    with pytest.raises(AssertionError):
        _ = thelper.data.SegmentationDataset(["0", "1"], "input", "label", "meta")
    with pytest.raises(AssertionError):
        _ = thelper.data.SegmentationDataset([], "input", "label")
    dataset = thelper.data.SegmentationDataset(["0", "1"], "input", "label")
    assert dataset.task.check_compat(thelper.tasks.Segmentation(["0", "1"], "input", "label"), exact=True)
    with pytest.raises(NotImplementedError):
        _ = dataset[0]


@pytest.fixture
def fake_image_root(request):
    def fin():
        shutil.rmtree(test_images_path, ignore_errors=True)
    fin()
    request.addfinalizer(fin)
    os.makedirs(test_images_path, exist_ok=True)
    for idx in range(10):
        open(os.path.join(test_images_path, str(idx) + ".jpg"), "a").close()
    open(os.path.join(test_images_path, "unknown.type"), "a").close()
    return test_images_path


@mock.patch.object(thelper.transforms.CenterCrop, "__call__")
def test_image_dataset(fake_op, fake_image_root, mocker):
    fake_imread = mocker.patch("cv2.imread")
    fake_imread.side_effect = lambda x: x
    with pytest.raises(AssertionError):
        _ = thelper.data.ImageDataset(None)
    with pytest.raises(AssertionError):
        _ = thelper.data.ImageDataset(os.path.join(fake_image_root, "0.jpg"))
    dataset = thelper.data.ImageDataset(fake_image_root, None, "0", "p", "i")
    assert dataset.task.check_compat(thelper.tasks.Task("0", None, ["p", "i"]))
    assert len(dataset) == 10
    with pytest.raises(AssertionError):
        _ = dataset[len(dataset)]
    sample = dataset[len(dataset) - 1]
    assert sample["i"] == 9
    assert sample == dataset[-1]
    fake_op.side_effect = lambda x: x
    dataset.transforms = thelper.transforms.Compose([thelper.transforms.CenterCrop(size=5)])
    sample = dataset[0]
    fake_op.assert_called_with(sample)
    assert len(dataset[0:2]) == 2
    with pytest.raises(AssertionError):
        fake_imread.side_effect = lambda x: None
        _ = dataset[0]


@pytest.fixture
def fake_image_folder_root(request):
    def fin():
        shutil.rmtree(test_folders_path, ignore_errors=True)
        shutil.rmtree(test_images_path, ignore_errors=True)
    fin()
    request.addfinalizer(fin)
    os.makedirs(test_folders_path, exist_ok=True)
    os.makedirs(test_images_path, exist_ok=True)
    for cls in range(10):
        os.makedirs(os.path.join(test_folders_path, str(cls)))
        open(os.path.join(test_folders_path, str(cls) + ".jpg"), "a").close()
        open(os.path.join(test_images_path, str(cls) + ".jpg"), "a").close()
        for idx in range(10):
            open(os.path.join(test_folders_path, str(cls), str(idx) + ".jpg"), "a").close()
        open(os.path.join(test_folders_path, str(cls), "dummy.notimg"), "a").close()
    # 100 images total, 10 in each category, with 10 extras to be ignored in root
    return test_folders_path


@mock.patch.object(thelper.transforms.CenterCrop, "__call__")
def test_image_folder_dataset(fake_op, fake_image_folder_root, mocker):
    fake_imread = mocker.patch("cv2.imread")
    fake_imread.side_effect = lambda x: x
    with pytest.raises(AssertionError):
        _ = thelper.data.ImageFolderDataset(None)
    with pytest.raises(AssertionError):
        _ = thelper.data.ImageFolderDataset(os.path.join(fake_image_folder_root, "0.jpg"))
    with pytest.raises(AssertionError):
        _ = thelper.data.ImageFolderDataset(test_images_path)
    dataset = thelper.data.ImageFolderDataset(fake_image_folder_root, None, "0", "1", "p", "i")
    classes = [str(idx) for idx in range(10)]
    assert dataset.task.check_compat(thelper.tasks.Classification(classes, "0", "1", ["p", "i"]))
    assert len(dataset) == 100
    with pytest.raises(AssertionError):
        _ = dataset[len(dataset)]
    sample = dataset[len(dataset) - 1]
    assert sample["i"] == 99
    assert sample == dataset[-1]
    fake_op.side_effect = lambda x: x
    dataset.transforms = thelper.transforms.Compose([thelper.transforms.CenterCrop(size=5)])
    sample = dataset[0]
    fake_op.assert_called_with(sample)
    assert len(dataset[0:2]) == 2
    with pytest.raises(AssertionError):
        fake_imread.side_effect = lambda x: None
        _ = dataset[0]
    for folder, subfolder, files in os.walk(fake_image_folder_root):
        for file in files:
            if file.endswith(".jpg"):
                os.remove(os.path.join(folder, file))
    with pytest.raises(AssertionError):
        _ = thelper.data.ImageFolderDataset(fake_image_folder_root)


@mock.patch.object(thelper.transforms.CenterCrop, "__call__")
def test_superres_dataset(fake_op, fake_image_folder_root, mocker):
    fake_imread = mocker.patch("cv2.imread")
    fake_imread.side_effect = lambda x: np.zeros((100, 100))
    fake_resize = mocker.patch("cv2.resize")
    fake_resize.side_effect = lambda x, *args, **kwargs: x
    fake_crop = mocker.patch("thelper.utils.safe_crop")
    fake_crop.side_effect = lambda x, *args, **kwargs: np.ones((50, 50))
    with pytest.raises(AssertionError):
        _ = thelper.data.SuperResFolderDataset(None)
    with pytest.raises(AssertionError):
        _ = thelper.data.SuperResFolderDataset(os.path.join(fake_image_folder_root, "0.jpg"))
    with pytest.raises(AssertionError):
        _ = thelper.data.SuperResFolderDataset(test_images_path)
    with pytest.raises(AssertionError):
        _ = thelper.data.SuperResFolderDataset(fake_image_folder_root, center_crop="potato")
    _ = thelper.data.SuperResFolderDataset(fake_image_folder_root, downscale_factor=4)
    with pytest.raises(AssertionError):
        _ = thelper.data.SuperResFolderDataset(fake_image_folder_root, downscale_factor=0.5)
    dataset = thelper.data.SuperResFolderDataset(fake_image_folder_root, downscale_factor=4,
                                                 lowres_image_key="0", highres_image_key="1",
                                                 path_key="p", idx_key="i", label_key="l")
    assert dataset.task.check_compat(thelper.tasks.SuperResolution("0", "1", ["p", "i", "l"]))
    assert len(dataset) == 100
    with pytest.raises(AssertionError):
        _ = dataset[len(dataset)]
    sample = dataset[len(dataset) - 1]
    assert sample["i"] == 99
    assert sample["p"] == dataset[-1]["p"]
    assert not fake_crop.called
    fake_resize.reset_mock()

    def resize_check(*args, **kwargs):
        if "fx" in kwargs and "fy" in kwargs:
            assert kwargs["fx"] == 1 / 4
            assert kwargs["fy"] == 1 / 4
            assert kwargs["dsize"] == (0, 0)
        else:
            assert kwargs["dsize"] == (50, 50)
        return np.zeros((25, 25))

    fake_resize.side_effect = resize_check
    dataset = thelper.data.SuperResFolderDataset(fake_image_folder_root, downscale_factor=4, center_crop=50,
                                                 lowres_image_key="0", highres_image_key="1")
    sample = dataset[-1]
    assert np.array_equal(sample["0"], np.zeros((25, 25)))
    assert np.array_equal(sample["1"], np.ones((50, 50)))
    assert fake_resize.call_count == 2
    assert fake_crop.call_count == 1
    fake_resize.reset_mock()
    dataset = thelper.data.SuperResFolderDataset(fake_image_folder_root, downscale_factor=4,
                                                 center_crop=50, rescale_lowres=False,
                                                 lowres_image_key="0", highres_image_key="1")
    sample = dataset[-1]
    assert np.array_equal(sample["0"], np.zeros((25, 25)))
    assert np.array_equal(sample["1"], np.ones((50, 50)))
    assert fake_resize.call_count == 1
    fake_op.side_effect = lambda x: x
    dataset.transforms = thelper.transforms.Compose([thelper.transforms.CenterCrop(size=5)])
    sample = dataset[0]
    fake_op.assert_called_with(sample)
    assert len(dataset[0:2]) == 2
    with pytest.raises(AssertionError):
        fake_imread.side_effect = lambda x: None
        _ = dataset[0]
    for folder, subfolder, files in os.walk(fake_image_folder_root):
        for file in files:
            if file.endswith(".jpg"):
                os.remove(os.path.join(folder, file))
    with pytest.raises(AssertionError):
        _ = thelper.data.SuperResFolderDataset(fake_image_folder_root, downscale_factor=4)
