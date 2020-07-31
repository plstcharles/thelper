import mock
import numpy as np

import thelper

test_save_path = ".pytest_cache"


class DummyDataset(thelper.data.Dataset):

    def __init__(self, image_size, sample_count, with_channels=False, transforms=None, deepcopy=None):
        super().__init__(transforms=transforms, deepcopy=deepcopy)
        self.samples = []
        pixel_count = int(np.prod(image_size))
        for sample_idx in range(sample_count):
            sample = np.arange(pixel_count).reshape(image_size)
            if with_channels:
                sample = np.expand_dims(sample, -1)
                sample = np.tile(sample, (1, 1, 3))
            self.samples.append(sample)
        self.task = thelper.tasks.Task("0")

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._getitems(idx)
        sample = {"0": self.samples[idx]}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


transform_call_count = 0


def dummy_stochastic_transformer(sample):
    global transform_call_count
    transform_call_count += 1
    if isinstance(sample["0"], list):  # hack for postproc test below
        sample["0"] = [p * 0 for p in sample["0"]]
    elif np.random.rand() < 0.5:
        sample["0"] = sample["0"] * -1
    return sample


@mock.patch.object(thelper.data.wrappers.patch.logger, "warning")
def test_splitter_config(mock_warning):
    wrapped_dataset_args = dict(
        dataset_type=DummyDataset, dataset_params=dict(
            image_size=(100, 120), sample_count=10, with_channels=True,
        )
    )
    aligned_no_channels = thelper.data.wrappers.patch.ImageSplitter(
        patch_size=(10, 10), patch_stride=(10, 10), patch_jitter=(0, 0),
        **wrapped_dataset_args, split_mode="stack",
    )
    assert mock_warning.call_count == 0
    assert len(aligned_no_channels) == 10
    assert aligned_no_channels.patch_count_per_image == 120
    assert aligned_no_channels.expected_input_shape == (100, 120, 3)
    assert aligned_no_channels.patch_coords[0][-1] == 90
    assert aligned_no_channels.patch_coords[1][-1] == 110
    unaligned_no_channels = thelper.data.wrappers.patch.ImageSplitter(
        patch_size=(10, 11), patch_stride=(10, 11), patch_jitter=(0, 0),
        **wrapped_dataset_args, split_mode="stack",
    )
    assert mock_warning.call_count == 1
    assert unaligned_no_channels.patch_count_per_image == 100
    assert unaligned_no_channels.expected_input_shape == (100, 120, 3)
    assert unaligned_no_channels.patch_coords[0][-1] == 90
    assert unaligned_no_channels.patch_coords[1][-1] == 99
    unaligned_no_channels = thelper.data.wrappers.patch.ImageSplitter(
        patch_size=(10, 10), patch_stride=(10, 7), patch_jitter=(0, 0),
        **wrapped_dataset_args, split_mode="stack",
    )
    assert mock_warning.call_count == 2
    assert unaligned_no_channels.patch_count_per_image == 160
    assert unaligned_no_channels.expected_input_shape == (100, 120, 3)
    assert unaligned_no_channels.patch_coords[0][-1] == 90
    assert unaligned_no_channels.patch_coords[1][-1] == 105
    unaligned_no_channels = thelper.data.wrappers.patch.ImageSplitter(
        patch_size=(10, 10), patch_stride=(7, 7), patch_jitter=(0, 0),
        **wrapped_dataset_args, split_mode="stack",
    )
    assert mock_warning.call_count == 4
    assert unaligned_no_channels.patch_count_per_image == 208
    assert unaligned_no_channels.expected_input_shape == (100, 120, 3)
    assert unaligned_no_channels.patch_coords[0][-1] == 84
    assert unaligned_no_channels.patch_coords[1][-1] == 105


def test_splitter_stack():
    dataset = thelper.data.wrappers.patch.ImageSplitter(
        patch_size=(10, 10), patch_stride=(10, 10), patch_jitter=(0, 0),
        dataset_type=DummyDataset, dataset_params=dict(
            image_size=(100, 100), sample_count=10, with_channels=False,
        ), split_mode="stack",
    )
    sample = dataset[0]
    assert "0" in sample and "patch_coords" in sample
    patch_stack = sample["0"]
    patch_coords_stack = sample["patch_coords"]
    assert len(patch_stack) == dataset.patch_count_per_image
    assert len(patch_coords_stack) == dataset.patch_count_per_image
    # now, let's check individual patches... (remember, the images are aranged matrices)
    # the 1st patch should be the top-left-most, on the first line
    assert patch_coords_stack[0] == (0, 0)
    assert patch_stack[0][0, 0] == 0 and patch_stack[0][-1, -1] == 909
    # the 10th patch should be the right-most edge, 1st line
    assert patch_coords_stack[9] == (0, 90)
    assert patch_stack[9][0, 0] == 90 and patch_stack[9][-1, -1] == 999
    # the 11th patch should be the left-most edge, 2nd line
    assert patch_coords_stack[10] == (10, 0)
    assert patch_stack[10][0, 0] == 1000 and patch_stack[10][-1, -1] == 1909


def test_splitter_iterate():
    dataset = thelper.data.wrappers.patch.ImageSplitter(
        patch_size=(5, 5), patch_stride=(1, 1), patch_jitter=(0, 0),
        dataset_type=DummyDataset, dataset_params=dict(
            image_size=(10, 10), sample_count=10, with_channels=False,
        ), split_mode="iterate",
    )
    assert len(dataset) == 10 * dataset.patch_count_per_image
    assert dataset[0]["0"].shape == (5, 5)
    for idx in range(5):
        assert dataset[idx]["0"][0, 0] == idx
        assert dataset[idx + 6]["0"][0, 0] == idx + 10
        assert dataset[idx + dataset.patch_count_per_image]["0"][0, 0] == idx


def test_splitter_random():
    dataset = thelper.data.wrappers.patch.ImageSplitter(
        patch_size=(5, 5), patch_stride=(1, 1), patch_jitter=(0, 0),
        dataset_type=DummyDataset, dataset_params=dict(
            image_size=(10, 10), sample_count=10, with_channels=False,
        ), split_mode="random",
    )
    assert len(dataset) == 10
    np.random.seed(0)
    sample0 = dataset[0]["0"]
    sample1 = dataset[0]["0"]
    assert sample0.shape == sample1.shape
    assert (sample0 != sample1).any()


def test_transforms():
    np.random.seed(0)
    args = dict(
        patch_size=(5, 5), patch_stride=(1, 1), patch_jitter=(0, 0),
        dataset_type=DummyDataset, dataset_params=dict(
            image_size=(30, 30), sample_count=30, with_channels=False,
        ), split_mode="stack", transforms=dummy_stochastic_transformer,
    )
    dataset = thelper.data.wrappers.patch.ImageSplitter(
        **args, transforms_mode="preproc",
    )
    # preproc: all value signs in a stack should be identical
    global transform_call_count
    transform_call_count = 0
    for _ in range(100):
        sample = dataset[np.random.randint(30)]["0"]
        assert all([(patch >= 0).all() for patch in sample]) or \
            all([(patch <= 0).all() for patch in sample])
    assert transform_call_count == 100
    dataset = thelper.data.wrappers.patch.ImageSplitter(
        **args, transforms_mode="per-patch",
    )
    # per-patch: some samples must have different signs
    transform_call_count = 0
    for _ in range(100):
        sample = dataset[np.random.randint(30)]["0"]
        assert any([(patch >= 0).all() for patch in sample]) and \
            any([(patch <= 0).all() for patch in sample])
    assert transform_call_count == 100 * dataset.patch_count_per_image
    dataset = thelper.data.wrappers.patch.ImageSplitter(
        **args, transforms_mode="postproc",
    )
    # postproc: all values should be zero (based on transform hack)
    transform_call_count = 0
    for _ in range(100):
        sample = dataset[np.random.randint(30)]["0"]
        assert all([(patch == 0).all() for patch in sample])
    assert transform_call_count == 100


def test_splitter_jitter():
    dataset = thelper.data.wrappers.patch.ImageSplitter(
        patch_size=(5, 5), patch_stride=(5, 5), patch_jitter=(4, 3),
        dataset_type=DummyDataset, dataset_params=dict(
            image_size=(20, 20), sample_count=5, with_channels=False,
        ), split_mode="iterate",
    )
    # first, just sample patches near image bounds to make sure OOB never crashes
    _ = [dataset[0]["0"] for _ in range(10000)]
    _ = [dataset[len(dataset) - 1]["0"] for _ in range(10000)]
    # now, sample 10 should be pretty centered, let's use it to check jitter bounds
    sample_stack = np.stack([np.asarray(dataset[10]["patch_coords"]) for _ in range(10000)])
    mean_coords = np.mean(sample_stack, axis=0)
    assert np.allclose(mean_coords, 10, atol=0.05)
    assert sample_stack[:, 0].min() == 6 and sample_stack[:, 0].max() == 14  # y-axis
    assert sample_stack[:, 1].min() == 7 and sample_stack[:, 1].max() == 13  # x-axis
