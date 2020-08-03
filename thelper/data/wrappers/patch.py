"""Wrapper module for patch-creation stuff."""

import logging
import typing

import numpy as np

import thelper.utils
from thelper.data.parsers import Dataset
from thelper.data.utils import _create_parser

logger = logging.getLogger(__name__)


class ImageSplitter(Dataset):
    """Utility wrapper used to split loaded images into patches.

    Supports jittering and per-patch augmentations. Will re-expose the patches under the
    same key as the wrapped dataset. Will leave the dataset's task untouched. Can optionally
    return a single patch per call (and iterate over all patches), a random patch per call, or
    all (stacked) patches per call.

    In order to support the `__len__` argument required for dataset parsers, the wrapped dataset
    must load images that are fixed-size (otherwise, there would be no way to predict the total
    sample count). If the provided patch_size and patch_stride arguments cannot be used to entirely
    slice the dataset's image shape, a warning will be printed on initialization.
    """

    supported_split_modes = ["stack", "iterate", "random"]
    supported_transforms_modes = ["preproc", "per-patch", "postproc"]

    def __init__(
            self,
            patch_size: typing.Sequence[int],
            patch_stride: typing.Sequence[int],
            patch_jitter: typing.Sequence[int],
            dataset_type: typing.Union[typing.AnyStr, typing.Type],
            dataset_params: typing.Optional[typing.Dict] = None,
            split_mode: typing.AnyStr = "stack",
            transforms_mode: typing.AnyStr = "preproc",
            transforms: typing.Optional[typing.Callable] = None,
            patch_coords_key: typing.AnyStr = "patch_coords",
            deepcopy: bool = False
    ):
        """Constructs the specified dataset parser & validates the splitting dimensions.

        Args:
            patch_size: a tuple of patch dimensions (all axes, but typically 2D).
            patch_stride: a tuple of patch strides (all axes, but typically 2D).
            patch_jitter: a tuple of patch jitter values (all axes, but typically 2D).
            dataset_type: the name (or type) of the parser to instantiate and wrap.
            dataset_params: the arguments to pass to the instantiated parser constructor.
            split_mode: the type of splitting to perform. Must be 'stack', 'iterate', or
                'random'. If 'stack' is chosen, the wrapper will stack all patches on a
                new (left-appended) sample dimension. If 'iterate' is chosen, the wrapper
                will return patches one-by-one, in order (rows first, then cols). If 'random'
                is chosen, a single patch will be returned per sample, and the dataset size
                will stay the same.
            transforms_mode: the application strategy for transforms. If 'preproc', the given
                transforms will be applied in the wrapped dataset. If 'per-patch', the
                transforms will be applied for each patch individually (potentially calling
                different operations each time). If 'postproc', the transforms will be applied
                to the output sample.
            transforms: function or object that should be applied to all patches in order
                to return the data in the requested transformed/augmented state.
            patch_coords_key: key (string) under which the cut out patch coordinates (top left
                corner-based) will be provided in the generated samples.
            deepcopy: specifies whether this dataset interface should be deep-copied inside
                :class:`thelper.data.loaders.LoaderFactory` so that it may be shared between
                different threads. This is false by default, as we assume datasets do not
                contain a state or buffer that might cause problems in multi-threaded data
                loaders.

        """
        super().__init__(transforms=transforms, deepcopy=deepcopy)
        assert split_mode in self.supported_split_modes, f"invalid split mode: {split_mode}"
        self.split_mode = split_mode
        assert transforms_mode in self.supported_transforms_modes, f"invalid transforms mode: {transforms_mode}"
        self.transforms_mode = transforms_mode
        assert patch_size and all([0 < p for p in patch_size]), f"invalid patch size: {patch_size}"
        assert patch_stride and all([0 < p for p in patch_stride]), f"invalid patch size: {patch_stride}"
        assert patch_jitter and all([0 <= p for p in patch_jitter]), f"invalid patch jitter: {patch_jitter}"
        assert len(patch_stride) == len(patch_size), "mismatched patch size/stride dimensions count"
        assert len(patch_jitter) == len(patch_size), "mismatched patch size/jitter dimensions count"
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_jitter = patch_jitter
        if isinstance(dataset_type, str):
            dataset_type = thelper.utils.import_class(dataset_type)  # pragma: no cover
        self.dataset_type = dataset_type
        self.dataset_params = dataset_params
        logger.debug(f"creating wrapped parser with type: {self.dataset_type}")
        self.wrapped_dataset, self.task = _create_parser(
            dataset_config={"type": self.dataset_type, "params": dataset_params},
            base_transforms=None if self.transforms_mode != "preproc" else self.transforms,
        )
        assert self.task is not None and self.task.input_key is not None, f"invalid task: {self.task}"
        self.patch_coords_key = patch_coords_key
        self.patch_coords = [[] for _ in range(len(self.patch_size))]  # per-dim, top-left patch coords
        self.patch_count_per_image = None
        self.expected_input_shape = None
        self._validate_split()
        # note: we keep a 'sample' object in case the underlying metadata needs to be split/sorted
        self.samples = self.wrapped_dataset.samples \
            if hasattr(self.wrapped_dataset, "samples") and self.wrapped_dataset.samples is not None \
            and len(self.wrapped_dataset.samples) == len(self.wrapped_dataset) else None
        if self.split_mode == "iterate":  # in this case, we need to dupe some sample metadata fields...
            self.samples = [self.samples[idx] for idx in range(len(self.wrapped_dataset))
                            for _ in range(self.patch_count_per_image)]

    def _validate_split(self):
        # first, we will get the 1st sample from the wrapped dataset to get the input shpae
        logger.debug("loading first sample for size checks...")
        sample = self.wrapped_dataset[0]
        assert isinstance(sample, dict), f"unexpected sample type: {type(sample)}"
        assert self.task.input_key in sample, "loaded sample did not possess image key"
        image = sample[self.task.input_key]
        assert isinstance(image, np.ndarray), "unexpected image type"
        in_shape = image.shape
        # validate that the input shape is not too small wrt to the patch size
        assert len(in_shape) == len(self.patch_size) or len(in_shape) == len(self.patch_size) + 1, \
            "unexpected image dim count; should be equal to patch size dim, or one less (for channels)"
        assert all([in_dim >= p_dim for in_dim, p_dim in zip(in_shape, self.patch_size)]), \
            "cannot split image with a patch size greater than the input"
        # now, run through all axes of the input shape and slice it according to patch dims
        for patch_dim, patch_size in enumerate(self.patch_size):
            coord_idx = 0
            while coord_idx + self.patch_size[patch_dim] <= in_shape[patch_dim]:
                self.patch_coords[patch_dim].append(coord_idx)
                coord_idx += self.patch_stride[patch_dim]
            if coord_idx != in_shape[patch_dim]:
                logger.warning(f"patch split @ dim[{patch_dim}] will not preserve whole input")
            assert len(self.patch_coords[patch_dim]), "messed up logic?"
        self.patch_count_per_image = np.prod([len(c) for c in self.patch_coords])
        assert self.patch_count_per_image >= 1
        self.expected_input_shape = in_shape

    def __len__(self):
        """Returns the total number of samples available from the wrapped dataset interface."""
        base_dataset_size = len(self.wrapped_dataset)
        if self.split_mode == "random" or self.split_mode == "stack":
            return base_dataset_size
        else:
            return base_dataset_size * self.patch_count_per_image

    def _get_patch_coords_from_index(self, patch_idx: int):
        out_coords = []
        for dim in reversed(range(len(self.patch_coords))):
            coords = self.patch_coords[dim]
            curr_coord_idx = patch_idx % len(coords)
            curr_coord = coords[curr_coord_idx]
            jitter = self.patch_jitter[dim]
            if jitter:
                offset = np.random.randint(-jitter, jitter + 1)
                patch_max_coord = self.expected_input_shape[dim] - self.patch_size[dim]
                curr_coord = min(max(curr_coord + offset, 0), patch_max_coord)
            out_coords = [curr_coord, *out_coords]
            patch_idx -= curr_coord_idx
            patch_idx //= len(coords)
        assert patch_idx == 0, "messed up logic?"
        return tuple(out_coords)

    def __getitem__(self, idx):
        """Returns the data sample (a dictionary) for a specific (0-based) index.

        This is where the actual image splitting happens. Unrelated dictionary keys/vals will
        be left untouched and returned.
        """
        if isinstance(idx, slice):
            return self._getitems(idx)  # todo: make more efficient for consecutive patches?
        if self.split_mode == "iterate":
            base_idx = idx // self.patch_count_per_image
        else:
            base_idx = idx
        assert base_idx < len(self.wrapped_dataset), "sample index is out-of-range"
        sample = self.wrapped_dataset[base_idx]
        assert isinstance(sample, dict), f"unexpected sample type: {type(sample)}"
        assert self.task.input_key in sample, "loaded sample did not possess image key"
        image = sample[self.task.input_key]
        assert isinstance(image, np.ndarray), "unexpected image type"
        assert image.shape == self.expected_input_shape, "unexpected input shape; should be constant!"
        if self.split_mode == "iterate" or self.split_mode == "random":
            if self.split_mode == "iterate":
                patch_idx = idx % self.patch_count_per_image
            else:  # self.split_mode == "random"
                patch_idx = np.random.randint(self.patch_count_per_image)
            patch_start_idx = self._get_patch_coords_from_index(patch_idx)
            patch_end_idx = [idx + offset for idx, offset in zip(patch_start_idx, self.patch_size)]
            patch_slices = tuple([slice(start, stop) for start, stop in zip(patch_start_idx, patch_end_idx)])
            sample[self.task.input_key] = image[patch_slices]
            if self.transforms_mode == "per-patch" and self.transforms:
                # should be same result as postproc, but do it here nonetheless, for good measure...
                sample = self.transforms(sample)
            sample[self.patch_coords_key] = patch_start_idx
        else:  # self.split_mode == "stack"
            patch_stack = []
            patch_coords_stack = []
            for patch_idx in range(self.patch_count_per_image):
                patch_start_idx = self._get_patch_coords_from_index(patch_idx)
                patch_end_idx = [idx + offset for idx, offset in zip(patch_start_idx, self.patch_size)]
                patch_slices = tuple([slice(start, stop) for start, stop in zip(patch_start_idx, patch_end_idx)])
                patch = image[patch_slices]
                if self.transforms_mode == "per-patch" and self.transforms:
                    # here, we create a fake sample to only transform the patch... not great, but oh well
                    patch = self.transforms({self.task.input_key: patch})[self.task.input_key]
                patch_stack.append(patch)
                patch_coords_stack.append(patch_start_idx)
            sample[self.task.input_key] = patch_stack
            sample[self.patch_coords_key] = patch_coords_stack
        if self.transforms_mode == "postproc" and self.transforms:
            sample = self.transforms(sample)
        return sample
