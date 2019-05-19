import numpy as np
import PIL.Image
import pytest
import torch

import thelper


def test_notransform():
    op = thelper.transforms.NoTransform()
    sample = np.random.randn(5, 5, 5)
    transf = op(sample)
    assert transf is sample
    transf_inv = op.invert(transf)
    assert transf_inv is sample
    assert "NoTransform" in repr(op)


def test_tonumpy():
    op = thelper.transforms.ToNumpy()
    sample = np.random.randn(5, 5, 5)
    cvt_sample = op(sample)
    assert cvt_sample is sample
    sample = torch.from_numpy(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
    cvt_sample = op(sample)
    assert isinstance(cvt_sample, np.ndarray)
    assert np.array_equal(cvt_sample, sample.numpy())
    orig_sample = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    sample = PIL.Image.fromarray(orig_sample)
    cvt_sample = op(sample)
    assert isinstance(cvt_sample, np.ndarray)
    assert np.array_equal(cvt_sample, orig_sample)
    op = thelper.transforms.ToNumpy(reorder_bgr=True)
    cvt_sample = op(sample)
    assert isinstance(cvt_sample, np.ndarray)
    assert np.array_equal(cvt_sample, orig_sample[..., ::-1])
    with pytest.raises(RuntimeError):
        _ = op.invert(cvt_sample)


def test_centercrop():
    with pytest.raises((AssertionError, ValueError)):
        _ = thelper.transforms.CenterCrop(size=-1, borderval=999)
    with pytest.raises((AssertionError, ValueError)):
        _ = thelper.transforms.CenterCrop(size=(10, 10, 10), borderval=999)
    op1 = thelper.transforms.CenterCrop(size=10, borderval=999)
    op2 = thelper.transforms.CenterCrop(size=(10, 10), borderval=999)
    assert op1.size == op2.size
    pass
