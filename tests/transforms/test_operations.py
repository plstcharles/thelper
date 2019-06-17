import cv2 as cv
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
    op2 = eval(repr(op))
    assert type(op) == type(op2)
    assert op.__dict__ == op2.__dict__


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
    op2 = eval(repr(op))
    assert type(op) == type(op2)
    assert op.__dict__ == op2.__dict__


def test_centercrop():
    with pytest.raises((AssertionError, ValueError)):
        _ = thelper.transforms.CenterCrop(size=-1, borderval=99)
    with pytest.raises((AssertionError, ValueError)):
        _ = thelper.transforms.CenterCrop(size=(10, 10, 10), borderval=99)
    op1 = thelper.transforms.CenterCrop(size=10, borderval=99)
    op2 = thelper.transforms.CenterCrop(size=(10, 10), borderval=99)
    assert op1.size == op2.size
    array = []
    idx = 0
    for i in range(12):
        array.append([])
        for j in range(12):
            array[i].append([i, j, idx])
            idx += 1
    sample = np.asarray(array)
    out1 = op1(sample)
    out2 = op2(sample)
    assert np.array_equal(out1, out2)
    assert out1.shape == (10, 10, 3)
    assert np.array_equal(out1[0, :, 0], [1] * 10)
    assert np.array_equal(out1[-1, :, 0], [10] * 10)
    assert np.array_equal(out1[:, 0, 1], [1] * 10)
    assert np.array_equal(out1[:, -1, 1], [10] * 10)
    assert out1[-1, -1, 2] == 130
    with pytest.raises((AssertionError, RuntimeError)):
        _ = op1.invert(out1)
    op3 = eval(repr(op1))
    assert type(op1) == type(op3)
    assert op1.__dict__ == op3.__dict__


def test_randomresizedcrop():
    with pytest.raises((AssertionError, ValueError)):
        _ = thelper.transforms.RandomResizedCrop(output_size=())
    op1 = thelper.transforms.RandomResizedCrop(output_size=10)
    op2 = thelper.transforms.RandomResizedCrop(output_size=(10, 10))
    assert op1.output_size == op2.output_size
    with pytest.raises((AssertionError, TypeError)):
        _ = thelper.transforms.RandomResizedCrop(output_size="something")
    op3 = thelper.transforms.RandomResizedCrop(output_size=None, input_size=(20, 30), ratio=(0.5, 1.0))
    op4 = thelper.transforms.RandomResizedCrop(output_size=None, input_size=(30, 20), ratio=0.66)
    assert op3.output_size == op4.output_size and op3.input_size == op4.input_size
    op5 = thelper.transforms.RandomResizedCrop(output_size=None, input_size=((30, 30), (20, 20)), ratio=None)
    with pytest.raises((AssertionError, ValueError)):
        _ = thelper.transforms.RandomResizedCrop(output_size=None, input_size=((20, 30), (20, 30)), ratio=0.66)
    op6 = thelper.transforms.RandomResizedCrop(output_size=None, input_size=((20, 30), (30, 20)), ratio=None)
    assert op5.output_size == op6.output_size and op5.input_size == op6.input_size
    with pytest.raises((AssertionError, TypeError)):
        _ = thelper.transforms.RandomResizedCrop(output_size=None, input_size="something")
    with pytest.raises((AssertionError, ValueError)):
        _ = thelper.transforms.RandomResizedCrop(output_size=None, input_size=(0.0, 1.0))
    op7 = thelper.transforms.RandomResizedCrop(output_size=None, input_size=(0.1, 1.0), probability=0.5)
    assert op7.probability == 0.5
    with pytest.raises((AssertionError, ValueError)):
        _ = thelper.transforms.RandomResizedCrop(output_size=None, input_size=(0.1, 1.0), probability=-1)
    op8 = thelper.transforms.RandomResizedCrop(output_size=None, flags="cv2.INTER_LINEAR")
    assert op8.flags == cv.INTER_LINEAR
