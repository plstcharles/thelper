# noinspection PyPackageRequirements
import mock
import numpy as np

import thelper
import thelper.transforms


# noinspection PyUnusedLocal
@mock.patch("thelper.transforms.CenterCrop", autospec=True)
@mock.patch("thelper.transforms.RandomResizedCrop", autospec=True)
def test_transform_pipeline_construct(fake_op_class1, fake_op_class2):
    _ = thelper.transforms.load_transforms([  # noqa: F841
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
    ])
    fake_op_class1.assert_called_once_with(output_size=[100, 100])
    fake_op_class2.assert_called_once_with(size=[200, 200])


# noinspection PyUnusedLocal
@mock.patch.object(thelper.transforms.CenterCrop, "__call__")
@mock.patch.object(thelper.transforms.RandomResizedCrop, "__call__")
def test_transform_stochastic_pipeline(fake_op1, fake_op2):
    fake_op1.side_effect = lambda x: x
    fake_op2.side_effect = lambda x: x
    sample = np.random.rand(5, 4, 3)
    transforms = thelper.transforms.load_transforms([
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
    ])
    out = transforms(sample)
    assert np.array_equal(out, sample)
