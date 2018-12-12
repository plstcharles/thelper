# noinspection PyPackageRequirements
import mock

import thelper
import thelper.utils as tu


# noinspection PyUnusedLocal
@mock.patch('thelper.utils.get_available_cuda_devices', return_value=[])    # no device available
@mock.patch('torch.load', return_value={'version': thelper.__version__})
def test_load_checkpoint_default_cpu_no_devices(mock_torch_load, mock_thelper_devices):
    checkpoint_file = 'dummy-checkpoint.pth'
    tu.load_checkpoint(checkpoint_file)
    mock_torch_load.assert_called_once_with(checkpoint_file, map_location='cpu')


# noinspection PyUnusedLocal
@mock.patch('thelper.utils.get_available_cuda_devices', return_value=[0])   # one device available
@mock.patch('torch.load', return_value={'version': thelper.__version__})
def test_load_checkpoint_not_default_cpu_with_devices(mock_torch_load, mock_thelper_devices):
    checkpoint_file = 'dummy-checkpoint.pth'
    tu.load_checkpoint(checkpoint_file)
    mock_torch_load.assert_called_once_with(checkpoint_file, map_location=None)
