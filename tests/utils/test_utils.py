import os

import mock

import thelper

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
MOD_DIR = os.path.join(CUR_DIR, "mod")
MOD_INIT = os.path.join(MOD_DIR, "__init__.py")
PKG_DIR = os.path.join(MOD_DIR, "pkg")
PKG_INIT = os.path.join(PKG_DIR, "__init__.py")


def test_import_class_fully_qualified():
    c = thelper.nn.Module
    cn = "thelper.nn.Module"
    cls = thelper.utils.import_class(cn)
    assert cls is c


def test_import_class_path_qualified():
    assert not os.path.isfile(MOD_INIT), f"File {MOD_INIT} must not exist for this test evaluation"
    assert not os.path.isfile(PKG_INIT), f"File {PKG_INIT} must not exist for this test evaluation"
    cn = os.path.join(CUR_DIR, "mod.pkg.loader.RandomLoader")
    cls = thelper.utils.import_class(cn)
    assert cls is not None
    assert issubclass(cls, thelper.data.ImageFolderDataset)


def test_import_multi_funcs():
    def callback(ref, key, fake):
        assert fake == "hello"
        ref[key] += 1
    func_config = [
        {"type": callback, "params": {"key": "a"}},
        {"type": callback, "params": {"key": "b"}},
        {"type": callback, "params": {"key": "c"}},
    ]
    func = thelper.utils.import_function(func_config, params={"fake": "hello"})
    refmap = {"a": 0, "b": 1, "c": 2}
    func(refmap)
    assert refmap["a"] == 1 and refmap["b"] == 2 and refmap["c"] == 3


# noinspection PyUnusedLocal
@mock.patch('thelper.utils.get_available_cuda_devices', return_value=[])    # no device available
@mock.patch('torch.load', return_value={'version': thelper.__version__})
def test_load_checkpoint_default_cpu_no_devices(mock_torch_load, mock_thelper_devices):
    checkpoint_file = 'dummy-checkpoint.pth'
    thelper.utils.load_checkpoint(checkpoint_file)
    mock_torch_load.assert_called_once_with(checkpoint_file, map_location='cpu')


# noinspection PyUnusedLocal
@mock.patch('thelper.utils.get_available_cuda_devices', return_value=[0])   # one device available
@mock.patch('torch.load', return_value={'version': thelper.__version__})
def test_load_checkpoint_not_default_cpu_with_devices(mock_torch_load, mock_thelper_devices):
    checkpoint_file = 'dummy-checkpoint.pth'
    thelper.utils.load_checkpoint(checkpoint_file)
    mock_torch_load.assert_called_once_with(checkpoint_file, map_location=None)
