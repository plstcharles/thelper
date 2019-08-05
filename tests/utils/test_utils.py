import thelper
import os

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
