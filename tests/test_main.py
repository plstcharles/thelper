import os
import shutil

import pytest

test_save_path = ".pytest_cache"

dummy_save_path = os.path.join(test_save_path, "dummy")
dummy_config_path = os.path.join(dummy_save_path, "dummy.json")
dummy_ckpt_path = os.path.join(dummy_save_path, "checkpoints", "dummy.ckpt")


def test_main_init(mocker):
    import thelper.__main__
    import thelper.cli
    _ = mocker.patch.object(thelper.cli, "main", return_value=1337)
    _ = mocker.patch.object(thelper.__main__, "__name__", "__main__")
    mock_exit = mocker.patch.object(thelper.__main__.sys, "exit")
    thelper.__main__.init()
    assert mock_exit.call_args[0][0] == 1337


@pytest.fixture
def dummy_config(request):
    def fin():
        shutil.rmtree(dummy_save_path, ignore_errors=True)
    fin()
    request.addfinalizer(fin)
    os.makedirs(os.path.join(dummy_save_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(dummy_save_path, "checkpoints"), exist_ok=True)
    open(dummy_config_path, "a").close()
    open(dummy_ckpt_path, "a").close()
    return


def test_main_args(dummy_config, mocker):
    mock_create = mocker.patch("thelper.cli.create_session")
    _ = mocker.patch("thelper.cli.resume_session")
    _ = mocker.patch("thelper.cli.visualize_data")
    _ = mocker.patch("thelper.cli.annotate_data")
    _ = mocker.patch("thelper.cli.split_data")
    _ = mocker.patch("thelper.cli.export_model")
    from thelper.cli import main
    assert main([]) == 1
    assert main(["--version"]) == 0
    with pytest.raises(AssertionError):
        main(["-v", "--silent", "new", dummy_config_path, dummy_save_path])
    config = {}

    def config_getter(*args, **kwargs):
        nonlocal config
        return config

    mock_json_load = mocker.patch("json.load")
    mock_json_load.side_effect = config_getter
    config = {"oii": "test"}
    assert main(["new", dummy_config_path, dummy_save_path]) == 0
    assert mock_create.called_with(config, dummy_save_path)
    assert main(["cl_new", dummy_config_path, dummy_save_path]) == 0
    config = {"trainer": {"device": "cpu"}}
    with pytest.raises(AssertionError):
        _ = main(["cl_new", dummy_config_path, dummy_save_path])
    assert main(["viz", dummy_config_path]) == 0
    assert main(["annot", dummy_config_path, dummy_save_path]) == 0
    assert main(["split", dummy_config_path, dummy_save_path]) == 0
    assert main(["export", dummy_config_path, dummy_save_path]) == 0
    _ = mocker.patch("thelper.utils.load_checkpoint")
    assert main(["resume", dummy_save_path]) == 0
    assert main(["resume", dummy_save_path, "-c=" + dummy_config_path]) == 0
    assert main(["resume", os.path.join(dummy_save_path, "checkpoints"), "-c=" + dummy_config_path]) == 0
    mock_query = mocker.patch("thelper.utils.query_string", return_value="dummy")
    mocker_getter = mocker.patch("thelper.utils.get_save_dir")
    assert main(["resume", "dummy_path", "-c=" + dummy_config_path]) == 0
    assert mock_query.call_count == 1
    assert mocker_getter.call_count == 1
    assert main(["resume", dummy_ckpt_path]) == 0
    assert main(["resume", dummy_ckpt_path, "-s=" + dummy_save_path]) == 0
