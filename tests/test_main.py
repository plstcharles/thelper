# noinspection PyPackageRequirements
import mock


def test_main_version():
    from thelper.cli import main
    main(["--version"])


def test_main_init():
    import thelper.__main__
    import thelper.cli
    with mock.patch.object(thelper.cli, "main", return_value=1337):
        with mock.patch.object(thelper.__main__, "__name__", "__main__"):
            with mock.patch.object(thelper.__main__.sys, "exit") as mock_exit:
                thelper.__main__.init()
                assert mock_exit.call_args[0][0] == 1337
