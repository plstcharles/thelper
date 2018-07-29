from thelper import longest
from thelper.cli import main


def test_main():
    main(["--version"])


def test_longest():
    assert longest([b'a',b'bc',b'abc'])==b'abc'
