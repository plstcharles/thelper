from thelper.infer.base import Tester


def test_infer_base_tester_cannot_run_train():
    t = Tester()
    try:
        t.train()
    except RuntimeError as ex:
        assert "Invalid call to 'train'" in str(ex) and "(Tester)" in str(ex), \
            "RuntimeError was raised, but not the expected one"
    else:
        raise AssertionError("Tester.train should not be allowed to be called")
