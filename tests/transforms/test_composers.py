import numpy as np

import thelper
import thelper.transforms


def fake_norm_op(sample):
    return ["norm", *sample]


def fake_norm_inv_op(sample):
    return ["norm_inv", *sample]


def fake_norm_repr_fn():
    return "Norm"


def fake_resize_op(sample):
    return ["resize", *sample]


def fake_resize_inv_op(sample):
    return ["resize_inv", *sample]


def fake_resize_repr_fn():
    return "Resize"


def test_default_compose(mocker):
    fake_norm = mocker.patch.object(thelper.transforms.operations.NormalizeMinMax, "__call__")
    fake_norm.side_effect = fake_norm_op
    fake_norm_inv = mocker.patch.object(thelper.transforms.operations.NormalizeMinMax, "invert")
    fake_norm_inv.side_effect = fake_norm_inv_op
    fake_norm_repr = mocker.patch.object(thelper.transforms.operations.NormalizeMinMax, "__repr__")
    fake_norm_repr.side_effect = fake_norm_repr_fn
    fake_resize = mocker.patch.object(thelper.transforms.operations.Resize, "__call__")
    fake_resize.side_effect = fake_resize_op
    fake_resize_inv = mocker.patch.object(thelper.transforms.operations.Resize, "invert")
    fake_resize_inv.side_effect = fake_resize_inv_op
    fake_resize_repr = mocker.patch.object(thelper.transforms.operations.Resize, "__repr__")
    fake_resize_repr.side_effect = fake_resize_repr_fn
    transforms = [
        {
            "operation": "thelper.transforms.operations.Resize",
            "params": {"dsize": (256, 256)}
        },
        {
            "operation": "thelper.transforms.operations.NormalizeMinMax",
            "params": {"min": 0, "max": 255}
        }
    ]
    composer = thelper.transforms.Compose(transforms=transforms)
    seed_call_count, epoch_call_count = 0, 0

    def fake_set_seed(seed):
        assert seed == 13
        nonlocal seed_call_count
        seed_call_count += 1

    def fake_set_epoch(epoch):
        assert epoch == 15
        nonlocal epoch_call_count
        epoch_call_count += 1

    setattr(composer[1], "set_seed", fake_set_seed)
    setattr(composer[1], "set_epoch", fake_set_epoch)
    sample_orig = ["orig"]
    sample_transf = composer(sample_orig)
    assert sample_transf == ["norm", "resize", "orig"]
    assert fake_norm.call_count == 1 and fake_resize.call_count == 1
    assert fake_norm_inv.call_count == 0 and fake_resize_inv.call_count == 0
    sample_transf_inv = composer.invert(sample_transf)
    assert sample_transf_inv == ["resize_inv", "norm_inv", "norm", "resize", "orig"]
    assert fake_norm.call_count == 1 and fake_resize.call_count == 1
    assert fake_norm_inv.call_count == 1 and fake_resize_inv.call_count == 1
    composer.set_seed(13)
    composer.set_epoch(15)
    assert seed_call_count == 1 and epoch_call_count == 1
    composer_repr = repr(composer)
    assert all([v in composer_repr for v in ["Compose", "Resize", "Norm"]])
    assert fake_norm_repr.call_count == 1 and fake_resize_repr.call_count == 1


def test_custom_compose(mocker):
    composers = thelper.transforms.CustomStepCompose(milestones={
        0: thelper.transforms.Resize(dsize=(16, 16)),
        2: thelper.transforms.Resize(dsize=(32, 32)),
        4: thelper.transforms.Resize(dsize=(64, 64)),
        8: thelper.transforms.Resize(dsize=(112, 112)),
        12: thelper.transforms.Resize(dsize=(160, 160)),
        15: thelper.transforms.Resize(dsize=(196, 196)),
        18: thelper.transforms.Resize(dsize=(224, 224)),
    }), thelper.transforms.CustomStepCompose(milestones={
        "2": [{"operation": "thelper.transforms.Resize", "params": {"dsize": (32, 32)}}],
        "4": [{"operation": "thelper.transforms.Resize", "params": {"dsize": (64, 64)}}],
        "8": [{"operation": "thelper.transforms.Resize", "params": {"dsize": (112, 112)}}],
        "12": [{"operation": "thelper.transforms.Resize", "params": {"dsize": (160, 160)}}],
        "15": [{"operation": "thelper.transforms.Resize", "params": {"dsize": (196, 196)}}],
        "18": [{"operation": "thelper.transforms.Resize", "params": {"dsize": (224, 224)}}]
    })

    def fake_set_seed(seed):
        assert seed == 13

    def fake_set_epoch(epoch):
        assert epoch >= 0

    def fake_resize_inv2(sample):
        assert sample == {}
        return "invert"

    fake_resize_inv = mocker.patch.object(thelper.transforms.operations.Resize, "invert")
    fake_resize_inv.side_effect = fake_resize_inv2
    for composer in reversed(composers):
        setattr(composer[1], "set_seed", fake_set_seed)
        setattr(composer[1], "set_epoch", fake_set_epoch)
        fake_resize_inv.reset_mock()
        for epoch in range(100):
            composer.set_seed(13)
            composer.step(epoch)
            img = np.zeros((np.random.randint(1, 1000), np.random.randint(1, 1000), 1), dtype=np.uint8)
            img_t = composer(img)
            closest_k = None
            for k, v in composer.milestones.items():
                if int(k) <= epoch:
                    closest_k = k
                else:
                    break
            if closest_k is None:
                assert img_t.shape == img.shape
            else:
                op = composer.milestones[closest_k]
                if isinstance(op, thelper.transforms.Resize):
                    assert img_t.shape[:2] == op.dsize
                else:
                    assert img_t.shape[:2] == op[0]["params"]["dsize"]
            if epoch >= 18:
                assert img_t.shape == (224, 224, 1)
        test = composer.invert({})
        assert fake_resize_inv.call_count == 1
        assert test == "invert"
