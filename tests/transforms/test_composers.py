# noinspection PyPackageRequirements
import numpy as np

import thelper
import thelper.transforms


def test_transform_custom_step_composers():
    transforms2 = thelper.transforms.CustomStepCompose(milestones={
        0: thelper.transforms.Resize(dsize=(16, 16)),
        2: thelper.transforms.Resize(dsize=(32, 32)),
        4: thelper.transforms.Resize(dsize=(64, 64)),
        8: thelper.transforms.Resize(dsize=(112, 112)),
        12: thelper.transforms.Resize(dsize=(160, 160)),
        15: thelper.transforms.Resize(dsize=(196, 196)),
        18: thelper.transforms.Resize(dsize=(224, 224)),
    }), thelper.transforms.CustomStepCompose(milestones={
        2: thelper.transforms.Resize(dsize=(32, 32)),
        4: thelper.transforms.Resize(dsize=(64, 64)),
        8: thelper.transforms.Resize(dsize=(112, 112)),
        12: thelper.transforms.Resize(dsize=(160, 160)),
        15: thelper.transforms.Resize(dsize=(196, 196)),
        18: thelper.transforms.Resize(dsize=(224, 224)),
    })
    for transforms in transforms2:
        for epoch in range(100):
            transforms.set_epoch(epoch)
            img = np.zeros((np.random.randint(1, 1000), np.random.randint(1, 1000), 1), dtype=np.uint8)
            img_t = transforms(img)
            closest_k = None
            for k, v in transforms.milestones.items():
                if k <= epoch:
                    closest_k = k
                else:
                    break
            if closest_k is None:
                assert img_t.shape == img.shape
            else:
                assert img_t.shape[:2] == transforms.milestones[closest_k].dsize
            if epoch >= 18:
                assert img_t.shape == (224, 224, 1)
