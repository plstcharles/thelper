# noinspection PyPackageRequirements
import torch

import thelper
import thelper.optim.schedulers


def test_transform_custom_step_schedulers():
    optim = torch.optim.SGD(torch.nn.Conv2d(5, 10, 3).parameters(), lr=777)
    schedulers = thelper.optim.CustomStepLR(optim, milestones={
        0: 1 / 16,
        2: 1 / 8,
        4: 1 / 4,
        8: 1 / 3,
        12: 2 / 5,
        15: 1 / 2
    }), thelper.optim.CustomStepLR(optim, milestones={
        2: 1 / 8,
        4: 1 / 4,
        8: 1 / 3,
        12: 2 / 5,
        15: 1 / 2
    })
    for scheduler in schedulers:
        for epoch in range(100):
            scheduler.step(epoch)
            closest_k = None
            for k, v in scheduler.milestones.items():
                if k <= epoch:
                    closest_k = k
                else:
                    break
            sched_lr = scheduler.get_lr()
            if closest_k is None:
                assert sched_lr == [777]
            else:
                assert sched_lr == [777 * scheduler.milestones[closest_k]]
            if epoch >= 18:
                assert sched_lr == [777 * (1 / 2)]
