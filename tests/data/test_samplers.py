import copy
import functools
import inspect

import numpy as np
import pytest

import thelper


@pytest.fixture
def fake_dataset():
    return np.random.permutation(10000), np.random.randint(5, size=10000)


def test_empty_weighted_subset_random(fake_dataset):
    sampler = thelper.data.samplers.WeightedSubsetRandomSampler([], [])
    assert len(sampler) == 0
    for _ in sampler:  # pragma: no cover
        assert False
    sampler = thelper.data.samplers.SubsetRandomSampler([])
    assert len(sampler) == 0
    for _ in sampler:  # pragma: no cover
        assert False
    sampler = thelper.data.samplers.SubsetSequentialSampler([])
    assert len(sampler) == 0
    for _ in sampler:  # pragma: no cover
        assert False
    sampler = thelper.data.samplers.FixedWeightSubsetSampler([], [],
                                                             weights={idx: 1.0 for idx in range(5)})
    assert len(sampler) == 0
    for _ in sampler:  # pragma: no cover
        assert False
    sampler = thelper.data.samplers.WeightedSubsetRandomSampler(fake_dataset[0], fake_dataset[1], scale=0.0)
    assert len(sampler) == 0
    for _ in sampler:  # pragma: no cover
        assert False
    sampler = thelper.data.samplers.FixedWeightSubsetSampler(fake_dataset[0], fake_dataset[1],
                                                             weights={idx: 0 for idx in range(5)})
    assert len(sampler) == 0
    for _ in sampler:  # pragma: no cover
        assert False


@pytest.mark.parametrize("debalance", [0.1, 0.01])
def test_weighted_subset_random_uniform(fake_dataset, debalance):
    prior_label_groups = {lbl: [] for lbl in range(5)}
    for idx, lbl in zip(*fake_dataset):
        prior_label_groups[lbl].append(idx)
    prior_label_weights = [len(prior_label_groups[lbl]) /
                           sum([len(v) for v in prior_label_groups.values()])
                           for lbl in prior_label_groups]
    assert all([abs(lbl - 0.2) < 0.05 for lbl in prior_label_weights]), \
        "bad rng fail, split is not uniform at all w/ randint"
    sampler = thelper.data.samplers.WeightedSubsetRandomSampler(
        indices=fake_dataset[0], labels=fake_dataset[1], stype="uniform", scale=1.0,
        seeds={"torch": 0}, epoch=0
    )
    assert len(sampler) == len(fake_dataset[0])
    post_label_groups = {lbl: [] for lbl in range(5)}
    for idx in sampler:
        post_label_groups[fake_dataset[1][idx]].append(fake_dataset[0][idx])
    post_label_weights = [len(post_label_groups[lbl]) /
                          sum([len(v) for v in post_label_groups.values()])
                          for lbl in post_label_groups]
    assert all([abs(lbl - 0.2) < 0.05 for lbl in post_label_weights]), "bad sampling"
    sampler = thelper.data.samplers.WeightedSubsetRandomSampler(
        indices=fake_dataset[0], labels=fake_dataset[1], stype="random", scale=1.0,
        seeds={"torch": 0}, epoch=0
    )
    assert len(sampler) == len(fake_dataset[0])
    post_label_groups = {lbl: [] for lbl in range(5)}
    for idx in sampler:
        post_label_groups[fake_dataset[1][idx]].append(fake_dataset[0][idx])
    post_label_weights = [len(post_label_groups[lbl]) /
                          sum([len(v) for v in post_label_groups.values()])
                          for lbl in post_label_groups]
    assert all([abs(lbl - 0.2) < 0.05 for lbl in post_label_weights]), "bad sampling"
    debalanced_labels = copy.deepcopy(fake_dataset[1])
    for idx, lbl in enumerate(debalanced_labels):
        if lbl == 0 and np.random.rand() > debalance:
            debalanced_labels[idx] = np.random.randint(4) + 1
    prior_label_groups = {lbl: [] for lbl in range(5)}
    for idx, lbl in zip(fake_dataset[0], debalanced_labels):
        prior_label_groups[lbl].append(idx)
    prior_label_weights = [len(prior_label_groups[lbl]) /
                           sum([len(v) for v in prior_label_groups.values()])
                           for lbl in prior_label_groups]
    assert abs(prior_label_weights[0] - 0.2) > 0.1, "bad debalancing"
    sampler = thelper.data.samplers.WeightedSubsetRandomSampler(
        indices=fake_dataset[0], labels=debalanced_labels, stype="uniform", scale=1.0,
        seeds={"torch": 0}, epoch=0
    )
    assert len(sampler) == len(fake_dataset[0])
    post_label_groups = {lbl: [] for lbl in range(5)}
    for idx in sampler:
        post_label_groups[fake_dataset[1][idx]].append(fake_dataset[0][idx])
    post_label_weights = [len(post_label_groups[lbl]) /
                          sum([len(v) for v in post_label_groups.values()])
                          for lbl in post_label_groups]
    assert all([abs(lbl - 0.2) < 0.1 for lbl in post_label_weights]), "bad sampling"


@pytest.fixture
def fake_multimodal_dataset():
    idxs = np.random.permutation(1000000)[0:10000]  # for subset test
    label_probs = [0.05, 0.25, 0.7]
    labels = np.random.multinomial(10000, label_probs)
    return idxs, [0] * labels[0] + [1] * labels[1] + [2] * labels[2]


def test_weighted_subset_root(fake_multimodal_dataset):
    assert len(np.unique(fake_multimodal_dataset[0])) == len(fake_multimodal_dataset[0])
    sampler = thelper.data.samplers.WeightedSubsetRandomSampler(
        indices=fake_multimodal_dataset[0], labels=fake_multimodal_dataset[1],
        stype="root2", scale=1.0
    )
    assert len(sampler) == len(fake_multimodal_dataset[0])
    root2_label_groups = {lbl: [] for lbl in range(3)}
    for sample_idx in sampler:
        array_idx, = np.where(fake_multimodal_dataset[0] == sample_idx)
        assert len(array_idx) == 1
        root2_label_groups[fake_multimodal_dataset[1][array_idx[0]]].append(sample_idx)
    root2_label_weights = [len(root2_label_groups[lbl]) /
                           sum([len(v) for v in root2_label_groups.values()])
                           for lbl in root2_label_groups]
    sampler = thelper.data.samplers.WeightedSubsetRandomSampler(
        indices=fake_multimodal_dataset[0], labels=fake_multimodal_dataset[1],
        stype="root8", scale=1.0
    )
    assert len(sampler) == len(fake_multimodal_dataset[0])
    root8_label_groups = {lbl: [] for lbl in range(3)}
    for sample_idx in sampler:
        array_idx, = np.where(fake_multimodal_dataset[0] == sample_idx)
        assert len(array_idx) == 1
        root8_label_groups[fake_multimodal_dataset[1][array_idx[0]]].append(sample_idx)
    root8_label_weights = [len(root8_label_groups[lbl]) /
                           sum([len(v) for v in root8_label_groups.values()])
                           for lbl in root8_label_groups]
    assert root8_label_weights[0] > root2_label_weights[0] and \
        root8_label_weights[1] > root2_label_weights[1] and \
        root8_label_weights[2] < root2_label_weights[2], \
        "root3 sampling distrib failed"
    sampler = thelper.data.samplers.WeightedSubsetRandomSampler(
        indices=fake_multimodal_dataset[0], labels=fake_multimodal_dataset[1],
        stype="root3", scale=1.5
    )
    assert len(sampler) == int(round(len(fake_multimodal_dataset[0]) * 1.5))


def test_custom_weighted_subset(fake_multimodal_dataset):
    assert len(np.unique(fake_multimodal_dataset[0])) == len(fake_multimodal_dataset[0])
    target_weights = {0: 5.0, 1: 1, 2: 0.25 / 0.7}  # essentially, bring everyone to 0.25
    sampler = thelper.data.samplers.FixedWeightSubsetSampler(
        indices=fake_multimodal_dataset[0], labels=fake_multimodal_dataset[1], weights=target_weights)
    assert abs(len(sampler) - len(fake_multimodal_dataset[0]) * 0.75) < 200
    label_groups = {lbl: [] for lbl in range(3)}
    for sample_idx in sampler:
        array_idx, = np.where(fake_multimodal_dataset[0] == sample_idx)
        assert len(array_idx) == 1
        label_groups[fake_multimodal_dataset[1][array_idx[0]]].append(sample_idx)
    label_weights = [len(label_groups[lbl]) / sum([len(v) for v in label_groups.values()])
                     for lbl in label_groups]
    assert all([abs(w - 0.333) < 0.05 for w in label_weights])


@pytest.mark.parametrize("sampler_type", [thelper.data.samplers.WeightedSubsetRandomSampler,
                                          thelper.data.samplers.FixedWeightSubsetSampler])
def test_subset_random_epoch_step(fake_dataset, sampler_type):
    if "weights" in inspect.signature(sampler_type).parameters:
        sampler_type = functools.partial(sampler_type, weights={idx: 1 for idx in range(5)})
    sampler = sampler_type(
        indices=fake_dataset[0], labels=fake_dataset[1],
        seeds={"torch": np.random.randint(1000)}, epoch=0
    )
    epoch0_label_groups = {lbl: [] for lbl in range(5)}
    for idx in sampler:
        epoch0_label_groups[fake_dataset[1][idx]].append(fake_dataset[0][idx])
    epoch1_label_groups = {lbl: [] for lbl in range(5)}
    for idx in sampler:
        epoch1_label_groups[fake_dataset[1][idx]].append(fake_dataset[0][idx])
    assert epoch0_label_groups != epoch1_label_groups
    sampler.set_epoch(0)
    epoch0_reset_label_groups = {lbl: [] for lbl in range(5)}
    for idx in sampler:
        epoch0_reset_label_groups[fake_dataset[1][idx]].append(fake_dataset[0][idx])
    assert epoch0_reset_label_groups == epoch0_label_groups
