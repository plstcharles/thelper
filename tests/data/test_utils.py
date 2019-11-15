import thelper


class DummyClassifDataset(thelper.data.Dataset):
    def __init__(self, nb_samples, nb_classes):
        super().__init__()
        # let's create a wierd deterministic label distribution...
        labels = []
        for idx in range(nb_classes):
            labels += [idx] * (idx + 1) ** 3
        assert nb_samples >= len(labels)
        pairs = []
        for idx in range(nb_samples):
            pairs.append((idx, labels[idx % len(labels)]))
        self.samples = [{"input": p[0], "label": p[1]} for p in pairs]
        self.task = thelper.tasks.Classification([str(idx) for idx in range(nb_classes)], "input", "label")

    def __getitem__(self, idx):
        return self.samples[idx]


def test_class_weights_uniform():
    dataset = DummyClassifDataset(10000, 5)
    label_map = {idx: [] for idx in range(5)}
    for sample in dataset:
        label_map[sample["label"]].append(sample["input"])
    init_weights = {str(idx): len(label_map[idx]) / len(dataset) for idx in label_map}
    task_label_counts = dataset.task.get_class_sizes(dataset)
    assert init_weights == {lbl: count / len(dataset) for lbl, count in task_label_counts.items()}
    out_weights = thelper.data.utils.get_class_weights(label_map, stype="uniform", invmax=False, norm=True)
    assert sum(out_weights.values()) == 1.0 and all([w == 0.2 for w in out_weights.values()])
    out_factors = thelper.data.utils.get_class_weights(label_map, stype="uniform", invmax=True, norm=False)
    assert sum(out_factors.values()) == 5.0 and all([w == 1.0 for w in out_factors.values()])


def test_class_weights_linear():
    dataset = DummyClassifDataset(10000, 5)
    label_map = dataset.task.get_class_sample_map(dataset)
    out_weights = thelper.data.utils.get_class_weights(label_map, stype="linear", invmax=False, norm=True)
    assert out_weights == {"0": 0.0045, "1": 0.036, "2": 0.1215, "3": 0.288, "4": 0.55}
    out_factors = thelper.data.utils.get_class_weights(label_map, stype="linear", invmax=True, norm=False)
    assert out_factors == {"0": 0.55 / 0.0045, "1": 0.55 / 0.036, "2": 0.55 / 0.1215, "3": 0.55 / 0.288, "4": 1}
    out_factors = thelper.data.utils.get_class_weights(label_map, stype="linear", invmax=True, norm=False, minw=1.5, maxw=10)
    assert out_factors == {"0": 10, "1": 10, "2": 0.55 / 0.1215, "3": 0.55 / 0.288, "4": 1.5}


def test_class_weights_root_vs_linear():
    dataset = DummyClassifDataset(10000, 5)
    label_map = dataset.task.get_class_sample_map(dataset)
    linear_weights = thelper.data.utils.get_class_weights(label_map, stype="linear", invmax=False, norm=True)
    root2_weights = thelper.data.utils.get_class_weights(label_map, stype="root2", invmax=False, norm=True)
    root3_weights = thelper.data.utils.get_class_weights(label_map, stype="root3", invmax=False, norm=True)
    for lbl in label_map:
        lbl_count = len(label_map[lbl])
        if lbl_count < len(dataset) / len(dataset.task.class_names):
            assert linear_weights[lbl] < root2_weights[lbl]
            assert root3_weights[lbl] > root2_weights[lbl]
        elif lbl_count > len(dataset) / len(dataset.task.class_names):
            assert linear_weights[lbl] > root2_weights[lbl]
            assert root3_weights[lbl] < root2_weights[lbl]
