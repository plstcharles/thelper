import logging
import copy

import numpy as np
import torch
import torch.utils.data.sampler

logger = logging.getLogger(__name__)


class WeightedSubsetRandomSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, indices, labels, stype="uniform", scale=1.0):
        super().__init__(None)
        if not isinstance(indices, list) or not isinstance(labels, list):
            raise AssertionError("expected indices and labels to be provided as lists")
        if len(indices) != len(labels):
            raise AssertionError("mismatched indices/labels list sizes")
        if not isinstance(scale, float) or scale < 0:
            raise AssertionError("invalid scale parameter; should be in [0,1]")
        self.nb_samples = int(round(len(indices) * scale))
        if self.nb_samples > 0:
            self.label_groups = {}
            if not isinstance(stype, str) or (stype not in ["uniform", "random"] and "root" not in stype):
                raise AssertionError("unexpected sampling type")
            if "root" in stype:
                self.pow = 1.0 / int(stype.split("root", 1)[1])  # will be the inverse power to use for rooting weights
                self.stype = "root"
            else:
                self.stype = stype
            self.indices = copy.deepcopy(indices)
            for idx, label in enumerate(labels):
                if label in self.label_groups:
                    self.label_groups[label].append(indices[idx])
                else:
                    self.label_groups[label] = [indices[idx]]
            if self.stype == "random":
                self.weights = [1.0 / len(self.label_groups[label]) for label in labels]
            else:
                if self.stype == "uniform":
                    self.label_weights = {label: 1.0 / len(self.label_groups) for label in self.label_groups}
                else:  # self.stype == "root"
                    self.label_weights = {label: (len(idxs) / len(labels)) ** self.pow for label, idxs in self.label_groups.items()}
                    tot_weight = sum([w for w in self.label_weights.values()])
                    self.label_weights = {label: weight / tot_weight for label, weight in self.label_weights.items()}
                self.label_counts = {}
                curr_nb_samples, max_sample_label = 0, None
                for label_idx, (label, indices) in enumerate(self.label_groups.items()):
                    self.label_counts[label] = int(self.nb_samples * self.label_weights[label])
                    curr_nb_samples += self.label_counts[label]
                    if max_sample_label is None or len(self.label_groups[label]) > len(self.label_groups[max_sample_label]):
                        max_sample_label = label
                if curr_nb_samples != self.nb_samples:
                    self.label_counts[max_sample_label] += self.nb_samples - curr_nb_samples

    def __iter__(self):
        if self.nb_samples == 0:
            return iter([])
        if self.stype == "random":
            return (self.indices[idx] for idx in torch.multinomial(self.weights, self.nb_samples, replacement=True))
        elif self.stype == "uniform" or self.stype == "root":
            indices = []
            for label, count in self.label_counts.items():
                max_samples = len(self.label_groups[label])
                while count > 0:
                    subidxs = torch.randperm(max_samples)
                    for subidx in range(min(count, max_samples)):
                        indices.append(self.label_groups[label][subidxs[subidx]])
                    count -= max_samples
            np.random.shuffle(indices)  # to make sure labels are still mixed up
            if len(indices) != self.nb_samples:
                raise AssertionError("messed up something internally...")
            return iter(indices)

    def __len__(self):
        return self.nb_samples
