import copy
import shutil

import torch

import thelper


dummy_dataset = torch.utils.data.TensorDataset(*[torch.Tensor([v]) for v in range(100)])


def test_data_loader_interface():
    pass

if __name__ == '__main__':
    test_data_loader_interface()
