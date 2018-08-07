from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import fnmatch
import logging
import pickle as pkl
import bz2
import torch
import numpy as np

logger = logging.getLogger(__name__)


def load_datasets_dataset(parameters):
    """
    Creates the dataloader associated to the features files path list
    :param parameters: Input parameters
    :return: Returns the dataloader associated with the features files path list
    """
    ds = DatasetsFromDirectories(parameters)
    n = ds.__len__()
    return n, DataLoader(dataset=ds, num_workers=parameters['num_workers'], batch_size=parameters['batch_size'],
                      shuffle=parameters['shuffle'])


def load_pca_data(pca_file_path, variance_cutoff=0.65):
    """
    Load a pca pickle file and return The U, S and V matrices and the index corresponding to the cumulative
        variance.
    :param pca_file_path:.
    :param variance_cutoff:
    :return: Returns U, S, V and the cut_idx
    """

    U, S, X_Mean = torch.load(pca_file_path)
    s = S.data.numpy()
    sum = np.sum(s)
    s_cum = np.cumsum(s)
    s_norm_cum = s_cum/sum
    s_norm_cum_cut = s_norm_cum[s_norm_cum <= variance_cutoff]
    cut_idx = len(s_norm_cum_cut)
    return U, S, X_Mean, cut_idx


class DatasetsFromDirectories(Dataset):
    """
    Dataset for managing the input of deep features
    """
    def __init__(self, parameters):
        super(DatasetsFromDirectories, self).__init__()

        file_paths_list = []
        for directory_path, dirs, file_paths in os.walk(parameters['features_datasets_directory']):
            for file_path in file_paths:
                if fnmatch.fnmatch(file_path, parameters['match_pattern']):
                    file_paths_list.append(os.path.join(directory_path,file_path ))

        n_files = len(file_paths_list)
        logger.info('number of files: %i' % n_files)

        if not n_files:
            raise Exception("no files!")

        self.filenames = file_paths_list

    def __getitem__(self, index):
        # load image
        fn = self.filenames[index]
        return fn

    def __len__(self):
        return len(self.filenames)
