import torch.utils.data as data
import numpy as np

class DataSpliter:
    """
    Splits the dataset into train and validation sets based on the proportion of validation set. 
    For test dataset, splitting is not performed.

    Returns
    -------
    list: list of datasets. For training, train and validation datasets. For testing, only the test dataset.
    """
    def __init__(self, dataset, is_train, validation_set_ratio=0.0):
        self.dataset = dataset
        self.is_train = is_train
        self.validation_set_ratio =validation_set_ratio

    def split(self):
        # training with train, val split
        if(self.is_train and self.validation_set_ratio > 0):
            # TODO: do a index based random split based on a seed received in configuration, so as to be consistent across multiple train runs.
            length = len(self.dataset)
            return data.random_split(self.dataset, [(1-self.validation_set_ratio)*length, self.validation_set_ratio * length])
        # no split needed
        return [self.dataset]

    @staticmethod
    def reduceSize(dataset, target_size):
        """
        Returns a subset dataset of given dataset containing first 0...target_size-1 indices.
        """
        return data.Subset(dataset, np.arange(target_size))
