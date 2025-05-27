from torch.utils.data import Dataset
import torch
import numpy as np


class BaseballDataset(Dataset):
    def __init__(self, X_file, Y_file):
        self.X = np.load(X_file)  # Load the NumPy file
        self.Y = np.load(Y_file)  # Load the NumPy file

    def __len__(self):
        return len(self.X)  # Return the number of samples in the data

    def __getitem__(self, index):
        sample_X = self.X[index]  # Get the sample at the specified index
        sample_Y = self.Y[index]
        return (
            torch.from_numpy(sample_X),
            torch.tensor(sample_Y),
        )  # Convert the sample to a PyTorch tensor
