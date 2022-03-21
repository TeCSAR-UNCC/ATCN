import torch
from torch.utils.data import Dataset
import numpy as np


class Pysionet(Dataset):
    def __init__(self, X, Y, input_size=25, conv2d=False, polarity_prob=0.5, training=False):
        self.X = X
        self.Y = Y
        self.size = len(X)
        self.input_size = input_size
        self.conv2d = conv2d
        self.training = training
        self.polarity_prob = polarity_prob

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x = self.X[index]
        x = x[:self.input_size]
        if self.training:
            rand = np.random.uniform(low=0, high=1, size=1)
            if rand <= self.polarity_prob:
                x = -1*x
        x = (x - x.min())/x.ptp()

        x = torch.from_numpy(x)
        x = torch.unsqueeze(x, 0)
        if self.conv2d:
            x = torch.unsqueeze(x, 0)
        
        y = torch.from_numpy(self.Y[:, index])
        y = y.squeeze(0)
        
        return x, y


if __name__ == "__main__":
    file_name = '/mnt/4tb/ECG/raw/data_weighted_3way_60i.h5'
    import data_generator
    X_train, Y_train, _, _, _ = data_generator.get_data(file_name)
    train_dataset = Pysionet(X_train, Y_train, training=True)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, pin_memory=True)

    (x, y) = next(iter(train_loader))

    print(x.shape)
    print(y.shape)
