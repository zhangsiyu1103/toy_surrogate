import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TrainDataset(Dataset):
    def __init__(self):
        self.samples = torch.load("./datasets/data/train.pt")

    def __len__(self):
        return len(self.samples["image"])

    def __getitem__(self, idx):
        sample = (self.samples["image"][idx], self.samples["category"][idx])
        return sample


class TestDataset(Dataset):
    def __init__(self):
        self.samples = torch.load("./datasets/data/test.pt")

    def __len__(self):
        return len(self.samples["image"])

    def __getitem__(self, idx):
        sample = (self.samples["image"][idx], self.samples["category"][idx])
        return sample

class Artificial(object):

    def __init__(self, batch_size = 16, num_workers=4):

        train_dataset = TrainDataset()
        test_dataset = TestDataset()


        self.train_loader_s = DataLoader(train_dataset, batch_size = 1,
            shuffle = True, num_workers = num_workers, pin_memory=True)
        self.test_loader_s = DataLoader(test_dataset, batch_size = 1,
            shuffle = True, num_workers = num_workers, pin_memory=True)
        self.train_loader = DataLoader(train_dataset, batch_size = batch_size,
            shuffle = True, num_workers = num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size = batch_size,
            shuffle = True, num_workers = num_workers, pin_memory=True)


