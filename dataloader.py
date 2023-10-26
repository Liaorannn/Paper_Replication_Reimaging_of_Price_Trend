"""
-*- coding: utf-8 -*-

@Author : Aoran,Li
@Time : 2023/10/25 13:43
@File : dataloader.py
"""
from __init__ import *
from utilis import *


class TrainValidSet(Dataset):
    def __init__(self, X, y, idx):
        self.X = torch.Tensor(X[idx])
        if len(self.X.shape) == 3:
            self.X = self.X.unsqueeze(dim=1)
        self.y = torch.Tensor(y.loc[idx, 'Ret_20d_label'].copy().values)
        if self.y.dtype == torch.float32:
            self.y = self.y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TestSet(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X[id])
        self.y = torch.Tensor(y['Ret_20d_label'].copy().values)
        if len(self.X.shape) == 3:
            self.X = self.X.unsqueeze(dim=1)
        if self.y.dtype == torch.float32:
            self.y = self.y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_train_valid_loader(image, label, train_idx, valid_idx, params):
    train_set = TrainValidSet(image, label, train_idx)
    valid_set = TrainValidSet(image, label, valid_idx)

    train_loader = DataLoader(train_set,
                              batch_size=params['BATCH_SIZE'],
                              num_workers=params['NUM_WORKERS'],
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_set,
                              batch_size=params['BATCH_SIZE'],
                              num_workers=params['NUM_WORKERS'],
                              shuffle=True,
                              pin_memory=True)

    return train_loader, valid_loader


def get_test_loader(image, label, params):
    test_set = TestSet(image, label)
    test_loader = DataLoader(test_set,
                             batch_size=params['BATCH_SIZE'],
                             num_workers=params['NUM_WORKERS'],
                             pin_memory=True)
    return test_loader
