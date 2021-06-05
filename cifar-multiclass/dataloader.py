# original code: https://gist.github.com/MattKleinsmith/5226a94bad5dd12ed0b871aed98cb123
# https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py
# This is an example for the CIFAR10 dataset (formerly CIFAR-10).
# There's a function for creating a train and validation iterator.
# There's also a function for creating a test iterator.
# Inspired by https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4

# Adapted for CIFAR10 by github.com/MatthewKleinsmith
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           augment=False,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=1,
                           pin_memory=True):
    """
    Utility function for loading and returning train and valid 
    multi-process iterators over the CIFAR10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize((0.1307,), (0.3081,))  # CIFAR10

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    # load the dataset
    # datasets.CIFAR100
    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                download=True, transform=train_transform)

    valid_dataset = datasets.CIFAR10(root=data_dir, train=True,
                download=True, transform=valid_transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                    batch_size=batch_size, sampler=train_sampler, 
                    num_workers=num_workers, pin_memory=pin_memory, drop_last = True)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                    batch_size=batch_size, sampler=valid_sampler, 
                    num_workers=num_workers, pin_memory=pin_memory)

    return (train_loader, valid_loader)
    
def get_test_loader(data_dir, 
                    batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process 
    test iterator over the CIFAR10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))  # CIFAR10

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.CIFAR10(root=data_dir,
                               train=False, 
                               download=True,
                               transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return data_loader


class OODDataset(Dataset):
    def __init__(self, lam):
        data_dir='./data/ood_lam_{}.pkl'.format(lam)
        with open(data_dir, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])
        y = torch.LongTensor([0])
        y = torch.squeeze(y)
        return x, y
    
# def get_oodloader():
# #     data = OODDataset(lam = 0.5)
#     data = OODDataset(lam = "mix")
#     loader = DataLoader(data, batch_size=128, shuffle=False)
#     return loader


# def get_test_loader(data_dir,
#                     batch_size,
#                     shuffle=True,
#                     num_workers=1,
#                     pin_memory=True):
#     """
#     Utility function for loading and returning a multi-process
#     test iterator over the CIFAR10 dataset.
#     If using CUDA, num_workers should be set to 1 and pin_memory to True.
#     Params
#     ------
#     - data_dir: path directory to the dataset.
#     - batch_size: how many samples per batch to load.
#     - shuffle: whether to shuffle the dataset after every epoch.
#     - num_workers: number of subprocesses to use when loading the dataset.
#     - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
#       True if using GPU.
#     Returns
#     -------
#     - data_loader: test set iterator.
#     """
#     normalize = transforms.Normalize((0.1307,), (0.3081,))  # CIFAR10
#
#     # define transform
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         normalize
#     ])
#
#     dataset = datasets.CIFAR10(root=data_dir,
#                                train=False,
#                                download=True,
#                                transform=transform)
#
#     data_loader = torch.utils.data.DataLoader(dataset,
#                                               batch_size=batch_size,
#                                               shuffle=shuffle,
#                                               num_workers=num_workers,
#                                               pin_memory=pin_memory)
#
#     return data_loader

def get_ood_loader_CIFAR100(data_dir,
                    batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    normalize = transforms.Normalize((0.1307,), (0.3081,))  # CIFAR10

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    dataset = datasets.CIFAR100(root=data_dir,
                               train=False,
                               download=True,
                               transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)

    return data_loader
