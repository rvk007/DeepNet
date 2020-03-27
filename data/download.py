import torch
import matplotlib.pyplot as plt
from torchvision import datasets


def download(train = True, transform = None):
    
    """
    Downloads the CIFAR-10 dataset

    Arguments:
        train: True to download train data, False to download test data
        transform : Transformation to be aaplied on downloaded data

    Returns:
        CIFAR-10 dataset 
    """

    # Download data
    return datasets.CIFAR10('./data', train=train, download=True, transform = transform)


