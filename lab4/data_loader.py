"""
Data loading utilities for CIFAR-10 and fake data.
"""

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import FakeData
from config import BATCH_SIZE, NUM_WORKERS, DATA_ROOT
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_data_loaders():
    transform = get_transform()
    
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, 
        train=True,
        download=True, 
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
  
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, 
        train=False,
        download=True, 
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    # Fake data for OOD detection
    fakeset = FakeData(
        size=1000, 
        image_size=(3, 32, 32), 
        transform=transform
    )
    fakeloader = torch.utils.data.DataLoader(
        fakeset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        persistent_workers=True if NUM_WORKERS > 0 else False
    )
    
    return trainloader, testloader, fakeloader, trainset.classes


def get_class_dict(classes):
    """
    Create a dictionary mapping class names to IDs.
    """
    return {class_name: id_class for id_class, class_name in enumerate(classes)}


def get_ood_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    svhn = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    loader = DataLoader(svhn, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader
