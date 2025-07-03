import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


def get_cifar10_transforms(augment_train=True):
    

    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010)
    )
    

    if augment_train:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    return transform_train, transform_val


def create_cifar10_loaders(config):
    

    transform_train, transform_val = get_cifar10_transforms(
        augment_train=config.get('augmentation', True)
    )
    

    train_dataset = datasets.CIFAR10(
        root=config['root'],
        train=True,
        download=True,
        transform=transform_train
    )
    
    val_dataset = datasets.CIFAR10(
        root=config['root'],
        train=False,
        download=True,
        transform=transform_val
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


def get_dataset_info(dataset_name="CIFAR10"):
    
    if dataset_name.upper() == "CIFAR10":
        return {
            "name": "CIFAR-10",
            "num_classes": 10,
            "input_shape": (3, 32, 32),
            "class_names": [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ],
            "train_samples": 50000,
            "test_samples": 10000,
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010)
        }
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


def create_data_loaders(config):
    
    dataset_config = config['dataset']
    dataset_name = dataset_config.get('name', 'CIFAR10')
    
    if dataset_name.upper() == 'CIFAR10':
        train_loader, val_loader = create_cifar10_loaders(dataset_config)
        dataset_info = get_dataset_info('CIFAR10')
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    print(f"Dataset: {dataset_info['name']}")
    print(f"Training samples: {dataset_info['train_samples']}")
    print(f"Validation samples: {dataset_info['test_samples']}")
    print(f"Number of classes: {dataset_info['num_classes']}")
    print(f"Input shape: {dataset_info['input_shape']}")
    
    return train_loader, val_loader, dataset_info


def show_sample_images(dataloader, num_samples=8):
    
    try:
        
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        
        
        images = images[:num_samples]
        labels = labels[:num_samples]
        
        
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
        images = images * std + mean
        images = torch.clamp(images, 0, 1)
        
    
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        class_names = get_dataset_info('CIFAR10')['class_names']
        
        for i in range(num_samples):
            img = images[i].permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].set_title(f'Class: {class_names[labels[i]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Cannot display images.")
    except Exception as e:
        print(f"Error displaying images: {e}")


def get_data_statistics(dataloader):
    
    total_samples = 0
    class_counts = {}
    
    for _, labels in dataloader:
        total_samples += labels.size(0)
        
        for label in labels:
            label_item = label.item()
            class_counts[label_item] = class_counts.get(label_item, 0) + 1
    

    class_percentages = {k: (v / total_samples) * 100 for k, v in class_counts.items()}
    
    return {
        'total_samples': total_samples,
        'class_counts': class_counts,
        'class_percentages': class_percentages,
        'num_classes': len(class_counts)
    }