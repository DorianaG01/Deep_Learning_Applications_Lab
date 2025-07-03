import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim import Adam, SGD
from tqdm import tqdm
from models.resnet import ResNet
from torch.utils.data import random_split
import wandb
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


Experiments = [
    {
        "name": "freeze_all_adam",
        "layers_to_freeze": ["input_adapter", "stage1"],
        "optimizer": "adam",
        "lr": 0.001,
        "description": "Freeze input_adapter + stage1, Adam optimizer"
    },
    {
        "name": "freeze_input_only_adam", 
        "layers_to_freeze": ["input_adapter"],
        "optimizer": "adam",
        "lr": 0.001,
        "description": "Freeze only input_adapter, Adam optimizer"
    },
    {
        "name": "no_freeze_adam",
        "layers_to_freeze": [],
        "optimizer": "adam", 
        "lr": 0.0005,  
        "description": "No freezing, Adam optimizer"
    },
    {
        "name": "freeze_all_sgd",
        "layers_to_freeze": ["input_adapter", "stage1"],
        "optimizer": "sgd",
        "lr": 0.01,
        "description": "Freeze input_adapter + stage1, SGD optimizer"
    },
    {
        "name": "freeze_input_only_sgd",
        "layers_to_freeze": ["input_adapter"],
        "optimizer": "sgd",
        "lr": 0.01,
        "description": "Freeze only input_adapter, SGD optimizer"
    },
    {
        "name": "no_freeze_sgd",
        "layers_to_freeze": [],
        "optimizer": "sgd",
        "lr": 0.005,  
        "description": "No freezing, SGD optimizer"
    }
]

batch_size = 128
num_classes = 100
num_epochs = 50
checkpoint_path = "/data01/dl24dorgio/dla/lab1/results/checkpoint/ResNet_Large_best.pth"

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

print("Loading datasets")
cifar100_train = datasets.CIFAR100(root="./data", train=True, transform=transform_train, download=True)
cifar100_test = datasets.CIFAR100(root="./data", train=False, transform=transform_test, download=True)

train_size = int(0.8 * len(cifar100_train))
val_size = len(cifar100_train) - train_size
cifar100_train, cifar100_val = random_split(cifar100_train, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(cifar100_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar100_val, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(cifar100_test, batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(cifar100_train)}")
print(f"Validation samples: {len(cifar100_val)}")
print(f"Test samples: {len(cifar100_test)}")

def create_model():

    model = ResNet(num_classes=10, depths=[7, 7], channels=[16, 32])
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.replace_classifier(num_classes=num_classes)
    model.to(device)  
    return model

def create_optimizer(model, optimizer_type, lr):
    if optimizer_type == "adam":
        return Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Optimizer {optimizer_type} non supportato")

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(config):
    print(f"\n{'='*60}")
    print(f"Starting experiment: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")
    
    wandb.init(
        project="cifar100-finetuning",
        name=config['name'],
        config={
            "architecture": "ResNet",
            "dataset": "CIFAR-100",
            "epochs": num_epochs,
            "batch_size": batch_size,
            "layers_to_freeze": config['layers_to_freeze'],
            "optimizer": config['optimizer'],
            "learning_rate": config['lr'],
            "description": config['description']
        },
        reinit=True
    )
    
    model = create_model()
    
    if config['layers_to_freeze']:
        model.freeze_layers(config['layers_to_freeze'])
        print(f"Frozen layers: {config['layers_to_freeze']}")
    else:
        print("No layers frozen - full fine-tuning")
    
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    wandb.config.update({
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "freeze_ratio": 1 - (trainable_params / total_params)
    })
    
    optimizer = create_optimizer(model, config['optimizer'], config['lr'])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            checkpoint_dir = f"./results/Experiments/{config['name']}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pth")
            
            wandb.run.summary["best_val_accuracy"] = best_val_acc
            wandb.run.summary["best_epoch"] = best_epoch
    

    print(f"\nTesting best model (epoch {best_epoch})...")
    checkpoint_dir = f"./results/Experiments/{config['name']}"
    model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pth"))
    test_loss, test_acc = evaluate_model(model, test_loader, device)
    
    print(f"Final Test Accuracy: {test_acc:.4f}")
    

    wandb.run.summary["test_accuracy"] = test_acc
    wandb.run.summary["test_loss"] = test_loss
    
    wandb.finish()
    
    return {
        'name': config['name'],
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'best_epoch': best_epoch,
        'trainable_params': trainable_params
    }

def main():

    print("Starting CIFAR-100 Fine-tuning Experiments")
    print(f"Total Experiments to run: {len(Experiments)}")
    
    results = []
    
    for i, config in enumerate(Experiments):
        
        print(f"Experiment {i+1}/{len(Experiments)}")
    
        try:
            result = run_experiment(config)
            results.append(result)
            
            print(f"Best Val Acc: {result['best_val_acc']:.4f}")
            print(f"Test Acc: {result['test_acc']:.4f}")
            
        except Exception as e:
            print(f"Error in experiment {config['name']}: {e}")
            continue

    print(f"\n{'='*80}")
    print("Final Results Summary")
    print(f"{'='*80}")
    
    results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    print(f"{'Rank':<4} {'Experiment':<25} {'Val Acc':<8} {'Test Acc':<9} {'Epoch':<6} {'Params':<10}")
    print("-" * 70)
    
    for i, result in enumerate(results):
        print(f"{i+1:<4} {result['name']:<25} {result['best_val_acc']:.4f}   {result['test_acc']:.4f}    {result['best_epoch']:<6} {result['trainable_params']:,}")
    
    print(f"\nBest performing experiment: {results[0]['name']}")
    print(f"Best test accuracy: {results[0]['test_acc']:.4f}")

if __name__ == "__main__":
    main()