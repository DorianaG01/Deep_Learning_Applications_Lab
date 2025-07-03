import torch
import torch.nn.functional as F
from tqdm import tqdm
import os


def evaluate_model(model, dataloader, device):
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return correct / total


def train_epoch_simple(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets in tqdm(train_loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches



def train_model_simple(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs):
 
    checkpoint_dir = os.path.join("results", "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_path = os.path.join(checkpoint_dir, f"{model.name}_best.pth")  

    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch_simple(model, train_loader, optimizer, device)
        
        val_acc = evaluate_model(model, val_loader, device)
        
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at {best_model_path} with accuracy {best_val_acc:.4f}")
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}: Train Loss = {train_loss:.4f}, Val Acc = {val_acc:.4f}")
        
        if scheduler:
            scheduler.step()
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
        'best_model_path': best_model_path  
    }

def create_optimizer_and_scheduler(model, config):
    optimizer_type = config.get('optimizer', 'sgd').lower()  

    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 5e-4)
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 5e-4)
        )
    else:
        raise ValueError(f"Optimizer '{optimizer_type}' not supported. Choose 'sgd' or 'adam'.")
    
    scheduler = None
    if 'scheduler' in config:
        sched_config = config['scheduler']
        if sched_config['type'] == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sched_config.get('milestones', [8, 15]),
                gamma=sched_config.get('gamma', 0.1)
            )
        if sched_config['type'] == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=sched_config.get('gamma', 0.9)
            )
        
    
    return optimizer, scheduler


def calculate_model_stats(model):
    
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'model_size_mb': model_size_mb,
        'name': getattr(model, 'name', model.__class__.__name__)
    }