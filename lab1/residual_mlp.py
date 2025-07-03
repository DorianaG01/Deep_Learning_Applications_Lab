import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt


class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_prob=0.2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return F.relu(x + residual)


class ResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth, dropout_prob=0.2):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, hidden_size, dropout_prob) for _ in range(depth)]
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.input_layer(x))
        for block in self.residual_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i - 1], hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(1)
        return self.layers(x)


def evaluate_model(model, dataloader, device, max_batches=None):
    
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0


def analyze_gradients(model):
    
    total_norm = 0.0
    layer_norms = {}
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm().item()
            total_norm += param_norm ** 2
            layer_norms[name] = param_norm
            param_count += 1
    
    total_norm = total_norm ** 0.5
    avg_norm = total_norm / param_count if param_count > 0 else 0
    
    return {
        'total_norm': total_norm,
        'avg_norm': avg_norm,
        'layer_norms': layer_norms
    }


def train_epoch_with_logging(model, train_loader, val_loader, optimizer, device, 
                           log_every=50, global_step_offset=0, track_gradients=False):
    
    model.train()
    
    train_losses = []
    val_accs = []
    steps = []
    gradient_stats = [] if track_gradients else None
    
    running_loss = 0.0
    local_step = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        
        
        if track_gradients and (local_step + 1) % log_every == 0:
            grad_stats = analyze_gradients(model)
            gradient_stats.append({
                'step': global_step_offset + local_step + 1,
                'total_norm': grad_stats['total_norm'],
                'avg_norm': grad_stats['avg_norm']
            })
        
        optimizer.step()
        running_loss += loss.item()
        local_step += 1


        if local_step % log_every == 0 or batch_idx == len(train_loader) - 1:
            train_loss = running_loss / min(log_every, local_step)
            val_acc = evaluate_model(model, val_loader, device, max_batches=15)
            
            global_step = global_step_offset + local_step
            steps.append(global_step)
            train_losses.append(train_loss)
            val_accs.append(val_acc)
            
            running_loss = 0.0
            model.train()

    result = {
        'steps': steps,
        'train_loss': train_losses,
        'val_acc': val_accs,
        'final_step': global_step_offset + local_step
    }
    
    if track_gradients and gradient_stats:
        result['gradient_stats'] = gradient_stats
    
    return result


def plot_with_gradients(results, save_path="./"):
    
    has_gradients = any('gradient_stats' in metrics for metrics in results.values())
    
    if has_gradients:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    

    ax1.set_title("Learning curves", fontsize=14)
    for model_name, metrics in results.items():
        if metrics['steps'] and metrics['train_loss']:
            ax1.plot(metrics['steps'], metrics['train_loss'], 
                   label=model_name, linewidth=2, alpha=0.8)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Validation curves", fontsize=14)
    for model_name, metrics in results.items():
        if metrics['steps'] and metrics['val_acc']:
            ax2.plot(metrics['steps'], metrics['val_acc'], 
                   label=model_name, linewidth=2, alpha=0.8)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    

    if has_gradients:
        ax3.set_title("Gradient Norms", fontsize=14)
        for model_name, metrics in results.items():
            if 'gradient_stats' in metrics and metrics['gradient_stats']:
                grad_steps = [g['step'] for g in metrics['gradient_stats']]
                grad_norms = [g['total_norm'] for g in metrics['gradient_stats']]
                if grad_steps and grad_norms:  
                    ax3.plot(grad_steps, grad_norms, label=model_name, linewidth=2, alpha=0.8)
        ax3.set_xlabel("Steps")
        ax3.set_ylabel("Gradient Norm")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')  
    
    plt.tight_layout()
    filename = "learning_curves_with_gradients.png" if has_gradients else "learning_curves.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def plot_learning_curves(results, save_path="./"):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
   
    ax1.set_title("Learning curves", fontsize=14)
    for model_name, metrics in results.items():
        if metrics['steps'] and metrics['train_loss']:
            ax1.plot(metrics['steps'], metrics['train_loss'], 
                   label=model_name, linewidth=2, alpha=0.8)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Validation curves", fontsize=14)
    for model_name, metrics in results.items():
        if metrics['steps'] and metrics['val_acc']:
            ax2.plot(metrics['steps'], metrics['val_acc'], 
                   label=model_name, linewidth=2, alpha=0.8)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150, bbox_inches='tight')
    plt.show()


def train_single_model(model, train_loader, val_loader, config, device, model_name):
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0)
    )
    
    all_steps = []
    all_train_losses = []
    all_val_accs = []
    all_gradient_stats = []
    
    global_step = 0
    best_val_acc = 0
    

    log_steps = config["training"].get("log_steps", 50)
    early_stopping = config["training"].get("early_stopping", False)
    track_gradients = config["training"].get("track_gradients", False)
    patience = config["training"].get("patience", 5)
    patience_counter = 0
    
    print(f"\nTraining {model_name}")
    print(f"Log every {log_steps} steps, Early stopping: {early_stopping}, Track gradients: {track_gradients}")
    
    for epoch in range(config["training"]["epochs"]):
        
        epoch_metrics = train_epoch_with_logging(
            model, train_loader, val_loader, optimizer, device,
            log_every=log_steps, global_step_offset=global_step, 
            track_gradients=track_gradients
        )
        
       
        all_steps.extend(epoch_metrics['steps'])
        all_train_losses.extend(epoch_metrics['train_loss'])
        all_val_accs.extend(epoch_metrics['val_acc'])
        
        if track_gradients and 'gradient_stats' in epoch_metrics:
            all_gradient_stats.extend(epoch_metrics['gradient_stats'])
        
        global_step = epoch_metrics['final_step']
        
        
        final_val_acc = evaluate_model(model, val_loader, device)
        

        if early_stopping:
            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        else:
            if final_val_acc > best_val_acc:
                best_val_acc = final_val_acc

        print(f"Epoch {epoch + 1}: Final Val Acc = {final_val_acc:.4f}")
    
    result = {
        'steps': all_steps,
        'train_loss': all_train_losses,
        'val_acc': all_val_accs
    }
    
    if track_gradients and all_gradient_stats:
        result['gradient_stats'] = all_gradient_stats
    
    return result, best_val_acc


def create_summary_table(results, best_accs):
    print("summary table")

    print(f"{'Model':<25} {'Best Val Acc':<15} {'Final Val Acc'}")
    print("-" * 60)
    
    for model_name in results.keys():
        metrics = results[model_name]
        best_acc = best_accs[model_name]
        final_acc = metrics['val_acc'][-1] if metrics['val_acc'] else 0
        
        print(f"{model_name:<25} {best_acc:<15.4f} {final_acc:.4f}")
    
    print("="*60)


def main():
    with open("config2.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root=config["dataset"]["root"], train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root=config["dataset"]["root"], train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["dataset"]["batch_size"])

    all_results = {}
    best_accuracies = {}

    for depth in config["training"]["depths"]:

        print(f"Training depth: {depth}")
       
        
        mlp_name = f"MLP_depth_{depth}"
        model_mlp = MLP(
            config["model"]["input_size"],
            [config["model"]["hidden_size"]] * depth,
            config["model"]["output_size"]
        ).to(device)
        
        mlp_results, mlp_best_acc = train_single_model(
            model_mlp, train_loader, val_loader, config, device, mlp_name
        )
        
        all_results[mlp_name] = mlp_results
        best_accuracies[mlp_name] = mlp_best_acc
        print(f"{mlp_name} - Best Val Acc: {mlp_best_acc:.4f}")
        
        res_name = f"ResidualMLP_depth_{depth}"
        model_res = ResidualMLP(
            config["model"]["input_size"],
            config["model"]["hidden_size"],
            config["model"]["output_size"],
            depth,
            config["training"]["dropout"]
        ).to(device)
        
        res_results, res_best_acc = train_single_model(
            model_res, train_loader, val_loader, config, device, res_name
        )
        
        all_results[res_name] = res_results
        best_accuracies[res_name] = res_best_acc
        print(f"{res_name} - Best Val Acc: {res_best_acc:.4f}")

    if config["training"].get("track_gradients", True):
        plot_with_gradients(all_results)
    else:
        plot_learning_curves(all_results)
    
    create_summary_table(all_results, best_accuracies)


if __name__ == "__main__":
    main()