import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F
import argparse

from models import CNN
from data_loader import get_data_loaders
from config import DEVICE

plt.switch_backend('Agg')


class AdversarialTrainer:
    def __init__(self, model, device=DEVICE):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
    def adversarial_training(self, trainloader, epochs=10, lr=0.001, eps=2/255, alpha=0.7, save_path=None):
        """Train model with adversarial examples."""
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Training with eps={eps:.4f}, alpha={alpha}, epochs={epochs}")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                adv_inputs = self._generate_adversarial_batch(inputs, labels, eps)
                
                mixed_inputs = alpha * inputs + (1 - alpha) * adv_inputs
                
                optimizer.zero_grad()
                outputs = self.model(mixed_inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 200 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            avg_loss = total_loss / len(trainloader)
            print(f'Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}')
        
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f'Model saved to: {save_path}')
    
    def _generate_adversarial_batch(self, inputs, labels, eps):
        """Generate adversarial examples using FGSM."""
        self.model.eval()
        adv_inputs = []
        
        for i in range(inputs.size(0)):
            x = inputs[i:i+1]
            y = labels[i:i+1]
            
            x.requires_grad = True
            output = self.model(x)
            loss = self.criterion(output, y)
            
            self.model.zero_grad()
            loss.backward()
            
            adv_x = x + eps * torch.sign(x.grad)
            adv_x = torch.clamp(adv_x, -1, 1)
            adv_inputs.append(adv_x.detach())
        
        self.model.train()
        return torch.cat(adv_inputs, dim=0)


def evaluate_robustness(model, testloader, eps_values=[1/255, 2/255, 4/255, 8/255]):
    """Evaluate model robustness."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    correct_clean = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            correct_clean += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    clean_accuracy = correct_clean / total_samples
    print(f"Clean accuracy: {clean_accuracy:.4f}")
    
    results = {'clean_accuracy': clean_accuracy, 'adversarial_accuracy': {}}
    
    for eps in eps_values:
        correct_adv = 0
        test_count = 0
        
        for batch_idx, (inputs, labels) in enumerate(testloader):
            if batch_idx >= 20: 
                break
                
            for i in range(inputs.size(0)):
                x = inputs[i:i+1].to(DEVICE)
                y = labels[i:i+1].to(DEVICE)
                
                x_adv = x.clone()
                x_adv.requires_grad = True
                
                output = model(x_adv)
                loss = criterion(output, y)
                
                model.zero_grad()
                loss.backward()
                
                x_adv = x_adv + eps * torch.sign(x_adv.grad)
                x_adv = torch.clamp(x_adv, -1, 1)
                
                with torch.no_grad():
                    adv_output = model(x_adv.detach())
                    adv_pred = adv_output.argmax(dim=1)
                    
                    if adv_pred == y:
                        correct_adv += 1
                
                test_count += 1
        
        adv_accuracy = correct_adv / test_count
        results['adversarial_accuracy'][eps] = adv_accuracy
        print(f"Eps {eps:.4f}: Adversarial accuracy = {adv_accuracy:.4f}")
    
    return results


def compute_roc_data(model, testloader, classes, eps=2/255, max_samples=1000):
    """Compute ROC data for clean and adversarial examples."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_clean_probs = []
    all_adv_probs = []
    all_labels = []
    sample_count = 0
    
    for inputs, labels in testloader:
        if sample_count >= max_samples:
            break
            
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        batch_size = min(inputs.size(0), max_samples - sample_count)
        inputs = inputs[:batch_size]
        labels = labels[:batch_size]
        
        with torch.no_grad():
            clean_outputs = model(inputs)
            clean_probs = F.softmax(clean_outputs, dim=1)
        
        adv_inputs = []
        for i in range(batch_size):
            x = inputs[i:i+1]
            y = labels[i:i+1]
            
            x_adv = x.clone()
            x_adv.requires_grad = True
            
            output = model(x_adv)
            loss = criterion(output, y)
            
            model.zero_grad()
            loss.backward()
            
            x_adv = x_adv + eps * torch.sign(x_adv.grad)
            x_adv = torch.clamp(x_adv, -1, 1)
            adv_inputs.append(x_adv.detach())
        
        adv_inputs = torch.cat(adv_inputs, dim=0)
        
        with torch.no_grad():
            adv_outputs = model(adv_inputs)
            adv_probs = F.softmax(adv_outputs, dim=1)
        
        all_clean_probs.append(clean_probs.cpu())
        all_adv_probs.append(adv_probs.cpu())
        all_labels.append(labels.cpu())
        
        sample_count += batch_size
    
    all_clean_probs = torch.cat(all_clean_probs, dim=0).numpy()
    all_adv_probs = torch.cat(all_adv_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    labels_binarized = label_binarize(all_labels, classes=range(len(classes)))
    
    return {
        'clean_probs': all_clean_probs,
        'adv_probs': all_adv_probs,
        'labels_binarized': labels_binarized,
        'n_classes': len(classes)
    }


def plot_roc_curves(roc_data, model_name="Model", save_path=None):
    """Plot macro and micro-averaged ROC curves."""
    clean_probs = roc_data['clean_probs']
    adv_probs = roc_data['adv_probs']
    labels_binarized = roc_data['labels_binarized']
    n_classes = roc_data['n_classes']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr_clean = []
    all_tpr_adv = []
    
    for i in range(n_classes):
        fpr_clean, tpr_clean, _ = roc_curve(labels_binarized[:, i], clean_probs[:, i])
        all_tpr_clean.append(np.interp(mean_fpr, fpr_clean, tpr_clean))
        
        fpr_adv, tpr_adv, _ = roc_curve(labels_binarized[:, i], adv_probs[:, i])
        all_tpr_adv.append(np.interp(mean_fpr, fpr_adv, tpr_adv))
    
    mean_tpr_clean = np.mean(all_tpr_clean, axis=0)
    mean_tpr_adv = np.mean(all_tpr_adv, axis=0)
    
    macro_auc_clean = auc(mean_fpr, mean_tpr_clean)
    macro_auc_adv = auc(mean_fpr, mean_tpr_adv)
    
    axes[0].plot(mean_fpr, mean_tpr_clean, 'b-', linewidth=2, 
                label=f'Clean (AUC = {macro_auc_clean:.3f})')
    axes[0].plot(mean_fpr, mean_tpr_adv, 'r-', linewidth=2,
                label=f'Adversarial (AUC = {macro_auc_adv:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'{model_name} - Macro-Average ROC')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    fpr_micro_clean, tpr_micro_clean, _ = roc_curve(labels_binarized.ravel(), clean_probs.ravel())
    fpr_micro_adv, tpr_micro_adv, _ = roc_curve(labels_binarized.ravel(), adv_probs.ravel())
    
    micro_auc_clean = auc(fpr_micro_clean, tpr_micro_clean)
    micro_auc_adv = auc(fpr_micro_adv, tpr_micro_adv)
    
    axes[1].plot(fpr_micro_clean, tpr_micro_clean, 'b-', linewidth=2,
                label=f'Clean (AUC = {micro_auc_clean:.3f})')
    axes[1].plot(fpr_micro_adv, tpr_micro_adv, 'r-', linewidth=2,
                label=f'Adversarial (AUC = {micro_auc_adv:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f'{model_name} - Micro-Average ROC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'macro_auc_clean': macro_auc_clean,
        'macro_auc_adv': macro_auc_adv,
        'micro_auc_clean': micro_auc_clean,
        'micro_auc_adv': micro_auc_adv
    }


def plot_robustness_comparison(standard_results, robust_results, save_path=None):
    """Plot comparison between standard and robust models."""
    eps_values = list(standard_results['adversarial_accuracy'].keys())
    
    standard_acc = [standard_results['adversarial_accuracy'][eps] for eps in eps_values]
    robust_acc = [robust_results['adversarial_accuracy'][eps] for eps in eps_values]
    
    plt.figure(figsize=(8, 6))
    plt.plot(eps_values, standard_acc, 'b-o', label='Standard Model', linewidth=2)
    plt.plot(eps_values, robust_acc, 'r-o', label='Robust Model', linewidth=2)
    plt.axhline(y=standard_results['clean_accuracy'], color='b', linestyle='--', alpha=0.7)
    plt.axhline(y=robust_results['clean_accuracy'], color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title('Adversarial Robustness Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate', 'compare', 'roc'], default='train')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eps', type=float, default=2/255)
    parser.add_argument('--alpha', type=float, default=0.7)
    
    args = parser.parse_args()
    
    trainloader, testloader, _, classes = get_data_loaders()
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    if args.mode == 'train':
        print("Training adversarially robust model...")
        robust_model = CNN()
        trainer = AdversarialTrainer(robust_model, DEVICE)
        save_path = f'./models/robust_cnn_eps_{args.eps:.4f}_alpha_{args.alpha}.pth'
        trainer.adversarial_training(trainloader, epochs=args.epochs, 
                                   eps=args.eps, alpha=args.alpha, save_path=save_path)
        print("Training completed!")
    
    elif args.mode == 'evaluate':
        print("Evaluating model robustness...")
        robust_model_path = f'./models/robust_cnn_eps_{args.eps:.4f}_alpha_{args.alpha}.pth'
        
        if not os.path.exists(robust_model_path):
            print(f"Model not found: {robust_model_path}")
            return
        
        robust_model = CNN()
        robust_model.load_state_dict(torch.load(robust_model_path, map_location=DEVICE))
        robust_model.to(DEVICE)
        
        eps_values = [0.5/255, 1/255, 2/255, 4/255, 8/255]
        results = evaluate_robustness(robust_model, testloader, eps_values)
    
    elif args.mode == 'roc':
        print("Computing ROC curves...")
        from config import CNN_EPOCHS, CNN_LR
        
        standard_model_path = f'./models/cifar10_CNN_{CNN_EPOCHS}_{CNN_LR}.pth'
        robust_model_path = f'./models/robust_cnn_eps_{args.eps:.4f}_alpha_{args.alpha}.pth'
        
        if os.path.exists(standard_model_path):
            standard_model = CNN()
            standard_model.load_state_dict(torch.load(standard_model_path, map_location=DEVICE))
            standard_model.to(DEVICE)
            
            standard_roc_data = compute_roc_data(standard_model, testloader, classes, eps=args.eps)
            standard_results = plot_roc_curves(standard_roc_data, "Standard Model", 
                                             "plots/roc_curves_standard.png")
            
            print(f"Standard - Clean: {standard_results['macro_auc_clean']:.3f}, "
                  f"Adversarial: {standard_results['macro_auc_adv']:.3f}")
        
        if os.path.exists(robust_model_path):
            robust_model = CNN()
            robust_model.load_state_dict(torch.load(robust_model_path, map_location=DEVICE))
            robust_model.to(DEVICE)
            
            robust_roc_data = compute_roc_data(robust_model, testloader, classes, eps=args.eps)
            robust_results = plot_roc_curves(robust_roc_data, "Robust Model", 
                                           "plots/roc_curves_robust.png")
            
            print(f"Robust - Clean: {robust_results['macro_auc_clean']:.3f}, "
                  f"Adversarial: {robust_results['macro_auc_adv']:.3f}")
    
    elif args.mode == 'compare':
        print("Comparing models...")
        from config import CNN_EPOCHS, CNN_LR
        
        standard_model_path = f'./models/cifar10_CNN_{CNN_EPOCHS}_{CNN_LR}.pth'
        robust_model_path = f'./models/robust_cnn_eps_{args.eps:.4f}_alpha_{args.alpha}.pth'
        
        if not os.path.exists(standard_model_path) or not os.path.exists(robust_model_path):
            print("Models not found. Train them first.")
            return
        
        standard_model = CNN()
        standard_model.load_state_dict(torch.load(standard_model_path, map_location=DEVICE))
        standard_model.to(DEVICE)
        
        robust_model = CNN()
        robust_model.load_state_dict(torch.load(robust_model_path, map_location=DEVICE))
        robust_model.to(DEVICE)
        
        eps_values = [0.5/255, 1/255, 2/255, 4/255]
        
        print("Evaluating standard model...")
        standard_results = evaluate_robustness(standard_model, testloader, eps_values)
        
        print("Evaluating robust model...")
        robust_results = evaluate_robustness(robust_model, testloader, eps_values)
        
        plot_robustness_comparison(standard_results, robust_results, 
                                 "plots/robustness_comparison.png")
        
        print(f"\nResults:")
        print(f"Standard: Clean={standard_results['clean_accuracy']:.3f}")
        print(f"Robust: Clean={robust_results['clean_accuracy']:.3f}")
        
        for eps in eps_values:
            std_acc = standard_results['adversarial_accuracy'][eps]
            rob_acc = robust_results['adversarial_accuracy'][eps]
            print(f"Eps {eps:.4f}: Standard={std_acc:.3f}, Robust={rob_acc:.3f}")


if __name__ == "__main__":
    main()