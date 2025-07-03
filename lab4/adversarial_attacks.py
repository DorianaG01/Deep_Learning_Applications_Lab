"""
Adversarial attacks implementation for CIFAR-10 models.
Includes FGSM (Fast Gradient Sign Method) for targeted and untargeted attacks.
"""

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from config import DEVICE

plt.switch_backend('Agg')


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)
    
    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class FGSMAttacker:
    """Fast Gradient Sign Method (FGSM) adversarial attack implementation."""
    
    def __init__(self, model, device=DEVICE):
    
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.inv_normalize = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
    def fgsm_attack(self, image, label, eps, targeted=False, target_label=None, max_iterations=100):
        
        x = image.clone().detach().to(self.device)
        y = label.clone().detach().to(self.device)
        
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        if len(y.shape) == 0:
            y = y.unsqueeze(0)
            
        x.requires_grad = True
        original_x = x.clone()
        
        with torch.no_grad():
            initial_output = self.model(x)
            initial_pred = initial_output.argmax(dim=1).item()
        
        if targeted and target_label == y.item():
            return {
                'adversarial_image': x,
                'original_image': original_x,
                'success': False,
                'iterations': 0,
                'reason': 'Target label same as ground truth'
            }
        
        if not targeted and initial_pred != y.item():
            return {
                'adversarial_image': x,
                'original_image': original_x,
                'success': False,
                'iterations': 0,
                'reason': 'Model already misclassifies the image'
            }
        
        for iteration in range(max_iterations):
            x.grad = None
            output = self.model(x)
            
            if targeted:
                target_tensor = torch.tensor([target_label]).to(self.device)
                loss = self.criterion(output, target_tensor)
                loss.backward()
                
                x = x - eps * torch.sign(x.grad)
            else:
                loss = self.criterion(output, y)
                loss.backward()
                
                x = x + eps * torch.sign(x.grad)
            
            x = x.detach()
            x.requires_grad = True
            
            with torch.no_grad():
                current_output = self.model(x)
                current_pred = current_output.argmax(dim=1).item()
                
                if targeted and current_pred == target_label:
                    return {
                        'adversarial_image': x.detach(),
                        'original_image': original_x,
                        'success': True,
                        'iterations': iteration + 1,
                        'final_prediction': current_pred,
                        'perturbation_budget': int(255 * (iteration + 1) * eps)
                    }
                elif not targeted and current_pred != y.item():
                    return {
                        'adversarial_image': x.detach(),
                        'original_image': original_x,
                        'success': True,
                        'iterations': iteration + 1,
                        'final_prediction': current_pred,
                        'perturbation_budget': int(255 * (iteration + 1) * eps)
                    }
        
        return {
            'adversarial_image': x.detach(),
            'original_image': original_x,
            'success': False,
            'iterations': max_iterations,
            'reason': f'Attack failed after {max_iterations} iterations'
        }
    
    def evaluate_epsilon_sensitivity(self, image, label, epsilon_values, targeted=False, target_label=None):
        """
        Evaluate attack success rate across different epsilon values.
        """
        results = {}
        
        for eps in epsilon_values:
            result = self.fgsm_attack(
                image=image,
                label=label,
                eps=eps,
                targeted=targeted,
                target_label=target_label
            )
            results[eps] = result
            
        return results
    
    def visualize_attack_result(self, attack_result, classes, save_path=None):
    
        if not attack_result['success']:
            print(f"Attack failed: {attack_result.get('reason', 'Unknown reason')}")
            return
        
        original_img = attack_result['original_image']
        adversarial_img = attack_result['adversarial_image']
        
        with torch.no_grad():
            orig_pred = self.model(original_img).argmax(dim=1).item()
            adv_pred = self.model(adversarial_img).argmax(dim=1).item()
        
        perturbation = adversarial_img - original_img
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        orig_display = self.inv_normalize(original_img.squeeze()).permute(1, 2, 0).detach().cpu()
        orig_display = torch.clamp(orig_display, 0, 1)
        axes[0].imshow(orig_display)
        axes[0].set_title(f'Original\n{classes[orig_pred]}')
        axes[0].axis('off')
        
        adv_display = self.inv_normalize(adversarial_img.squeeze()).permute(1, 2, 0).detach().cpu()
        adv_display = torch.clamp(adv_display, 0, 1)
        axes[1].imshow(adv_display)
        axes[1].set_title(f'Adversarial\n{classes[adv_pred]}')
        axes[1].axis('off')
        
        pert_display = self.inv_normalize(perturbation.squeeze()).permute(1, 2, 0).detach().cpu()
        axes[2].imshow(pert_display, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[2].set_title('Perturbation\n(magnified)')
        axes[2].axis('off')
        
        pert_mag = perturbation.squeeze().mean(0).detach().cpu()
        im = axes[3].imshow(pert_mag, cmap='hot')
        axes[3].set_title('Perturbation\nMagnitude')
        axes[3].axis('off')
        plt.colorbar(im, ax=axes[3])
        
        plt.suptitle(f'FGSM Attack - {attack_result["iterations"]} iterations, '
                    f'Budget: {attack_result.get("perturbation_budget", "N/A")}/255')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Attack visualization saved: {save_path}")
        else:
            plt.show()
    
    def plot_epsilon_analysis(self, epsilon_results, save_path=None):
        """
        Plot attack success rate vs epsilon values.
        """
        epsilons = list(epsilon_results.keys())
        success_rates = [1 if result['success'] else 0 for result in epsilon_results.values()]
        iterations = [result['iterations'] for result in epsilon_results.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(epsilons, success_rates, 'bo-')
        ax1.set_xlabel('Epsilon')
        ax1.set_ylabel('Attack Success')
        ax1.set_title('Attack Success vs Epsilon')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        successful_eps = [eps for eps, result in epsilon_results.items() if result['success']]
        successful_iters = [result['iterations'] for result in epsilon_results.values() if result['success']]
        
        if successful_eps:
            ax2.plot(successful_eps, successful_iters, 'ro-')
            ax2.set_xlabel('Epsilon')
            ax2.set_ylabel('Iterations to Success')
            ax2.set_title('Iterations vs Epsilon (Successful Attacks)')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No successful attacks', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('No Successful Attacks')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Epsilon analysis saved: {save_path}")
        else:
            plt.show()