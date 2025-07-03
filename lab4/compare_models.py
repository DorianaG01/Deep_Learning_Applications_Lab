"""
Simplified script to compare CNN and Autoencoder OOD detection methods.
"""
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_auc_score, roc_curve
from models import CNN, Autoencoder
from data_loader import get_data_loaders
from evaluation import compute_ood_scores, compute_ae_reconstruction_scores, max_logit_score, max_softmax_score
from config import DEVICE, CNN_EPOCHS, CNN_LR, AE_EPOCHS, AE_LR

plt.switch_backend('Agg')


def load_trained_models():
    """Load pre-trained CNN and Autoencoder models."""
    cnn_model = CNN()
    cnn_path = f'./models/cifar10_CNN_{CNN_EPOCHS}_{CNN_LR}.pth'
    
    ae_model = Autoencoder()
    ae_path = f'./models/cifar10_Autoencoder_{AE_EPOCHS}_{AE_LR}.pth'
    
    if not os.path.exists(cnn_path) or not os.path.exists(ae_path):
        print(f"Models not found. Train them first.")
        return None, None
    
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    ae_model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
    
    cnn_model.to(DEVICE).eval()
    ae_model.to(DEVICE).eval()
    
    print("Models loaded successfully.")
    return cnn_model, ae_model


def compare_ood_methods(cnn_model, ae_model, testloader, fakeloader):
    """Compare all OOD detection methods."""
    print("Computing OOD scores...")
    
    # Compute scores for all methods
    test_logit = compute_ood_scores(cnn_model, testloader, max_logit_score)
    fake_logit = compute_ood_scores(cnn_model, fakeloader, max_logit_score)
    
    test_softmax = compute_ood_scores(cnn_model, testloader, max_softmax_score)
    fake_softmax = compute_ood_scores(cnn_model, fakeloader, max_softmax_score)
    
    test_ae = compute_ae_reconstruction_scores(ae_model, testloader)
    fake_ae = compute_ae_reconstruction_scores(ae_model, fakeloader)
    
    methods = {
        'Max Logit': (test_logit, fake_logit),
        'Max Softmax': (test_softmax, fake_softmax),
        'Autoencoder': (test_ae, fake_ae)
    }
    
    results = {}
    for method_name, (test_scores, fake_scores) in methods.items():
        all_scores = torch.cat([test_scores, fake_scores])
        all_labels = torch.cat([torch.ones_like(test_scores), torch.zeros_like(fake_scores)])
        auc = roc_auc_score(all_labels.cpu().numpy(), all_scores.cpu().numpy())
        
        results[method_name] = {
            'auc': auc,
            'all_scores': all_scores,
            'all_labels': all_labels
        }
        print(f"{method_name} AUC: {auc:.4f}")
    
    return results


def plot_comparison(results):
    """Plot comparison of all methods."""
    os.makedirs('plots', exist_ok=True)
    
    methods = list(results.keys())
    aucs = [results[method]['auc'] for method in methods]
    
    # Bar plot for AUC scores
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, aucs, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('AUC Score')
    plt.title('OOD Detection Performance Comparison')
    plt.ylim(0, 1)
    
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/ood_methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # ROC curves comparison
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'red', 'green']
    
    for i, (method_name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(result['all_labels'].cpu().numpy(),
                               result['all_scores'].cpu().numpy())
        auc = result['auc']
        plt.plot(fpr, tpr, color=colors[i], linewidth=2,
                label=f'{method_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/roc_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Plots saved: ood_methods_comparison.png, roc_curves_comparison.png")


def main():
    """Main comparison function."""
    print("Comparing OOD detection methods...")
    
    _, testloader, fakeloader, classes = get_data_loaders()
    
    cnn_model, ae_model = load_trained_models()
    if cnn_model is None or ae_model is None:
        return
    
    results = compare_ood_methods(cnn_model, ae_model, testloader, fakeloader)
    
    plot_comparison(results)
    
    best_method = max(results.keys(), key=lambda k: results[k]['auc'])
    print(f"Best method: {best_method} (AUC: {results[best_method]['auc']:.4f})")
    
    print("Comparison completed!")


if __name__ == "__main__":
    main()