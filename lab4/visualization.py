import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import random
import os
from config import PLOT_BINS, ALPHA

plt.switch_backend('Agg')

os.makedirs('plots', exist_ok=True)


def plot_sample_image(dataloader, title="Sample Image", save_path=None):
    """Plot a sample image from the dataloader."""
    for data in dataloader:
        x, y = data
        plt.figure(figsize=(6, 6))
        plt.imshow(x[0].permute(1, 2, 0))
        plt.title(title)
        plt.axis('off')
        
        if save_path is None:
            save_path = f"plots/{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: {save_path}")
        break


def plot_model_predictions(model, dataloader, classes, device='cuda', save_path=None):
    """
    Plot model predictions for a random sample.
    """
    model.eval()
    for data in dataloader:
        x, y = data
        break
    
    k = random.randint(0, x.shape[0] - 1)
    print(f'Ground Truth: {y[k]}, {classes[y[k]]}')
    
    with torch.no_grad():
        output = model(x.to(device))
    

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.bar(np.arange(10), output[k].detach().cpu())
    plt.title('Logits')
    plt.xlabel('Class')
    plt.ylabel('Value')
    

    plt.subplot(1, 3, 2)
    softmax_probs = F.softmax(output, dim=1)
    plt.bar(np.arange(10), softmax_probs[k].detach().cpu())
    plt.title('Softmax Probabilities')
    plt.xlabel('Class')
    plt.ylabel('Probability')

    plt.subplot(1, 3, 3)
    plt.imshow(x[k].permute(1, 2, 0))
    plt.title(f'Image: {classes[y[k]]}')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        data_type = "test" if "cifar" in str(type(dataloader.dataset)).lower() else "fake"
        save_path = f"plots/model_predictions_{data_type}_sample_{k}.png"
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(cm, classes, save_path=None):
    """
    Plot confusion matrix.
    """
    cm_norm = cm.astype(np.float32)
    cm_norm /= cm_norm.sum(axis=1, keepdims=True)
    cm_norm = (100 * cm_norm).astype(np.int32)
    
    disp = metrics.ConfusionMatrixDisplay(cm_norm, display_labels=classes)
    disp.plot()
    
    if save_path is None:
        save_path = "plots/confusion_matrix.png"
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_ood_score_distributions(in_dist_scores, ood_scores, method_name="", save_path=None):
    """
    Plot distributions of OOD scores for in-distribution and OOD data.
    """
    plt.figure(figsize=(12, 4))
    

    plt.subplot(1, 2, 1)
    plt.plot(sorted(in_dist_scores.cpu()), label='In-distribution (Test)')
    plt.plot(sorted(ood_scores.cpu()), label='Out-of-distribution (Fake)')
    plt.xlabel('Sample Index')
    plt.ylabel('OOD Score')
    plt.title(f'{method_name} - Sorted Scores')
    plt.legend()
    

    plt.subplot(1, 2, 2)
    plt.hist(in_dist_scores.cpu(), density=True, alpha=ALPHA, bins=PLOT_BINS, 
             label='In-distribution (Test)')
    plt.hist(ood_scores.cpu(), density=True, alpha=ALPHA, bins=PLOT_BINS, 
             label='Out-of-distribution (Fake)')
    plt.xlabel('OOD Score')
    plt.ylabel('Density')
    plt.title(f'{method_name} - Score Distributions')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path is None:
        method_clean = method_name.lower().replace(' ', '_')
        save_path = f"plots/ood_distributions_{method_clean}.png"
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_reconstruction_comparison(model, dataloader, num_samples=5, save_path=None):

    model.eval()
    
    for data in dataloader:
        x, _ = data
        break
    
    with torch.no_grad():
        _, reconstructed = model(x.cuda())
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        
        axes[0, i].imshow(x[i].permute(1, 2, 0))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
    
        axes[1, i].imshow(reconstructed[i].cpu().permute(1, 2, 0))
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = "plots/autoencoder_reconstruction.png"
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved: {save_path}")