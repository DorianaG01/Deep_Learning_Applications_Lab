"""
Evaluation utilities for model testing and OOD detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from config import DEVICE
import matplotlib.pyplot as plt
import os

def evaluate_cnn(model, testloader, classes):
    """
    Evaluate CNN model on test data.
    """
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            predicted = outputs.argmax(1)
            
            y_pred.append(predicted)
            y_true.append(labels)
    
    y_pred_tensor = torch.cat(y_pred)
    y_true_tensor = torch.cat(y_true)
    
    accuracy = (y_pred_tensor == y_true_tensor).float().mean().item()
    cm = metrics.confusion_matrix(y_true_tensor.cpu(), y_pred_tensor.cpu())
    
    print(f'Accuracy: {accuracy:.4f}')
    
    # Per-class accuracy
    cm_norm = cm.astype(np.float32)
    cm_norm /= cm_norm.sum(axis=1, keepdims=True)
    per_class_acc = np.diag(cm_norm).mean()
    print(f'Per class accuracy: {per_class_acc:.4f}')
    
    return accuracy, cm, y_true_tensor, y_pred_tensor


def max_logit_score(logits):
    """Maximum logit score for OOD detection."""
    return logits.max(dim=1)[0]


def max_softmax_score(logits, temperature=1.0):
    """Maximum softmax score for OOD detection."""
    softmax_probs = F.softmax(logits / temperature, dim=1)
    return softmax_probs.max(dim=1)[0]


def compute_ood_scores(model, dataloader, score_function):
    """
    Compute OOD scores for a dataset.
    """
    model.eval()
    scores = []
    
    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            batch_scores = score_function(outputs)
            scores.append(batch_scores)
    
    return torch.cat(scores)


def compute_ae_reconstruction_scores(model, dataloader):
    """
    Compute reconstruction error scores for autoencoder-based OOD detection.
    """
    model.eval()
    scores = []
    mse_loss = torch.nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(DEVICE)
            _, reconstructed = model(inputs)
            
            mse = mse_loss(inputs, reconstructed)
            reconstruction_error = mse.mean(dim=[1, 2, 3])
            
            # Use negative error as score (higher is better for in-distribution)
            scores.append(-reconstruction_error)
    
    return torch.cat(scores)


def evaluate_ood_detection(in_dist_scores, ood_scores, save_path=None):
    """
    Evaluate OOD detection performance using ROC curve.
    """
    plt.switch_backend('Agg')
    
    os.makedirs('plots', exist_ok=True)
    

    all_scores = torch.cat([in_dist_scores, ood_scores])
    labels = torch.cat([
        torch.ones_like(in_dist_scores),  # In-distribution = 1
        torch.zeros_like(ood_scores)      # OOD = 0
    ])
    
    roc_display = metrics.RocCurveDisplay.from_predictions(
        labels.cpu(), 
        all_scores.cpu()
    )
    
    if save_path is None:
        save_path = "plots/roc_curve.png"
    
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved ROC curve: {save_path}")
    
    return roc_display