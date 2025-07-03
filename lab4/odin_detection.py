import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from collections import defaultdict

from config import DEVICE
from data_loader import get_data_loaders, get_ood_loader
from models import CNN
import argparse



def odin_score(model, inputs, temperature=1.0, epsilon=0.0):
    model.eval()
    inputs = inputs.clone().detach().to(DEVICE)
    inputs.requires_grad = True

    outputs = model(inputs)
    outputs = outputs / temperature
    softmax_outputs = nn.Softmax(dim=1)(outputs)
    max_softmax, pseudo_labels = torch.max(softmax_outputs, dim=1)

    loss = nn.CrossEntropyLoss()(outputs, pseudo_labels)
    model.zero_grad()
    loss.backward()
    grad = inputs.grad.data

    perturbed_inputs = inputs - epsilon * torch.sign(grad)
    perturbed_inputs = torch.clamp(perturbed_inputs, -1, 1)

    with torch.no_grad():
        outputs_perturbed = model(perturbed_inputs)
        outputs_perturbed = outputs_perturbed / temperature
        softmax_outputs = nn.Softmax(dim=1)(outputs_perturbed)
        confidence_scores, _ = torch.max(softmax_outputs, dim=1)

    return confidence_scores.cpu()


def odin_grid_search(model, dataloader_in, dataloader_ood, T_values, eps_values, max_batches=10):
    results = []
    best_scores_in, best_scores_ood = [], []

    for T in T_values:
        for eps in eps_values:
            scores_in, scores_ood = [], []

            for i, (x, _) in enumerate(dataloader_in):
                if i >= max_batches: break
                scores = odin_score(model, x, temperature=T, epsilon=eps)
                scores_in.extend(scores.numpy())

            for i, (x, _) in enumerate(dataloader_ood):
                if i >= max_batches: break
                scores = odin_score(model, x, temperature=T, epsilon=eps)
                scores_ood.extend(scores.numpy())

            labels = [1]*len(scores_in) + [0]*len(scores_ood)
            all_scores = scores_in + scores_ood
            auroc = roc_auc_score(labels, all_scores)
            results.append((T, eps, auroc))

            print(f"T={T}, eps={eps:.4f}, AUROC={auroc:.4f}")

    
            if len(best_scores_in) == 0 or auroc > max(r[2] for r in results[:-1]):
                best_scores_in = scores_in
                best_scores_ood = scores_ood

    best = max(results, key=lambda x: x[2])
    print(f"\n Best ODIN setting: T={best[0]}, eps={best[1]:.4f}, AUROC={best[2]:.4f}")
    return best, results, best_scores_in, best_scores_ood


def plot_grid_search(results):
    auroc_by_T = defaultdict(lambda: {'eps': [], 'auroc': []})
    for T, eps, auroc in results:
        auroc_by_T[T]['eps'].append(eps)
        auroc_by_T[T]['auroc'].append(auroc)

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 5))
    for T, values in sorted(auroc_by_T.items()):
        plt.plot(values['eps'], values['auroc'], marker='o', label=f'T={T}')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('AUROC')
    plt.title('ODIN Grid Search: AUROC vs Epsilon')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/odin_grid_search.png', dpi=150)
    plt.close()
    print(" Grid search plot salvato: plots/odin_grid_search.png")


def plot_roc_curve(scores_in, scores_ood):
    scores = np.array(scores_in + scores_ood)
    labels = np.array([1]*len(scores_in) + [0]*len(scores_ood))

    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best ODIN Setting')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/odin_roc_curve.png', dpi=150)
    plt.close()
    print(" ROC curve salvata: plots/odin_roc_curve.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ODIN OOD Detection")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--max_batches", type=int, default=10, help="Batches to evaluate per dataset")
    args = parser.parse_args()

    model = CNN()
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)

    _, testloader_in, _, _ = get_data_loaders()
    testloader_ood = get_ood_loader()

    T_values = [1, 10, 100]
    eps_values = [0.0, 0.001, 0.002, 0.004]

    best_params, results, best_in, best_ood = odin_grid_search(
        model, testloader_in, testloader_ood, T_values, eps_values, max_batches=args.max_batches
    )

    plot_grid_search(results)
    plot_roc_curve(best_in, best_ood)
