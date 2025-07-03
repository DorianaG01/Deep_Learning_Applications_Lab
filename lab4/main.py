import torch
import argparse
import os
from models import CNN, Autoencoder
from data_loader import get_data_loaders
from training import train_cnn, train_autoencoder
from evaluation import (
    evaluate_cnn, compute_ood_scores, compute_ae_reconstruction_scores,
    evaluate_ood_detection, max_logit_score, max_softmax_score
)
from visualization import (
    plot_confusion_matrix, plot_ood_score_distributions,
    plot_sample_image, plot_reconstruction_comparison
)
from config import DEVICE, CNN_EPOCHS, CNN_LR, AE_EPOCHS, AE_LR
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

plt.switch_backend('Agg')


def plot_cnn_roc_only(testloader, fakeloader):
    """Load CNN and plot only ROC curve."""
    print("Plotting CNN ROC curve...")
    
    cnn_model = CNN()
    cnn_path = f'./models/cifar10_CNN_{CNN_EPOCHS}_{CNN_LR}.pth'
    
    if not os.path.exists(cnn_path):
        print(f"CNN model not found: {cnn_path}")
        print("Train CNN first with: python main.py --mode cnn")
        return
    
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    cnn_model.to(DEVICE).eval()
    
    os.makedirs('plots', exist_ok=True)
    
    test_scores = compute_ood_scores(cnn_model, testloader, max_logit_score)
    fake_scores = compute_ood_scores(cnn_model, fakeloader, max_logit_score)
    
    evaluate_ood_detection(test_scores, fake_scores, "plots/cnn_roc_curve.png")
    
    all_scores = torch.cat([test_scores, fake_scores])
    all_labels = torch.cat([torch.ones_like(test_scores), torch.zeros_like(fake_scores)])
    auc = roc_auc_score(all_labels.cpu().numpy(), all_scores.cpu().numpy())
    
    print(f"CNN AUC: {auc:.4f}")
    print("ROC curve saved to: plots/cnn_roc_curve.png")
    
    plot_ood_score_distributions(test_scores, fake_scores, "CNN Max Logit", 
                                "plots/cnn_ood_distributions.png")
    
    return auc


def train_cnn_pipeline(trainloader, testloader, fakeloader, classes):
    """Train and evaluate CNN."""
    print("Training CNN...")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    cnn_model = CNN()
    save_path = f'./models/cifar10_CNN_{CNN_EPOCHS}_{CNN_LR}.pth'
    train_cnn(cnn_model, trainloader, epochs=CNN_EPOCHS, lr=CNN_LR, save_path=save_path, device=DEVICE)
    
    accuracy, cm, _, _ = evaluate_cnn(cnn_model, testloader, classes)
    plot_confusion_matrix(cm, classes, "plots/cnn_confusion_matrix.png")
    
    test_scores = compute_ood_scores(cnn_model, testloader, max_logit_score)
    fake_scores = compute_ood_scores(cnn_model, fakeloader, max_logit_score)
    
    plot_ood_score_distributions(test_scores, fake_scores, "CNN Max Logit", 
                                "plots/cnn_ood_distributions.png")
    evaluate_ood_detection(test_scores, fake_scores, "plots/cnn_roc_curve.png")
    
    print(f"CNN completed. Accuracy: {accuracy:.4f}")
    return cnn_model


def train_ae_pipeline(trainloader, testloader, fakeloader):
    """Train and evaluate Autoencoder."""
    print("Training Autoencoder...")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    ae_model = Autoencoder()
    train_autoencoder(ae_model, trainloader, epochs=AE_EPOCHS, lr=AE_LR, device=DEVICE)
    
    save_path = f'./models/cifar10_Autoencoder_{AE_EPOCHS}_{AE_LR}.pth'
    torch.save(ae_model.state_dict(), save_path)
    
    plot_reconstruction_comparison(ae_model, testloader, save_path="plots/ae_reconstruction.png")
    
    test_scores = compute_ae_reconstruction_scores(ae_model, testloader)
    fake_scores = compute_ae_reconstruction_scores(ae_model, fakeloader)
    
    plot_ood_score_distributions(test_scores, fake_scores, "Autoencoder", 
                                "plots/ae_ood_distributions.png")
    evaluate_ood_detection(test_scores, fake_scores, "plots/ae_roc_curve.png")
    
    print("Autoencoder completed.")
    return ae_model


def compare_models(testloader, fakeloader):
    """Load and compare all trained models."""
    print("Comparing models...")
    
    
    cnn_model = CNN()
    cnn_path = f'./models/cifar10_CNN_{CNN_EPOCHS}_{CNN_LR}.pth'
    ae_model = Autoencoder()
    ae_path = f'./models/cifar10_Autoencoder_{AE_EPOCHS}_{AE_LR}.pth'
    
    if not os.path.exists(cnn_path):
        print(f"CNN model not found: {cnn_path}")
        return
    if not os.path.exists(ae_path):
        print(f"Autoencoder model not found: {ae_path}")
        return
    
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=DEVICE))
    ae_model.load_state_dict(torch.load(ae_path, map_location=DEVICE))
    cnn_model.to(DEVICE).eval()
    ae_model.to(DEVICE).eval()
    
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
    for name, (test_scores, fake_scores) in methods.items():
        all_scores = torch.cat([test_scores, fake_scores])
        all_labels = torch.cat([torch.ones_like(test_scores), torch.zeros_like(fake_scores)])
        auc = roc_auc_score(all_labels.cpu().numpy(), all_scores.cpu().numpy())
        results[name] = auc
        print(f"{name} AUC: {auc:.4f}")
    
    plt.figure(figsize=(8, 5))
    methods = list(results.keys())
    aucs = list(results.values())
    bars = plt.bar(methods, aucs, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('AUC Score')
    plt.title('OOD Detection Performance Comparison')
    plt.ylim(0, 1)
    
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    best_method = max(results.keys(), key=lambda k: results[k])
    print(f"Best method: {best_method} (AUC: {results[best_method]:.4f})")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='CIFAR-10 OOD Detection')
    parser.add_argument('--mode', choices=['cnn', 'ae', 'compare', 'all', 'roc-cnn'], default='all',
                        help='What to run')
    parser.add_argument('--epochs-cnn', type=int, default=None, help='CNN training epochs')
    parser.add_argument('--epochs-ae', type=int, default=None, help='Autoencoder training epochs')
    
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    
    trainloader, testloader, fakeloader, classes = get_data_loaders()
    
    global CNN_EPOCHS, AE_EPOCHS
    if args.epochs_cnn:
        CNN_EPOCHS = args.epochs_cnn
    if args.epochs_ae:
        AE_EPOCHS = args.epochs_ae
    
    if args.mode != 'roc-cnn':
        os.makedirs('plots', exist_ok=True)
        plot_sample_image(testloader, "Sample CIFAR-10", "plots/sample_cifar10.png")
        plot_sample_image(fakeloader, "Sample Fake", "plots/sample_fake.png")
    
    if args.mode in ['cnn', 'all']:
        train_cnn_pipeline(trainloader, testloader, fakeloader, classes)
    
    if args.mode in ['ae', 'all']:
        train_ae_pipeline(trainloader, testloader, fakeloader)
    
    if args.mode in ['compare', 'all']:
        compare_models(testloader, fakeloader)
    
    if args.mode == 'roc-cnn':
        plot_cnn_roc_only(testloader, fakeloader)
    
    print("Completed! Check plots/ directory for results.")


if __name__ == "__main__":
    main()