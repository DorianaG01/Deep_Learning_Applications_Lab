import torch
import os
import numpy as np
from models import CNN
from data_loader import get_data_loaders, get_class_dict
from adversarial_attacks import FGSMAttacker
from config import DEVICE, CNN_EPOCHS, CNN_LR
import argparse


def load_trained_cnn():
    """Load the trained CNN model."""
    model_path = f'./models/cifar10_CNN_{CNN_EPOCHS}_{CNN_LR}.pth'
    if not os.path.exists(model_path):
        print(f"CNN model not found at: {model_path}")
        print("Train the model first with: python main.py --mode cnn")
        return None
    
    model = CNN()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()
    print("CNN model loaded successfully.")
    return model


def test_single_attack(model, testloader, classes, class_dict):
    """Test FGSM attack on a single sample."""
    print("Testing single FGSM attack...")
    
    for x_batch, y_batch in testloader:
        break
    
    sample_id = 0
    eps = 1/255
    target_label = class_dict['deer']
    
    x = x_batch[sample_id].to(DEVICE)
    y = y_batch[sample_id].to(DEVICE)
    
    attacker = FGSMAttacker(model, DEVICE)
    
    # Untargeted attack
    result = attacker.fgsm_attack(x, y, eps, targeted=False)
    print(f"Untargeted attack - Success: {result['success']}")
    
    # Targeted attack
    result = attacker.fgsm_attack(x, y, eps, targeted=True, target_label=target_label)
    print(f"Targeted attack - Success: {result['success']}")
    
    if result['success']:
        print(f"Final prediction: {classes[result['final_prediction']]}")
    
    os.makedirs('plots', exist_ok=True)
    save_path = f"plots/fgsm_attack_sample_{sample_id}.png"
    attacker.visualize_attack_result(result, classes, save_path)


def test_epsilon_sensitivity(model, testloader, classes, class_dict):
    """Test attack success across different epsilon values."""
    print("Testing epsilon sensitivity...")
    
    for x_batch, y_batch in testloader:
        break
    
    sample_id = 0
    x = x_batch[sample_id].to(DEVICE)
    y = y_batch[sample_id].to(DEVICE)
    
    epsilon_values = [0.5/255, 1/255, 2/255, 4/255, 8/255, 16/255]
    target_label = class_dict['deer']
    
    attacker = FGSMAttacker(model, DEVICE)
    
    # Untargeted attacks
    untargeted_results = attacker.evaluate_epsilon_sensitivity(
        x, y, epsilon_values, targeted=False
    )
    
    print("Untargeted Attack Results:")
    for eps, result in untargeted_results.items():
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f" ε={eps:.4f}: {status}")
    
    # Targeted attacks
    targeted_results = attacker.evaluate_epsilon_sensitivity(
        x, y, epsilon_values, targeted=True, target_label=target_label
    )
    
    print("Targeted Attack Results:")
    for eps, result in targeted_results.items():
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f" ε={eps:.4f}: {status}")
    
  
    os.makedirs('plots', exist_ok=True)
    attacker.plot_epsilon_analysis(
        untargeted_results,
        save_path="plots/epsilon_analysis_untargeted.png"
    )
    attacker.plot_epsilon_analysis(
        targeted_results,
        save_path="plots/epsilon_analysis_targeted.png"
    )


def test_multiple_samples(model, testloader, classes, class_dict, num_samples=10):
    """Test attacks on multiple samples to get statistics."""
    print(f"Testing multiple samples ({num_samples} samples)...")
    
    attacker = FGSMAttacker(model, DEVICE)
    eps = 2/255
    target_label = class_dict['deer']
    
    untargeted_successes = 0
    targeted_successes = 0
    samples_tested = 0
    
    for x_batch, y_batch in testloader:
        for i in range(min(num_samples, len(x_batch))):
            if samples_tested >= num_samples:
                break
                
            x = x_batch[i].to(DEVICE)
            y = y_batch[i].to(DEVICE)
            
            # Skip if target equals ground truth
            if y.item() == target_label:
                continue
            
            # Untargeted attack
            result_untargeted = attacker.fgsm_attack(x, y, eps, targeted=False)
            if result_untargeted['success']:
                untargeted_successes += 1
            
            # Targeted attack
            result_targeted = attacker.fgsm_attack(x, y, eps, targeted=True, target_label=target_label)
            if result_targeted['success']:
                targeted_successes += 1
            
            samples_tested += 1
        
        if samples_tested >= num_samples:
            break
    
    print(f"Untargeted attacks: {untargeted_successes}/{samples_tested} ({100*untargeted_successes/samples_tested:.1f}%)")
    print(f"Targeted attacks: {targeted_successes}/{samples_tested} ({100*targeted_successes/samples_tested:.1f}%)")


def main():

    parser = argparse.ArgumentParser(description='Test adversarial attacks on CNN')
    parser.add_argument('--test', choices=['single', 'epsilon', 'multiple', 'all'],
                        default='all', help='Type of test to run')
    parser.add_argument('--samples', type=int, default=10,
                        help='Number of samples for multiple test')
    
    args = parser.parse_args()
    
    model = load_trained_cnn()
    if model is None:
        return
    
    _, testloader, _, classes = get_data_loaders()
    class_dict = get_class_dict(classes)
    
    if args.test in ['single', 'all']:
        test_single_attack(model, testloader, classes, class_dict)
    
    if args.test in ['epsilon', 'all']:
        test_epsilon_sensitivity(model, testloader, classes, class_dict)
    
    if args.test in ['multiple', 'all']:
        test_multiple_samples(model, testloader, classes, class_dict, args.samples)
    
    print("Adversarial testing completed!")
    print("Check plots/ directory for visualizations.")


if __name__ == "__main__":
    main()