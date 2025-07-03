# Lab 4: Out-of-Distribution Detection
A comprehensive implementation of Out-of-Distribution (OOD) detection methods on CIFAR-10, including CNN-based confidence scoring, autoencoder reconstruction, adversarial training, and advanced ODIN detection.

##  Project Overview

This laboratory explores multiple approaches to detect when input data comes from a different distribution than the training data. We implement and compare five main methods:

1. **CNN Max Logit**: Using maximum logit values as confidence scores
2. **CNN Max Softmax**: Using maximum softmax probabilities  
3. **Autoencoder Reconstruction**: Using reconstruction error as anomaly score
4. **Adversarial Training**: Robust models for improved OOD detection
5. **ODIN Method**: Out-of-DIstribution detector for Neural networks

##  Features

### Core OOD Detection Methods
- **CNN Confidence Scoring**: Multiple confidence-based metrics
- **Autoencoder Anomaly Detection**: Reconstruction error analysis
- **ODIN Implementation**: Temperature scaling and input preprocessing
- **Comparative Analysis**: ROC curves and AUC evaluation

### Adversarial Training Pipeline
- **FGSM Attack Implementation**: Fast Gradient Sign Method
- **Adversarial Training**: Mixed clean/adversarial example training
- **Robustness Evaluation**: Multiple epsilon values testing
- **Attack Visualization**: Individual attack result analysis

### Comprehensive Evaluation
- **ROC Analysis**: Detailed performance curves
- **Method Comparison**: Side-by-side evaluation
- **Statistical Analysis**: Score distributions and separability
- **Visual Results**: Confusion matrices and example plots

##  Installation

1. Navigate to the lab directory:
```bash
cd lab4
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify CUDA availability (optional):
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

##  Quick Start

### Complete Pipeline
```bash
# Train all models and run complete analysis
python main.py --mode all --epochs-cnn 30 --epochs-ae 20
```

### Individual Components
```bash
# Train CNN only
python main.py --mode cnn --epochs-cnn 30

# Train Autoencoder only
python main.py --mode ae --epochs-ae 20

# Compare all OOD detection methods
python compare_models.py

# Plot CNN ROC curve only
python main.py --mode roc-cnn
```

### Adversarial Training
```bash
# Train adversarially robust CNN
python adversarial_robustness.py --mode train --epochs 15 --eps 0.008

# Compare standard vs robust models
python adversarial_robustness.py --mode compare

# Generate ROC curves for adversarial analysis
python adversarial_robustness.py --mode roc
```

### ODIN Method
```bash
# Run ODIN detection with hyperparameter search
python odin_detection.py
```

### Attack Analysis
```bash
# Test individual attacks
python test_adversarial.py --test single

# Analyze epsilon sensitivity
python test_adversarial.py --test epsilon

# Test multiple samples
python test_adversarial.py --test multiple --samples 50
```

##  Project Structure

```
lab4/
├── README.md                    # This documentation
├── requirements.txt             # Python dependencies
├── config.py                   # Configuration settings
│
├── models.py                   # CNN and Autoencoder architectures
├── data_loader.py              # CIFAR-10 and fake data loading
├── training.py                 # Training functions
├── evaluation.py               # OOD evaluation utilities
├── visualization.py            # Plotting and visualization
│
├── main.py                     # Main OOD detection script
├── compare_models.py           # Model comparison analysis
├── odin_detection.py           # ODIN method implementation
│
├── adversarial_attacks.py      # FGSM attack implementation
├── adversarial_robustness.py   # Adversarial training pipeline
├── test_adversarial.py         # Adversarial testing utilities
│
├── models/                     # Pre-trained model weights
│   ├── cifar10_CNN_30_0.0001.pth              # Standard CNN
│   ├── cifar10_Autoencoder_20_0.0001.pth      # Autoencoder
│   ├── robust_cnn_eps_0.0020_alpha_0.3.pth    # Robust CNN (ε=0.002)
│   ├── robust_cnn_eps_0.0040_alpha_0.5.pth    # Robust CNN (ε=0.004)
│   └── robust_cnn_eps_0.0078_alpha_0.7.pth    # Robust CNN (ε=0.008)
│
└── plots/                      # Generated visualizations
    ├── cnn_confusion_matrix.png           # CNN classification results
    ├── cnn_roc_curve.png                  # CNN OOD detection ROC
    ├── ae_roc_curve.png                   # Autoencoder OOD detection ROC
    ├── comparison.png                     # Method comparison
    ├── odin_roc_curve.png                 # ODIN method results
    ├── roc_curves_standard.png            # Standard model ROC analysis
    ├── ae_reconstruction.png              # Autoencoder reconstructions
    ├── cnn_ood_distributions.png          # Score distributions
    ├── ae_ood_distributions.png           # AE score distributions
    ├── odin_grid_search.png               # ODIN hyperparameter search
    ├── fgsm_targeted_attack_sample_0.png  # Attack visualization
    ├── sample_cifar10.png                 # CIFAR-10 samples
    └── sample_fake.png                    # Fake data samples
```
##  License

This project is part of academic coursework and is available under the MIT License. See the [LICENSE](LICENSE) file for details.

---

##  Summary

This Lab 4 implementation provides a comprehensive exploration of Out-of-Distribution detection methods, combining theoretical understanding with practical implementation. The included pre-trained models and visualizations enable immediate experimentation and analysis, while the modular code structure supports extension and customization for research purposes.
