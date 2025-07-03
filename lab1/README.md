# Lab 1: ResNet Paper Replication 🧪

A comprehensive implementation of experiments inspired by the seminal ResNet paper:

> **[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)**  
> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, CVPR 2016

This laboratory demonstrates the key insights of ResNet through a series of progressively complex experiments, from simple MLPs on MNIST to advanced transfer learning on CIFAR datasets.

##  Objective

Replicate and understand the core findings of the ResNet paper:
- **Deeper networks don't guarantee better performance** without proper architecture
- **Residual connections enable training of very deep networks**
- **Skip connections solve the vanishing gradient problem**
- **Transfer learning benefits from well-trained feature extractors**

##  Project Structure

```
lab1/
├── __init__.py                       # Package initialization
├── config2.yaml / my_config.yaml    # Configuration files
├── main.py                          # Main orchestration script
├── models/
│   ├── model_factor.py              # Factory pattern for model creation
│   ├── residual_mlp.py              # Exercise 1.1 + 1.2 implementations
│   ├── plain_cnn.py                 # Plain CNN for Exercise 1.3
│   └── resnet.py                    # ResNet for Exercise 1.3
├── extract_features.py              # Feature extraction for Exercise 2.1
├── fine_tuning.py                   # Fine-tuning for Exercise 2.1
├── results/
│   ├── checkpoint/                  # Saved model checkpoints
│   └── training_results.png         # Training visualizations
└── utils/
    ├── data_loader.py               # Data loading utilities
    ├── training.py                  # Training pipeline
    └── visualization.py             # Plotting and analysis tools
```

##  Exercises Overview

### Exercise 1.1: Baseline MLP 
**Dataset**: MNIST (28×28 grayscale, 10 classes)  
**File**: `models/residual_mlp.py`

Implement a simple Multi-Layer Perceptron to establish baseline performance on MNIST digit classification.

**Architecture**:
```
Input (784) → Hidden Layer(s) → Output (10)
```

**Key Metrics**: Training/validation loss and accuracy across epochs

---

### Exercise 1.2: Residual MLP 
**Dataset**: MNIST (28×28 grayscale, 10 classes)  
**File**: `models/residual_mlp.py`

Compare standard MLP vs MLP with residual connections. Analyze gradient flow and training stability.

**Architecture**:
```
Input → [Residual Block + Skip Connection] × N → Output
```

**Analysis**:
- Performance comparison across different network depths
- Gradient magnitude analysis to demonstrate improved flow
- Training convergence comparison

---

### Exercise 1.3: CNN Comparison 🖼
**Dataset**: CIFAR-10 (32×32 RGB, 10 classes)  
**Files**: `models/plain_cnn.py`, `models/resnet.py`

Demonstrate that deeper CNNs without residual connections don't always perform better, while ResNets consistently improve with depth.

**Architectures**:
- **PlainCNN**: Standard convolutional layers without skip connections
- **ResNet**: Convolutional layers with residual blocks

**Key Comparison**: Performance vs network depth for both architectures

---

### Exercise 2.1: Transfer Learning 
**Datasets**: CIFAR-10 → CIFAR-100  
**Files**: `extract_features.py`, `fine_tuning.py`

Explore transfer learning strategies using a ResNet trained on CIFAR-10.

**Two-Step Process**:
1. **Feature Extraction**: Use pre-trained ResNet as feature extractor
   - Load ResNet trained on CIFAR-10 from checkpoint
   - Extract features from CIFAR-100 using frozen ResNet (before classifier)
   - Train Linear SVM on extracted features as baseline
   
2. **Fine-tuning**: Comprehensive comparison of fine-tuning strategies
   - Replace classifier layer (10 → 100 classes)
   - Compare different freezing strategies: all layers, input only, no freezing
   - Test both Adam and SGD optimizers
   - Track experiments with Weights & Biases integration

##  Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib pyyaml scikit-learn numpy tqdm wandb
```

**Note**: The fine-tuning experiments use Weights & Biases for experiment tracking. Create a free account at [wandb.ai](https://wandb.ai) and run `wandb login` before executing `fine_tuning.py`.

### Running Experiments

#### 1. MLP Experiments (MNIST)
```bash
# Run both baseline and residual MLP comparison
python main.py --exercise 1 --dataset mnist
```

#### 2. CNN Comparison (CIFAR-10)
```bash
# Compare PlainCNN vs ResNet
python main.py --exercise 1.3 --dataset cifar10
```

#### 3. Transfer Learning (CIFAR-10 → CIFAR-100)
```bash
# Step 1: Extract features and establish SVM baseline
python extract_features.py

# Step 2: Run comprehensive fine-tuning experiments
python fine_tuning.py
```

**Fine-tuning experiments include**:
- 6 different configurations testing freezing strategies and optimizers
- Automatic experiment tracking with Weights & Biases
- Comprehensive comparison of Adam vs SGD optimizers
- Results saved in `./results/Experiments/` with individual checkpoints

## ⚙ Configuration

Modify `config2.yaml` or `my_config.yaml` to customize:

```yaml
# Example configuration
model:
  depths: [2, 5, 7]           # Network depths to compare
  channels: [16, 32, 64]      # Channel progression
  initial_channels: 16        # Starting channels
  num_classes: 10             # Output classes

dataset:
  name: "CIFAR10"
  root: "./data"
  batch_size: 128
  augmentation: true

training:
  epochs: 20
  learning_rate: 0.1
  optimizer: "sgd"
  scheduler:
    type: "multistep"
    milestones: [8, 15]
    gamma: 0.1
```

##  Key Features

###  Factory Pattern
```python
from models.model_factor import create_model

# Easy model creation
mlp = create_model("MLP", "Small", num_classes=10)
resnet = create_model("ResNet", "Large", num_classes=100)
```

###  Unified Training Pipeline
```python
from utils.training import train_model_simple

# Same interface for all model types
results = train_model_simple(model, train_loader, val_loader, optimizer, scheduler, device, epochs)
```

###  Comprehensive Visualization
```python
from utils.visualization import plot_results, create_simple_report

# Automatic plotting and reporting
plot_results(results, save_path="./results/comparison.png")
create_simple_report(results)
```

###  Flexible Data Loading
```python
from utils.data_loader import create_data_loaders

# Supports MNIST, CIFAR-10, CIFAR-100
train_loader, val_loader, dataset_info = create_data_loaders(config)
```

##  Monitoring and Analysis

### Weights & Biases Integration (Implemented)
Your fine-tuning experiments automatically log to W&B:
```python
# Already integrated in fine_tuning.py
wandb.init(project="cifar100-finetuning", name=experiment_name)
```

**Tracked Metrics**:
- Training/validation loss and accuracy per epoch
- Learning rate scheduling
- Trainable vs frozen parameters count
- Best validation accuracy and corresponding epoch
- Final test accuracy for each experiment

### Tensorboard Alternative
```bash
# If you prefer Tensorboard for other experiments
tensorboard --logdir=./results/tensorboard
```

##  References

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- [PyTorch ResNet Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)


