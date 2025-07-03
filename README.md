# Lab 1: ResNet Paper Replication 

This laboratory demonstrates the key insights of ResNet through a series of progressively complex experiments, from simple MLPs on MNIST to advanced transfer learning on CIFAR datasets.

##  Objective

Replicate and understand the core findings of the ResNet paper:
- **Deeper networks don't guarantee better performance** without proper architecture
- **Residual connections enable training of very deep networks**
- **Skip connections solve the vanishing gradient problem**
- **Transfer learning benefits from well-trained feature extractors**

##  Experiments & Results

### Exercise 1.1-1.2: MLP vs ResidualMLP (MNIST) 

**Objective**: Demonstrate that residual connections solve vanishing gradient problems in deep networks

<div align="center">
<img src="lab1/results/MLP_comparison.png" alt="MLP vs ResidualMLP Comparison" width="55%">
</div>

**Key Findings**:
- **Standard MLP (16 layers)**: Severe training instability, poor convergence (~60-80% accuracy)
- **ResidualMLP (16 layers)**: Stable training, consistent convergence (>95% accuracy)
- **Critical Insight**: Skip connections enable training of much deeper networks without degradation

---

### Exercise 1.3: PlainCNN vs ResNet (CIFAR-10) 

**Objective**: Demonstrate degradation problem in deep CNNs and ResNet's solution

<div align="center">
<img src="lab1/results/CNN_comparison.png" alt="CNN Architecture Comparison" width="55%">
</div>

**Key Findings**:
- **PlainCNN Large**: Performance degrades significantly with depth (~68% accuracy)
- **ResNet Large**: Consistent improvement with depth (~78% accuracy)
- **Training Dynamics**: ResNets show smoother loss curves and faster convergence

| Architecture | Size | Depth | Final Accuracy | Training Convergence |
|--------------|------|-------|----------------|---------------------|
| PlainCNN | Small | [2,2] | ~76% | Stable |
| PlainCNN | Medium | [5,5] | ~74% | Some degradation |
| PlainCNN | Large | [7,7] | **~68%** | **Clear degradation** |
| ResNet | Small | [2,2] | ~76% | Stable |
| ResNet | Medium | [5,5] | ~77% | Improved |
| ResNet | Large | [7,7] | **~78%** | **Best performance** |

** Critical Insight**: PlainCNNs exhibit the "degradation problem" - deeper networks perform worse even on training data. ResNets completely solve this issue.

---

### Exercise 2.1: Transfer Learning (CIFAR-10 → CIFAR-100) 

**Objective**: Explore transfer learning strategies and fine-tuning approaches

#### Phase 1: Feature Extraction Baseline
- **Method**: Linear SVM on frozen ResNet features
- **Result**: **18.5% accuracy** on CIFAR-100
- **Insight**: Pre-trained features contain useful transferable information

#### Phase 2: Fine-tuning Strategy Comparison

<div align="center" style="font-size: 12px;" >

| Rank | Strategy | Optimizer | Test Accuracy | Trainable Params |
|------|----------|-----------|---------------|------------------|
|  | Freeze Input Only | Adam | **53.30%** | 162,180 |
|  | No Freeze | Adam | **53.06%** | 162,644 |
|  | Freeze Input Only | SGD | **50.64%** | 162,180 |
|  | No Freeze | SGD | **49.05%** | 162,644 |
|  | Freeze All Early | SGD | **47.89%** | 129,188 |
|  | Freeze All Early | Adam | **47.86%** | 129,188 |

</div>


---
