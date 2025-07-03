"""
Configuration file for the CIFAR-10 OOD Detection project.
"""

import os
import torch
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


BATCH_SIZE = 128
NUM_WORKERS = 4
DATA_ROOT = './data'


CNN_EPOCHS = 30
AE_EPOCHS = 20
CNN_LR = 0.0001
AE_LR = 0.0001

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

PLOT_BINS = 25
ALPHA = 0.5