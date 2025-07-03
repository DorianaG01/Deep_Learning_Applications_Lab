"""
PlainCNN implementation without skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PlainCNN(nn.Module):
  
    def __init__(self,
                 num_classes: int = 10,
                 depths: List[int] = [1, 1],
                 channels: List[int] = [16, 32],
                 initial_channels: int = 16,
                 input_rgb: bool = True,
                 name: str = "PlainCNN"):
        super(PlainCNN, self).__init__()
        
        in_channels = 3 if input_rgb else 1
        self.name = name
        self.depths = depths
        self.channels = channels
        self.current_channels = initial_channels
        
    
        self.input_adapter = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )
        
    
        self.stage1 = self._make_stage(channels[0], depths[0], stride=2)  # 32x32 -> 16x16
        self.stage2 = self._make_stage(channels[1], depths[1], stride=2)  # 16x16 -> 8x8
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(channels[-1], num_classes)
        
    def _make_stage(self, out_channels: int, depth: int, stride: int):
        
        layers = []
        
    
        layers.append(self._make_block(self.current_channels, out_channels, stride))
        self.current_channels = out_channels
        
    
        for _ in range(1, depth):
            layers.append(self._make_block(out_channels, out_channels, stride=1))
            
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels: int, out_channels: int, stride: int):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
    
        x = self.input_adapter(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x