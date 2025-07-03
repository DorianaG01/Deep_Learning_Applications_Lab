"""
ResNet implementation with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock
from typing import List


class ResNet(nn.Module):
    
    def __init__(self,
                 num_classes: int = 10,
                 depths: List[int] = [1, 1],
                 channels: List[int] = [16, 32],
                 initial_channels: int = 16,
                 input_rgb: bool = True,
                 name: str = "ResNet"):
        super(ResNet, self).__init__()
        
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

        self.stage1 = self._make_stage(channels[0], depths[0], stride=2)
        self.stage2 = self._make_stage(channels[1], depths[1], stride=2)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(channels[-1], num_classes)

    def _make_stage(self, out_channels: int, num_blocks: int, stride: int):
        
        downsample = None
        if self.current_channels != out_channels or stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.current_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [BasicBlock(self.current_channels, out_channels, stride, downsample)]
        self.current_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
    
        x = self.input_adapter(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        
        if return_features:
            return x

        x = self.classifier(x)
        return x
    
    def replace_classifier(self, num_classes):
        
        self.classifier = nn.Linear(self.channels[-1], num_classes)


    def freeze_layers(self, layers_to_freeze):
        
        for name, param in self.named_parameters():
            if any(layer in name for layer in layers_to_freeze):
                param.requires_grad = False

                
