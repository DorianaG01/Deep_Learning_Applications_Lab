"""
Training utilities for CNN and Autoencoder models.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def train_cnn(model, trainloader, epochs=50, lr=0.0001, save_path=None, device='cpu'):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')
    
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f'Model saved to {save_path}')


def train_autoencoder(model, trainloader, epochs=20, lr=0.0001, device='cpu'):
    """
    Train the Autoencoder model.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, _ = data  
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            encoded, decoded = model(inputs)
            loss = criterion(inputs, decoded)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(trainloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')