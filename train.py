import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    '''Train one epoch for given model.'''
    model.train()
    train_loss = 0.0
    total_samples = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(images)
        total_samples += len(images)
    return train_loss / total_samples


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    '''Evaluate the model with given data.'''
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predicted = outputs.argmax(dim=1)
            total_correct += (labels == predicted).sum().item()
            val_loss += loss.item() * len(images)
            total_samples += len(images)
    return val_loss / total_samples, total_correct / total_samples