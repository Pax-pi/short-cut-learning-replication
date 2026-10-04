import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    '''Train one epoch for given model.'''
    model.train()
    train_loss = 0.0
    total_samples = 0
    for images, labels, _ in dataloader:
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
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predicted = outputs.argmax(dim=1)
            total_correct += (labels == predicted).sum().item()
            val_loss += loss.item() * len(images)
            total_samples += len(images)
    return val_loss / total_samples, total_correct / total_samples

def evaluate_cross_group(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> dict:
    '''Evaluate the model with given data with cross group metrics.'''
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_samples = 0
    G0_correct = 0
    G1_correct = 0
    G2_correct = 0
    G3_correct = 0
    total_G0 = 0
    total_G1 = 0
    total_G2 = 0
    total_G3 = 0
    with torch.no_grad():
        for images, labels, place in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            place = place.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predicted = outputs.argmax(dim=1)
            correct = labels == predicted
            total_correct += correct.sum().item()
            val_loss += loss.item() * len(images)
            total_samples += len(images)
            G0_correct += (correct & (labels == 0) & (place == 0)).sum().item()
            G1_correct += (correct & (labels == 0) & (place == 1)).sum().item()
            G2_correct += (correct & (labels == 1) & (place == 0)).sum().item()
            G3_correct += (correct & (labels == 1) & (place == 1)).sum().item()
            total_G0 += ((labels == 0) & (place == 0)).sum().item()
            total_G1 += ((labels == 0) & (place == 1)).sum().item()
            total_G2 += ((labels == 1) & (place == 0)).sum().item()
            total_G3 += ((labels == 1) & (place == 1)).sum().item()
    metrics = {}
    metrics['Avg_Loss'] = val_loss / total_samples 
    metrics['Accuracy'] = total_correct / total_samples
    metrics['G0_Accuracy'] = G0_correct / total_G0
    metrics['G1_Accuracy'] = G1_correct / total_G1
    metrics['G2_Accuracy'] = G2_correct / total_G2
    metrics['G3_Accuracy'] = G3_correct / total_G3
    metrics['WGA'] = min(metrics['G0_Accuracy'], metrics['G1_Accuracy'], metrics['G2_Accuracy'], metrics['G3_Accuracy'])
    metrics['GAP'] =  metrics['Accuracy'] - metrics['WGA']
    return metrics