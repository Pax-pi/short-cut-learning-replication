import time
from pathlib import Path

import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SCLRDataset
from train import train_one_epoch, evaluate

DATA_ROOT = Path('data/waterbird_complete95_forest2water2')
METADATA_PATH = DATA_ROOT / 'metadata.csv'
NUM_EPOCHS = 100
BATCH_SIZE = 32
LR = 0.001
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 2


def main():
    metadata = pd.read_csv(METADATA_PATH)
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    device = torch.device('mps')
    model.to(device)
    train_dataset = SCLRDataset(metadata, DATA_ROOT, 'train')
    val_dataset = SCLRDataset(metadata, DATA_ROOT, 'val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    best_val_loss = float('inf')
    val_acc_at_best_loss = 0.0
    
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model=model, 
            dataloader=train_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device
        )
        val_loss, val_acc = evaluate(
            model=model, 
            dataloader=val_loader, 
            criterion=criterion, 
            device=device
        )
        val_accuracy = val_acc * 100
        
        epoch_time = time.time() - start_time
        
        print(
            f'Epoch [{epoch + 1}/{NUM_EPOCHS}] | '
            f'Time taken: {epoch_time:.1f}s | '
            f'Training Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | '
            f'Validation Accuracy: {val_accuracy:.2f}%'
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_acc_at_best_loss = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New record! Saving weights. Val Loss: {val_loss:.4f}')
            
    model.load_state_dict(torch.load('best_model.pth'))
    print(f'Training complete. Loaded checkpoint with the lowest validation loss. Accuracy: {val_acc_at_best_loss:.2f}%.')
        
        
if __name__ == '__main__':
    main()