from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


SPLIT_MAP = {
    'train': 0,
    'val': 1,
    'test': 2,
}


class SCLRDataset(Dataset):
    
    def __init__(self, df: pd.DataFrame, dataroot: Path, split: Literal['train', 'val', 'test']) -> None:
        """Initialize an instance"""
        self.dataroot = dataroot
        self.df = df[df.split == SPLIT_MAP[split]]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def __len__(self) -> int:
        """Return the length"""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and return a sample from the dataset.
        
        Args:
            idx: Positional index of the sample within the selected split.
            
        Returns:
            A tuple containing the image, target, and place tensors.
            The image has shape (3, 224, 224) and dtype torch.float32.
            The target is a scalar torch.long tensor, where 0 represents landbirds and 1 represents waterbirds.
            The place is a scalar torch.long tensor, where 0 represents land backgrounds and 1 represents water backgrounds.
        """
        file_id = self.df.img_filename.iloc[idx]
        target = self.df.y.iloc[idx]
        place = self.df.place.iloc[idx]
        
        target_tensor = torch.tensor(target, dtype=torch.long)
        place_tensor = torch.tensor(place, dtype=torch.long)
        
        file_path = self.dataroot / file_id
        
        img = Image.open(file_path).convert('RGB')
        img = self.transform(img)
        return img, target_tensor, place_tensor  # type: ignore