import pandas as pd
import torch
import torchvision.transforms as transforms
from pathlib import Path
from torch.utils.data import Dataset
from typing import Literal
from PIL import Image


SPLIT_MAP = {
    "train": 0,
    "val": 1,
    "test": 2,
}


class SCLRDataset(Dataset):
    
    def __init__(self, df: pd.DataFrame, dataroot: Path, split: Literal["train", "val", "test"]) -> None:
        '''Initialize an instance'''
        self.dataroot = dataroot
        self.df = df[df.split == SPLIT_MAP[split]]
        self.transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        
    def __len__(self) -> int:
        '''Return the length'''
        return len(self.df)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''Get item'''
        file_id = self.df.img_filename.iloc[idx]
        target = self.df.y.iloc[idx]
        
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        file_path = self.dataroot / file_id
        
        img = Image.open(file_path).convert("RGB")
        img = self.transform(img)
        return img, target_tensor  # type: ignore