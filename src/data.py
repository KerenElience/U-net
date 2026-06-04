"""
- UnetData
    - train
        - raw
        - segment
    - valid
        - xxx
        - xxx
    - test
        - xxx
        - xxx
"""

import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from utils.utils import keep_image_size_open

class UnetDataset(Dataset):
    def __init__(self, path, datatype = "train"):
        super().__init__()
        self.path = path
        self.namepath = os.path.join(path, "Segmentation", f"{datatype}.txt")
        self.name = self.inital()
    
    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        name = self.name[index]
        segment_path = os.path.join(self.path, "SegmentationObject", f"{name}.png")
        raw_path = os.path.join(self.path, "JPEGImages", f"{name}.jpg")
        segment_img = keep_image_size_open(segment_path)
        image = keep_image_size_open(raw_path)
        trans = self.transform()
        return trans(image), trans(segment_img)
    
    def transform(self):
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        return trans
    
    def inital(self):
        name = []
        with open(self.namepath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    name.append(line)
        return name
            
