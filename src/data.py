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
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.name = os.listdir(os.path.join(path, "segment"))
    
    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        name = self.name[index]
        segment_path = os.path.join(self.path, "raw", name)
        raw_path = os.path.join(self.path, "segment", name.replace(".png", ".jpg"))
        segment_img = keep_image_size_open(segment_path)
        image = keep_image_size_open(raw_path)
        return self.transform(image), self.transform(segment_img)
    
    def transform(self):
        trans = transforms.Compose([
            transforms.ToTensor()
        ])
        return trans
