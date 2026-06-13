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
import albumentations as A
from torch.utils.data import Dataset
from utils.utils import read_image

MAX_HEIGHT, MAX_WIDTH = 256, 256
train_trans = A.Compose([
    A.RandomResizedCrop((MAX_HEIGHT, MAX_WIDTH)),
    # A.Resize(MAX_HEIGHT, MAX_WIDTH),
    A.CoarseDropout(),
    A.ColorJitter(),
    A.ElasticTransform(),
    A.HorizontalFlip(p = 0.5),
    A.GaussNoise(p = 0.2),
    A.ToTensorV2()
])

val_trans = A.Compose([
    A.Resize(MAX_HEIGHT, MAX_WIDTH),
    A.ToTensorV2()
])

class UnetDataset(Dataset):
    def __init__(self, path, datatype = "train"):
        super().__init__()
        self.path = path
        self.namepath = os.path.join(path, "Segmentation", f"{datatype}.txt")
        self.name = self.inital()
        self.transformer = train_trans if datatype == "train" else val_trans
    
    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        name = self.name[index]
        segment_path = os.path.join(self.path, "SegmentationClass", f"{name}.png")
        raw_path = os.path.join(self.path, "JPEGImages", f"{name}.jpg")
        image, segment_img = read_image(raw_path, segment_path)
        augmented = self.transformer(image = image, mask = segment_img)
        img, mask = augmented["image"], augmented["mask"]
        img = img / 255
        mask = mask.long()
        return img, mask
    
    def inital(self):
        name = []
        with open(self.namepath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line:
                    name.append(line)
        return name
