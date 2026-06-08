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
import cv2
from torch.utils.data import Dataset
from utils.utils import read_image

MAX_HEIGHT, MAX_WIDTH = 256, 256
train_trans = A.Compose([
    A.Resize(288, 288),
    A.RandomCrop(MAX_HEIGHT, MAX_WIDTH),
    A.HorizontalFlip(p = 0.5),
    A.RandomRotate90(p = 0.1),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.2),
    A.RandomBrightnessContrast(p=0.1),
    A.GaussNoise(p=0.1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255),
    A.ToTensorV2()
])

val_trans = A.Compose([
    A.Resize(MAX_HEIGHT, MAX_WIDTH, interpolation=cv2.INTER_NEAREST),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255),
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
