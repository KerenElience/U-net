import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = Path(ROOT)
runpath = ROOT / "run"
if not runpath.exists():
    runpath.mkdir()

def read_image(raw_path, segment_path):
    img = Image.open(raw_path).convert("RGB")
    mask = Image.open(segment_path)
    return np.array(img), np.array(mask)

# def read_image_cv(raw_path, segment_path):
#     img = cv2.imread(raw_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     mask = cv2.imread(segment_path, cv2.IMREAD_GRAYSCALE)
#     return img, mask