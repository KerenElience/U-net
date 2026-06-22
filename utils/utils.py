import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from functools import partial
from joblib import Parallel, delayed

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

def calc_rare_sample_weight(dataset, precent, rare_threshold = 0.01, ignore_index = [0],
                            rare_weight = 2.0, normal_weight = 1.0,
                            n_jobs = -1):
    rare_indices = [n for n, v in enumerate(precent) if v < rare_threshold and n not in ignore_index]

    def get_mask(idx):
        return dataset[idx][1]
    
    def _check_rare(idx, rare_indices):
        mask = get_mask(idx)
        unique = np.unique(mask)
        return rare_weight if len(np.intersect1d(unique, rare_indices)) > 0 else normal_weight
    
    mask_idx = list(range(len(dataset)))
    check_fun = partial(_check_rare, rare_indices = rare_indices)
    sample_weight = Parallel(n_jobs=n_jobs)(
        delayed(check_fun)(i) for i in mask_idx
    )
    return sample_weight

def calc_classes_weight_by_pixel(dataset):
    cls = np.array()
    for _, mask in dataset:
        mask[mask == 255] = 0
        cls.append(np.bincount(mask.flatten(), minlength=21))
    
    precent = cls.sum(axis =0 )/cls.sum()
    weight = np.clip( np.median(precent) / (precent + 1e-6), 0.1, 10)
    return precent, weight