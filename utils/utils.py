import os
from pathlib import Path
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = Path(ROOT)
runpath = ROOT / "run"
if not runpath.exists():
    runpath.mkdir()
    
def keep_image_size_open(path, size = (256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new("RGB", (temp, temp), (0, 0 ,0))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask