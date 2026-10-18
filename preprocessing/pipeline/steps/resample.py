import sys
from pathlib import Path

current = Path.cwd()
root = current.parent.parent
sys.path.append(root)

from lib import resample_image

class Resample:
    def __init__(
        self, 
        target_spacing,
        ):
        self.target_spacing = target_spacing

    def step(self, mask, image):
        mask = resample_image(
            mask,
            self.target_spacing,
            is_mask=True
        )
        image = resample_image(
            image,
            self.target_spacing,
            is_mask=False
        )
        return mask, image