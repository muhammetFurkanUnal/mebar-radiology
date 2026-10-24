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

    def step(self, ctx):
        ctx.mask = resample_image(
            ctx.mask,
            self.target_spacing,
            is_mask=True
        )
        ctx.image = resample_image(
            ctx.image,
            self.target_spacing,
            is_mask=False
        )
        return ctx