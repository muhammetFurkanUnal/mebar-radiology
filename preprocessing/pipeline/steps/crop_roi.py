import sys
from pathlib import Path

current = Path.cwd()
root = current.parent.parent
sys.path.append(root)

from lib import crop_ROI


class CropROI:
    def __init__(
        self, 
        volume_mm,
        out_path=None
        ):
        self.volume_mm = volume_mm
        self.out_path = out_path

    def step(self, ctx):
        ctx.mask, ctx.image = crop_ROI(
            ctx.mask,
            ctx.image,
            self.volume_mm,
            self.out_path
        )
        return ctx