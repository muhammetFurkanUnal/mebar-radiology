import sys
from pathlib import Path

current = Path.cwd()
root = current.parent.parent
sys.path.append(root)

from lib import bias_field_correction


class BiasFieldCorrection:
    def __init__(
        self, 
        shrink_factor,
        out_path=None
        ):
        self.shrink_factor = shrink_factor
        self.out_path = out_path

    def step(self, ctx):
        ctx.image = bias_field_correction(
            ctx.image,
            self.shrink_factor,
            self.out_path
        )
        return ctx