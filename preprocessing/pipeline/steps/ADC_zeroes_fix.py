import sys
from pathlib import Path

current = Path.cwd()
root = current.parent.parent
sys.path.append(root)

from lib import ADC_intensity_threshold


class ADCZeroesFix:
    def __init__(
        self, 
        lower,
        uppper,
        out_path=None
        ):
        self.lower = lower
        self.upper = upper
        self.out_path = out_path

    def step(self, ctx):
        ctx.image = ADC_intensity_threshold(
            ctx.image,
            self.lower,
            self.upper,
            self.out_path
        )
        return ctx