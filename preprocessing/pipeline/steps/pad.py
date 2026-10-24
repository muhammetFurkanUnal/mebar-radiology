import sys
from pathlib import Path

current = Path.cwd()
root = current.parent.parent
sys.path.append(root)

from lib import pad


class Pad:
    def __init__(
        self, 
        target_size,
        out_path=None
        ):
        self.target_size = target_size
        self.out_path = out_path

    def step(self, ctx):
        ctx.mask, ctx.image = pad(
            ctx.mask,
            ctx.image,
            self.target_size,
            self.out_path
        )
        return ctx