import sys
from pathlib import Path

current = Path.cwd()
root = current.parent.parent
sys.path.append(root)

from lib import zscore_norm


class ZScoreNorm:
    def __init__(
        self, 
        out_path=None
        ):
        self.out_path = out_path

    def step(self, ctx):
        ctx.image = zscore_norm(
            ctx.image,
            self.out_path
        )
        return ctx