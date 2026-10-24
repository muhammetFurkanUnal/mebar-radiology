import os

import SimpleITK as sitk


class Write:
    def step(self, ctx):
        for path in (ctx.mask_out_path, ctx.image_out_path):
            folder = os.path.dirname(path)
            if folder:
                os.makedirs(folder, exist_ok=True)
        sitk.WriteImage(ctx.mask, ctx.mask_out_path)
        sitk.WriteImage(ctx.image, ctx.image_out_path)
        return ctx
