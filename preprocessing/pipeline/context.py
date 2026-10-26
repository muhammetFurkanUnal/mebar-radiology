from dataclasses import dataclass, field

import SimpleITK as sitk


# A container that carries all the information of a single sample.
# Each pipeline step receives this container, modifies it, and gives it back.
@dataclass
class SampleContext:
    sample_id: str          # unique id of the sample
    modality: str           # image type: ADC or T2
    mask: sitk.Image        # tumor mask (image)
    image: sitk.Image       # the actual image
    mask_out_path: str = "" # file path where the mask will be saved
    image_out_path: str = ""# file path where the image will be saved
    meta: dict = field(default_factory=dict)  # free space to hold extra data later


# Extracts the measurements of one image into a single dict.
# Thanks to `prefix`, the mask and the image sit in the same dict without clashing.
def _img_metrics(image: sitk.Image, prefix: str) -> dict:
    size = image.GetSize()
    spacing = image.GetSpacing()
    mm = [s * sp for s, sp in zip(size, spacing)]
    return {
        f"{prefix}_shape_x": size[0],
        f"{prefix}_shape_y": size[1],
        f"{prefix}_shape_z": size[2],
        f"{prefix}_spacing_x": spacing[0],
        f"{prefix}_spacing_y": spacing[1],
        f"{prefix}_spacing_z": spacing[2],
        f"{prefix}_mm_x": mm[0],
        f"{prefix}_mm_y": mm[1],
        f"{prefix}_mm_z": mm[2],
        f"{prefix}_voxels": size[0] * size[1] * size[2],
        f"{prefix}_volume_mm3": mm[0] * mm[1] * mm[2],
    }


# Builds one row for a checkpoint: sample info + the mask/image measurements at that moment.
# `stage` tells at which pipeline step this measurement was taken.
def sample_metrics(ctx: SampleContext, stage: str) -> dict:
    row = {
        "stage": stage,
        "sample_id": ctx.sample_id,
        "modality": ctx.modality,
        "image_out_path": ctx.image_out_path,
        "mask_out_path": ctx.mask_out_path,
    }
    # row.update(_img_metrics(ctx.image, "image"))
    # row.update(_img_metrics(ctx.mask, "mask"))
    return row
