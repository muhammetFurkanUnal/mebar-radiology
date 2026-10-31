import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd, EnsureTyped,
    RandFlipd, RandAffined,
    RandScaleIntensityd, RandShiftIntensityd, RandAdjustContrastd,
    RandGaussianNoised, RandGaussianSmoothd, RandBiasFieldd
)
from config import config

cfg = config.transform

train_transform = Compose([
    EnsureChannelFirstd(keys=["image", "mask"], channel_dim=0),
    EnsureTyped(keys=["image", "mask"], dtype=torch.float32),
    RandFlipd(keys=["image", "mask"], prob=cfg.rnd_h_flip_p, spatial_axis=1),
    RandAffined(
        keys=["image", "mask"],
        prob=cfg.rnd_aff_p,
        rotate_range=np.deg2rad(cfg.rnd_rot_deg),
        scale_range=(cfg.rnd_down_scale, cfg.rnd_up_scale),
        translate_range=(cfg.translate_down, cfg.translate_up),
        mode=("bilinear", "nearest"),
        padding_mode=("border", "zeros"),
    ),
    RandScaleIntensityd(keys=["image"], factors=cfg.intensity_scale_factor, prob=cfg.intensity_scale_p),
    RandShiftIntensityd(keys=["image"], offsets=cfg.shift_intensity_offsets, prob=cfg.shift_intensity_p),
    RandAdjustContrastd(keys=["image"], gamma=cfg.adjust_contrast_gamma, retain_stats=True, prob=cfg.adjust_contrast_p),
    RandGaussianNoised(keys=["image"], mean=cfg.gaussian_noise_mean, std=cfg.gaussian_noise_std, prob=cfg.gaussian_noise_p),
    RandGaussianSmoothd(keys=["image"], sigma_x=cfg.gaussian_smooth_sigma_x, sigma_y=cfg.gaussian_smooth_sigma_y, prob=cfg.gaussian_smooth_p),
    RandBiasFieldd(keys=["image"], coeff_range=cfg.bias_field_coeff_range, degree=cfg.bias_field_degree, prob=cfg.bias_field_p),
])

test_transform = Compose([
    EnsureChannelFirstd(keys=["image", "mask"], channel_dim=0),
    EnsureTyped(keys=["image", "mask"], dtype=torch.float32),
])
