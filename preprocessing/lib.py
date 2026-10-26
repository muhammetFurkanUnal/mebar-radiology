from __future__ import annotations
from pathlib import Path
import SimpleITK as sitk
import numpy as np


def all_samples_loadable(root: str | Path = "data/raw") -> bool:
    """Recursively check that every .nrrd file under `root` opens correctly.

    Files that fail to open are printed to the terminal. If everything loads,
    nothing is printed and True is returned; otherwise False.
    """
    ok = True
    for path in Path(root).rglob("*.nrrd"):
        try:
            img = sitk.ReadImage(str(path))
            sitk.GetArrayViewFromImage(img)  # force access to pixel data
        except Exception as exc:
            print(f"[FAILED] {path}: {type(exc).__name__}: {exc}")
            ok = False
    print("all sample loadable.")
    return ok

def same_grid(a, b, tol=1e-5):
        import numpy as np
        return (a.GetSize() == b.GetSize()
                and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=tol)
                and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=tol)
                and np.allclose(a.GetDirection(), b.GetDirection(), atol=tol))


def foreground_mask(img: sitk.Image) -> sitk.Image:
    """Binary tissue mask (1 = tissue, 0 = background), same grid as img."""
    img_f = sitk.Cast(img, sitk.sitkFloat32)

    # Otsu: below threshold -> 0 (background), above -> 1 (tissue)
    mask = sitk.OtsuThreshold(img_f, 0, 1, 200)

    # keep only the largest connected component (removes noise specks)
    cc = sitk.ConnectedComponent(mask)
    cc = sitk.RelabelComponent(cc, sortByObjectSize=True)
    mask = cc == 1

    # close small gaps and fill internal holes (dark tissue, fluid, etc.)
    mask = sitk.BinaryMorphologicalClosing(mask, [3] * img.GetDimension())
    mask = sitk.BinaryFillhole(mask)

    return sitk.Cast(mask, sitk.sitkUInt8)


def resample_image(
    image_image:sitk.Image,
    target_spacing,
    is_mask,
    out_path=None
    ) -> sitk.Image:
    original_spacing = image_image.GetSpacing()
    original_size = image_image.GetSize()
    
    # new tensor dimensions
    new_size = [
        int(round(org_spc * org_size / target_spc))
        for org_spc, org_size, target_spc in zip(original_spacing, original_size, target_spacing)
    ]
    
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputDirection(image_image.GetDirection())
    resample_filter.SetOutputOrigin(image_image.GetOrigin())
    # if some voxels goes above boundary, we set a default value for erroneous voxels
    resample_filter.SetDefaultPixelValue(0) 
    # identity transformation, we gotta give a type of transformation but we actually don't need, so identity
    resample_filter.SetTransform(sitk.Transform())
    if is_mask:
        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample_filter.SetInterpolator(sitk.sitkBSpline)


    # now apply the filter we built
    resampled_image = resample_filter.Execute(image_image)

    if out_path:
        sitk.WriteImage(resampled_image, out_path)

    return resampled_image


def crop_ROI(
    mask_image:sitk.Image,
    image_image:sitk.Image,
    mm_dims,
    out_path=None
    ):

    # method receives one mask and one image
    # crops the tumor area in the volume of given mm dimensions, centering the tumor
    #
    # args:
    # out_path is a list
    # first index is mask save path
    # second index is image save path
    # returns a list
    # first index is mask sitk.Image
    # second index is image sitk.Image

    if not same_grid(mask_image, image_image):
        raise ValueError("[crop] mask and image are not in the same grid")

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_image)
    bbox = stats.GetBoundingBox(1)
    x0, y0, z0, x_len, y_len, z_len = bbox
    x_cen, y_cen, z_cen = x0 + x_len/2, y0+y_len/2, z0+z_len/2

    vox_dims = [
        int(round(dim/spacing))
        for spacing, dim in zip(image_image.GetSpacing(), mm_dims)
    ]

    vox_dims = [
        min(size, vox) for size, vox in zip(image_image.GetSize(), vox_dims)
    ]

    # clamping - limiting upper and lower bound to be inside of the image fov
    x_size, y_size, z_size = image_image.GetSize()
    x0 = int(max(0, min(x_size-vox_dims[0], x_cen-vox_dims[0]/2)))
    y0 = int(max(0, min(y_size-vox_dims[1], y_cen-vox_dims[1]/2)))
    z0 = int(max(0, min(z_size-vox_dims[2], z_cen-vox_dims[2]/2)))

    ROI_mask = sitk.RegionOfInterest(mask_image, (vox_dims[0], vox_dims[1], vox_dims[2]), (x0, y0, z0))
    ROI_image = sitk.RegionOfInterest(image_image, (vox_dims[0], vox_dims[1], vox_dims[2]), (x0, y0, z0))

    if out_path:
        sitk.WriteImage(ROI_mask, out_path[0])
        sitk.WriteImage(ROI_image, out_path[1])

    return ROI_mask, ROI_image


def crop_area_highlight(
    mask_image:sitk.Image,
    image_image:sitk.Image,
    mm_dims,
    out_path=None
    ):

    if not same_grid(mask_image, image_image):
        raise ValueError("[crop area highlight] mask and image are not in the same grid")
    

    # calculate tumor centered bbox starts and dims 
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_image)
    bbox = stats.GetBoundingBox(1)
    x0, y0, z0, x_len, y_len, z_len = bbox
    x_cen, y_cen, z_cen = x0+x_len/2, y0+y_len/2, z0+z_len/2

    vox_dims = [
        int(round(dim/spacing))
        for spacing, dim in zip(image_image.GetSpacing(), mm_dims)
    ]
    vox_dims = [
        min(size, vox) for size, vox in zip(image_image.GetSize(), vox_dims)
    ]
    
    # clamping - limiting upper and lower bound to be inside of the image fov
    x_size, y_size, z_size = image_image.GetSize()
    x0 = int(max(0, min(x_size-vox_dims[0], x_cen-vox_dims[0]/2)))
    y0 = int(max(0, min(y_size-vox_dims[1], y_cen-vox_dims[1]/2)))
    z0 = int(max(0, min(z_size-vox_dims[2], z_cen-vox_dims[2]/2)))

    # create crop area mask
    bbox_mask = sitk.Image(image_image.GetSize(), sitk.sitkUInt8)
    bbox_mask.CopyInformation(image_image)

    # fill the mask
    bbox_mask[x0:x0 + vox_dims[0], y0:y0 + vox_dims[1], z0:z0 + vox_dims[2]] = 1
    return bbox_mask, image_image


def pad(
    mask_image: sitk.Image,
    image_image: sitk.Image,
    target_size,
    out_path=None,
):
    """Center both images inside a black background of `target_size`.

    Each image is padded symmetrically with zeros (black) so it sits in the
    middle of the output volume. If any dimension of `target_size` is smaller
    than the corresponding image size, a ValueError is raised.
    `out_path` is expected to be a [mask_path, image_path] pair, matching
    `crop_ROI`; when given, the padded images are written to those paths.
    """

    if not same_grid(mask_image, image_image):
        raise ValueError("[pad] mask and image are not in the same grid")

    target_size = [int(s) for s in target_size]

    for image in (mask_image, image_image):
        if any(t < s for t, s in zip(target_size, image.GetSize())):
            raise ValueError(
                f"target_size {tuple(target_size)} is smaller than image size {tuple(image.GetSize())}"
            )

    padded = []
    for image in (mask_image, image_image):
        size = image.GetSize()
        pad_lower = [(t - s) // 2 for t, s in zip(target_size, size)]
        pad_upper = [(t - s) - pl for t, s, pl in zip(target_size, size, pad_lower)]
        padded.append(sitk.ConstantPad(image, pad_lower, pad_upper, 0))

    if out_path:
        sitk.WriteImage(padded[0], out_path[0])
        sitk.WriteImage(padded[1], out_path[1])

    return padded[0], padded[1]


def bias_field_correction(
    image_image:sitk.Image,
    shrink_factor,
    out_path=None
    ):

    f_image = sitk.Cast(image_image, sitk.sitkFloat32)

    # foreground mask
    fg_mask = sitk.OtsuThreshold(f_image, 0, 1, 200)

    shrinked_image = sitk.Shrink(f_image, [shrink_factor]*f_image.GetDimension())
    shrinked_mask = sitk.Shrink(fg_mask, [shrink_factor]*f_image.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.Execute(shrinked_image, shrinked_mask)

    log_bias_field = corrector.GetLogBiasFieldAsImage(f_image)
    corrected = f_image / sitk.Exp(log_bias_field)

    if out_path:
        sitk.WriteImage(corrected, out_path)

    return corrected


def zscore_norm(
    image_image:sitk.Image,
    out_path=None
    ):
    f_image:sitk.Image = sitk.Cast(image_image, sitk.sitkFloat32)
    f_array = sitk.GetArrayFromImage(f_image)

    fg_mask = foreground_mask(image_image)
    mask_array = sitk.GetArrayViewFromImage(fg_mask).astype(bool)

    mean, std = f_array[mask_array].mean(), f_array[mask_array].std()
    f_array = (f_array - mean) / (std + 1e-8)

    normed = sitk.GetImageFromArray(f_array)
    normed.CopyInformation(image_image)

    if out_path:
        sitk.WriteImage(normed, out_path)

    return normed


def apply_zscore_norm_adc(
    out_path
    ):
    pass


def T2_intensity_threshold(
    image_image: sitk.Image,
    lower: float,
    upper: float,
    out_path=None,
) -> sitk.Image:
    """Clip intensities to the `lower`/`upper` percentiles of the foreground.

    `lower` and `upper` are percent values in [0, 100] (e.g. 0.05 -> 0.05th
    percentile, 99.5 -> 99.5th percentile). Values below the lower percentile
    and above the upper percentile are clamped, so extreme intensity outliers
    are removed before downstream normalization.
    """
    f_image = sitk.Cast(image_image, sitk.sitkFloat32)
    f_array = sitk.GetArrayFromImage(f_image)

    fg_mask = foreground_mask(image_image)
    mask_array = sitk.GetArrayViewFromImage(fg_mask).astype(bool)
    foreground = f_array[mask_array]

    low = np.percentile(foreground, lower)
    high = np.percentile(foreground, upper)

    f_array[mask_array] = np.clip(f_array[mask_array], low, high)

    clipped = sitk.GetImageFromArray(f_array)
    clipped.CopyInformation(image_image)

    if out_path:
        sitk.WriteImage(clipped, out_path)

    return clipped


def ADC_intensity_threshold(
    image_image: sitk.Image,
    lower: float,
    upper: float,
    out_path=None,
) -> sitk.Image:
    """Clamp intensities to the absolute `[lower, upper]` range.

    Unlike `T2_intensity_threshold` (which uses percentiles), `lower` and
    `upper` here are raw intensity bounds (e.g. 0 and 3000).
    """
    f_image = sitk.Cast(image_image, sitk.sitkFloat32)
    f_array = sitk.GetArrayFromImage(f_image)

    f_array = np.clip(f_array, lower, upper)

    clipped = sitk.GetImageFromArray(f_array)
    clipped.CopyInformation(image_image)

    if out_path:
        sitk.WriteImage(clipped, out_path)

    return clipped


def handle_adc_scaling_issue(
    image_image:sitk.Image,
    scale,
    out_path=None
    ):
    # first handle small-large scaled imagings difference
    # second crop out of possible bound values [0, 3000]
    pass



if __name__ == "__main__":
    all_samples_loadable()


