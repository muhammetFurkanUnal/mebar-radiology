from __future__ import annotations
from pathlib import Path
import SimpleITK as sitk


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
    resample_filter.SetDefaultPixelValue(image_image.GetPixelIDValue()) 
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

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_image)
    bbox = stats.GetBoundingBox(1)
    x0, y0, z0, x_len, y_len, z_len = bbox

    x_cen, y_cen, z_cen = x0 + x_len/2, y0+y_len/2, z0+z_len/2

    vox_dims = [
        int(round(dim/spacing))
        for spacing, dim in zip(image_image.GetSpacing(), mm_dims)
    ]

    x0, y0, z0 = int(round(x_cen-vox_dims[0]/2)), int(round(y_cen-vox_dims[1]/2)), int(round(z_cen-vox_dims[2]/2))

    # debugging
    print("original image size: ", image_image.GetSize())
    print("voxel dimensions: ", vox_dims)
    print("center of the tumor: ", x_cen, y_cen, z_cen)
    print("starting point of the crop: ", x0, y0, z0)
    print("spacing of the image: ", image_image.GetSpacing())

    ROI_mask = sitk.RegionOfInterest(mask_image, (vox_dims[0], vox_dims[1], vox_dims[2]), (x0, y0, z0))
    ROI_image = sitk.RegionOfInterest(image_image, (vox_dims[0], vox_dims[1], vox_dims[2]), (x0, y0, z0))

    if out_path:
        sitk.WriteImage(ROI_mask, out_path[0])
        sitk.WriteImage(ROI_image, out_path[1])


    return ROI_mask, ROI_image


def apply_bias_field_correction(
    out_path
    ):
    pass


def apply_zscore_norm_t2(
    out_path
    ):
    pass


def apply_zscore_norm_adc(
    out_path
    ):
    pass


def intensity_threshold(
    lower,
    upper,
    out_path
    ):
    pass


def handle_adc_scaling_issue(
    out_path
    ):
    # first handle small-large scaled imagings difference
    # second crop out of possible bound values [0, 3000]
    pass



if __name__ == "__main__":
    all_samples_loadable()


