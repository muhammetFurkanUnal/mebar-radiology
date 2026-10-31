"""Slice 3D medical volumes into 2D images and build a slice manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk


AXIS = "axial"
LABEL_COLUMN = "response_group"
REQUIRED_COLUMNS = ("sample_id", "image_path", "mask_path", LABEL_COLUMN)
SOURCE_SHAPE_COLUMNS = (
    "sample_id",
    "image_path",
    "mask_path",
    "image_shape_z",
    "image_shape_y",
    "image_shape_x",
    LABEL_COLUMN,
)
SLICE_MANIFEST_COLUMNS = (
    "sample_id",
    LABEL_COLUMN,
    "axis",
    "slice_count",
    "slice_position",
    "has_tumor",
    "image_path",
    "mask_path",
)


def _repo_root(manifest_path: Path) -> Path:
    # The manifest lives under <repo>/preprocessing/configurations/<name>/.
    return manifest_path.resolve().parents[3]


def _resolve_source_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return _repo_root(manifest_path) / path


def _read_pair(manifest_path: Path, sample_id: str) -> tuple[Path, Path]:
    # Returns the image and mask paths for one sample from the manifest.
    manifest = pd.read_csv(manifest_path)
    rows = manifest[manifest["sample_id"] == sample_id]

    if len(rows) == 0:
        raise ValueError(f"sample_id not found in manifest: {sample_id}")
    if len(rows) > 1:
        raise ValueError(f"sample_id is not unique in manifest: {sample_id}")

    row = rows.iloc[0]
    image_path = _resolve_source_path(manifest_path, row["image_path"])
    mask_path = _resolve_source_path(manifest_path, row["mask_path"])

    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"mask not found: {mask_path}")

    return image_path, mask_path


def slice_volume(manifest_path: str | Path, sample_id: str) -> dict[str, list[np.ndarray]]:
    """Turns one 3D volume into axial 2D image and mask slices."""
    manifest_path = Path(manifest_path).resolve()

    image_path, mask_path = _read_pair(manifest_path, sample_id)
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

    if image.ndim != 3:
        raise ValueError(f"expected a 3D image, got shape {image.shape}")
    if image.shape != mask.shape:
        raise ValueError(f"image/mask shape mismatch: {image.shape} vs {mask.shape}")

    if image.shape[0] == 0:
        raise ValueError("volume contains no slices")

    return {
        "images": [image[index] for index in range(image.shape[0])],
        "masks": [mask[index] for index in range(mask.shape[0])],
    }


def build_2d_dataset(config_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    """Slices every volume in a configuration folder and writes one slice manifest."""
    config_dir = Path(config_dir).resolve()
    manifest_path = config_dir / "images_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    output_dir = Path(output_dir).resolve() if output_dir else config_dir / "images_2d"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError(f"manifest is missing columns: {missing}")

    sample_ids = manifest["sample_id"].astype(str).tolist()
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("source manifest contains duplicate sample_id rows")

    response_groups = dict(zip(sample_ids, manifest[LABEL_COLUMN]))

    all_rows: list[dict] = []
    seen: set[str] = set()

    for sample_id in sample_ids:
        if sample_id in seen:
            raise ValueError(f"sample processed more than once: {sample_id}")
        seen.add(sample_id)

        slices = slice_volume(manifest_path, sample_id)
        images = slices["images"]
        masks = slices["masks"]
        slice_count = len(images)

        if len(masks) != slice_count:
            raise RuntimeError(
                f"image/mask slice count mismatch for {sample_id}: {len(masks)} != {slice_count}"
            )

        slice_dir = output_dir / sample_id / AXIS
        slice_dir.mkdir(parents=True, exist_ok=True)

        for index, (image_slice, mask_slice) in enumerate(zip(images, masks)):
            image_filename = f"{sample_id}_{AXIS}_{index + 1:03d}.npy"
            mask_filename = f"{sample_id}_{AXIS}_{index + 1:03d}_mask.npy"
            image_path = slice_dir / image_filename
            mask_path = slice_dir / mask_filename
            np.save(image_path, image_slice)
            np.save(mask_path, mask_slice)

            all_rows.append(
                {
                    "sample_id": sample_id,
                    LABEL_COLUMN: response_groups[sample_id],
                    "axis": AXIS,
                    "slice_index (1-first)": index + 1,
                    "slice_count": slice_count,
                    "slice_position": f"{index + 1}/{slice_count}",
                    "has_tumor": bool(mask_slice.any()),
                    "image_path": image_path.relative_to(output_dir).as_posix(),
                    "mask_path": mask_path.relative_to(output_dir).as_posix(),
                }
            )

        # Make sure every image and mask slice was actually written for this sample.
        for index in range(slice_count):
            expected_image = slice_dir / f"{sample_id}_{AXIS}_{index + 1:03d}.npy"
            expected_mask = slice_dir / f"{sample_id}_{AXIS}_{index + 1:03d}_mask.npy"
            if not expected_image.is_file():
                raise RuntimeError(f"slice image missing after write: {expected_image}")
            if not expected_mask.is_file():
                raise RuntimeError(f"slice mask missing after write: {expected_mask}")

    slice_manifest_path = output_dir / "manifest.csv"
    pd.DataFrame(all_rows).to_csv(slice_manifest_path, index=False)
    return slice_manifest_path


def verify_2d_dataset(source_manifest_path: str | Path, slice_manifest_path: str | Path) -> list[str]:
    """Checks a built 2D slice dataset against its source 3D manifest."""
    source = pd.read_csv(source_manifest_path)
    slices = pd.read_csv(slice_manifest_path)
    errors: list[str] = []

    missing_source = [column for column in SOURCE_SHAPE_COLUMNS if column not in source.columns]
    missing_slices = [column for column in SLICE_MANIFEST_COLUMNS if column not in slices.columns]
    if missing_source:
        return [f"source manifest missing columns: {missing_source}"]
    if missing_slices:
        return [f"slice manifest missing columns: {missing_slices}"]

    source["sample_id"] = source["sample_id"].astype(str)
    slices["sample_id"] = slices["sample_id"].astype(str)

    try:
        source["image_shape_z"] = source["image_shape_z"].astype(int)
        slices["slice_count"] = slices["slice_count"].astype(int)
    except (ValueError, TypeError) as error:
        return [f"slice count columns are not integer: {error}"]

    if source["sample_id"].duplicated().any():
        errors.append("source manifest has duplicate sample_id rows")
    if slices.duplicated(subset=["sample_id", "slice_position"]).any():
        errors.append("slice manifest has duplicate (sample_id, slice_position) rows")

    source_ids = set(source["sample_id"])
    slice_ids = set(slices["sample_id"])
    if source_ids != slice_ids:
        errors.append(
            f"sample id mismatch: missing={sorted(source_ids - slice_ids)[:5]}, "
            f"extra={sorted(slice_ids - source_ids)[:5]}"
        )

    if (slices["axis"] != AXIS).any():
        errors.append(f"slice manifest contains axes other than {AXIS}")

    source_labels = source.drop_duplicates(subset="sample_id").set_index("sample_id")[LABEL_COLUMN]
    for sample_id, group in slices.groupby("sample_id"):
        expected_label = source_labels.get(sample_id)
        if group[LABEL_COLUMN].isna().any():
            errors.append(f"missing {LABEL_COLUMN} for {sample_id}")
        elif (group[LABEL_COLUMN] != expected_label).any():
            errors.append(f"{LABEL_COLUMN} mismatch for {sample_id}")

    expected_total = int(source["image_shape_z"].sum())
    if len(slices) != expected_total:
        errors.append(
            f"total slice count mismatch: manifest rows={len(slices)}, source z sum={expected_total}"
        )

    slice_counts = slices.groupby("sample_id").size()
    for sample_id, z in source[["sample_id", "image_shape_z"]].itertuples(index=False):
        if slice_counts.get(sample_id, 0) != z:
            errors.append(
                f"slice count mismatch for {sample_id}: manifest={slice_counts.get(sample_id, 0)}, source z={z}"
            )
        sample_rows = slices[slices["sample_id"] == sample_id]
        if not (sample_rows["slice_count"] == z).all():
            errors.append(f"slice_count column mismatch for {sample_id}")

    base = Path(slice_manifest_path).resolve().parent

    for _, row in slices.iterrows():
        image_file = base / row["image_path"]
        mask_file = base / row["mask_path"]

        if not image_file.is_file():
            errors.append(f"image file missing: {row['image_path']}")
        if not mask_file.is_file():
            errors.append(f"mask file missing: {row['mask_path']}")
            continue

        try:
            mask = np.load(mask_file, mmap_mode="r")
        except (OSError, ValueError) as error:
            errors.append(f"failed to read mask: {row['mask_path']} ({error})")
            continue

        if not np.isin(mask, [0, 1]).all():
            errors.append(f"mask is not binary: {row['mask_path']}")
        if bool(mask.any()) != bool(row["has_tumor"]):
            errors.append(f"has_tumor mismatch: {row['mask_path']}")

    for sample_id, y, x in source[["sample_id", "image_shape_y", "image_shape_x"]].itertuples(index=False):
        sample_rows = slices[slices["sample_id"] == sample_id]
        if sample_rows.empty:
            continue

        row = sample_rows.iloc[0]
        image_file = base / row["image_path"]
        mask_file = base / row["mask_path"]
        if not image_file.is_file() or not mask_file.is_file():
            continue

        try:
            image = np.load(image_file, mmap_mode="r")
            mask = np.load(mask_file, mmap_mode="r")
        except (OSError, ValueError) as error:
            errors.append(f"failed to read first slice for {sample_id}: {error}")
            continue

        expected_shape = (int(y), int(x))
        if image.shape != expected_shape:
            errors.append(f"image slice shape mismatch for {sample_id}: {image.shape} != {expected_shape}")
        if mask.shape != image.shape:
            errors.append(f"image/mask slice shape mismatch for {sample_id}: {image.shape} vs {mask.shape}")

    referenced = {
        str((base / row[column]).resolve())
        for _, row in slices.iterrows()
        for column in ("image_path", "mask_path")
    }
    actual = {str(path.resolve()) for path in base.rglob("*.npy")}
    orphans = actual - referenced
    if orphans:
        errors.append(f"{len(orphans)} unreferenced .npy files found")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a 2D slice dataset from a preprocessed images manifest.")
    parser.add_argument("manifest_path", help="Path to the preprocessed images manifest CSV.")
    parser.add_argument("output_dir", nargs="?", default=None, help="Optional output directory for slices and manifest.")
    parser.add_argument("--verify", action="store_true", help="Verify an existing slice dataset instead of building.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path).resolve()
    if not manifest_path.is_file():
        parser.error(f"manifest not found: {manifest_path}")

    config_dir = manifest_path.parent
    output_dir = Path(args.output_dir).resolve() if args.output_dir else config_dir / "images_2d"

    if args.verify:
        slice_manifest_path = output_dir / "manifest.csv"
        if not slice_manifest_path.is_file():
            parser.error(f"slice manifest not found: {slice_manifest_path}")

        errors = verify_2d_dataset(manifest_path, slice_manifest_path)
        if errors:
            print("verification failed:")
            for error in errors:
                print(f"  - {error}")
            return 1

        print("verification passed")
        return 0

    slice_manifest_path = build_2d_dataset(config_dir, output_dir)
    print(f"Wrote slice manifest: {slice_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

