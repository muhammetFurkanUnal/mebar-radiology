#!/usr/bin/env python3
"""Build a manifest for the processed image/mask pairs of a configuration."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parents[2]
DEFAULT_IMAGES = CONFIG_DIR / "images"
DEFAULT_OUTPUT = CONFIG_DIR / "images_manifest.csv"
DEFAULT_SOURCE_MANIFEST = REPO_ROOT / "data" / "manifest.csv"

MODALITIES = ("ADC", "T2")
MASK_SUFFIX = "_mask"
HEADER_BYTES = 128 * 1024
ORIGIN_TOLERANCE = 1e-5
SOURCE_FIELDS = ("batch_id", "response_group", "patient_name", "patient_id")

METADATA_FIELDS = (
    "dimension", "shape_x", "shape_y", "shape_z", "dtype", "encoding",
    "spacing_x_mm", "spacing_y_mm", "spacing_z_mm", "origin", "space",
)

COLUMNS = (
    ["sample_id", "modality", *SOURCE_FIELDS, "image_path", "mask_path", "image_filename", "mask_filename"]
    + [f"image_{field}" for field in METADATA_FIELDS]
    + [f"mask_{field}" for field in METADATA_FIELDS]
    + ["shape_matches", "spacing_matches", "origin_matches", "is_valid_sample", "validation_errors", "manifest_created_at"]
)


def rel(path: Path, root: Path) -> str:
    """Return a path relative to the repository root when possible."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def numbers(text: str | None) -> list[float]:
    """Extract all numbers from a header value."""
    return [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text or "")]


def vector_norm(text: str) -> float | None:
    values = numbers(text)
    return math.sqrt(sum(value * value for value in values)) if len(values) == 3 else None


def header_value(header: str, field: str) -> str | None:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", header, re.MULTILINE)
    return match.group(1).strip() if match else None


def read_nrrd_header(path: Path) -> dict[str, object]:
    """Read only the NRRD header; voxels are never loaded."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(HEADER_BYTES)
        header = raw.split(b"\n\n", 1)[0].decode("latin-1").replace("\r", "")
        if not header.startswith("NRRD"):
            return {"read_error": "not_a_nrrd_header"}
        sizes = (header_value(header, "sizes") or "").split()
        directions = re.findall(r"\([^)]*\)|none", header_value(header, "space directions") or "", re.I)
        spacing = [vector_norm(value) if value.lower() != "none" else None for value in directions]
        return {
            "dimension": header_value(header, "dimension"),
            "shape_x": int(sizes[0]) if len(sizes) > 0 else None,
            "shape_y": int(sizes[1]) if len(sizes) > 1 else None,
            "shape_z": int(sizes[2]) if len(sizes) > 2 else None,
            "dtype": header_value(header, "type"),
            "encoding": header_value(header, "encoding"),
            "spacing_x_mm": spacing[0] if len(spacing) > 0 else None,
            "spacing_y_mm": spacing[1] if len(spacing) > 1 else None,
            "spacing_z_mm": spacing[2] if len(spacing) > 2 else None,
            "origin": header_value(header, "space origin"),
            "space": header_value(header, "space"),
            "read_error": None,
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return {"read_error": f"header_read_error:{type(error).__name__}"}


def metadata(prefix: str, values: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{field}": values.get(field) for field in METADATA_FIELDS}


def close(first: object, second: object, tolerance: float = 1e-6) -> bool | None:
    if first is None or second is None:
        return None
    return abs(float(first) - float(second)) <= tolerance


def same_vector(first: object, second: object, tolerance: float = ORIGIN_TOLERANCE) -> bool | None:
    if first is None or second is None:
        return None
    left, right = numbers(str(first)), numbers(str(second))
    if len(left) != 3 or len(right) != 3:
        return None
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def split_filename(path: Path) -> tuple[str | None, str | None]:
    """Return the sample id and the file kind (modality or 'mask')."""
    if path.suffix.lower() != ".nrrd":
        return None, None
    stem = path.stem
    if stem.endswith(MASK_SUFFIX):
        return stem[: -len(MASK_SUFFIX)], "mask"
    for modality in MODALITIES:
        suffix = f"_{modality}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)], modality
    return None, None


def load_source(path: Path) -> dict[str, dict[str, str]]:
    """Load the raw manifest so sample ids can carry their labels."""
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle)}


def build_row(
    root: Path,
    sample_id: str,
    modality: str,
    image: Path,
    mask: Path | None,
    source: dict[str, str] | None,
    created_at: str,
) -> dict[str, object]:
    errors: list[str] = []
    if mask is None:
        errors.append("missing_mask_file")

    image_meta = read_nrrd_header(image)
    mask_meta = read_nrrd_header(mask) if mask else {"read_error": None}
    if image_meta.get("read_error"):
        errors.append(f"image_{image_meta['read_error']}")
    if mask_meta.get("read_error"):
        errors.append(f"mask_{mask_meta['read_error']}")

    image_shape = tuple(image_meta.get(key) for key in ("shape_x", "shape_y", "shape_z"))
    mask_shape = tuple(mask_meta.get(key) for key in ("shape_x", "shape_y", "shape_z"))
    shape_matches = image_shape == mask_shape if mask and all(image_shape) and all(mask_shape) else None
    if shape_matches is False:
        errors.append("image_mask_shape_mismatch")

    checks = [close(image_meta.get(f"spacing_{axis}_mm"), mask_meta.get(f"spacing_{axis}_mm")) for axis in ("x", "y", "z")]
    spacing_matches = all(checks) if mask and all(check is not None for check in checks) else None
    if spacing_matches is False:
        errors.append("image_mask_spacing_mismatch")

    origin_matches = same_vector(image_meta.get("origin"), mask_meta.get("origin")) if mask else None
    if origin_matches is False:
        errors.append("image_mask_origin_mismatch")

    source = source or {}
    row: dict[str, object] = {
        "sample_id": sample_id,
        "modality": modality,
        **{field: source.get(field) for field in SOURCE_FIELDS},
        "image_path": rel(image, root),
        "mask_path": rel(mask, root) if mask else None,
        "image_filename": image.name,
        "mask_filename": mask.name if mask else None,
        "shape_matches": shape_matches,
        "spacing_matches": spacing_matches,
        "origin_matches": origin_matches,
        "is_valid_sample": not errors,
        "validation_errors": ";".join(errors),
        "manifest_created_at": created_at,
    }
    row.update(metadata("image", image_meta))
    row.update(metadata("mask", mask_meta))
    return row


def build_manifest(images_dir: Path, root: Path, source: dict[str, dict[str, str]]):
    """Pair processed images with their masks and validate each pair."""
    images: dict[tuple[str, str], Path] = {}
    masks: dict[str, Path] = {}
    unknown: list[str] = []

    for path in sorted(images_dir.iterdir()):
        if not path.is_file():
            continue
        sample_id, kind = split_filename(path)
        if kind == "mask":
            masks[sample_id] = path
        elif kind in MODALITIES:
            images[(sample_id, kind)] = path
        elif path.suffix.lower() == ".nrrd":
            unknown.append(path.name)

    created_at = datetime.now(UTC).isoformat()
    rows = [
        build_row(root, sample_id, modality, image, masks.get(sample_id), source.get(sample_id), created_at)
        for (sample_id, modality), image in sorted(images.items())
    ]
    orphan_masks = sorted(set(masks) - {sample_id for sample_id, _ in images})
    return rows, unknown, orphan_masks


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a manifest for processed image/mask pairs.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    args = parser.parse_args()

    images_dir = args.images if args.images.is_absolute() else Path.cwd() / args.images
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    source = load_source(args.source_manifest)
    rows, unknown, orphan_masks = build_manifest(images_dir, REPO_ROOT, source)

    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    invalid = [row for row in rows if not row["is_valid_sample"]]
    print(f"Wrote {len(rows)} sample rows to {output}")
    print(f"Valid rows: {len(rows) - len(invalid)} | Rows with validation findings: {len(invalid)}")
    for error, count in sorted(Counter(row["validation_errors"] for row in invalid).items()):
        print(f"  {count}x {error}")
    if orphan_masks:
        print(f"Masks without an image: {len(orphan_masks)} ({', '.join(orphan_masks[:5])})")
    if unknown:
        print(f"Unrecognized nrrd files: {len(unknown)} ({', '.join(unknown[:5])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
