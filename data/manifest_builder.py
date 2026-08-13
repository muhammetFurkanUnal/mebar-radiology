#!/usr/bin/env python3
"""Build an EDA-ready NRRD manifest from the configured raw batches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path


# Add each newly delivered batch here; paths are relative to the repository root.
PART_DIRS = [Path("data/raw/batch-1"), Path("data/raw/batch-2")]
MODALITIES = ("ADC", "T2")
MASK_NAME = re.compile(r"segmentation|label|mask|(?:^|[-_])seg(?:[_.-]|$)", re.I)
HEADER_BYTES = 128 * 1024

COLUMNS = [
    "sample_id", "batch_id", "response_group", "patient_name", "patient_id", "modality",
    "patient_path", "modality_path", "nesting_depth", "image_path", "mask_path",
    "image_filename", "mask_filename", "nrrd_file_count", "other_file_count",
    "image_candidate_count", "mask_candidate_count",
    "image_dimension", "image_shape_x", "image_shape_y", "image_shape_z", "image_dtype",
    "image_encoding", "image_spacing_x_mm", "image_spacing_y_mm", "image_spacing_z_mm", "image_origin",
    "mask_dimension", "mask_shape_x", "mask_shape_y", "mask_shape_z", "mask_dtype",
    "mask_encoding", "mask_spacing_x_mm", "mask_spacing_y_mm", "mask_spacing_z_mm", "mask_origin",
    "shape_matches", "spacing_matches", "origin_matches", "is_valid_sample", "validation_errors",
    "manifest_created_at",
]


def rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def stable_id(*values: str) -> str:
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()[:16]


def normalized_group_name(name: str) -> str:
    """Normalize Turkish uppercase spelling such as İNTERMEDİATE."""
    normalized = unicodedata.normalize("NFD", name)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def header_value(header: str, field: str) -> str | None:
    match = re.search(rf"^{re.escape(field)}:\s*(.+)$", header, re.MULTILINE)
    return match.group(1).strip() if match else None


def vector_norm(text: str) -> float | None:
    values = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]
    return math.sqrt(sum(v * v for v in values)) if len(values) == 3 else None


def read_nrrd_header(path: Path) -> dict[str, object]:
    """Read NRRD header only; image voxels are never loaded into memory."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(HEADER_BYTES)
        header = raw.split(b"\n\n", 1)[0].decode("latin-1").replace("\r", "")
        if not header.startswith("NRRD"):
            return {"read_error": "not_a_nrrd_header"}
        sizes = (header_value(header, "sizes") or "").split()
        directions = re.findall(r"\([^)]*\)|none", header_value(header, "space directions") or "", re.I)
        spacing = [vector_norm(v) if v.lower() != "none" else None for v in directions]
        return {
            "dimension": header_value(header, "dimension"),
            "shape_x": int(sizes[0]) if len(sizes) > 0 else None,
            "shape_y": int(sizes[1]) if len(sizes) > 1 else None,
            "shape_z": int(sizes[2]) if len(sizes) > 2 else None,
            "dtype": header_value(header, "type"), "encoding": header_value(header, "encoding"),
            "spacing_x_mm": spacing[0] if len(spacing) > 0 else None,
            "spacing_y_mm": spacing[1] if len(spacing) > 1 else None,
            "spacing_z_mm": spacing[2] if len(spacing) > 2 else None,
            "origin": header_value(header, "space origin"), "read_error": None,
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        return {"read_error": f"header_read_error:{type(error).__name__}"}


def metadata(prefix: str, values: dict[str, object]) -> dict[str, object]:
    fields = ("dimension", "shape_x", "shape_y", "shape_z", "dtype", "encoding", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm", "origin")
    return {f"{prefix}_{field}": values.get(field) for field in fields}


def close(first: object, second: object, tolerance: float = 1e-6) -> bool | None:
    if first is None or second is None:
        return None
    return abs(float(first) - float(second)) <= tolerance


def modality_dirs(patient_dir: Path, modality: str) -> list[Path]:
    # This accepts accidental nesting such as PATIENT/PATIENT/ADC.
    return sorted(path for path in patient_dir.rglob(modality) if path.is_dir())


def build_row(root: Path, batch: Path, group: Path, patient: Path, modality: str, created_at: str) -> dict[str, object]:
    errors: list[str] = []
    folders = modality_dirs(patient, modality)
    folder = folders[0] if len(folders) == 1 else None
    if not folders:
        errors.append("missing_modality_directory")
    elif len(folders) > 1:
        errors.append(f"ambiguous_modality_directories:{len(folders)}")

    files = sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: p.name) if folder else []
    nrrds = [p for p in files if p.suffix.lower() == ".nrrd"]
    masks = [p for p in nrrds if MASK_NAME.search(p.name)]
    images = [p for p in nrrds if p not in masks]
    if len(nrrds) != 2:
        errors.append(f"nrrd_file_count:{len(nrrds)}")
    if len(files) != len(nrrds):
        errors.append(f"non_nrrd_file_count:{len(files) - len(nrrds)}")
    if len(images) != 1:
        errors.append(f"image_candidate_count:{len(images)}")
    if len(masks) != 1:
        errors.append(f"mask_candidate_count:{len(masks)}")

    image, mask = (images[0] if len(images) == 1 else None), (masks[0] if len(masks) == 1 else None)
    image_meta = read_nrrd_header(image) if image else {"read_error": None}
    mask_meta = read_nrrd_header(mask) if mask else {"read_error": None}
    if image_meta.get("read_error"):
        errors.append(f"image_{image_meta['read_error']}")
    if mask_meta.get("read_error"):
        errors.append(f"mask_{mask_meta['read_error']}")
    image_shape = tuple(image_meta.get(k) for k in ("shape_x", "shape_y", "shape_z"))
    mask_shape = tuple(mask_meta.get(k) for k in ("shape_x", "shape_y", "shape_z"))
    shape_matches = image_shape == mask_shape if image and mask and all(image_shape) and all(mask_shape) else None
    if shape_matches is False:
        errors.append("image_mask_shape_mismatch")
    checks = [close(image_meta.get(f"spacing_{axis}_mm"), mask_meta.get(f"spacing_{axis}_mm")) for axis in ("x", "y", "z")]
    spacing_matches = all(checks) if image and mask and all(c is not None for c in checks) else None
    if spacing_matches is False:
        errors.append("image_mask_spacing_mismatch")
    origin_matches = image_meta.get("origin") == mask_meta.get("origin") if image_meta.get("origin") and mask_meta.get("origin") else None
    if origin_matches is False:
        errors.append("image_mask_origin_mismatch")

    batch_id, response_group = batch.name, normalized_group_name(group.name)
    row: dict[str, object] = {
        "sample_id": stable_id(batch_id, response_group, patient.name, modality),
        "batch_id": batch_id, "response_group": response_group, "patient_name": patient.name,
        "patient_id": stable_id(batch_id, response_group, patient.name), "modality": modality,
        "patient_path": rel(patient, root), "modality_path": rel(folder, root) if folder else None,
        "nesting_depth": len(folder.relative_to(patient).parts) - 1 if folder else None,
        "image_path": rel(image, root) if image else None, "mask_path": rel(mask, root) if mask else None,
        "image_filename": image.name if image else None, "mask_filename": mask.name if mask else None,
        "nrrd_file_count": len(nrrds), "other_file_count": len(files) - len(nrrds),
        "image_candidate_count": len(images), "mask_candidate_count": len(masks),
        "shape_matches": shape_matches, "spacing_matches": spacing_matches, "origin_matches": origin_matches,
        "is_valid_sample": not errors, "validation_errors": ";".join(errors), "manifest_created_at": created_at,
    }
    row.update(metadata("image", image_meta)); row.update(metadata("mask", mask_meta))
    return row


def build_manifest(root: Path) -> list[dict[str, object]]:
    created_at = datetime.now(UTC).isoformat()
    rows: list[dict[str, object]] = []
    for configured in PART_DIRS:
        batch = configured if configured.is_absolute() else root / configured
        if not batch.is_dir():
            raise FileNotFoundError(f"Batch directory not found: {batch}")
        for group in sorted(p for p in batch.iterdir() if p.is_dir()):
            for patient in sorted(p for p in group.iterdir() if p.is_dir()):
                for modality in MODALITIES:
                    rows.append(build_row(root, batch, group, patient, modality, created_at))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an EDA-ready NRRD manifest.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/manifest.csv"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = build_manifest(root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="raise")
        writer.writeheader(); writer.writerows(rows)
    invalid = [row for row in rows if not row["is_valid_sample"]]
    print(f"Wrote {len(rows)} modality rows for {len(rows) // 2} patients to {output}")
    print(f"Valid rows: {len(rows) - len(invalid)} | Rows with validation findings: {len(invalid)}")
    for error, count in sorted(Counter(row["validation_errors"] for row in invalid).items()):
        print(f"  {count}x {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
