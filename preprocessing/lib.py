"""Preprocessing utilities."""

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
    return ok

if __name__ == "__main__":
    all_samples_loadable()