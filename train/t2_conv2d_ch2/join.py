import argparse
from pathlib import Path
import pandas as pd


def join(results_csv_path: str | Path, manifest_csv_path: str | Path, output_csv_path: str | Path) -> pd.DataFrame:
    results_csv_path = Path(results_csv_path)
    manifest_csv_path = Path(manifest_csv_path)
    output_csv_path = Path(output_csv_path)

    results = pd.read_csv(results_csv_path)
    manifest = pd.read_csv(manifest_csv_path)

    results["sample_id"] = results["sample_id"].astype(str)
    manifest["sample_id"] = manifest["sample_id"].astype(str)

    # Merge slice scores with manifest tumor indicator
    merged = results.merge(
        manifest[["sample_id", "slice_index (1-first)", "has_tumor"]],
        left_on=["sample_id", "slice_idx"],
        right_on=["sample_id", "slice_index (1-first)"],
        how="left",
    ).drop(columns=["slice_index (1-first)"])

    # Sort to preserve slice ordering per patient
    merged = merged.sort_values(["sample_id", "slice_idx"]).reset_index(drop=True)

    rows = []
    for sample_id, group in merged.groupby("sample_id", sort=False):
        row = {
            "sample_id": sample_id,
            "label": int(group["label"].iloc[0]),
        }
        for _, slice_row in group.iterrows():
            s_idx = int(slice_row["slice_idx"])
            row[f"slice_{s_idx}_index"] = s_idx
            row[f"slice_{s_idx}_score_0"] = float(slice_row["score_0"])
            row[f"slice_{s_idx}_score_1"] = float(slice_row["score_1"])
            row[f"slice_{s_idx}_has_tumor"] = bool(slice_row["has_tumor"])
        rows.append(row)

    wide_df = pd.DataFrame(rows)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(output_csv_path, index=False)
    print(f"Transformed {len(merged)} slices into {len(wide_df)} sample rows (70 columns) and saved to: {output_csv_path}")
    return wide_df


def main():
    parser = argparse.ArgumentParser(description="Join inference results with manifest CSV.")
    parser.add_argument("results_csv_path", type=str, help="Path to results CSV file")
    parser.add_argument("manifest_csv_path", type=str, help="Path to manifest CSV file")
    parser.add_argument("output_csv_path", type=str, help="Path to output CSV file")
    args = parser.parse_args()

    join(args.results_csv_path, args.manifest_csv_path, args.output_csv_path)


if __name__ == "__main__":
    main()
