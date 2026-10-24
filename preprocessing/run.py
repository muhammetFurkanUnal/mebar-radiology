from lib import (
    all_samples_loadable
)
import pandas as pd
from pathlib import Path
from config import load_config, CONFIGS, NoAlignmentConfig
from pipeline.pipeline import Pipeline
from pipeline.context import SampleContext
import SimpleITK as sitk
import sys
import traceback
from random import shuffle

class SampleProvider:
    def __init__(self, config, provide_random:bool):
        manifest_path = config.manifest_path
        manifest_df = pd.read_csv(manifest_path)
        self.sample_paths = []
        for idx, row in manifest_df.iterrows():
            if row["sample_id"] in config.excluded_sample_ids:
                continue
            self.sample_paths.append([
                {
                    "sample_id":row["sample_id"],
                    "path":row["mask_path"],
                    "modality":row["modality"]
                },
                {
                    "sample_id":row["sample_id"],
                    "path":row["image_path"],
                    "modality":row["modality"]
                }
            ])
        
        self.current = 0
        if provide_random:
            shuffle(self.sample_paths)

    def provide(self):
        if not self.can_provide():
            return None
        samples = []
        samples.append(
            {
                "sample_id":self.sample_paths[self.current][0]["sample_id"],
                "image":sitk.ReadImage(self.sample_paths[self.current][0]["path"]),
                "modality":self.sample_paths[self.current][0]["modality"],
            }
        )
        samples.append(
            {
                "sample_id":self.sample_paths[self.current][1]["sample_id"],
                "image":sitk.ReadImage(self.sample_paths[self.current][1]["path"]),
                "modality":self.sample_paths[self.current][1]["modality"],
            }
        )
        self.current += 1
        return samples

    def can_provide(self)->bool:
        if self.current >= len(self.sample_paths):
            return False
        return True


def check_before_pipeline():
    check = all_samples_loadable()
    return check


def run_pipeline(config, n):

    records = []
    successes = []
    failures = []
    adc_pipeline = Pipeline(config, "ADC", records)
    t2_pipeline = Pipeline(config, "T2", records)

    provider = SampleProvider(config, False)

    loop_counter = 0

    while provider.can_provide():
        # prepare images
        mask_packet, image_packet = provider.provide()
        mask_sample_id = mask_packet["sample_id"]
        image_sample_id = image_packet["sample_id"]

        # mask-image id check
        if mask_sample_id != image_sample_id:
            print(f"id mismatch while processing sample with \nmask id:{mask_sample_id}\nimage_id:{image_sample_id}\n")
            failures.append({"id": mask_sample_id, "modality": mask_packet["modality"], "error": "mask/image id mismatch"})
            break

        # mask-image modality check
        if mask_packet["modality"] != image_packet["modality"]:
            print(f"modality mismatch while processing sample with id:{mask_sample_id}\n")
            failures.append({"id": mask_sample_id, "modality": mask_packet["modality"], "error": "mask/image modality mismatch"})
            break
        modality = image_packet["modality"]

        mask = mask_packet["image"]
        image = image_packet["image"]

        mask_out_path = f"{config.output_folder_path}/{mask_sample_id}_mask.nrrd"
        image_out_path = f"{config.output_folder_path}/{image_sample_id}_{modality}.nrrd"

        ctx = SampleContext(
            sample_id=image_sample_id,
            modality=modality,
            mask=mask,
            image=image,
            mask_out_path=mask_out_path,
            image_out_path=image_out_path,
        )

        # pass image and mask though pipeline
        try:
            if modality == "ADC":
                ctx = adc_pipeline.run_pipeline(ctx)
            else:
                ctx = t2_pipeline.run_pipeline(ctx)
        except Exception as e:
            print(f"[ERROR] sample {image_sample_id} ({modality}): {e}")
            failures.append({"id": image_sample_id, "modality": modality, "error": traceback.format_exc()})
        else:
            successes.append(image_sample_id)

        loop_counter += 1
        if loop_counter >= n:
            break

    df = pd.DataFrame(records)
    df.to_csv(Path(config.output_folder_path).parent / "pipeline_checkpoints.csv", index=False)

    # SUMMARY
    print("=============== run summary ===============")
    print(f"total attempted : {loop_counter}")
    print(f"completed ok    : {len(successes)}")
    print(f"failed          : {len(failures)}")

    if failures:
        print("\nfailed samples:")
        for failure in failures:
            print(f"  - {failure['id']} ({failure['modality']}): {failure['error'].strip().splitlines()[-1]}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run.py <configuration>")
        print("Example: python run.py no_alignment")
        exit()

    # if not check_before_pipeline():
    #     print("pipeline could not get started.")
    #     exit()


    config_name = sys.argv[1]
    n = int(sys.argv[2])

    config:NoAlignmentConfig = load_config(config_name)

    run_pipeline(config, n)