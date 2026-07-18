"""One-time data-prep step: extract only the sampled subset of train2020
images from the downloaded zip, not all ~45k -- see common/config.py's
TRAIN_* constants for the sampling strategy (every coat/tie image, plus a
random sample of the rest) and indexer/dataset.py for how it's used.

Selective extraction (not `unzip train2020.zip`) because we only need
~12k of the 45k images; extracting all of them would cost disk space and
time for images that will never be indexed.

Run once after downloading train2020.zip (see README.md):
    python -m indexer.extract_train_subset
"""
import os
import zipfile

from common.config import RAW_DIR, TRAIN_IMAGE_DIR
from indexer.dataset import train2020_sample_ids

ZIP_PATH = os.path.join(RAW_DIR, "train2020.zip")


def run():
    if not os.path.exists(ZIP_PATH):
        raise SystemExit(f"{ZIP_PATH} not found -- download it first (see README.md)")

    print("Selecting sample (coat/tie images + random sample)...")
    keep_ids = train2020_sample_ids()
    print(f"  {len(keep_ids)} images to extract")

    os.makedirs(TRAIN_IMAGE_DIR, exist_ok=True)
    already = set(os.listdir(TRAIN_IMAGE_DIR))

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        # zip entries are typically "train/<file>.jpg" -- build id lookup by
        # filename since that's what's inside the zip, not the numeric id
        wanted_files = _wanted_file_names(keep_ids)
        extracted = 0
        for name in names:
            base = os.path.basename(name)
            if base in wanted_files and base not in already:
                with zf.open(name) as src, open(os.path.join(TRAIN_IMAGE_DIR, base), "wb") as dst:
                    dst.write(src.read())
                extracted += 1
                if extracted % 1000 == 0:
                    print(f"  extracted {extracted}...")

    print(f"Done. Extracted {extracted} new images to {TRAIN_IMAGE_DIR}")


def _wanted_file_names(keep_ids):
    import json
    from common.config import TRAIN_ANNOTATIONS_PATH
    with open(TRAIN_ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {img["file_name"] for img in raw["images"] if img["id"] in keep_ids}


if __name__ == "__main__":
    run()
