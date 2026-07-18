"""Loads Fashionpedia's COCO-style annotation files and exposes them per-image.

Data lives in one place (data/raw). This module is the only thing that knows
the annotation files' shape — everything downstream (embedding, indexing,
retrieval) works with the plain `ImageRecord` objects this returns.

Two sources are merged: val2020 (the small, originally-used slice) and
train2020 (much larger — see common/config.py's TRAIN_* constants for why).
train2020 image ids are offset to a disjoint range so they can never collide
with val2020 ids when both are combined into one corpus.
"""
import json
import os
import random
from dataclasses import dataclass, field

from common.config import (
    ANNOTATIONS_PATH, IMAGE_DIR, TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGE_DIR,
    TRAIN_PRIORITY_CATEGORIES, TRAIN_RANDOM_SAMPLE_SIZE, TRAIN_SAMPLE_SEED,
)

TRAIN_ID_OFFSET = 100_000_000  # comfortably above any real Fashionpedia id


@dataclass
class GarmentInstance:
    category: str          # e.g. "shirt, blouse"
    supercategory: str     # e.g. "upperbody"
    bbox: list             # [x, y, w, h]
    segmentation: list      # polygon(s), COCO format


@dataclass
class ImageRecord:
    image_id: int
    file_name: str
    path: str
    width: int
    height: int
    instances: list = field(default_factory=list)  # list[GarmentInstance]


def _load_source(annotations_path, image_dir, id_offset=0, keep_image_ids=None):
    """One annotation file + image dir -> list[ImageRecord]. id_offset keeps
    ids from different sources disjoint. keep_image_ids, if given, restricts
    to only those (pre-offset) image ids -- used to apply the train2020
    sampling strategy without loading instances for images we'll discard."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    categories = {c["id"]: c for c in raw["categories"]}

    images_by_id = {}
    for img in raw["images"]:
        if keep_image_ids is not None and img["id"] not in keep_image_ids:
            continue
        path = os.path.join(image_dir, img["file_name"])
        if not os.path.exists(path):
            continue
        images_by_id[img["id"]] = ImageRecord(
            image_id=img["id"] + id_offset,
            file_name=img["file_name"],
            path=path,
            width=img["width"],
            height=img["height"],
        )

    for ann in raw["annotations"]:
        rec = images_by_id.get(ann["image_id"])
        if rec is None:
            continue
        cat = categories[ann["category_id"]]
        rec.instances.append(
            GarmentInstance(
                category=cat["name"],
                supercategory=cat["supercategory"],
                bbox=ann["bbox"],
                segmentation=ann["segmentation"],
            )
        )

    return list(images_by_id.values())


def train2020_sample_ids():
    """(pre-offset) image ids to include from train2020: every image with a
    TRAIN_PRIORITY_CATEGORIES instance, plus a reproducible random sample of
    the rest, capped at TRAIN_RANDOM_SAMPLE_SIZE."""
    with open(TRAIN_ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    categories = {c["id"]: c["name"] for c in raw["categories"]}

    priority_ids, all_ids = set(), set()
    for ann in raw["annotations"]:
        all_ids.add(ann["image_id"])
        if categories[ann["category_id"]] in TRAIN_PRIORITY_CATEGORIES:
            priority_ids.add(ann["image_id"])

    remaining = sorted(all_ids - priority_ids)
    rng = random.Random(TRAIN_SAMPLE_SEED)
    rng.shuffle(remaining)
    sampled = set(remaining[:TRAIN_RANDOM_SAMPLE_SIZE])

    return priority_ids | sampled


def load_dataset():
    """Returns list[ImageRecord] for every annotated image that exists on
    disk, merging val2020 (always) with train2020 (if downloaded -- see
    README.md for the fetch step; degrades gracefully to val2020-only
    otherwise)."""
    records = _load_source(ANNOTATIONS_PATH, IMAGE_DIR)

    if os.path.exists(TRAIN_ANNOTATIONS_PATH) and os.path.exists(TRAIN_IMAGE_DIR):
        keep_ids = train2020_sample_ids()
        records += _load_source(TRAIN_ANNOTATIONS_PATH, TRAIN_IMAGE_DIR,
                                 id_offset=TRAIN_ID_OFFSET, keep_image_ids=keep_ids)

    return records


def category_vocab():
    """All known garment category names, e.g. for the query parser's keyword
    list. Both sources share Fashionpedia's fixed 46-category schema, so
    val2020 alone is sufficient (and fast) to enumerate them."""
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return sorted({c["name"] for c in raw["categories"]})


def load_unannotated_images():
    """Every val_test2020 image with NO Fashionpedia instance annotation.

    Fashionpedia's segmentation labels only cover val2020 within the
    val_test2020.zip archive; the rest (the test2020 portion) has no
    category/color ground truth, but is still real fashion photos worth
    having in the searchable corpus for whole-image/scene/style/weather
    matching — see indexer/build_index.py's second indexing pass. (train2020
    is fully annotated, so it has no unannotated counterpart to add here.)
    """
    annotated = {r.file_name for r in _load_source(ANNOTATIONS_PATH, IMAGE_DIR)}
    unannotated = []
    for file_name in sorted(os.listdir(IMAGE_DIR)):
        if file_name not in annotated:
            unannotated.append(os.path.join(IMAGE_DIR, file_name))
    return unannotated
