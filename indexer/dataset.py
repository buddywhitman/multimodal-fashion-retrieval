"""Loads the Fashionpedia COCO-style annotation file and exposes it per-image.

Data lives in one place (data/raw). This module is the only thing that knows
the annotation file's shape — everything downstream (embedding, indexing,
retrieval) works with the plain `ImageRecord` objects this returns.
"""
import json
import os
from dataclasses import dataclass, field

from common.config import ANNOTATIONS_PATH, IMAGE_DIR


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


def load_dataset():
    """Returns list[ImageRecord] for every annotated image that exists on disk."""
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    categories = {c["id"]: c for c in raw["categories"]}

    images_by_id = {}
    for img in raw["images"]:
        path = os.path.join(IMAGE_DIR, img["file_name"])
        if not os.path.exists(path):
            continue
        images_by_id[img["id"]] = ImageRecord(
            image_id=img["id"],
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


def category_vocab():
    """All known garment category names, e.g. for the query parser's keyword list."""
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return sorted({c["name"] for c in raw["categories"]})


def load_unannotated_images():
    """Every image on disk that has NO Fashionpedia instance annotations.

    Fashionpedia's segmentation labels only cover a subset (val2020) of the
    images shipped in val_test2020.zip; the rest (the test2020 portion) have
    no category/color ground truth, but are still real fashion photos worth
    having in the searchable corpus for whole-image/scene/style/weather
    matching — see indexer/build_index.py's second indexing pass.
    """
    annotated = {r.file_name for r in load_dataset()}
    unannotated = []
    for file_name in sorted(os.listdir(IMAGE_DIR)):
        if file_name not in annotated:
            unannotated.append(os.path.join(IMAGE_DIR, file_name))
    return unannotated
