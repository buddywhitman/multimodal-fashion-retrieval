"""Part A: The Indexer — two complementary representations per image.

1. IMAGE level (fashionpedia_images): the whole image embedded with fashion-CLIP,
   plus zero-shot scene/style/weather tags and the bag of (category, color)
   attributes and layered-garment relations. Captures context, setting, and
   overall vibe.

2. GARMENT level (fashionpedia_garments): every garment instance cropped from
   its bbox and embedded separately, tagged with its category and mask-derived
   color. This is the compositionality fix — "red tie" can match the *tie crop*
   and "white shirt" the *shirt crop* independently, which a single pooled
   image vector fundamentally cannot distinguish from "white tie + red shirt".

A second pass indexes every image that has no Fashionpedia instance
annotations (whole-image embedding + scene/style/weather only, no garment
crops/attributes) — this grows the searchable corpus from the ~1158 annotated
images to the full ~3200 on disk, giving pure-CLIP/context queries more to
find without needing labels that don't exist for those images.

Both live in Chroma (persistent, one pip install, metadata filtering built in).

Run: python -m indexer.build_index
"""
import json
import os

import chromadb
from PIL import Image
from tqdm import tqdm

from common.config import (
    CATEGORY_CACHE_PATH, CHROMA_DIR, CROP_PAD_FRAC, GARMENT_COLLECTION,
    IMAGE_COLLECTION, MIN_CROP_PX, SCENE_PROMPTS, STYLE_PROMPTS, WEATHER_PROMPTS,
)
from indexer.color_extract import extract_colors
from indexer.dataset import category_vocab, load_dataset, load_unannotated_images
from indexer.embed import embed_images
from indexer.relations import extract_layered_pairs
from indexer.scene_tag import tag_images

BATCH_IMAGES = 32


def _crop_garment(image, bbox):
    x, y, w, h = bbox
    px, py = w * CROP_PAD_FRAC, h * CROP_PAD_FRAC
    left, top = max(0, int(x - px)), max(0, int(y - py))
    right, bottom = min(image.width, int(x + w + px)), min(image.height, int(y + h + py))
    if right - left < MIN_CROP_PX or bottom - top < MIN_CROP_PX:
        return None
    return image.crop((left, top, right, bottom))


def _fresh_collection(client, name):
    if name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    return client.create_collection(name, metadata={"hnsw:space": "cosine"})


def _tag_batch(pil_images):
    embeddings = embed_images(pil_images)
    tags = tag_images(embeddings, SCENE_PROMPTS, STYLE_PROMPTS, WEATHER_PROMPTS)
    return embeddings, tags


def build():
    os.makedirs(os.path.dirname(CATEGORY_CACHE_PATH), exist_ok=True)
    with open(CATEGORY_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(category_vocab(), f)

    records = load_dataset()
    unannotated_paths = load_unannotated_images()
    print(f"Indexing {len(records)} annotated images ({sum(len(r.instances) for r in records)} garment instances) "
          f"+ {len(unannotated_paths)} unannotated images")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    img_col = _fresh_collection(client, IMAGE_COLLECTION)
    gar_col = _fresh_collection(client, GARMENT_COLLECTION)

    # ---- pass 1: annotated images (full structured metadata) ----
    for i in tqdm(range(0, len(records), BATCH_IMAGES), desc="annotated"):
        batch = records[i:i + BATCH_IMAGES]
        pil_images = [Image.open(r.path).convert("RGB") for r in batch]
        img_embeddings, tags = _tag_batch(pil_images)

        img_ids, img_metas = [], []
        crops, crop_meta = [], []

        for rec, img, (scene, style, weather) in zip(batch, pil_images, tags):
            categories, colors = [], []
            for j, inst in enumerate(rec.instances):
                # 1 or 2 colors: a second is reported only when the garment
                # shows real internal color spread (striped/color-blocked) --
                # see indexer/color_extract.py. Both register as (category,
                # color) pairs so either matches a query, same as a garment
                # with two visually distinct instances would.
                inst_colors = extract_colors(img, inst.segmentation)
                for color in inst_colors:
                    categories.append(inst.category)
                    colors.append(color)

                crop = _crop_garment(img, inst.bbox)
                if crop is not None:
                    crops.append(crop)
                    crop_meta.append({
                        "image_id": rec.image_id, "file_name": rec.file_name,
                        "path": rec.path, "category": inst.category, "color": inst_colors[0],
                        "instance_idx": j,
                    })

            relations = extract_layered_pairs(rec.instances)

            img_ids.append(str(rec.image_id))
            img_metas.append({
                "file_name": rec.file_name, "path": rec.path,
                "categories": "|".join(categories), "colors": "|".join(colors),
                "pairs": "|".join(f"{c}::{col}" for c, col in zip(categories, colors)),
                "relations": "|".join(f"{a}::{b}" for a, b in relations),
                "scene": scene, "style": style, "weather": weather,
            })

        img_col.add(ids=img_ids, embeddings=img_embeddings.tolist(), metadatas=img_metas)

        if crops:
            crop_embeddings = embed_images(crops)
            gar_ids = [f"{m['image_id']}_{m['instance_idx']}" for m in crop_meta]
            gar_col.add(ids=gar_ids, embeddings=crop_embeddings.tolist(), metadatas=crop_meta)

    # ---- pass 2: unannotated images (whole-image + tags only) ----
    # negative ids so they can never collide with a real Fashionpedia image_id
    next_id = -1
    for i in tqdm(range(0, len(unannotated_paths), BATCH_IMAGES), desc="unannotated"):
        batch_paths = unannotated_paths[i:i + BATCH_IMAGES]
        pil_images = [Image.open(p).convert("RGB") for p in batch_paths]
        img_embeddings, tags = _tag_batch(pil_images)

        img_ids, img_metas = [], []
        for path, (scene, style, weather) in zip(batch_paths, tags):
            img_ids.append(str(next_id))
            next_id -= 1
            img_metas.append({
                "file_name": os.path.basename(path), "path": path,
                "categories": "", "colors": "", "pairs": "", "relations": "",
                "scene": scene, "style": style, "weather": weather,
            })
        img_col.add(ids=img_ids, embeddings=img_embeddings.tolist(), metadatas=img_metas)

    print(f"Done. images={img_col.count()}  garments={gar_col.count()}")


if __name__ == "__main__":
    build()
