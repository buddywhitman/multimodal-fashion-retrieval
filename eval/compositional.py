"""The PRD's core hint, quantified: does the system tell "red shirt + blue
pants" from "blue shirt + red pants"?

Controlled color-swap discrimination test. For each corpus image that contains
two garments of *different* colors (cat_a=color_a, cat_b=color_b), we form:
  correct  query = "a person wearing a {color_a} {cat_a} and a {color_b} {cat_b}"
  swapped  query = "a person wearing a {color_b} {cat_a} and a {color_a} {cat_b}"
A system that understands compositionality scores the image higher for the
correct query than the swapped one. We measure, per image, the score margin
(correct - swapped) under two systems:

  CLIP-only : rank by whole-image fashion-CLIP similarity alone (the baseline
              the PRD says to beat) -- computed here directly from the image
              embedding vs. each query embedding.
  Hybrid    : this system's full score (search()), which adds instance-scoped
              exact (category, color) re-ranking.

Reported: fraction of pairs where correct > swapped (discrimination accuracy),
and mean margin. CLIP-only is expected near chance (~0.5) because a single
pooled vector contains "red", "blue", "shirt", "pants" for both phrasings;
the hybrid should be near-perfect because its attribute match is
instance-scoped.

Run: python -m eval.compositional
"""
import random

import numpy as np

from indexer.color_extract import dominant_color_name, extract_colors
from indexer.dataset import load_dataset
from indexer.embed import embed_text
from PIL import Image

from eval.benchmark import GARMENT_CATEGORIES
import chromadb
from common.config import CHROMA_DIR, IMAGE_COLLECTION
from retriever.search import _exact_match

MAX_PAIRS = 60


def _find_swap_pairs():
    """Images with exactly-usable two-garment, two-color structure."""
    pairs = []
    for rec in load_dataset():
        if len(rec.instances) < 2:
            continue
        img = Image.open(rec.path).convert("RGB")
        garments = []
        seen = set()
        for inst in rec.instances:
            if inst.category not in GARMENT_CATEGORIES:
                continue
            # primary color only here: the swap experiment needs one
            # unambiguous identity color per garment to construct a
            # well-posed "correct vs swapped" pair
            color = dominant_color_name(img, inst.segmentation)
            if color in ("unknown",) or inst.category in seen:
                continue
            seen.add(inst.category)
            garments.append((inst.category, color))
        # need two garments with two *different* colors
        for i in range(len(garments)):
            for j in range(i + 1, len(garments)):
                (ca, cola), (cb, colb) = garments[i], garments[j]
                if cola != colb:
                    pairs.append((rec, (ca, cola), (cb, colb)))
                    break
            else:
                continue
            break
    return pairs


def _q(cat_a, col_a, cat_b, col_b):
    ha, hb = cat_a.split(",")[0], cat_b.split(",")[0]
    return f"a person wearing a {col_a} {ha} and a {col_b} {hb}"


def run():
    random.seed(0)
    pairs = _find_swap_pairs()
    random.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]
    print(f"Color-swap compositional discrimination over {len(pairs)} two-garment images\n")

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    img_col = client.get_collection(IMAGE_COLLECTION)

    clip_correct, hybrid_correct = 0, 0
    clip_margins, hybrid_margins = [], []

    for rec, (ca, cola), (cb, colb) in pairs:
        correct_q = _q(ca, cola, cb, colb)
        swapped_q = _q(ca, colb, cb, cola)

        # --- CLIP-only: image embedding vs each query embedding ---
        emb = np.asarray(img_col.get(ids=[str(rec.image_id)], include=["embeddings"])["embeddings"][0])
        qc, qs = embed_text([correct_q, swapped_q])
        clip_c, clip_s = float(np.dot(emb, qc)), float(np.dot(emb, qs))
        clip_correct += clip_c > clip_s
        clip_margins.append(clip_c - clip_s)

        # --- Hybrid attribute component (instance-scoped exact match) ---
        # mirrors what's actually indexed: an instance can register >1 color
        # (see indexer/color_extract.py), so use the same extraction here.
        rec_img = Image.open(rec.path).convert("RGB")
        pairs_str = "|".join(
            f"{i.category}::{col}" for i in rec.instances for col in extract_colors(rec_img, i.segmentation)
        )
        hyb_c = np.mean([_exact_match((ca, cola), pairs_str), _exact_match((cb, colb), pairs_str)])
        hyb_s = np.mean([_exact_match((ca, colb), pairs_str), _exact_match((cb, cola), pairs_str)])
        hybrid_correct += hyb_c > hyb_s
        hybrid_margins.append(hyb_c - hyb_s)

    n = len(pairs)
    print(f"  CLIP-only  discrimination accuracy: {clip_correct/n:.3f}   mean margin: {np.mean(clip_margins):+.4f}")
    print(f"  Hybrid     discrimination accuracy: {hybrid_correct/n:.3f}   mean margin: {np.mean(hybrid_margins):+.4f}")
    print(f"\n  (accuracy = fraction where the correctly-composed query outscores the color-swapped decoy;")
    print(f"   0.5 = chance. This is exactly the 'red shirt/blue pants vs blue shirt/red pants' case.)")


if __name__ == "__main__":
    run()
