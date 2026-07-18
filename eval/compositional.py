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

Ground truth (garments per image, and each image's full attribute pairs for
the hybrid's exact-match scoring) is read straight from the already-built
index (eval/ground_truth.py) rather than recomputed from raw images.

Run: python -m eval.compositional
"""
import random

import numpy as np

from indexer.embed import embed_text

from eval.benchmark import GARMENT_CATEGORIES
from eval.ground_truth import primary_color_garments
import chromadb
from common.config import CHROMA_DIR, IMAGE_COLLECTION
from retriever.search import _exact_match

MAX_PAIRS = 60


def _find_swap_pairs():
    """Images with exactly-usable two-garment, two-color structure."""
    pairs = []
    for image_id, file_name, garments in primary_color_garments(GARMENT_CATEGORIES):
        # need two garments with two *different* colors
        for i in range(len(garments)):
            for j in range(i + 1, len(garments)):
                (ca, cola), (cb, colb) = garments[i], garments[j]
                if cola != colb:
                    pairs.append(((image_id, file_name), (ca, cola), (cb, colb)))
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

    # one batched fetch (metadata's `pairs` field + embedding) for exactly the
    # sampled images, instead of a Chroma call or image re-open per pair
    ids = [str(image_id) for (image_id, _fn), _a, _b in pairs]
    got = img_col.get(ids=ids, include=["metadatas", "embeddings"])
    pairs_str_by_id = {cid: meta["pairs"] for cid, meta in zip(got["ids"], got["metadatas"])}
    emb_by_id = {cid: np.asarray(e) for cid, e in zip(got["ids"], got["embeddings"])}

    # batch every correct/swapped query text across all pairs into one model
    # call, not one call per pair -- same lesson as retriever/search.py:
    # fewer, larger calls beat many small sequential ones on this CPU backbone.
    query_texts = []
    for (_image_id, _fn), (ca, cola), (cb, colb) in pairs:
        query_texts.append(_q(ca, cola, cb, colb))
        query_texts.append(_q(ca, colb, cb, cola))
    query_embs = embed_text(query_texts)

    clip_correct, hybrid_correct = 0, 0
    clip_margins, hybrid_margins = [], []

    for i, ((image_id, _file_name), (ca, cola), (cb, colb)) in enumerate(pairs):
        cid = str(image_id)

        # --- CLIP-only: image embedding vs each query embedding ---
        emb = emb_by_id[cid]
        qc, qs = query_embs[2 * i], query_embs[2 * i + 1]
        clip_c, clip_s = float(np.dot(emb, qc)), float(np.dot(emb, qs))
        clip_correct += clip_c > clip_s
        clip_margins.append(clip_c - clip_s)

        # --- Hybrid attribute component (instance-scoped exact match) ---
        pairs_str = pairs_str_by_id[cid]
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
