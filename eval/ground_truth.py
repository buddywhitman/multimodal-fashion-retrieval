"""Shared ground-truth loader for eval scripts.

Every eval script that needs "what (category, color) pairs actually exist in
the corpus" used to recompute them by re-opening every image and re-running
segmentation-mask color extraction from scratch -- redundant work, since
indexer/build_index.py already computed and stored exactly this in Chroma's
image collection (`pairs` metadata: "category::color|category::color|...",
one entry per color per garment instance, primary color first -- see
indexer/color_extract.py's extract_colors()).

Reading it back from the index instead of recomputing it is not just faster
(no image I/O, no mask math) -- it also removes a whole class of possible
bug: eval "ground truth" recomputed independently could silently drift from
what's actually indexed if the two code paths ever diverged. Reading the same
data both places makes that impossible by construction.

Run standalone for a quick sanity/timing check: python -m eval.ground_truth
"""
import time
from collections import defaultdict

import chromadb

from common.config import CHROMA_DIR, IMAGE_COLLECTION


def load_indexed_records():
    """Returns list[(image_id, file_name, ordered_pairs, scene)] for every
    ANNOTATED image (unannotated ones -- negative ids, see build_index.py --
    have no `pairs` and are skipped). ordered_pairs is
    list[(category, color)], primary color first per instance, exactly as
    stored at index time. One Chroma call, no image I/O.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    img_col = client.get_collection(IMAGE_COLLECTION)
    got = img_col.get(include=["metadatas"])

    records = []
    for cid, meta in zip(got["ids"], got["metadatas"]):
        if not meta.get("pairs"):
            continue  # unannotated image or one with zero garment instances
        pairs = []
        for p in meta["pairs"].split("|"):
            if "::" in p:
                cat, col = p.split("::", 1)
                pairs.append((cat, col))
        records.append((cid, meta.get("file_name"), pairs, meta.get("scene")))
    return records


def ground_truth_pairs(exclude_unknown=True):
    """(category, color) -> set of file_names. Drop-in replacement for the
    per-eval-script versions that used to recompute this from raw images."""
    truth = defaultdict(set)
    for _image_id, file_name, pairs, _scene in load_indexed_records():
        for cat, col in pairs:
            if exclude_unknown and col == "unknown":
                continue
            truth[(cat, col)].add(file_name)
    return truth


def ground_truth_triples(exclude_unknown=True):
    """(category, color, scene) -> set of file_names."""
    truth = defaultdict(set)
    for _image_id, file_name, pairs, scene in load_indexed_records():
        for cat, col in pairs:
            if exclude_unknown and col == "unknown":
                continue
            truth[(cat, col, scene)].add(file_name)
    return truth


def primary_color_garments(garment_categories):
    """list[(image_id, file_name, list[(category, primary_color)])], deduped
    to the first (primary) color per category per image, restricted to
    `garment_categories`. Used by eval/compositional.py, which needs one
    unambiguous identity color per garment to construct a well-posed
    color-swap pair."""
    out = []
    for image_id, file_name, pairs, _scene in load_indexed_records():
        seen = set()
        garments = []
        for cat, col in pairs:
            if cat not in garment_categories or cat in seen or col == "unknown":
                continue
            seen.add(cat)
            garments.append((cat, col))
        if garments:
            out.append((image_id, file_name, garments))
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    truth = ground_truth_pairs()
    dt = time.perf_counter() - t0
    print(f"load_indexed_records + ground_truth_pairs: {dt:.3f}s for {len(truth)} (category, color) combos")
    print(f"(previously: several minutes, re-opening every image and recomputing color extraction each run)")
