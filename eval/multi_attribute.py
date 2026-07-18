"""Tests the PRD's own literal example of "Context Awareness" verbatim:
multi-attribute queries combining color + clothing type + location. This is
the exact 3-attribute combination named in the PRD's Part B requirements —
previously only 2-attribute (category+color) and pure-scene queries had been
benchmarked; this closes that gap directly.

Method, same corpus-grounded approach as eval/benchmark.py: for every
(category, color, scene) triple that occurs >=2 times, build the natural
3-attribute query ("a person wearing a {color} {category} {scene_phrase}")
and measure P@5/R@5 against the true positive images.

Also answers a sharper question than raw P@5: does the location term actually
*help* discriminate, or is it dead weight? For each triple, we also run the
2-attribute query (color+category only, no location) and compare. If location
is doing real work, the 3-attribute query should rank true positives higher
(or equal) on average -- especially for combos where color+category alone is
ambiguous (many images share that garment+color across different scenes).

Run: python -m eval.multi_attribute
"""
import random
from collections import defaultdict

import chromadb
from PIL import Image

from common.config import CHROMA_DIR, GARMENT_COLLECTION, IMAGE_COLLECTION
from eval.benchmark import GARMENT_CATEGORIES
from indexer.color_extract import extract_colors
from indexer.dataset import load_dataset
from retriever.search import search

MIN_SUPPORT = 2
K = 5
SAMPLE_SIZE = 150  # this runs 2 searches/triple; sampled (like ablate_weights.py) to keep runtime sane

SCENE_PHRASES = {
    "office": "in a modern office", "street": "on an urban street",
    "park": "in a park", "home": "at home", "studio": "in a studio",
    "runway": "on a runway", "beach": "at the beach", "cafe": "in a cafe",
    "gym": "at the gym",
}


def _ground_truth_triples():
    """(category, color, scene) -> set of file_names."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    img_col = client.get_collection(IMAGE_COLLECTION)
    scene_by_id = {}

    truth = defaultdict(set)
    for rec in load_dataset():
        if not rec.instances:
            continue
        if rec.image_id not in scene_by_id:
            meta = img_col.get(ids=[str(rec.image_id)], include=["metadatas"])["metadatas"][0]
            scene_by_id[rec.image_id] = meta["scene"]
        scene = scene_by_id[rec.image_id]

        img = Image.open(rec.path).convert("RGB")
        for inst in rec.instances:
            if inst.category not in GARMENT_CATEGORIES:
                continue
            for color in extract_colors(img, inst.segmentation):
                if color == "unknown":
                    continue
                truth[(inst.category, color, scene)].add(rec.file_name)
    return truth


def run():
    truth = _ground_truth_triples()
    all_triples = [(t, files) for t, files in truth.items() if len(files) >= MIN_SUPPORT]
    random.seed(0)
    triples = random.sample(all_triples, min(SAMPLE_SIZE, len(all_triples)))
    print(f"{len(all_triples)} (category, color, scene) triples with >= {MIN_SUPPORT} true positives; "
          f"sampling {len(triples)} (seed=0)\n")

    p3_list, r3_list, p2_list = [], [], []
    for (cat, color, scene), true_files in triples:
        head = cat.split(",")[0]
        phrase = SCENE_PHRASES.get(scene, f"in a {scene}")
        query3 = f"a person wearing a {color} {head} {phrase}"
        query2 = f"a person wearing a {color} {head}"

        r3, _ = search(query3, k=K)
        r2, _ = search(query2, k=K)
        hits3 = {r["file_name"] for r in r3} & true_files
        hits2 = {r["file_name"] for r in r2} & true_files

        p3, r3_recall = len(hits3) / K, len(hits3) / min(K, len(true_files))
        p2 = len(hits2) / K
        p3_list.append(p3); r3_list.append(r3_recall); p2_list.append(p2)

        print(f"  3attr P@5={p3:.2f} R@5={r3_recall:.2f}  |  2attr(no location) P@5={p2:.2f}  "
              f"n_true={len(true_files):>3}  '{query3}'")

    n = len(p3_list)
    print(f"\n  3-attribute (color+type+location): mean P@{K}={sum(p3_list)/n:.3f}  mean R@{K}={sum(r3_list)/n:.3f}")
    print(f"  2-attribute (color+type only):     mean P@{K}={sum(p2_list)/n:.3f}")
    delta = sum(p3_list)/n - sum(p2_list)/n
    print(f"  location term effect on P@{K}: {delta:+.3f}  ({'helps' if delta > 0.005 else 'no measurable effect' if abs(delta) <= 0.005 else 'hurts'})")


if __name__ == "__main__":
    run()
