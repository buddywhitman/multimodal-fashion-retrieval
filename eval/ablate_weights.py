"""Ablation over the hybrid-scoring weights (W_CLIP/W_COMP/W_SCENE) against a
fixed random sample of corpus-grounded GARMENT queries, to check whether the
config.py defaults are actually a good choice or just an untested guess.

Reuses eval.benchmark's ground-truth construction but monkeypatches the
weight globals per run (they're read at call-time inside retriever.search,
so reassigning the module attributes before each search() call is enough --
no re-import needed).

Run: python -m eval.ablate_weights
"""
import random

import retriever.search as searcher
from eval.benchmark import GARMENT_CATEGORIES, ground_truth_pairs

search = searcher.search

SAMPLE_SIZE = 120  # up from the original 40 -- a much larger slice of the ~190 GARMENT combos now available
K = 5

CONFIGS = [
    ("current (0.50/0.35/0.15)", 0.50, 0.35, 0.15),
    ("clip-heavy (0.70/0.20/0.10)", 0.70, 0.20, 0.10),
    ("comp-heavy (0.35/0.55/0.10)", 0.35, 0.55, 0.10),
    ("comp-only (0.10/0.80/0.10)", 0.10, 0.80, 0.10),
    ("equal (0.34/0.33/0.33)", 0.34, 0.33, 0.33),
]


def run():
    truth = ground_truth_pairs()
    combos = [(pair, files) for pair, files in truth.items()
              if len(files) >= 3 and pair[0] in GARMENT_CATEGORIES]
    random.seed(0)
    sample = random.sample(combos, min(SAMPLE_SIZE, len(combos)))
    print(f"Ablating over {len(sample)} random GARMENT combos (seed=0)\n")

    for label, w_clip, w_comp, w_scene in CONFIGS:
        searcher.W_CLIP, searcher.W_COMP, searcher.W_SCENE = w_clip, w_comp, w_scene
        precisions, recalls = [], []
        for (cat, color), true_files in sample:
            head = cat.split(",")[0]
            query = f"a person wearing a {color} {head}"
            results, _ = search(query, k=K)
            hits = {r["file_name"] for r in results} & true_files
            precisions.append(len(hits) / K)
            recalls.append(len(hits) / min(K, len(true_files)))
        print(f"  {label:32}  mean P@{K}={sum(precisions)/len(precisions):.3f}  mean R@{K}={sum(recalls)/len(recalls):.3f}")


if __name__ == "__main__":
    run()
