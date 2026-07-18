"""Reports whether ground truth even exists in the corpus for each eval
query's exact attribute combination — the honest ceiling check.

A retrieval system cannot return an image that doesn't exist. Before scoring
Precision@k, this tells you whether a P@k < 1.0 means "the retriever is
wrong" or "no such image is in this corpus" — the difference matters when
deciding whether to keep tuning the algorithm or accept the dataset's limit.

Run: python -m eval.coverage
"""
from collections import Counter

from eval.ground_truth import load_indexed_records


def build_pair_counts():
    counts = Counter()
    for _image_id, _file_name, pairs, _scene in load_indexed_records():
        for cat, color in pairs:
            counts[(cat, color)] += 1
    return counts


CHECKS = [
    ("A person in a bright yellow raincoat.", ("coat", "yellow")),
    ("Someone wearing a blue shirt sitting on a park bench.", ("shirt, blouse", "blue")),
    ("A red tie and a white shirt in a formal setting. (tie half)", ("tie", "red")),
    ("A red tie and a white shirt in a formal setting. (shirt half)", ("shirt, blouse", "white")),
]


def run():
    counts = build_pair_counts()
    print("Ground-truth coverage for each eval query's exact attribute combo:\n")
    for label, pair in CHECKS:
        n = counts.get(pair, 0)
        verdict = "EXISTS" if n > 0 else "ABSENT FROM CORPUS"
        print(f"  {pair}: {n} instance(s) -- {verdict}")
        print(f"    ({label})\n")


if __name__ == "__main__":
    run()
