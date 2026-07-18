"""Reports whether ground truth even exists in the corpus for each eval
query's attribute combination — the honest ceiling check.

A retrieval system cannot return an image that doesn't exist. Before scoring
Precision@k, this tells you whether a P@k < 1.0 means "the retriever is
wrong" or "no such image is in this corpus" — the difference matters when
deciding whether to keep tuning the algorithm or accept the dataset's limit.

Checks both EXACT (the literal color word) and FAMILY (any color in the same
chromatic family, e.g. "cornflower blue" for a "blue" query -- see
common/colors.py) coverage. Family matters because the color namer is a
~800-name XKCD-derived palette (finer than the query vocabulary), so a real
"yellow"-family coat is very likely named something more specific
("mustard", "maize") rather than the bare word "yellow" -- exact-only
checking would misreport real coverage as absent.

Run: python -m eval.coverage
"""
from collections import Counter

from common.colors import same_family
from eval.ground_truth import load_indexed_records


def build_pair_counts():
    counts = Counter()
    for _image_id, _file_name, pairs, _scene in load_indexed_records():
        for cat, color in pairs:
            counts[(cat, color)] += 1
    return counts


CHECKS = [
    ("A person in a bright yellow raincoat.", "coat", "yellow"),
    ("Someone wearing a blue shirt sitting on a park bench.", "shirt, blouse", "blue"),
    ("A red tie and a white shirt in a formal setting. (tie half)", "tie", "red"),
    ("A red tie and a white shirt in a formal setting. (shirt half)", "shirt, blouse", "white"),
]


def run():
    counts = build_pair_counts()
    print("Ground-truth coverage for each eval query's attribute combo:\n")
    for label, cat, color in CHECKS:
        exact_n = counts.get((cat, color), 0)
        family_matches = {c: n for (c2, c), n in counts.items() if c2 == cat and same_family(c, color)}
        family_n = sum(family_matches.values())

        verdict = "EXISTS (exact)" if exact_n > 0 else (
            "EXISTS (family only)" if family_n > 0 else "ABSENT FROM CORPUS")
        print(f"  ({cat}, {color}): exact={exact_n}  family={family_n}  -- {verdict}")
        if family_n > 0 and exact_n == 0:
            top = sorted(family_matches.items(), key=lambda x: -x[1])[:5]
            print(f"    top family matches: {top}")
        print(f"    ({label})\n")


if __name__ == "__main__":
    run()
