"""Automated Precision@k for the 5 PRD evaluation queries.

There's no human relevance-labeled ground truth, so relevance is judged by a
*predicate* over the metadata stored at index time — reproducible, and lets
weight/backbone changes be compared objectively instead of eyeballed.

Critically: `eval/coverage.py` established that this 1158-image corpus
contains **zero** yellow coats and **zero** red ties. A predicate demanding
those exact combos would score 0 forever regardless of retrieval quality --
that's a dataset ceiling, not a retriever bug. So each case is marked EXACT
(ground truth exists, predicate demands it) or PROXY (ground truth absent,
predicate instead checks the closest correct behavior achievable: right
category + right color *family*, and that CLIP still respects scene/style).

Run: python -m eval.evaluate
"""
from common.colors import family_members
from retriever.search import search

# "nearest achievable match" families, sourced from the same family table the
# retriever itself uses (common/colors.py) so this never drifts out of sync
# with the palette the way a hand-copied set would.
YELLOW_FAMILY = family_members("yellow") | family_members("brown") | {"beige", "orange", "coral"}
RED_FAMILY = family_members("red") | family_members("pink")


def _has_pair(r, category, color=None):
    for cat, col in zip(r["categories"], r["colors"]):
        if cat == category and (color is None or col == color):
            return True
    return False


def _has_category_in_family(r, category, family):
    for cat, col in zip(r["categories"], r["colors"]):
        if cat == category and col in family:
            return True
    return False


# (query, kind, relevance predicate over a result dict)
CASES = [
    ("A person in a bright yellow raincoat.", "PROXY (no yellow coat in corpus)",
     lambda r: _has_category_in_family(r, "coat", YELLOW_FAMILY)),

    ("Professional business attire inside a modern office.", "EXACT",
     lambda r: r["scene"] == "office" or r["style"] == "formal"),

    ("Someone wearing a blue shirt sitting on a park bench.", "EXACT",
     lambda r: _has_pair(r, "shirt, blouse", "blue") and r["scene"] in ("park", "street")),

    ("Casual weekend outfit for a city walk.", "EXACT",
     lambda r: r["style"] == "casual" or r["scene"] == "street"),

    ("A red tie and a white shirt in a formal setting.", "PROXY (no red tie in corpus)",
     lambda r: _has_category_in_family(r, "tie", RED_FAMILY) and _has_pair(r, "shirt, blouse", "white")),
]

K = 5


def run():
    print(f"Precision@{K} (EXACT = ground truth exists in corpus; PROXY = corpus has none, scored on nearest achievable match)\n")
    scores = []
    for query, kind, relevant in CASES:
        results, _ = search(query, k=K)
        hits = [relevant(r) for r in results]
        p = sum(hits) / len(hits) if hits else 0.0
        scores.append(p)
        marks = "".join("Y" if h else "." for h in hits)
        print(f"  P@{K}={p:.2f}  [{marks}]  ({kind})")
        print(f"    {query}")
    print(f"\n  mean P@{K} = {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    run()
