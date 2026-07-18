"""Automated Precision@k for the 5 PRD evaluation queries.

There's no human relevance-labeled ground truth, so relevance is judged by a
*predicate* over the metadata stored at index time — reproducible, and lets
weight/backbone changes be compared objectively instead of eyeballed.

The color namer (common/colors.py) is a ~800-name XKCD-derived palette, finer
than the small set of color *words* a query uses -- a real yellow coat is
very likely named "mustard" or "maize" in the index, not the bare word
"yellow". So every color match here is checked by chromatic *family*
(`same_family`, the same mechanism the retriever itself uses for graded
partial credit), not exact string equality -- exact-only checking would
misreport real, substantial ground truth as absent (verified directly via
`eval/coverage.py`: after merging in Fashionpedia's train2020 split, this
corpus has 71 yellow-family coats, 652 blue-family shirts, and 127
red-family ties -- all previously zero or near-zero in the smaller val2020-
only corpus).

Run: python -m eval.evaluate
"""
from common.colors import same_family
from retriever.search import search


def _has_pair(r, category, color=None):
    for cat, col in zip(r["categories"], r["colors"]):
        if cat == category and (color is None or same_family(col, color)):
            return True
    return False


# (query, relevance predicate over a result dict)
CASES = [
    ("A person in a bright yellow raincoat.",
     lambda r: _has_pair(r, "coat", "yellow")),

    ("Professional business attire inside a modern office.",
     lambda r: r["scene"] == "office" or r["style"] == "formal"),

    ("Someone wearing a blue shirt sitting on a park bench.",
     lambda r: _has_pair(r, "shirt, blouse", "blue") and r["scene"] in ("park", "street")),

    ("Casual weekend outfit for a city walk.",
     lambda r: r["style"] == "casual" or r["scene"] == "street"),

    ("A red tie and a white shirt in a formal setting.",
     lambda r: _has_pair(r, "tie", "red") and _has_pair(r, "shirt, blouse", "white")),
]

K = 5


def run():
    print(f"Precision@{K} (color matches are chromatic-family-aware -- see module docstring)\n")
    scores = []
    for query, relevant in CASES:
        results, _ = search(query, k=K)
        hits = [relevant(r) for r in results]
        p = sum(hits) / len(hits) if hits else 0.0
        scores.append(p)
        marks = "".join("Y" if h else "." for h in hits)
        print(f"  P@{K}={p:.2f}  [{marks}]")
        print(f"    {query}")
    print(f"\n  mean P@{K} = {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    run()
