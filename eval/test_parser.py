"""Calibration + regression test for the zero-shot vocabulary resolver.

Two things must both hold for the embedding fallback to be a net win:
  1. RECALL: garment words absent from the hardcoded synonym dict
     ("windbreaker", "parka", "loafers"...) resolve to a sensible category.
  2. PRECISION: non-garment words ("person", "walking", "morning"...) do NOT
     get spuriously resolved to a category (which would inject a false
     compositional constraint and hurt retrieval).

This is what justifies SIM_THRESHOLD in retriever/vocab_resolver.py — set it
too low and precision fails, too high and recall fails. Run this after any
threshold change.

Run: python -m eval.test_parser
"""
from retriever.query_parser import CATEGORY_SYNONYMS, parse

# garment words deliberately NOT in CATEGORY_SYNONYMS -> should resolve to a
# plausible category (we accept any reasonable upper/outer/footwear mapping,
# since "windbreaker" legitimately reads as coat OR jacket)
SHOULD_RESOLVE = {
    "windbreaker": {"coat", "jacket"},
    "parka": {"coat", "jacket"},
    "anorak": {"coat", "jacket"},
    "loafers": {"shoe"},
    "sandals": {"shoe"},
    "turtleneck": {"sweater", "top, t-shirt, sweatshirt", "shirt, blouse"},
    "leggings": {"tights, stockings", "pants"},
    "overalls": {"jumpsuit", "pants"},
}

# non-garment words -> must NOT resolve to any garment category
SHOULD_NOT_RESOLVE = [
    "a person walking in the morning sunshine",
    "someone standing near a building downtown",
    "a cheerful relaxed weekend vibe",
]


def _garment_cats(query):
    return {cat for cat, _ in parse(query)["garments"]}


def run():
    assert not (set(SHOULD_RESOLVE) & set(CATEGORY_SYNONYMS)), "test words leaked into the dict"

    recall_ok = 0
    print("RECALL — novel garment words should resolve:")
    for word, acceptable in SHOULD_RESOLVE.items():
        cats = _garment_cats(f"a person wearing a {word}")
        hit = bool(cats & acceptable)
        recall_ok += hit
        print(f"  {'OK ' if hit else 'MISS'} '{word}' -> {cats or '(unresolved)'}  (accept {acceptable})")

    print("\nPRECISION — non-garment sentences should resolve to NO garment:")
    precision_ok = 0
    for sent in SHOULD_NOT_RESOLVE:
        cats = _garment_cats(sent)
        clean = len(cats) == 0
        precision_ok += clean
        print(f"  {'OK ' if clean else 'FALSE+'} '{sent}' -> {cats or '(none)'}")

    print(f"\nrecall {recall_ok}/{len(SHOULD_RESOLVE)}   precision {precision_ok}/{len(SHOULD_NOT_RESOLVE)}")
    assert recall_ok >= len(SHOULD_RESOLVE) - 1, "zero-shot recall regressed"
    assert precision_ok == len(SHOULD_NOT_RESOLVE), "zero-shot precision regressed (false positives)"
    print("OK")


if __name__ == "__main__":
    run()
