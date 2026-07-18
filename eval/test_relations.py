"""Regression test for query-side layering-direction detection
(retriever/query_parser.py's finalize()) -- the word-position logic that
turns "a jacket over a shirt" into a directed relation, not just "these two
co-occur". The depth model itself (indexer/depth_relations.py) that verifies
z-order at index time was validated manually against a real corpus image (a
jacket visibly worn over a t-shirt -- correctly ordered); this test locks in
the cheap, deterministic parsing half.

Run: python -m eval.test_relations
"""
from retriever.query_parser import parse

CASES = [
    ("a jacket layered over a shirt", ("jacket", "shirt, blouse"), "jacket"),
    ("a shirt under a jacket", ("jacket", "shirt, blouse"), "jacket"),
    ("a cardigan over a top", ("cardigan", "top, t-shirt, sweatshirt"), "cardigan"),
    ("a top under a cardigan", ("cardigan", "top, t-shirt, sweatshirt"), "cardigan"),
]

UNDIRECTED_CASES = [
    "a jacket and a shirt layered together",
]

NO_RELATION_CASES = [
    "a red tie and a white shirt in a formal setting",
    "a person in a bright yellow raincoat",
]


def run():
    ok = 0
    total = 0

    for query, expected_pair, expected_over in CASES:
        total += 1
        rel = parse(query)["relation"]
        got_ok = rel is not None and rel["pair"] == tuple(sorted(expected_pair)) and rel["over"] == expected_over
        print(f"  {'OK' if got_ok else 'MISS':4} '{query}' -> {rel}")
        ok += got_ok

    for query in UNDIRECTED_CASES:
        total += 1
        rel = parse(query)["relation"]
        got_ok = rel is not None and rel["over"] is None
        print(f"  {'OK' if got_ok else 'MISS':4} '{query}' -> {rel}  (expected undirected)")
        ok += got_ok

    for query in NO_RELATION_CASES:
        total += 1
        rel = parse(query)["relation"]
        got_ok = rel is None
        print(f"  {'OK' if got_ok else 'MISS':4} '{query}' -> {rel}  (expected None)")
        ok += got_ok

    print(f"\n{ok}/{total} passed")
    assert ok == total, "relation-direction parsing regression"


if __name__ == "__main__":
    run()
