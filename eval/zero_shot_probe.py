"""Qualitative zero-shot probe: queries using words that appear in NEITHER the
category/color vocabulary NOR the scene/style prompt bank (common/config.py).
These can only be answered by fashion-CLIP's raw semantic similarity -- no
structured signal in this codebase recognizes any of these words. This is
what "zero-shot capability" in the PRD's evaluation criteria actually means:
generalizing past every explicit label the system was given.

Not scored (no cheap ground truth exists for open-ended style language) --
printed for manual inspection alongside eval/run_eval_queries.py's contact
sheets.

Run: python -m eval.zero_shot_probe
"""
from retriever.search import search

PROBES = [
    "an elegant evening gown for a gala",
    "someone dressed for a rainy autumn commute",
    "a minimalist monochrome outfit",
    "streetwear with an oversized silhouette",
    "a cozy knit sweater for a winter morning",
]

K = 5


def run():
    for q in PROBES:
        results, parsed = search(q, k=K)
        print(f"\n'{q}'")
        print(f"  parsed structured signal: {parsed}  <- note how little of the query this covers")
        for r in results[:3]:
            print(f"    score={r['score']:.3f}  {r['file_name']}  scene={r['scene']} style={r['style']} cats={r['categories'][:3]}")


if __name__ == "__main__":
    run()
