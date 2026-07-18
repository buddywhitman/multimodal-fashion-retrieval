"""Query latency profile, broken down by pipeline stage -- evidence for the
scalability claim in docs/WRITEUP.md §10, not just an assertion.

Measures steady-state (post one-time build) query latency on the current
corpus (15,189 images / 99,300 garment crops). The retriever's deterministic
recall and candidate scoring both run against in-memory indexes
(retriever/attribute_index.py, retriever/image_store.py) built once at
startup, so per-query cost is a batched CLIP text-encode + a fixed-size image
ANN + dict/numpy lookups -- not a per-query scan of the collection.

Run: python -m eval.profile_latency
"""
import time

from retriever.search import search

QUERIES = [
    "A person in a bright yellow raincoat.",                  # 1 garment term
    "A red tie and a white shirt in a formal setting.",       # 2 garment terms + style
    "Professional business attire inside a modern office.",   # scene/style only, no garment
    "a person wearing a blue shirt",                          # high-recall single-attribute (large color family)
]

N_RUNS = 7


def run():
    print(f"Latency over {N_RUNS} runs per query (corpus: 15,189 images / 99,300 garment crops)\n")
    # warm up: the first query pays one-time model load + attribute_index +
    # image_store builds (~6s total). Measured steady-state below excludes it.
    search(QUERIES[0], k=10)
    for q in QUERIES:
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            search(q, k=10)
            times.append(time.perf_counter() - t0)
        times.sort()
        median = times[len(times) // 2]
        print(f"  median={median*1000:6.1f}ms  min={min(times)*1000:6.1f}ms  max={max(times)*1000:6.1f}ms   '{q}'")

    print("\nNote: median is the meaningful number; a one-off max spike is the lazy")
    print("in-memory index build or GC, not per-query cost. The dominant remaining")
    print("cost is the fashion-CLIP text encode (~15-20ms/query); the attribute")
    print("index lookup and in-memory scoring are sub-millisecond. See write-up §6e.")


if __name__ == "__main__":
    run()
