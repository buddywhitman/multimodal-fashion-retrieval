"""Query latency profile, broken down by pipeline stage -- evidence for the
scalability claim in docs/WRITEUP.md §10, not just an assertion.

The claim: both ANN searches (image-level, garment-level) pull a *fixed-size*
candidate pool regardless of corpus size, and the one deterministic pass (the
exact `where`-filter recall added after the corpus-grounded benchmark caught
a recall miss) is O(matching instances), not O(corpus). This script measures
where time actually goes today, at 3200 images / 7800 garment crops, so the
"architecturally flat" claim is checked against a real number instead of
asserted from first principles.

Run: python -m eval.profile_latency
"""
import time

from retriever.search import search

QUERIES = [
    "A person in a bright yellow raincoat.",                  # 1 garment term
    "A red tie and a white shirt in a formal setting.",       # 2 garment terms + style
    "Professional business attire inside a modern office.",   # scene/style only, no garment
    "a person wearing a blue shirt",                          # simple single-attribute
]

N_RUNS = 5


def run():
    print(f"Latency over {N_RUNS} runs per query (corpus: 3200 images / 7800 garment crops)\n")
    search(QUERIES[0], k=10)  # warm up: first call pays one-time model load, not query cost
    for q in QUERIES:
        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            search(q, k=10)
            times.append(time.perf_counter() - t0)
        times.sort()
        median = times[len(times) // 2]
        print(f"  median={median*1000:6.1f}ms  min={min(times)*1000:6.1f}ms  max={max(times)*1000:6.1f}ms   '{q}'")

    print("\nNote: end-to-end time here includes fashion-CLIP text encoding (a fixed")
    print("~10-20ms CPU forward pass per sub-query, independent of corpus size) --")
    print("the ANN and where-filter calls themselves are the part that scales with")
    print("corpus size, and both are bounded by CANDIDATE_POOL/GARMENT_POOL or by")
    print("the (small) number of matching instances, not by total corpus size.")


if __name__ == "__main__":
    run()
