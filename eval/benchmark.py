"""Broader retrieval benchmark, grounded in the corpus itself rather than 5
hand-picked prompts.

Method: for every (category, color) combo that actually occurs >= MIN_SUPPORT
times in the corpus, synthesize a natural-language query ("a person wearing a
{color} {category}") and check whether the system finds the known true-
positive images. Ground truth here is derived from Fashionpedia's own
segmentation+category labels (not hand-labeled by us), so this measures real
retrieval quality across dozens of query points instead of 5 anecdotes — and
because these exact (category, color) *strings* are never fed to the model as
training labels, finding them from a synthesized sentence is a genuine
zero-shot-generalization test, not a lookup.

Two reported slices, because Fashionpedia's 46 categories mix actual garment
types (coat, dress, jacket...) with small parts/embellishments (rivet, bead,
sequin, epaulette...) that aren't "clothing types" in the PRD's sense at all —
averaging them together would understate performance on the thing the
assignment actually asks about:
  GARMENT  — the wearable clothing-type categories (what a query realistically asks for)
  PART     — hardware/decoration details (embellishment recognition, a much harder task)

"unknown"-colored instances (mask rasterization failed — crowd/RLE or
degenerate polygons) are excluded: they're a labeling artifact, not a query
anyone would type.

Ground truth is read straight from the already-built index
(eval/ground_truth.py) rather than recomputed from raw images -- ~0.3s
instead of several minutes, since indexer/build_index.py already did this
work once at index time.

Run: python -m eval.benchmark
"""
from collections import defaultdict

from eval.ground_truth import ground_truth_pairs as _indexed_ground_truth_pairs
from retriever.search import search

MIN_SUPPORT = 3
K = 5

GARMENT_CATEGORIES = {
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket",
    "vest", "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape",
    "tie", "hat", "shoe", "bag, wallet", "glasses", "scarf", "glove", "sock",
    "tights, stockings", "headband, head covering, hair accessory", "belt", "umbrella",
}


def ground_truth_pairs():
    """(category, color) -> set of file_names that truly contain that combo.

    An instance can contribute more than one color (see
    indexer/color_extract.py -- ~25% of garments show real internal color
    spread), so ground truth matches what's actually indexed: a combo is a
    true positive if *either* color matches, exactly like the indexed
    (category, color) pairs used at query time.
    """
    return _indexed_ground_truth_pairs(exclude_unknown=True)


def run():
    truth = ground_truth_pairs()
    combos = [(pair, files) for pair, files in truth.items() if len(files) >= MIN_SUPPORT]
    combos.sort(key=lambda x: -len(x[1]))
    print(f"{len(combos)} (category, color) combos with >= {MIN_SUPPORT} true positives\n")

    by_slice = defaultdict(lambda: ([], []))  # slice -> (precisions, recalls)
    for (cat, color), true_files in combos:
        head = cat.split(",")[0]
        query = f"a person wearing a {color} {head}"
        results, _ = search(query, k=K)
        retrieved = {r["file_name"] for r in results}
        hits = retrieved & true_files

        p = len(hits) / K
        r = len(hits) / min(K, len(true_files))
        slice_name = "GARMENT" if cat in GARMENT_CATEGORIES else "PART"
        by_slice[slice_name][0].append(p)
        by_slice[slice_name][1].append(r)
        by_slice["ALL"][0].append(p)
        by_slice["ALL"][1].append(r)
        print(f"  [{slice_name:7}] P@{K}={p:.2f} R@{K}={r:.2f}  n_true={len(true_files):>3}  '{query}'")

    print()
    for slice_name in ["GARMENT", "PART", "ALL"]:
        ps, rs = by_slice[slice_name]
        if ps:
            print(f"  {slice_name:7} (n={len(ps):>3}):  mean P@{K}={sum(ps)/len(ps):.3f}  mean R@{K}={sum(rs)/len(rs):.3f}")


if __name__ == "__main__":
    run()
