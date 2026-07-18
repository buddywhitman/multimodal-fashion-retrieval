"""In-memory inverted index: (category, color) -> set of image ids.

The retriever's deterministic "exact recall" step needs, per parsed
(category, color-family): every image containing a garment of that category
in that color family. Doing this as a Chroma `where`-filter (`{"$and":
[{"category": cat}, {"color": {"$in": [~100 family colors]}}]}`) costs
~160ms/query because Chroma scans the full 99k-garment collection for the
match -- and that scan cost is independent of how the filter is written
(measured: a scalar-field match and a Python-side filter both cost the same,
since the scan, not the filter evaluation, dominates).

Building the same lookup once in memory (paged load of the garment metadata,
~4s one-time) turns each per-query lookup into dict access: measured 1.08ms
vs. ~160ms, a ~150x reduction on the single dominant latency stage. Memory is
trivial (~7k keys, 99k id entries). For a long-running service the build cost
amortizes to nothing; for the eval suite it happens once per process.

Kept separate from search.py so the index build is lazy and cached, and so
the retriever degrades gracefully -- if for some reason it can't build (empty
collection), callers can fall back to a Chroma filter.
"""
from collections import defaultdict

import chromadb

from common.config import CHROMA_DIR, GARMENT_COLLECTION

_PAGE = 5000  # stays under Chroma's SQL-variable cap for a single get()

_by_cat_color = None   # {(category, color): set(image_id)}
_by_cat = None         # {category: set(image_id)}


def _build():
    global _by_cat_color, _by_cat
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    gar = client.get_collection(GARMENT_COLLECTION)
    n = gar.count()

    by_cat_color = defaultdict(set)
    by_cat = defaultdict(set)
    offset = 0
    while offset < n:
        batch = gar.get(limit=_PAGE, offset=offset, include=["metadatas"])
        for m in batch["metadatas"]:
            iid = str(m["image_id"])
            by_cat_color[(m["category"], m["color"])].add(iid)
            by_cat[m["category"]].add(iid)
        offset += _PAGE

    _by_cat_color = by_cat_color
    _by_cat = by_cat


def _ensure():
    if _by_cat_color is None:
        _build()


def image_ids_for(category, colors=None):
    """Image ids of every garment of `category` whose color is in `colors`
    (an iterable of color names -- typically a whole chromatic family). If
    `colors` is None, every image with a garment of that category."""
    _ensure()
    if colors is None:
        return set(_by_cat.get(category, set()))
    ids = set()
    for c in colors:
        ids |= _by_cat_color.get((category, c), set())
    return ids
