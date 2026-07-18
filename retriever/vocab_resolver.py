"""Zero-shot resolution of unknown garment words to known categories.

The hardcoded CATEGORY_SYNONYMS map in query_parser.py covers common words but
is by definition a closed list — "windbreaker", "parka", "loafers", "gown" not
already listed would lose their compositional signal. This resolves such words
the same way the rest of the system works: by fashion-CLIP text similarity,
making the *parser itself* zero-shot for garment types.

The hard part is precision, not recall: fashion-CLIP happily returns a
high "a photo of a {w}" similarity for non-garment words too ("near", "person",
"standing" all score as high as real garments against category prototypes —
verified in eval/test_parser.py). A plain similarity threshold therefore
injects false garment constraints and *hurts* retrieval.

Fix: a two-prototype zero-shot classifier. Embed the bare candidate word and
compare its best similarity to the garment categories against its best
similarity to a bank of non-garment concept prototypes ("a place", "an
action", "a person", "a time of day"...). Accept only when garments win by a
margin. This cleanly separates real garment words (which are more garment-like
than place/action/person-like) from filler words (which aren't) — see the
calibration in eval/test_parser.py.

Kept separate from query_parser so the common fast path (words already in the
dict) never touches the model; the resolver only loads it on an actual miss,
and caches all prototype embeddings across calls.
"""
import numpy as np

from indexer.dataset import category_vocab
from indexer.embed import embed_text

# Margin by which garment-similarity must beat distractor-similarity.
# Calibrated in eval/test_parser.py: real garment synonyms clear +0.05
# comfortably; the hardest filler words ("near", "park") top out at ~+0.03.
GARMENT_MARGIN = 0.05

# non-garment concept prototypes the candidate is scored against
_DISTRACTORS = [
    "a place or location", "an action or activity", "a person", "a time of day",
    "a feeling or mood", "the weather", "a building", "a direction", "a color",
]

# cheap guard: words that are never garments, skipped before any embedding
_STOPWORDS = {
    "a", "an", "the", "person", "someone", "wearing", "worn", "with", "and",
    "in", "on", "at", "for", "of", "to", "her", "his", "their", "outfit",
    "look", "day", "walk", "walking", "sitting", "standing", "inside", "near",
}

_cat_names = None
_cat_emb = None
_dis_emb = None


def _ensure():
    global _cat_names, _cat_emb, _dis_emb
    if _cat_names is None:
        _cat_names = category_vocab()
        _cat_emb = embed_text([f"a photo of a {c.split(',')[0]}" for c in _cat_names])
        _dis_emb = embed_text(_DISTRACTORS)


def resolve_unknown_garments(words, known_indices):
    """words: token list. known_indices: positions already matched. Returns
    list of (index, category) for words that resolve as garments."""
    candidates = [(i, w) for i, w in enumerate(words)
                  if i not in known_indices and len(w) >= 4 and w not in _STOPWORDS]
    if not candidates:
        return []

    _ensure()
    embs = embed_text([w for _, w in candidates])  # bare words, [K, D]
    cat_sims = embs @ _cat_emb.T   # [K, C]
    dis_sims = embs @ _dis_emb.T   # [K, D]

    resolved = []
    for crow, drow, (idx, _w) in zip(cat_sims, dis_sims, candidates):
        best = int(crow.argmax())
        if float(crow[best]) - float(drow.max()) >= GARMENT_MARGIN:
            resolved.append((idx, _cat_names[best]))
    return resolved
