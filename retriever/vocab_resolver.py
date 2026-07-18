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

Split into find_candidates() (pure string matching, no model calls) and
classify_candidates() (pure vector math given precomputed embeddings) so a
caller (retriever/search.py) can embed the candidate words in the SAME batched
model call as everything else it needs that query -- see search.py's docstring
for why that matters (measured ~3x fewer, larger, batched forward passes vs.
many small sequential ones on CPU).
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


def ensure_prototypes():
    """Load+cache the category/distractor prototype embeddings. Idempotent --
    a real model call only on the first-ever invocation in this process."""
    global _cat_names, _cat_emb, _dis_emb
    if _cat_names is None:
        _cat_names = category_vocab()
        _cat_emb = embed_text([f"a photo of a {c.split(',')[0]}" for c in _cat_names])
        _dis_emb = embed_text(_DISTRACTORS)


def find_candidates(words, known_indices):
    """Pure string filtering, no model calls. Returns list[(index, word)]
    worth embedding and classifying."""
    return [(i, w) for i, w in enumerate(words)
            if i not in known_indices and len(w) >= 4 and w not in _STOPWORDS]


def classify_candidates(candidates, embeddings):
    """candidates: list[(index, word)]; embeddings: precomputed [K, D] array
    in the same order (the caller embeds these -- typically batched together
    with other query text, see retriever/search.py). Returns list[(index,
    category)] for the ones that clear the garment-vs-distractor margin.
    Pure vector math -- no model call here."""
    if not candidates:
        return []
    ensure_prototypes()
    cat_sims = embeddings @ _cat_emb.T   # [K, C]
    dis_sims = embeddings @ _dis_emb.T   # [K, D]

    resolved = []
    for crow, drow, (idx, _w) in zip(cat_sims, dis_sims, candidates):
        best = int(crow.argmax())
        if float(crow[best]) - float(drow.max()) >= GARMENT_MARGIN:
            resolved.append((idx, _cat_names[best]))
    return resolved


def resolve_unknown_garments(words, known_indices):
    """Convenience one-shot wrapper (embeds candidates itself, one model
    call). Used by callers that don't need to batch with other text --
    e.g. query_parser.parse()'s simple/standalone path."""
    candidates = find_candidates(words, known_indices)
    if not candidates:
        return []
    embeddings = embed_text([w for _, w in candidates])
    return classify_candidates(candidates, embeddings)
