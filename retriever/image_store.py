"""In-memory image metadata + embedding store.

The retriever needs, for every candidate image, its stored metadata (the
`pairs`/`relations`/scene/style/weather fields used for scoring) and its
whole-image embedding (for the `image_sim` term). Candidates surfaced by the
attribute index (retriever/attribute_index.py) but not by the semantic ANN
pool arrive without either, so search.py used to fetch them with a per-query
`img_col.get(missing, include=["metadatas","embeddings"])` -- measured ~52ms
for a high-recall query (a "blue"-family shirt pulls ~500 candidate images).

Loading the whole image collection once into memory (1.7s one-time, ~31MB for
15k images at 512-dim float32) makes those lookups instant: a full-corpus
score is a single numpy matmul (~1.5ms for all 15k). Same tradeoff as
attribute_index.py -- trivial at this corpus size, amortized to nothing for a
long-running service; at ~1M images you'd instead rely on Chroma's
server/sharded mode (the write-up's §10 scaling story), which is a client
swap, not an architecture change.
"""
import numpy as np

import chromadb

from common.config import CHROMA_DIR, IMAGE_COLLECTION

_PAGE = 3000

_meta_by_id = None      # {image_id(str): metadata dict}
_emb_by_id = None       # {image_id(str): np.ndarray [D] float32}


def _build():
    global _meta_by_id, _emb_by_id
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    img_col = client.get_collection(IMAGE_COLLECTION)
    n = img_col.count()

    meta_by_id, emb_by_id = {}, {}
    offset = 0
    while offset < n:
        batch = img_col.get(limit=_PAGE, offset=offset, include=["metadatas", "embeddings"])
        for cid, meta, emb in zip(batch["ids"], batch["metadatas"], batch["embeddings"]):
            meta_by_id[cid] = meta
            emb_by_id[cid] = np.asarray(emb, dtype=np.float32)
        offset += _PAGE

    _meta_by_id, _emb_by_id = meta_by_id, emb_by_id


def _ensure():
    if _meta_by_id is None:
        _build()


def get_meta(image_id):
    _ensure()
    return _meta_by_id.get(image_id)


def image_sim(image_id, q_emb):
    """Cosine similarity of the stored (already L2-normalized) image embedding
    with a normalized query embedding. Returns 0.0 for an unknown id."""
    _ensure()
    emb = _emb_by_id.get(image_id)
    if emb is None:
        return 0.0
    return float(np.dot(emb, q_emb))
