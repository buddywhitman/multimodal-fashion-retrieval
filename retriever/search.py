"""Part B: The Retriever — compositional hybrid search.

Pipeline per query:
  1. IMAGE search: top image-level matches by fashion-CLIP similarity
     (carries scene / context / overall vibe).
  2. GARMENT search: for each parsed (category, color) sub-query, ANN-search the
     garment-crop collection and pull the *parent images* into the candidate
     set. This is the recall half of the compositionality fix — it surfaces
     images where the target garment is small/off-center and the whole-image
     vector missed it, without scanning the whole corpus (matters at 1M).
  3. SCORE every candidate:
       image_sim   — fashion-CLIP whole-image cosine
       comp_score  — mean over sub-queries of the best *exact* (category, color)
                     match among that image's garments, with a bonus if the
                     query also asked for a layering relation ("a jacket over
                     a shirt") and the image's bbox-derived layered-pairs
                     metadata confirms those two garments co-occur in the
                     same body region. Exact, instance-scoped attributes are
                     what keep "red tie + white shirt" from scoring the same
                     as "white tie + red shirt" (the precision half of the fix).
       tag_score   — does the image's zero-shot scene/style/weather tag match
                     the query's parsed scene/style/weather words
  4. Blend with config weights, renormalized over whichever signals the query
     actually produced (a pure-scene query isn't taxed for lacking garments).

If parsing yields nothing, this degrades to pure image similarity — the hybrid
machinery never blocks a zero-shot query, it only sharpens one it understands.

Text embedding is batched, not scattered across N sequential model calls. A
query like "a red tie and a white shirt in a formal setting" needs the main
query embedding, a zero-shot-resolution embedding ("setting"), and two
garment sub-query embeddings ("a photo of a red tie", "a photo of a white
shirt") -- four separate CLIP forward passes if done naively. Measured on
this CPU-only backbone: 4 sequential single-item calls took ~44ms median: one
batched 4-item call took ~14ms (~3x) -- the per-call Python/tokenizer/model
overhead dominates at this batch size, not the matmuls. So this module embeds
the query + any zero-shot candidates together (query_parser.tokenize +
vocab_resolver.classify_candidates take the embeddings rather than fetching
their own), and separately batches all garment sub-query texts into one call.
"""
import numpy as np

from common.colors import family_members, same_family
from common.config import (
    CANDIDATE_POOL, CHROMA_DIR, GARMENT_COLLECTION, GARMENT_POOL,
    IMAGE_COLLECTION, W_CLIP, W_COMP, W_SCENE,
)
from indexer.embed import embed_text
from retriever import attribute_index as attr_index
from retriever import image_store
from retriever.query_parser import finalize, garment_query_text, tokenize
from retriever.vocab_resolver import classify_candidates

import chromadb

_client = None


def _cols():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client.get_collection(IMAGE_COLLECTION), _client.get_collection(GARMENT_COLLECTION)


def _exact_match(subq, pairs):
    """Best match of one (cat, color) sub-query against an image's garments.

    1.0  exact category + exact color label
    0.8  exact category + color in the same chromatic family (e.g. query
         "blue", garment labeled "denim" or "navy" -- see common/colors.py)
    0.6  exact category, unrelated/no color match
    """
    qcat, qcolor = subq
    best = 0.0
    for p in pairs.split("|"):
        if "::" not in p:
            continue
        cat, col = p.split("::", 1)
        if cat != qcat:
            continue
        if qcolor is None:
            best = max(best, 1.0)
        elif qcolor == col:
            best = max(best, 1.0)
        elif same_family(qcolor, col):
            best = max(best, 0.8)
        else:
            best = max(best, 0.6)
    return best


def _relation_match(relation, relations_str, relations_directed_str):
    """relation: {"pair": (catA, catB), "over": cat_or_None} from
    query_parser.finalize(). Returns a score: 0 no match, 0.6 co-occurrence
    only (undirected -- the pair layers together, but the query either didn't
    specify direction or we couldn't verify it), 1.0 confirmed direction
    (indexer/depth_relations.py's real z-order agrees with the query)."""
    if relation is None:
        return 0.0

    pair_hit = False
    for p in relations_str.split("|"):
        if "::" not in p:
            continue
        a, b = p.split("::", 1)
        if tuple(sorted((a, b))) == relation["pair"]:
            pair_hit = True
            break
    if not pair_hit:
        return 0.0
    if relation["over"] is None:
        return 0.6

    for p in relations_directed_str.split("|"):
        if ">" not in p:
            continue
        over, under = p.split(">", 1)
        if {over, under} == set(relation["pair"]) and over == relation["over"]:
            return 1.0
    return 0.6  # co-occurs, but direction unverified or the query's asserted direction doesn't match


def search(query: str, k: int = 10, fuzzy_garment: bool = False):
    img_col, gar_col = _cols()

    # ---- 0. parse, batching the query embedding with any zero-shot
    # vocabulary candidates instead of embedding them separately ----
    words, tokens, zs_candidates = tokenize(query)
    batch_texts = [query] + [w for _, w in zs_candidates]
    batch_embs = embed_text(batch_texts)
    q_emb = batch_embs[0]
    resolved_zs = classify_candidates(zs_candidates, batch_embs[1:]) if zs_candidates else []
    parsed = finalize(words, tokens, resolved_zs)

    # ---- 1. image-level candidate pool ----
    img_res = img_col.query(
        query_embeddings=[q_emb.tolist()],
        n_results=min(CANDIDATE_POOL, img_col.count()),
        include=["metadatas", "distances"],
    )
    candidates = {}  # image_id -> {meta, image_sim}
    for cid, meta, dist in zip(img_res["ids"][0], img_res["metadatas"][0], img_res["distances"][0]):
        candidates[cid] = {"meta": meta, "image_sim": 1.0 - dist}

    # ---- 2. garment-level recall: pull parent images of matching garments ----
    #  (a) EXACT recall on (category, color-family) via an in-memory inverted
    #      index (retriever/attribute_index.py). This is deterministic recall:
    #      the (category, color) attributes are ground truth from Fashionpedia's
    #      segmentation masks, so there's no reason to depend on CLIP-crop-
    #      embedding ranking a small crop highly enough to surface it (verified
    #      failure mode: a torso "shirt, blouse" crop ranked 142nd of 7800 for
    #      its own correct text query). The index replaced a ~160ms Chroma
    #      `$in`-over-~100-family-colors scan with a ~1ms dict lookup (§6e of
    #      the write-up).
    #  (b) OPTIONAL fuzzy ANN search on garment-crop embeddings, for recall
    #      beyond exact family matches. Measured to add negligibly to benchmark
    #      recall (P@5 0.812 vs 0.810 with/without) because (a) already pulls
    #      every family true positive, while costing ~120ms/query -- so it's
    #      off by default and kept only as an escape hatch.
    for (cat, color) in parsed["garments"]:
        family = family_members(color) if color is not None else None
        for cid in attr_index.image_ids_for(cat, family):
            candidates.setdefault(cid, {"meta": None, "image_sim": None})

    if fuzzy_garment and parsed["garments"]:
        garment_texts = [garment_query_text(cat, color) for cat, color in parsed["garments"]]
        garment_embs = embed_text(garment_texts)
        for sub_emb in garment_embs:
            gres = gar_col.query(
                query_embeddings=[sub_emb.tolist()],
                n_results=min(GARMENT_POOL, gar_col.count()),
                include=["metadatas"],
            )
            for meta in gres["metadatas"][0]:
                candidates.setdefault(str(meta["image_id"]), {"meta": None, "image_sim": None})

    # fill missing image metadata/sim for garment-sourced candidates from the
    # in-memory image store (retriever/image_store.py) -- replaces a per-query
    # Chroma get() of ~500 images (metadata + embeddings) with instant dict
    # lookups + a numpy dot (§6e of the write-up).
    for cid, c in candidates.items():
        if c["meta"] is None:
            c["meta"] = image_store.get_meta(cid)
            c["image_sim"] = image_store.image_sim(cid, q_emb)

    # ---- 3. score ----
    relation = parsed["relation"]
    has_comp = len(parsed["garments"]) > 0
    has_tag = bool(parsed["scenes"] or parsed["styles"] or parsed["weathers"])
    w_clip, w_comp, w_tag = W_CLIP, W_COMP * has_comp, W_SCENE * has_tag
    total_w = w_clip + w_comp + w_tag

    results = []
    for cid, c in candidates.items():
        meta = c["meta"]
        if meta is None:
            continue
        image_sim = c["image_sim"] if c["image_sim"] is not None else 0.0

        comp_score = 0.0
        if has_comp:
            comp_score = float(np.mean([_exact_match(g, meta["pairs"]) for g in parsed["garments"]]))
            if relation is not None:
                rel_score = _relation_match(relation, meta.get("relations", ""),
                                             meta.get("relations_directed", ""))
                # 0.6 co-occurs undirected, 1.0 confirmed z-order -- scaled to
                # a max +0.2 bonus so it sharpens ranking without dominating
                # the per-garment attribute match this sits on top of
                comp_score = min(1.0, comp_score + 0.2 * rel_score)

        tag_score = 0.0
        if has_tag:
            hits = ((meta.get("scene") in parsed["scenes"]) + (meta.get("style") in parsed["styles"])
                    + (meta.get("weather") in parsed["weathers"]))
            denom = (len(parsed["scenes"]) > 0) + (len(parsed["styles"]) > 0) + (len(parsed["weathers"]) > 0)
            tag_score = hits / denom if denom else 0.0

        final = (w_clip * image_sim + w_comp * comp_score + w_tag * tag_score) / total_w
        results.append({
            "image_id": cid,
            "file_name": meta["file_name"], "path": meta["path"],
            "categories": meta["categories"].split("|") if meta["categories"] else [],
            "colors": meta["colors"].split("|") if meta["colors"] else [],
            "scene": meta.get("scene"), "style": meta.get("style"), "weather": meta.get("weather"),
            "image_sim": round(image_sim, 4), "comp_score": round(comp_score, 4),
            "tag_score": round(tag_score, 4), "score": round(final, 4),
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k], parsed
