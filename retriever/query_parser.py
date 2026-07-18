"""Decompose a free-text query into structured signals:

  garments : list[(category, color_or_None)]   -> per-garment (compositional) search
  scenes   : list[scene_label]                 -> scene re-rank
  styles   : list[style_label]                 -> style re-rank
  weathers : list[weather_label]                -> weather re-rank
  relation : (category_a, category_b) or None   -> layered-garment re-rank

Deliberately a shallow keyword scan, not an LLM call. It only ever *adds*
re-ranking signal on top of CLIP; when it extracts nothing (a truly novel
description), the retriever falls back to pure image similarity. So the parser
can miss without breaking zero-shot — it just stops contributing a boost.
"""
import json
import re

from common.colors import PALETTE, canonical_color
from common.config import CATEGORY_CACHE_PATH

# words people type -> Fashionpedia category label strings
CATEGORY_SYNONYMS = {
    "shirt": "shirt, blouse", "blouse": "shirt, blouse",
    "t-shirt": "top, t-shirt, sweatshirt", "tshirt": "top, t-shirt, sweatshirt",
    "top": "top, t-shirt, sweatshirt", "sweatshirt": "top, t-shirt, sweatshirt",
    "hoodie": "top, t-shirt, sweatshirt",
    "coat": "coat", "raincoat": "coat", "overcoat": "coat", "trench": "coat",
    "jacket": "jacket", "blazer": "jacket",
    "vest": "vest", "waistcoat": "vest",
    "pants": "pants", "trousers": "pants", "jeans": "pants", "chinos": "pants",
    "shorts": "shorts", "skirt": "skirt", "dress": "dress", "gown": "dress",
    "tie": "tie", "necktie": "tie", "bowtie": "tie",
    "glasses": "glasses", "sunglasses": "glasses",
    "hat": "hat", "cap": "hat", "beanie": "hat",
    "shoe": "shoe", "shoes": "shoe", "sneakers": "shoe", "boots": "shoe",
    "bag": "bag, wallet", "purse": "bag, wallet",
    "sweater": "sweater", "jumper": "sweater", "cardigan": "cardigan",
    "scarf": "scarf", "belt": "belt", "glove": "glove", "gloves": "glove",
    "headband": "headband, head covering, hair accessory",
    "tights": "tights, stockings", "stockings": "tights, stockings", "stocking": "tights, stockings",
    "jumpsuit": "jumpsuit", "cape": "cape", "umbrella": "umbrella",
}

# scene/style trigger words -> tag labels used at index time
SCENE_KEYWORDS = {
    "office": "office", "workplace": "office", "meeting": "office",
    "street": "street", "city": "street", "urban": "street", "sidewalk": "street",
    "park": "park", "garden": "park", "bench": "park", "outdoors": "park",
    "home": "home", "bedroom": "home", "indoors": "home",
    "runway": "runway", "catwalk": "runway", "studio": "studio",
}
STYLE_KEYWORDS = {
    "formal": "formal", "business": "formal", "professional": "formal",
    "suit": "formal", "elegant": "formal",
    "casual": "casual", "weekend": "casual", "everyday": "casual", "relaxed": "casual",
    "coat": "outerwear", "raincoat": "outerwear", "jacket": "outerwear",
    "sport": "sporty", "sporty": "sporty", "athletic": "sporty", "gym": "sporty",
}
WEATHER_KEYWORDS = {
    "rain": "rainy", "rainy": "rainy", "raincoat": "rainy", "umbrella": "rainy", "wet": "rainy",
    "sunny": "sunny", "sunshine": "sunny", "clear": "sunny",
    "cold": "cold", "winter": "cold", "snow": "cold", "snowy": "cold", "chilly": "cold",
}
# phrases indicating two garments are worn together/layered (see
# indexer/relations.py for what this can and can't actually tell you)
RELATION_KEYWORDS = {"layered", "layering", "over", "under", "tucked", "untucked"}


def _load_categories():
    with open(CATEGORY_CACHE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def parse(query: str, zero_shot_vocab: bool = True):
    text = query.lower()
    words = re.findall(r"[a-z][a-z\-]*", text)
    cat_vocab = set(_load_categories())

    tokens = []  # (index, kind, value)
    matched_idx = set()
    for i, w in enumerate(words):
        wc = canonical_color(w)
        if wc in PALETTE or w in PALETTE:
            tokens.append((i, "color", canonical_color(wc)))
        if w in CATEGORY_SYNONYMS:
            tokens.append((i, "category", CATEGORY_SYNONYMS[w]))
            matched_idx.add(i)
        elif w in cat_vocab:
            tokens.append((i, "category", w))
            matched_idx.add(i)

    # any word already classified as scene/style/weather/relation vocabulary
    # is not a candidate for garment resolution -- computed here (before the
    # resolver call) specifically so the resolver doesn't waste an embedding
    # call re-litigating "office" or "layered" as a possible garment.
    non_garment_idx = {i for i, w in enumerate(words)
                        if w in SCENE_KEYWORDS or w in STYLE_KEYWORDS
                        or w in WEATHER_KEYWORDS or w in RELATION_KEYWORDS}

    # zero-shot fallback: resolve garment words the dict missed, by embedding
    # similarity to known categories (retriever/vocab_resolver.py). Only runs
    # when there's an unmatched word to resolve, so the common path stays fast.
    if zero_shot_vocab:
        from retriever.vocab_resolver import resolve_unknown_garments
        known = {i for i, _kind, _val in tokens} | non_garment_idx
        for idx, cat in resolve_unknown_garments(words, known):
            tokens.append((idx, "category", cat))

    # pair each category with the nearest preceding-or-adjacent color word
    garments, used = [], set()
    for idx, kind, val in tokens:
        if kind != "category":
            continue
        best, best_dist = None, 4
        for cidx, ckind, cval in tokens:
            if ckind == "color" and cidx not in used and 0 <= idx - cidx < best_dist:
                best, best_dist = (cidx, cval), idx - cidx
        if best:
            used.add(best[0])
            garments.append((val, best[1]))
        else:
            garments.append((val, None))

    scenes = sorted({SCENE_KEYWORDS[w] for w in words if w in SCENE_KEYWORDS})
    styles = sorted({STYLE_KEYWORDS[w] for w in words if w in STYLE_KEYWORDS})
    weathers = sorted({WEATHER_KEYWORDS[w] for w in words if w in WEATHER_KEYWORDS})

    relation = None
    if any(w in RELATION_KEYWORDS for w in words) and len(garments) >= 2:
        relation = tuple(sorted((garments[0][0], garments[1][0])))

    return {"garments": garments, "scenes": scenes, "styles": styles,
            "weathers": weathers, "relation": relation}


def garment_query_text(category, color):
    """A natural sub-query for garment-crop search, e.g. 'a photo of a red tie'."""
    head = category.split(",")[0]
    return f"a photo of a {color} {head}" if color else f"a photo of a {head}"
