"""Zero-shot scene, style & weather tagging — one generic mechanism, three axes.

Each image's CLIP embedding is cosine-matched against a small prompt bank
(config.SCENE_PROMPTS / STYLE_PROMPTS / WEATHER_PROMPTS). The top label per
axis is stored as metadata, giving the retriever a structured handle on
queries that carry location/style/weather language but no garment/color term
at all ("...modern office", "casual weekend...city walk", "a rainy day").

Adding a new axis (e.g. a future city/place-type bank) is "add a prompt dict
to config.py and one line here" — the tagging and scoring machinery is shared.
"""
import numpy as np

from indexer.embed import embed_text

_cache = {}  # id(prompts dict) -> (keys, embeddings)


def _prompt_embeddings(prompts: dict):
    key = id(prompts)
    if key not in _cache:
        keys = list(prompts.keys())
        _cache[key] = (keys, embed_text([prompts[k] for k in keys]))
    return _cache[key]


def zero_shot_tag(image_embeddings: np.ndarray, prompts: dict):
    """image_embeddings [N, D] normalized -> list of top label per image for this axis."""
    keys, prompt_emb = _prompt_embeddings(prompts)
    sims = image_embeddings @ prompt_emb.T  # [N, P]
    return [keys[i] for i in sims.argmax(axis=1)]


def tag_images(image_embeddings, scene_prompts, style_prompts, weather_prompts):
    """Convenience wrapper: tag all three axes at once.

    Returns list of (scene, style, weather) tuples, one per image.
    """
    scenes = zero_shot_tag(image_embeddings, scene_prompts)
    styles = zero_shot_tag(image_embeddings, style_prompts)
    weathers = zero_shot_tag(image_embeddings, weather_prompts)
    return list(zip(scenes, styles, weathers))
