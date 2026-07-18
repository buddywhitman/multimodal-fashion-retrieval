"""Shared paths and constants. Single source of truth for both indexer and retriever."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
IMAGE_DIR = os.path.join(RAW_DIR, "val_test2020")  # unzipped images live here
ANNOTATIONS_PATH = os.path.join(RAW_DIR, "instances_attributes_val2020.json")

CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
IMAGE_COLLECTION = "fashionpedia_images"      # whole-image embeddings (scene/style/context)
GARMENT_COLLECTION = "fashionpedia_garments"  # per-garment crop embeddings (compositionality)
CATEGORY_CACHE_PATH = os.path.join(DATA_DIR, "categories.json")

# --- Model -----------------------------------------------------------------
# Fashion-domain CLIP: trained on ~800k fashion image/text pairs, so it grounds
# garment types, colors, and styles far better than general clip-ViT-B-32.
# embed.py is the only module that touches the model — swapping this to a
# stronger checkpoint (Marqo-FashionSigLIP, ViT-L variants) is a one-line change.
CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"
EMBED_BATCH = 32

# --- Region (garment crop) indexing ---------------------------------------
# Pad each garment bbox before cropping so context (collar line, adjacent
# fabric) survives — a bare-tight tie crop embeds poorly.
CROP_PAD_FRAC = 0.15
MIN_CROP_PX = 24  # skip degenerate slivers (buttons, zippers seen edge-on)

# --- Zero-shot scene / style / weather tagging -----------------------------
# Each image is tagged once at index time by cosine-matching its embedding
# against these prompt banks (indexer/scene_tag.py — one generic function,
# reused for all three axes). Gives structured signal to queries that carry
# location/style/weather language but no garment/color term at all
# ("...modern office", "casual weekend...city walk", "a rainy day").
#
# "Locations (cities, places)" from the future-work list is implemented as
# place *types*, not city names: nothing in a street-style photo reveals
# which city it was taken in, so a city-name prompt bank would just be
# fabricating structure CLIP can't actually ground. Place type (office vs.
# park vs. beach) is the granularity that's honestly inferable from pixels.
SCENE_PROMPTS = {
    "office": "a photo taken inside a modern office",
    "street": "a photo on an urban city street",
    "park": "a photo in a park or garden outdoors",
    "home": "a photo inside a home or bedroom",
    "studio": "a studio fashion photo on a plain background",
    "runway": "a fashion runway or catwalk photo",
    "beach": "a photo outdoors at a beach or by the water",
    "cafe": "a photo inside a cafe or restaurant",
    "gym": "a photo inside a gym or fitness studio",
}
STYLE_PROMPTS = {
    "formal": "formal business attire, a suit and tie",
    "casual": "casual everyday clothing, a t-shirt or hoodie",
    "outerwear": "wearing a heavy coat or outerwear",
    "sporty": "sporty athletic activewear",
}
WEATHER_PROMPTS = {
    "rainy": "a rainy day, wet pavement, someone wearing a raincoat or holding an umbrella",
    "sunny": "a bright sunny day with clear skies",
    "cold": "a cold winter day, someone bundled up in heavy outerwear",
    "indoor": "an indoor photo with no visible weather",
}

# --- Hybrid scoring ---------------------------------------------------------
# final = W_CLIP*image_sim + W_COMP*compositional + W_SCENE*scene_style_weather_match
# Weights renormalize over whichever signals a given query actually produces,
# so a pure-scene query isn't penalized for having no garment terms.
# Chosen by ablation (eval/ablate_weights.py): comp-heavy outperformed the
# naive 0.50/0.35/0.15 guess because exact (category, color) matches are
# ground truth from segmentation masks -- more trustworthy than CLIP's fuzzy
# similarity once a candidate is already known to have the right garment/color.
W_CLIP = 0.35
W_COMP = 0.55
W_SCENE = 0.10

CANDIDATE_POOL = 60      # image-level ANN pool
GARMENT_POOL = 40        # per-sub-query garment ANN pool (for recall)
