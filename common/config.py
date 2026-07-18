"""Shared paths and constants. Single source of truth for both indexer and retriever."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
IMAGE_DIR = os.path.join(RAW_DIR, "val_test2020")  # unzipped images live here
ANNOTATIONS_PATH = os.path.join(RAW_DIR, "instances_attributes_val2020.json")

# The PRD names Fashionpedia as "one such dataset", not a mandate to use only
# its small val2020 annotated slice (1158 images). Fashionpedia's much larger
# train2020 split -- same format, same license, ~45k annotated images -- is
# used here specifically to guarantee the 5 PRD eval queries have real ground
# truth (val2020 alone has zero yellow coats and zero red ties; train2020 has
# 3124 coat instances and 1457 tie instances to draw from). Optional: code
# degrades gracefully (falls back to val2020-only) if these aren't present.
TRAIN_IMAGE_DIR = os.path.join(RAW_DIR, "train2020")
TRAIN_ANNOTATIONS_PATH = os.path.join(RAW_DIR, "instances_attributes_train2020.json")
# guaranteed-coverage categories: every train2020 image containing one of
# these is included, regardless of the random sample below
TRAIN_PRIORITY_CATEGORIES = {"coat", "tie"}
# plus a random sample of other train2020 images, for general corpus depth
# (not just the 2 categories the PRD eval queries happen to need) -- capped
# to keep indexing time tractable (GPU embedding + cropped color extraction
# together run at roughly 35-40ms/image, so ~12k images is ~7-8 minutes)
TRAIN_RANDOM_SAMPLE_SIZE = 7500
TRAIN_SAMPLE_SEED = 0

CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
IMAGE_COLLECTION = "fashionpedia_images"      # whole-image embeddings (scene/style/context)
GARMENT_COLLECTION = "fashionpedia_garments"  # per-garment crop embeddings (compositionality)
CATEGORY_CACHE_PATH = os.path.join(DATA_DIR, "categories.json")

# --- Model -----------------------------------------------------------------
# Fashion-domain CLIP: trained on ~800k fashion image/text pairs, so it
# grounds garment types, colors, and styles far better than generic CLIP.
# embed.py is the only module that touches the model. CLIP_BACKEND matters:
# checkpoints are packaged either HuggingFace-native ("transformers") or
# OpenCLIP-native ("open_clip", loaded via the "hf-hub:" convention) --
# verified directly that loading an OpenCLIP-packaged checkpoint through
# transformers.CLIPModel silently produces an untrained model (hundreds of
# MISSING/UNEXPECTED weight keys), so getting this right isn't optional.
#
# A larger checkpoint (Marqo/marqo-fashionCLIP, "open_clip" backend -- both
# paths are implemented and tested in embed.py) was directly measured
# against this one on eval/compare_backbones.py's color-swap discrimination
# task (the metric that matters most for this project) and did NOT
# outperform it here: 0.633 vs 0.650 accuracy, within noise for n=60 --
# despite Marqo's own reported +57% benchmark improvement on their eval set,
# which evidently doesn't transfer to this corpus/task. Kept the original
# checkpoint on that evidence rather than switching on a vendor's benchmark
# claim; the dual-backend infrastructure this investigation produced is a
# real, tested capability for whichever future checkpoint someone tries.
CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"
CLIP_BACKEND = "transformers"
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
# Chosen by ablation (eval/ablate_weights.py), re-run on the merged 15k-image
# corpus. The bigger, more diverse corpus makes CLIP's whole-image similarity
# noisier relative to the exact (category, color) match (ground truth from
# segmentation masks), so the optimum shifted further toward W_COMP: a sweep
# of 0.35→0.10 W_CLIP showed garment P@5 rising 0.785→0.805 and plateauing at
# W_CLIP=0.20 (no further gain from 0.15 or 0.10). Picked the plateau knee
# rather than pushing W_CLIP lower, to keep CLIP weight at 2x W_SCENE as a
# robustness margin for pure-scene queries (where W_COMP zeroes out and the
# W_CLIP:W_SCENE ratio is all that's left) -- verified pure-scene ranking is
# unchanged across the whole 0.35→0.10 range. CLIP-heavy (0.70) collapsed to
# 0.433 on this corpus, confirming the direction.
W_CLIP = 0.20
W_COMP = 0.70
W_SCENE = 0.10

CANDIDATE_POOL = 60      # image-level ANN pool
GARMENT_POOL = 40        # per-sub-query garment ANN pool (for recall)
