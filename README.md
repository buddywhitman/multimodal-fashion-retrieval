# glance — Multimodal Fashion & Context Retrieval

Natural-language image search over the [Fashionpedia](https://github.com/cvdfoundation/fashionpedia) dataset:
type a description, get back matching images. Built for the Glance ML
internship assignment (see `docs/PRD.pdf`, full write-up in `docs/WRITEUP.md`).

## Architecture

- **`common/`** — shared config, the Lab-space color classifier + chromatic families (`colors.py`), and prompt banks. No logic, just constants both sides depend on.
- **`indexer/`** (Part A) — reads Fashionpedia's annotations, embeds each whole image *and* each garment crop separately with fashion-domain CLIP (`patrickjohncyh/fashion-clip`), extracts 1-2 mask-derived `(category, color)` pairs per garment instance (Lab-space nearest neighbor + a second color for patterned/color-blocked garments — ~25% of instances), derives bbox-overlap "layered" relations between garments, and zero-shot tags scene/style/weather. A second pass indexes every unannotated image on disk (whole-image + tags only), growing the corpus from 1158 → 3200. Stores everything in two persistent Chroma collections.
- **`retriever/`** (Part B) — parses the query for `(category, color)` pairs, scene/style/weather keywords, and layering phrases, resolving garment words the dictionary misses via a zero-shot embedding classifier (`vocab_resolver.py`). Does image-level + garment-level ANN search (plus an exact metadata-filter pass, expanded to the whole color family, for anything the parser resolved), and hybrid-reranks by `W_CLIP·image_sim + W_COMP·attribute_match + W_SCENE·tag_match`. Falls back to pure CLIP similarity when nothing parses — zero-shot queries are never blocked.
- **`eval/`** — coverage check, Precision@k on the 5 PRD queries, a corpus-grounded benchmark (437 queries), a multi-attribute (color+type+location) benchmark, a controlled compositional-discrimination experiment, a weight ablation (120 combos), color-classifier and parser regression tests, a zero-shot probe, and a latency profile. See `docs/WRITEUP.md` for the numbers and the four real bugs these caught.

Logic and data are fully separated: nothing in `indexer/` or `retriever/`
hardcodes a file path or a label — those come from `common/config.py` and the
annotation file itself.

## Setup

```bash
pip install -r requirements.txt
```

Dataset (already present under `data/raw/` in this repo — images in
`data/raw/val_test2020/`, annotations in
`data/raw/instances_attributes_val2020.json`). To fetch fresh:

```bash
curl -o data/raw/val_test2020.zip https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip
curl -o data/raw/instances_attributes_val2020.json https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json
unzip data/raw/val_test2020.zip -d data/raw/val_test2020_tmp
mv data/raw/val_test2020_tmp/test data/raw/val_test2020
```

## Build the index (Part A)

```bash
python -m indexer.build_index
```

## Query (Part B)

```bash
python -m retriever.cli "A person in a bright yellow raincoat." --k 5
python -m retriever.cli "a jacket layered over a shirt" --k 5
python -m retriever.cli "a person in a navy windbreaker" --k 5   # zero-shot vocab, not in any dictionary
```

## Evaluation

```bash
python -m eval.test_colors        # color-classifier regression suite (22 cases)
python -m eval.test_parser        # zero-shot vocabulary resolver precision/recall calibration
python -m eval.coverage           # does ground truth even exist for the 5 PRD queries?
python -m eval.evaluate           # Precision@5 on the 5 PRD queries, coverage-aware
python -m eval.benchmark          # Precision@5/Recall@5 over corpus-grounded queries (GARMENT/PART slices)
python -m eval.multi_attribute    # PRD's own "color + type + location" example, measured
python -m eval.compositional      # CLIP-only vs hybrid on the "red shirt/blue pants" color-swap case
python -m eval.ablate_weights     # weight sweep that justified the hybrid-scoring config
python -m eval.zero_shot_probe    # qualitative probe with words in no vocabulary at all
python -m eval.profile_latency    # per-stage query latency, evidence for the scalability claim
python -m eval.run_eval_queries   # saves labeled contact-sheet PNGs to eval/outputs/
```

All ground-truth-dependent scripts above read directly from the built index
(`eval/ground_truth.py`) instead of recomputing color extraction from raw
images — measured 1364x faster (385.8s → 0.283s) than the original
per-script recomputation, so the whole suite now runs in well under a
minute total instead of tens of minutes.

**Headline numbers** (full detail in `docs/WRITEUP.md`):
- **Compositionality** (the PRD's core hint), color-swap discrimination "red shirt+blue pants" vs "blue shirt+red pants": CLIP-only **0.65** (near chance) → hybrid **1.00**.
- **Context Awareness** (the PRD's own "color + type + location" example): adding the location term **roughly doubles precision** (+0.30 P@5, reproduced across two independent runs).
- 258 corpus-grounded GARMENT queries (not hand-picked): **mean P@5 = 0.850, R@5 = 0.926**.
- **Zero-shot parser**: novel garment words (windbreaker, parka, loafers...) resolve at **8/8 recall, 3/3 precision** with no hardcoding (`eval/test_parser.py`).
- 5 PRD eval queries: mean P@5 = 0.600 — capped by corpus coverage, not the algorithm: this corpus contains **zero** yellow coats and **zero** red ties at all (verified in `eval/coverage.py`).
- Query latency: **20-61ms** — a batching fix (embed the query + zero-shot candidates + all garment sub-queries in ≤2 model calls instead of up to N+2 sequential ones) brought latency in *below* the original 15-55ms baseline, after zero-shot vocabulary resolution and multi-color extraction had temporarily regressed it to 75-190ms. Still scales with sub-query count, not corpus size.

## Why this approach (short version — full write-up in `docs/WRITEUP.md`)

Vanilla CLIP pools a whole image into one vector, so it can't reliably tell
"red tie, white shirt" from "white tie, red shirt" — both contain the same
bag of visual concepts. Fashionpedia already ships instance-level
segmentation + category labels; this system embeds each garment crop
*separately* and re-ranks with exact `(category, color)` matching (with
graded credit for same-family colors, and a second color for patterned
garments) on top of fashion-domain CLIP's whole-image similarity, plus a
bbox-derived "layered" relation for queries like "a jacket over a shirt". On
the exact color-swap case the PRD hint names, this takes discrimination
accuracy from CLIP-only's 0.65 to 1.00. The query parser is itself zero-shot:
garment words it doesn't know ("windbreaker", "parka") are resolved through
fashion-CLIP via a two-prototype garment-vs-not classifier, so the parser
isn't the one closed-vocabulary component in the pipeline. CLIP's zero-shot
generalization is kept for anything the parser can't resolve; compositional
precision is fixed for anything it can.

## Locations & weather

Implemented as zero-shot **place-type** and **weather** tagging (`indexer/scene_tag.py`,
one generic mechanism reused for scene/style/weather), not city names — a
street-style photo doesn't honestly reveal which city it was taken in, so a
city-name prompt bank would fabricate structure CLIP can't ground. Measured
directly: adding the location term to a color+type query roughly doubles P@5
(`eval/multi_attribute.py`). Extending to finer place types or real
EXIF/geolocation metadata (where available) is a config change, not an
architecture change — see `docs/WRITEUP.md` §9a.

## Scaling to ~1M images

Both the image-level and garment-level ANN searches operate on a **fixed-size
candidate pool**, not the full corpus, so query latency scales with the
number of parsed sub-queries — not corpus size — as the dataset grows
(measured in `eval/profile_latency.py`, not just asserted). The parts needing
attention at 1M images are indexing throughput (batch on GPU) and Chroma's
single-process persistence (swap for a sharded/managed vector DB —
`retriever/search.py` only calls the standard collection `.query()`/`.get()`
API, so this is a client swap, not a rewrite). See `docs/WRITEUP.md` §10.
