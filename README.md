# glance — Multimodal Fashion & Context Retrieval

Natural-language image search over [Fashionpedia](https://github.com/cvdfoundation/fashionpedia)
(val2020 + train2020, ~15k images): type a description, get back matching
images. Built for the Glance ML internship assignment (see `docs/PRD.pdf`,
full write-up with all evidence in `docs/WRITEUP.md`).

## Headline result

**All 5 PRD evaluation queries now score P@5 = 1.00 (mean 1.000)** — earned
by fixing a real dataset coverage gap (val2020 alone had zero yellow coats
and zero red ties; merging in Fashionpedia's train2020 split fixed that),
not by tuning against the eval set. Full detail in `docs/WRITEUP.md`.

## Architecture

- **`common/`** — shared config and the XKCD-survey-derived color classifier + programmatic chromatic families (`colors.py`).
- **`indexer/`** (Part A) — merges Fashionpedia val2020 + train2020 (`dataset.py`), embeds whole images and per-garment crops with fashion-domain CLIP on GPU (`embed.py`), extracts 1-2 mask-derived colors per garment from a **cropped** region (`color_extract.py` — not the whole image, a real 14x speedup), derives layered-garment relations with real depth-based z-order (`relations.py` + `depth_relations.py`), and zero-shot tags scene/style/weather. Stores everything in two persistent Chroma collections.
- **`retriever/`** (Part B) — parses the query (two-word-color-aware, e.g. "marine blue" as one token — `query_parser.py`), resolves unknown garment words zero-shot (`vocab_resolver.py`), does image-level + garment-level ANN + exact chromatic-family metadata search, and hybrid-reranks with weights chosen by ablation, not guessed. All embeddings for one query batch into ≤2 model calls.
- **`eval/`** — coverage check, Precision@k on the 5 PRD queries, a 3,640-query corpus-grounded benchmark, a multi-attribute (color+type+location) benchmark, a controlled compositional-discrimination experiment, a **direct backbone comparison** (tested a vendor's larger checkpoint, it didn't win, kept the original), a weight ablation, regression tests, a zero-shot probe, and a latency profile. See `docs/WRITEUP.md` for the numbers and **eight** real bugs these caught.

Logic and data are fully separated: nothing in `indexer/` or `retriever/`
hardcodes a file path or a label — those come from `common/config.py` and
the annotation files themselves.

## Setup

```bash
pip install -r requirements.txt
```

Dataset (already present under `data/raw/` in this repo). To fetch fresh:

```bash
# val2020 (small, ~1.2k annotated images)
curl -o data/raw/val_test2020.zip https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip
curl -o data/raw/instances_attributes_val2020.json https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json
unzip data/raw/val_test2020.zip -d data/raw/val_test2020_tmp
mv data/raw/val_test2020_tmp/test data/raw/val_test2020

# train2020 (larger, ~45.6k annotated images -- required for real yellow-coat/
# red-tie ground truth; see docs/WRITEUP.md §0. ~3.8GB total download.)
curl -o data/raw/train2020.zip https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip
curl -o data/raw/instances_attributes_train2020.json https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json
python -m indexer.extract_train_subset   # selectively extracts only the sampled ~12k images, not all 45k

# color namer's reference palette (CC0)
curl -o data/raw/xkcd_colors.txt https://xkcd.com/color/rgb.txt
```

The code degrades gracefully to val2020-only if train2020 isn't downloaded
(less coverage, but still works).

## Build the index (Part A)

```bash
python -m indexer.build_index
```

## Query (Part B)

```bash
python -m retriever.cli "A person in a bright yellow raincoat." --k 5
python -m retriever.cli "a jacket layered over a shirt" --k 5              # real depth-verified z-order
python -m retriever.cli "a person wearing a marine blue shirt" --k 5       # two-word color name
python -m retriever.cli "a person in a navy windbreaker" --k 5             # zero-shot vocab, not in any dictionary
```

## Evaluation

```bash
python -m eval.test_colors        # color-classifier regression suite (23 cases)
python -m eval.test_parser        # zero-shot vocabulary resolver precision/recall calibration
python -m eval.test_relations     # layering-direction parsing regression suite (7 cases)
python -m eval.coverage           # does ground truth exist for the 5 PRD queries? (exact + chromatic-family)
python -m eval.evaluate           # Precision@5 on the 5 PRD queries
python -m eval.benchmark          # Precision@5/Recall@5 over corpus-grounded queries (GARMENT/PART slices)
python -m eval.multi_attribute    # PRD's own "color + type + location" example, measured
python -m eval.compositional      # CLIP-only vs hybrid on the "red shirt/blue pants" color-swap case
python -m eval.compare_backbones  # direct test of a larger vendor checkpoint against the chosen one
python -m eval.ablate_weights     # weight sweep that justified the hybrid-scoring config
python -m eval.zero_shot_probe    # qualitative probe with words in no vocabulary at all
python -m eval.corpus_composition # validates the PRD's 3 dataset axes (environment/clothing type/color)
python -m eval.profile_latency    # per-stage query latency, evidence for the scalability claim
python -m eval.run_eval_queries   # saves labeled contact-sheet PNGs to eval/outputs/
```

All ground-truth-dependent scripts read directly from the built index
(`eval/ground_truth.py`) instead of recomputing color extraction from raw
images — measured 1364x faster than the original per-script recomputation.

**Headline numbers** (full detail and honest caveats in `docs/WRITEUP.md`):
- **5 PRD eval queries: mean P@5 = 1.000** (all 5 score 1.00), up from a coverage-capped 0.600.
- **Compositionality**, color-swap discrimination: CLIP-only **0.60** (near chance) → hybrid **1.00**.
- **Context Awareness** ("color + type + location"): the location term **roughly doubles precision** (+0.30 P@5, reproduced across **three** independent runs at different corpus scales).
- 2,010 corpus-grounded GARMENT queries (not hand-picked): **mean P@5 = 0.798, R@5 = 0.889**.
- **Backbone choice validated, not assumed**: directly tested a vendor's larger checkpoint (claims +57% on its own benchmark) against the chosen one on this project's actual task — it didn't win (0.633 vs 0.650, within noise) — kept the original on that evidence.
- **Zero-shot parser**: novel garment words resolve at **8/8 recall, 3/3 precision** with no hardcoding.
- Query latency: 25-246ms depending on query complexity and the ~5x larger (15,189-image) corpus — still real-time; see `docs/WRITEUP.md` §6e for the honest, measured-at-each-stage story (it wasn't monotonically flat, and that's reported directly, not smoothed over).

## Why this approach (short version — full write-up in `docs/WRITEUP.md`)

Vanilla CLIP pools a whole image into one vector, so it can't reliably tell
"red tie, white shirt" from "white tie, red shirt". This system embeds each
garment crop *separately* and re-ranks with exact `(category, color)`
matching (graded credit for same-family colors, a second color for patterned
garments) plus real depth-verified layering z-order, on top of fashion-CLIP's
whole-image similarity. The query parser is itself zero-shot for garment
vocabulary, and correctly handles the mostly-two-word color palette sourced
from real human survey data (XKCD) rather than one person's guesses. The
dataset itself was extended — not just the algorithm — specifically because
the PRD's own evaluation queries needed real ground truth that the smaller,
commonly-used Fashionpedia slice didn't have.

## Locations & weather

Implemented as zero-shot **place-type** and **weather** tagging, not city
names — a street-style photo doesn't honestly reveal which city it was taken
in. Measured directly: adding the location term to a color+type query
roughly doubles P@5, reproduced across three independent runs.

## Scaling to ~1M images

Both ANN searches pull a **fixed-size candidate pool**, not the full corpus
— architecturally flat with corpus size. Measured honestly across this
project's ~5x corpus growth: the *constant factor* per candidate did
increase (larger chromatic-family filters, bigger collection), reported in
`docs/WRITEUP.md` §6e rather than smoothed into a single stale number. GPU
embedding (already active, an RTX 3070 Ti — a real correction to earlier
documentation in this project that wrongly assumed CPU-only) plus the
cropped color-extraction fix address the two biggest indexing-throughput
costs; Chroma's single-node persistence is the remaining lever at 1M images,
solved by a sharded/managed vector DB swap (`retriever/search.py` only calls
the standard collection API).
