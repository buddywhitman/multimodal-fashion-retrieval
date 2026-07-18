# glance — Multimodal Fashion & Context Retrieval

Natural-language image search over [Fashionpedia](https://github.com/cvdfoundation/fashionpedia):
type a description, get back matching images — understanding what someone is
wearing, where they are, and the overall vibe. Full technical write-up in
[`docs/WRITEUP.md`](docs/WRITEUP.md).

## Architecture

Two pipelines, logic fully separated from data (everything path/label-specific
lives in `common/config.py` and the annotation files):

- **`common/`** — shared config and the CIE-Lab color classifier with programmatic chromatic families (`colors.py`).
- **`indexer/`** (Part A) — merges Fashionpedia val2020 + a train2020 sample (`dataset.py`), embeds whole images and per-garment crops with fashion-domain CLIP on GPU (`embed.py`), extracts mask-derived colors (`color_extract.py`), derives depth-verified layering z-order (`relations.py` + `depth_relations.py`), and zero-shot tags scene/style/weather (`scene_tag.py`). Two persistent Chroma collections.
- **`retriever/`** (Part B) — parses the query into structured signals (`query_parser.py`), resolves unknown garment words zero-shot (`vocab_resolver.py`), combines whole-image CLIP recall with deterministic attribute recall via in-memory indexes (`attribute_index.py`, `image_store.py`), and hybrid-reranks with ablation-chosen weights (`search.py`).
- **`eval/`** — the 5 PRD queries, a corpus-grounded benchmark, a compositional-discrimination experiment, a multi-attribute (color+type+location) test, a backbone comparison, a weight ablation, a zero-shot probe, regression tests, and a latency profile.

## Headline numbers

- **5 PRD evaluation queries: mean P@5 = 1.00.**
- **Compositionality** ("red tie/white shirt" vs. "white tie/red shirt" discrimination): whole-image CLIP **0.60** (near chance) → hybrid **1.00**.
- **Context awareness** (color + type + location): the location term roughly doubles precision (+0.30 P@5).
- 2,010 corpus-grounded GARMENT queries: **P@5 = 0.817, R@5 = 0.913** (against a metric ceiling of 0.888).
- **Zero-shot parser**: novel garment words resolve at **8/8 recall, 3/3 precision**.
- Query latency: **~30–34 ms median**.

## Setup

```bash
pip install -r requirements.txt
```

Fetch the dataset (Fashionpedia val2020 + train2020, plus the color palette):

```bash
# val2020 (small annotated split)
curl -o data/raw/val_test2020.zip https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip
curl -o data/raw/instances_attributes_val2020.json https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json
unzip data/raw/val_test2020.zip -d data/raw/val_test2020_tmp && mv data/raw/val_test2020_tmp/test data/raw/val_test2020

# train2020 (larger annotated split; ~3.8GB total)
curl -o data/raw/train2020.zip https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip
curl -o data/raw/instances_attributes_train2020.json https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json
python -m indexer.extract_train_subset   # extracts only the sampled subset, not all 45k images

# color-namer reference palette (CC0)
curl -o data/raw/xkcd_colors.txt https://xkcd.com/color/rgb.txt
```

The code runs on val2020 alone if train2020 isn't downloaded (less coverage).

## Build the index (Part A)

```bash
python -m indexer.build_index
```

## Query (Part B)

```bash
python -m retriever.cli "A person in a bright yellow raincoat." --k 5
python -m retriever.cli "a jacket layered over a shirt" --k 5              # depth-verified z-order
python -m retriever.cli "a person wearing a marine blue shirt" --k 5       # multi-word color name
python -m retriever.cli "a person in a navy windbreaker" --k 5             # zero-shot garment vocabulary
```

## Evaluation

```bash
python -m eval.test_colors        # color-classifier regression suite
python -m eval.test_parser        # zero-shot vocabulary resolver precision/recall
python -m eval.test_relations     # layering-direction parsing regression suite
python -m eval.coverage           # ground-truth coverage for the 5 PRD queries (exact + family)
python -m eval.evaluate           # Precision@5 on the 5 PRD queries
python -m eval.benchmark          # P@5/R@5 over corpus-grounded queries (GARMENT/PART slices)
python -m eval.multi_attribute    # color + type + location, measured
python -m eval.compositional      # CLIP-only vs hybrid color-swap discrimination
python -m eval.compare_backbones  # chosen backbone vs a larger alternative
python -m eval.ablate_weights     # hybrid-scoring weight sweep
python -m eval.zero_shot_probe    # queries with words in no vocabulary at all
python -m eval.corpus_composition # coverage across environment / clothing type / color axes
python -m eval.profile_latency    # per-stage query latency
python -m eval.run_eval_queries   # labeled contact-sheet PNGs to eval/outputs/
```

Ground-truth-dependent scripts read directly from the built index
(`eval/ground_truth.py`), so the whole suite runs in well under a minute.

## Why this approach

Vanilla CLIP pools a whole image into one vector and so can't reliably tell
"red tie, white shirt" from "white tie, red shirt". This system embeds each
garment crop separately and re-ranks with exact `(category, color)` matching
(graded credit for same-family colors, a second color for patterned garments)
plus depth-verified layering, on top of fashion-CLIP's whole-image similarity.
The query parser is itself zero-shot for garment vocabulary, and CLIP's
zero-shot generalization is the fallback for anything the structured parser
can't resolve. See [`docs/WRITEUP.md`](docs/WRITEUP.md) for the full design and
all benchmarks.
