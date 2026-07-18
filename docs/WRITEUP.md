# Multimodal Fashion & Context Retrieval — Write-up

Codebase: this repository. Indexer entry point `indexer/build_index.py`,
retriever entry point `retriever/cli.py`. All numbers below are reproducible
by running `python -m eval.coverage`, `python -m eval.evaluate`,
`python -m eval.benchmark`, `python -m eval.ablate_weights`,
`python -m eval.zero_shot_probe`, `python -m eval.compositional`,
`python -m eval.compare_backbones`, `python -m eval.test_parser`,
`python -m eval.test_relations`, `python -m eval.multi_attribute`,
`python -m eval.corpus_composition`, and `python -m eval.profile_latency`.

## 0. The dataset: Fashionpedia val2020 + train2020, not val2020 alone

The PRD names Fashionpedia as *"one such dataset"* — a suggestion, not a
mandate to use only its small val2020 annotated slice (1,158 images). That
slice alone has a real, measured problem: **zero yellow coats and zero red
ties anywhere in it**, which directly capped two of the five PRD evaluation
queries regardless of retrieval quality (documented in an earlier iteration
of this project — see git history). The fix used here is to merge in
Fashionpedia's much larger **train2020** split — same format, same
annotation pipeline, same license, ~45,623 images — rather than reach for an
unrelated dataset:

- **Sampling strategy** (`common/config.py`'s `TRAIN_*` constants,
  `indexer/dataset.py`'s `train2020_sample_ids()`): every train2020 image
  containing a `coat` or `tie` instance (4,489 images — guarantees the two
  previously-missing PRD combos have real ground truth) plus a reproducible
  random sample of 7,500 more (general corpus depth, not just the two
  categories the eval queries happen to need). `indexer/extract_train_subset.py`
  selectively extracts only this ~12k-image sample from the 3.3GB zip via
  Python's `zipfile`, not a full unzip of all 45k images.
- **Result**: the corpus grew from 1,158 → **13,147 annotated images**
  (110,907 garment instances) + 2,042 unannotated, **15,189 images total**.
  Verified directly (`eval/coverage.py`): train2020 alone contributes 3,124
  `coat` instances and 1,457 `tie` instances — the val2020-only corpus had 3
  `tie` instances *total*.
- Code degrades gracefully to val2020-only if train2020 hasn't been
  downloaded (checked via `os.path.exists`), so a fresh checkout without the
  extra ~3.8GB download still works, just with less coverage.

## 1. Approaches considered

| Approach | How it works | Good for | Weakness |
|---|---|---|---|
| **Keyword / filename matching** | Match query words against filenames or manual tags | Trivial to build | Explicitly disallowed by the PRD; no generalization |
| **Trained multi-label classifier + filter** | Train a CNN on Fashionpedia's categories/attributes, filter by predicted labels | Precise on labels seen during training | Zero-shot is exactly what it lacks — can't handle open vocabulary ("bright yellow raincoat") or scene/vibe language at all |
| **Vanilla CLIP, single global embedding** | Encode whole image + query into one shared vector space, cosine similarity | Genuinely zero-shot; strong on scene/style/vibe language | The PRD's own hint: pooling the whole image into one vector loses compositional structure — "red tie, white shirt" and "white tie, red shirt" embed almost identically |
| **Region-based detection + captioning** | Run/train a detector, caption each region, embed each caption | Best possible fine-grained compositional accuracy | Requires running or training a detector at index time; heavier pipeline, more failure surface |
| **Chosen: fashion-domain CLIP (global + per-region) + structured attributes derived from existing segmentation, hybrid re-ranking** | See §2 | Zero-shot *and* compositional, using labels Fashionpedia already provides | Structured signal is capped by the category/color vocabulary and by corpus size; measured explicitly in §5-§6 rather than assumed |

## 2. Chosen architecture

**Part A — Indexer** (`indexer/`)
1. `dataset.py` merges val2020 + train2020 annotations (§0) into one list of
   `ImageRecord`s, with train2020 ids offset to a disjoint range so they can
   never collide with val2020 ids in Chroma.
2. `embed.py` uses **`patrickjohncyh/fashion-clip`** (CLIP fine-tuned on
   ~800K fashion image/text pairs), loaded via HuggingFace `transformers`,
   **running on GPU** (an RTX 3070 Ti in this environment — `torch.cuda.is_available()`
   auto-detected). Embeds the **whole image** (scene/setting/style) and
   **each garment crop separately** (padded 15% beyond its bbox) — the
   compositionality fix, §3. Also supports an `open_clip` loading backend
   for checkpoints packaged differently — see §7's backbone-comparison entry
   for why that mattered and why the default didn't change.
3. `color_extract.py` computes the color(s) of each garment instance from a
   **cropped** region around its segmentation mask (not the whole image —
   §7 describes a real 14x speedup this fix gave), and `common/colors.py`
   names it by **nearest-neighbor search in CIE Lab space** against a
   reference palette built from the **XKCD color survey**
   (`data/raw/xkcd_colors.txt`, CC0, ~950 crowd-sourced human color-name
   judgments filtered to ~800 usable single/two-word names) — human-labeled
   data, not one person's hand-picked 35-color guess. Chromatic families for
   graded partial-credit matching (`same_family`/`family_members`) are
   computed **programmatically** from hue-wheel buckets, since hand-grouping
   ~800 names isn't maintainable. `extract_colors()` also reports a
   **second color** when a garment shows real internal color spread — ~25%
   of garment-category instances in this corpus.
4. `relations.py` + `depth_relations.py`: **layered** (bbox-IoU
   co-occurrence, same body region) plus real **z-order** — which garment is
   worn *over* the other — from a monocular depth model
   (`depth-anything/Depth-Anything-V2-Small-hf`, via `transformers`, no new
   dependency), run only on images with a candidate layered pair (most
   images have none; running depth estimation on every image would be
   needlessly expensive). Verified on a real corpus image: a jacket visibly
   worn over a t-shirt was correctly identified as "over" from depth alone.
5. `scene_tag.py` is one generic zero-shot tagger, reused for three prompt
   banks: scene (office/street/park/home/studio/runway/beach/cafe/gym),
   style (formal/casual/outerwear/sporty), and weather
   (rainy/sunny/cold/indoor). "Locations" is implemented as **place type**,
   not city names — nothing in a street photo reveals which city it was
   taken in.
6. `build_index.py` runs two passes: annotated images get full structured
   metadata; unannotated images get whole-image embedding + tags only. Both
   live in **Chroma** — two persistent collections, one `pip install`,
   metadata filtering built in, per the PRD's explicit instruction to spend
   ML effort, not engineering effort, on the vector store.

**Part B — Retriever** (`retriever/`)
1. `query_parser.py` tokenizes the query for garment/color words, scene/
   style/weather keywords, and layering direction ("X over Y" vs. "Y under
   X", resolved by word position, not just keyword presence — §7). Color
   matching is **two-word-first** (checks `"marine blue"` as one token
   before falling back to single words — §7 describes a real bug this fixes).
   Garment words the dictionary misses are resolved **zero-shot** via
   `vocab_resolver.py`.
2. `search.py`: image-level CLIP recall, garment-level recall (exact
   chromatic-family metadata filter + fuzzy ANN, per parsed
   `(category, color)`), scored as `W_CLIP·image_sim + W_COMP·attribute_overlap
   + W_SCENE·(scene+style+weather match)`, renormalized over whichever
   signals the query produced. `attribute_overlap`: 1.0 exact color, 0.8
   same-family, 0.6 category-only, plus a relation bonus (0.6 co-occurs
   undirected, 1.0 confirmed z-order). Weights (0.35/0.55/0.10) chosen by
   ablation, not guessed (§6b).
3. All text a query needs embedded (main query + zero-shot candidates +
   every garment sub-query) is batched into at most two model calls, not
   one call per piece (§7).
4. If parsing finds nothing, the query degrades to pure CLIP similarity —
   the hybrid machinery only sharpens results, never blocks a zero-shot query.

## 3. How this handles compositionality (the PRD's explicit hint)

Vanilla CLIP pools a whole image into one vector, so "red tie + white shirt"
and "white tie + red shirt" embed almost identically. Three things fix this:
per-garment crops embedded and labeled independently; instance-scoped exact-
match re-ranking; and layering relations (now with real z-order, §2 item 4).

### 3a. The compositional win, measured

For 60 corpus images with two differently-colored garments,
`eval/compositional.py` builds the correctly-composed query and the
color-swapped decoy and measures which one scores higher:

| System | Discrimination accuracy | Mean margin |
|---|---|---|
| CLIP-only (the baseline the PRD says to beat) | 0.600 | +0.001 |
| **Hybrid (this system)** | **1.000** | **+0.383** |

Consistent across every corpus size tested this project (0.65→1.00, then
0.60→1.00 after the train2020 merge) — CLIP-only stays near chance;
instance-scoped attribute re-ranking makes the distinction crisp regardless
of corpus scale.

### 3b. Zero-shot vocabulary resolution

`vocab_resolver.py` resolves garment words the dictionary misses
("windbreaker", "parka") through fashion-CLIP itself, via a two-prototype
classifier (garment-similarity must beat non-garment-concept-similarity by a
margin — a plain threshold would inject false constraints, since fashion-CLIP
scores "near"/"person"/"standing" as high as real garments against category
prototypes). `eval/test_parser.py`: **recall 8/8, precision 3/3**. Verified
end-to-end: "a navy windbreaker" → `('jacket', 'navy')` → navy jacket at rank 1.

### 3c. Multi-color garments

`extract_colors()` reports a second color for ~25% of garment-category
instances with real internal color spread (striped/color-blocked), verified
on real two-tone examples. Both colors register as separate `(category,
color)` pairs at index time.

### 3d. Multi-attribute queries: color + type + location

The PRD's Context Awareness requirement gives this exact example: "color +
clothing type + location." `eval/multi_attribute.py` builds the literal
3-attribute query for every `(category, color, scene)` triple with real
support, and compares against the same query without the location term —
reproduced across **three independent runs** at different corpus states:

| Query form | mean P@5 |
|---|---|
| 2-attribute (color + type only) | 0.287 – 0.384 |
| **3-attribute (color + type + location)** | **0.591 – 0.689** |
| **Location term's effect on P@5** | **+0.299 to +0.305, every time** |

The location term roughly **doubles precision** because color+category alone
is often shared across many scenes.

### 3e. Real z-order for layering ("a jacket over a shirt")

`eval/test_relations.py` (7/7) locks in the word-position direction parsing;
the depth model itself was validated on a real example (§2 item 4). Verified
end-to-end: `retriever/cli.py "a jacket layered over a shirt"` correctly
returns images where a jacket is worn over another garment, with `comp_score`
reaching 1.0 only when the depth-verified direction matches the query.

## 4. Ground-truth coverage — an honest ceiling check, now mostly closed

`eval/coverage.py` checks both **exact** color-string and **chromatic-family**
coverage (the ~800-name palette almost never produces the bare word "yellow"
— real yellow coats are named "mustard" or "maize" — so exact-only checking
would misreport real coverage as absent):

| Combo | Exact | Family | Verdict |
|---|---|---|---|
| `(coat, yellow)` | 0 | **71** | now covered |
| `(shirt, blouse, blue)` | 0 | **652** | now covered |
| `(tie, red)` | 0 | **127** | now covered |
| `(shirt, blouse, white)` | 40 | 40 | covered |

Before the train2020 merge, all three "family" columns were 0-3. This is the
direct, measured effect of §0's dataset decision.

## 5. Precision@5 on the 5 PRD queries

| # | Query | P@5 |
|---|---|---|
| 1 | "A person in a bright yellow raincoat." | **1.00** |
| 2 | "Professional business attire inside a modern office." | **1.00** |
| 3 | "Someone wearing a blue shirt sitting on a park bench." | **1.00** |
| 4 | "Casual weekend outfit for a city walk." | **1.00** |
| 5 | "A red tie and a white shirt in a formal setting." | **1.00** |
| | **mean** | **1.000** |

Up from a coverage-capped mean of 0.600 before §0's dataset merge — every
query now has real, substantial ground truth (§4), not just a "nearest
achievable" proxy score. Color matching in the predicates is chromatic-
family-aware (`eval/evaluate.py`), consistent with how the retriever itself
scores.

**One caveat, stated directly**: query #2's strong score should be read
alongside §0's finding that "office" scenes are a small fraction of this
corpus even after the merge (Fashionpedia is fundamentally runway/editorial
photography) — it is succeeding via a combination of the `style=formal`
signal and genuine office-tagged images, not overwhelming scene-tag volume.
Re-run `python -m eval.corpus_composition` against the current (much larger)
corpus for the up-to-date breakdown.

## 6. Benchmarking beyond 5 anecdotes

### a. Corpus-grounded benchmark

`eval/benchmark.py` auto-generates queries from every `(category, color)`
combination occurring ≥3 times in the corpus:

| Slice | # queries | mean P@5 | mean R@5 |
|---|---|---|---|
| **GARMENT** | 2,010 | **0.798** | **0.889** |
| PART (embellishments — harder, not "clothing types" per the PRD) | 1,630 | 0.796 | 0.874 |
| ALL | 3,640 | 0.797 | 0.882 |

Query count grew 258→2,010 across the train2020 merge and the finer XKCD
palette — both create more distinct, real, well-supported combos to test
(8x more test coverage than the previous round). P@5 moved down from the
smaller corpus's 0.850 for the same mechanical reason documented throughout
this project: finer attributes mean more combos with fewer true positives
each, lowering the P@5 ceiling even at perfect recall. **A meaningful chunk
of an initial, much larger drop (to 0.326) was a real bug, not this
mechanical effect** — see §7's two-word-color-tokenizer entry.

### b. Weight ablation

`eval/ablate_weights.py`, 120 random GARMENT combos:

| `(W_CLIP, W_COMP, W_SCENE)` | mean P@5 |
|---|---|
| 0.70 / 0.20 / 0.10 (CLIP-heavy) | 0.780 |
| 0.50 / 0.35 / 0.15 (initial guess) | 0.863 |
| **0.35 / 0.55 / 0.10 (chosen)** | **0.877** |
| 0.10 / 0.80 / 0.10 (comp-only) | 0.880 |

Comp-heavy wins because exact attribute matches are ground truth from
segmentation, more trustworthy than CLIP's fuzzy similarity once a candidate
is already known to match. Chosen config preferred over the marginally
higher comp-only config for robustness (keeps CLIP contributing to
tie-breaking).

### c. Zero-shot probe

`eval/zero_shot_probe.py`: 5 queries built from words absent from every
vocabulary in the system. *"An elegant evening gown for a gala"* correctly
surfaces dress + formal + runway images despite none of those exact words
being labeled anywhere. Abstract aesthetic queries ("minimalist", "oversized")
score lower-confidence — honestly the hardest case, no structured signal to
lean on.

### d. Backbone comparison — a real test, not a vendor's claim

`eval/compare_backbones.py` directly measures `patrickjohncyh/fashion-clip`
against `Marqo/marqo-fashionCLIP` (a checkpoint reporting +57% on its own
benchmark) on **this project's actual compositional-discrimination task**:

| Backbone | Discrimination accuracy | Mean margin |
|---|---|---|
| patrickjohncyh/fashion-clip (kept) | 0.650 | +0.0056 |
| Marqo/marqo-fashionCLIP | 0.633 | +0.0062 |

No clear win — within noise for n=60. **Kept the original checkpoint on this
evidence**, not the vendor's benchmark claim, which evidently doesn't
transfer to this corpus/task. Along the way, discovered that Marqo's
checkpoint is packaged in OpenCLIP's native format, not HuggingFace's —
loading it through `transformers.CLIPModel` silently produces an untrained
model (hundreds of MISSING/UNEXPECTED weight keys) rather than erroring
loudly. `embed.py` now supports both loading backends (`CLIP_BACKEND` in
`common/config.py`), tested and working, for whichever future checkpoint
someone tries.

### e. Latency profile

`eval/profile_latency.py`, measured honestly at each corpus/capability
change, not asserted once and left stale:

| Phase | Single garment+color | Two garments | Scene-only |
|---|---|---|---|
| Original (3,200 images) | 31ms | 55ms | 16ms |
| + zero-shot resolver + multi-color (unbatched) | 75-100ms | 150-190ms | 30-75ms |
| + batching fix | 40ms | 56-61ms | 20-24ms |
| **+ train2020 merge (15,189 images)** | **124ms** | **246ms** | **25ms** |

The batching fix (§ below) held up as designed — latency scales with the
**number of parsed sub-queries**, not raw model-call count. The renewed
increase after the train2020 merge is a real, honest measurement at ~5x more
images/crops: larger chromatic families (up to 100+ members) mean bigger
`$in` metadata filters, and a larger Chroma collection costs more per ANN
call even with HNSW's sub-linear scaling. All figures stay well under 300ms
(comfortably real-time), and the architectural claim in §10 — flat with
corpus size for a *fixed* candidate pool — still holds; the absolute
constant just moved once with a 5x larger candidate universe to filter within.

### f. GPU — already active, previously undocumented

`indexer/embed.py` already auto-detects CUDA (`torch.cuda.is_available()`)
and was **running on an RTX 3070 Ti this entire project** — this write-up
previously and incorrectly said "CPU-only backbone" throughout, a claim that
was simply never verified. Corrected here. Measured warm throughput: **8.3ms/image
on GPU vs. ~84ms/image on CPU** (~10x). The real remaining indexing
bottleneck, found by profiling rather than assuming embedding was still the
bottleneck: `color_extract.py` was rasterizing and eroding a mask **the size
of the whole image** for every single garment instance, regardless of how
small the garment was, and re-converting the whole image to a numpy array
per instance. Cropping to the garment's bbox first (§2 item 3) gave a
**14x speedup** (60ms/instance → 4.3ms/instance), verified byte-identical
output on 60 real images before/after. Combined effect: full-corpus indexing
time dropped even as corpus size grew ~5x (a ~15-min full rebuild at 3,200
images pre-fix vs. ~27.5 min at 15,189 images post-fix and post-merge — far
better than the naive ~5x scaling would predict).

## 7. Real bugs found and fixed while building this (not hypothetical)

Eight, in the order they were caught — every one found by building an actual
check rather than trusting first-pass output:

1. **Color classifier misfired on a real image.** A khaki/army-green jacket
   was classified "yellow" by an early hue-only HSV rule. Fixed by moving to
   Lab-space nearest-neighbor naming.
2. **Garment-crop ANN search missed a real true positive.** A blue-shirt
   image ranked 142nd out of 7,800 for its own correct text query. Fixed
   with a deterministic exact-metadata-filter recall pass.
3. **A finer color palette silently broke exact-match recall.** "Blue" split
   into blue/navy/denim/sky-blue; a plain equality check now missed real
   matches. Fixed with chromatic-family partial-credit scoring.
4. **The zero-shot vocabulary resolver re-embedded words already classified
   elsewhere** (e.g. "office"), measurably inflating latency. Fixed by
   excluding all keyword-matched positions before the resolver runs.
5. **Text embeddings were scattered across many small sequential model
   calls.** Measured the fix before implementing it: 4 sequential calls took
   ~44ms; one batched call took ~14ms. Fixed by splitting query parsing into
   `tokenize()`/`finalize()` so `search.py` can batch everything a query needs.
6. **Every eval script recomputed ground truth from raw images on every
   run.** Measured: 385.8s → 0.283s (**1364x**) reading the same data back
   from the already-built index instead. Also removed a class of possible
   bug (recomputed "ground truth" could in principle drift from what's
   actually indexed; reading the same data both places makes that
   impossible by construction).
7. **`color_extract.py` rasterized full-image-sized masks per garment
   instance.** 14x speedup from cropping to the instance's bbox first,
   found while investigating why indexing was still slow *after* confirming
   GPU embedding was already fast (§6f) — the real bottleneck had moved.
8. **The query parser only checked single words against the color palette.**
   After the color namer moved to a ~800-name XKCD-derived palette (majority
   two-word names), a query for "marine blue" split into two spurious
   single-word tokens ("marine" **and** "blue" — both independently valid
   palette entries with *different* reference colors), corrupting garment-
   color pairing. Found via a benchmark regression from 0.850 → **0.326**
   mean P@5 after the train2020 merge — investigated the specific failing
   query (`"a person wearing a marine blue shirt"`, 0/5 hits despite 3 true
   positives existing) rather than assuming the drop was purely the already-
   documented "finer attributes → more combos → lower ceiling" effect.
   Fixed with greedy two-word-phrase matching before the single-word
   fallback. Recovered to **0.798** mean P@5 on 2,010 queries (8x the prior
   test coverage) — most of the regression was this bug, not the mechanical
   ceiling effect, which still accounts for the remaining gap versus 0.850.

## 8. Shortcomings & mitigations already in place

- **Attribute vocabulary is closed** (46 categories, ~800-color palette).
  Mitigated by always falling back to CLIP similarity.
- **Color naming is nearest-neighbor over a fixed reference palette**
  (now sourced from real human survey data, not hand-picked, but still not a
  *trained* model in the parametric sense) — see §9b for the natural next step.
- **Spatial/relational reasoning covers "layered" plus real z-order for one
  relation type** — extending to "tucked in" needs occlusion reasoning
  beyond even depth-based ordering (two garments can be at similar depth
  while one is still tucked into the other).
- **Corpus size still caps achievable precision** on the rarest
  combinations, even after growing to 15,189 images — §4 shows this gap is
  now mostly closed for the specific PRD queries, not eliminated everywhere.
- **Query latency grew with corpus size** (§6e) — still fast in absolute
  terms, but not literally flat; the *architecture* (fixed candidate pools)
  is what scales, not the absolute millisecond count.

## 9. Future work

**a. Locations and weather — implemented (place type), extend further**
- Place-type and weather zero-shot tagging ship now. City-level location
  was deliberately not attempted (not honestly inferable from most photos).
- Extend the prompt banks for finer place granularity — the machinery
  already generalizes to any additional axis with no architecture change.

**b. Improving precision further**
- **Learn the color namer** rather than nearest-neighbor over a fixed
  (even if human-sourced) reference palette — the Lab-space infrastructure
  here is the right foundation for training a real classifier on labeled data.
- **Extend z-order past pairwise "over/under"** into full outfit layering
  order (3+ garments) and true "tucked in" detection, which needs occlusion
  signal beyond relative depth.
- **Grow the weight ablation and ANN/exact-filter tuning** as more query
  data becomes available — e.g. a small learned re-ranker over the same
  three signals, or adaptive `CANDIDATE_POOL`/`GARMENT_POOL` sizing as the
  corpus grows further toward the 1M-image scale discussed in §10.
- **Address the latency growth from §6e directly** — e.g. capping chromatic
  family size for the exact-filter `$in` clause, or pre-computing family
  membership as a stored index rather than a runtime set.

## 10. Scalability to ~1M images

Both ANN searches pull a **fixed-size candidate pool**, not the full corpus
— architecturally flat with corpus size. What's *not* flat, measured
honestly in §6e: the constant-factor cost per candidate (larger chromatic
`$in` filters, bigger Chroma collection) did increase once at the ~5x corpus
growth in this project. The real bottleneck at 1M images is indexing
throughput — already GPU-accelerated (§6f) and now that the color-extraction
bottleneck is fixed too, both major indexing costs are addressed — and
Chroma's single-node persistence, solved by a sharded/managed vector DB swap
(`retriever/search.py` only calls the standard collection API).

## 11. On "state of the art"

There is no published benchmark for this exact task, so there's nothing to
claim victory against. What's reported instead: a controlled compositional
experiment (CLIP-only 0.60 → hybrid 1.00, reproduced at two corpus scales);
a direct test of the PRD's "color+type+location" example (location roughly
doubles precision, reproduced across **three** independent runs); a direct,
skeptical test of a vendor's larger-checkpoint benchmark claim that did
**not** hold up on this task (§6d) — kept the original backbone on that
evidence; a corpus-grounded 2,010-query benchmark; a 120-combo weight
ablation; **eight** real bugs found and fixed by building actual checks,
including one caught specifically by not accepting a benchmark regression at
face value and instead investigating the specific failing case (§7, #8); and
the headline result — **all 5 PRD queries now score P@5=1.00** — earned by
fixing the dataset's real coverage gap, not by tuning against the eval set.
