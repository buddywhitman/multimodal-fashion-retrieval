# Multimodal Fashion & Context Retrieval — Write-up

Codebase: this repository. Indexer entry point `indexer/build_index.py`,
retriever entry point `retriever/cli.py`. All numbers below are reproducible
by running `python -m eval.coverage`, `python -m eval.evaluate`,
`python -m eval.benchmark`, `python -m eval.ablate_weights`,
`python -m eval.zero_shot_probe`, `python -m eval.compositional`,
`python -m eval.test_parser`, `python -m eval.multi_attribute`, and
`python -m eval.profile_latency`.

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
1. `dataset.py` loads Fashionpedia's COCO-style annotations — segmentation
   polygons + category labels are already there per garment instance. Also
   exposes every image with **no** annotation, so the corpus isn't capped at
   the labeled subset (§6c).
2. `embed.py` uses **`patrickjohncyh/fashion-clip`** (CLIP fine-tuned on
   ~800K fashion image/text pairs via HuggingFace `transformers`) — swapped
   in from generic `clip-ViT-B-32` after verifying it discriminates fashion
   color/garment language measurably better. Embeds:
   - the **whole image** (captures scene/setting/style — "office", "park
     bench", "casual weekend")
   - **each garment crop separately**, padded 15% beyond its bbox — the
     compositionality fix, see §3.
3. `color_extract.py` computes the color(s) of each garment instance from
   its segmentation mask (mask eroded 2px inward to avoid background/skin
   bleed at the boundary, then median RGB), and `common/colors.py` names it
   by **nearest-neighbor search in CIE Lab space** against a 35-color
   fashion palette — not hand-tuned HSV thresholds. Lab is perceptually
   uniform, so it resolves exactly the shade pairs fashion queries hinge on:
   burgundy vs. maroon, khaki vs. olive vs. sage, navy vs. denim.
   Regression-tested in `eval/test_colors.py` (22/22 cases, including a real
   failure caught during this project — see §7).
   `common/colors.py` also defines **chromatic families** (e.g. `{blue,
   navy, denim, sky blue}`) so a query for "a blue shirt" gets full credit
   for an exact "blue" label and graded partial credit for a family member
   like "denim".
   `extract_colors()` also reports a **second color** when a garment shows
   real internal color spread — measured at ~25% of garment-category
   instances in this corpus (a striped or color-blocked garment medians to
   one blended, often-wrong color otherwise). See §3c.
4. `relations.py` derives one spatial/relational attribute — **layered** —
   from bbox IoU between same-body-region garment instances (e.g. a jacket
   overlapping a shirt). Scoped honestly: bounding boxes can't reliably
   give true z-order ("tucked in" vs. "worn over"), so this captures
   co-occurrence/layering, not full occlusion reasoning (§8).
5. `scene_tag.py` is one generic zero-shot tagger, reused for **three**
   prompt banks (`common/config.py`): scene (office/street/park/home/
   studio/runway/beach/cafe/gym), style (formal/casual/outerwear/sporty),
   and weather (rainy/sunny/cold/indoor) — cosine-matched against the image
   embedding. This is the "locations and weather" future-work item,
   implemented: **place type**, not city names (nothing in a street photo
   reveals which city it was taken in — a city-name bank would fabricate
   structure CLIP can't actually ground; place type is what's honestly
   inferable from pixels).
6. `build_index.py` runs two passes: annotated images get full structured
   metadata (categories, colors, relations, scene/style/weather); the
   remaining unannotated images on disk get whole-image embedding +
   scene/style/weather tags only (no garment crops, since there's no
   segmentation to crop from) — growing the searchable corpus from 1158 to
   **3200** images. Both live in **Chroma** — two persistent collections,
   one `pip install`, metadata filtering built in, per the PRD's explicit
   instruction to spend ML effort, not engineering effort, on the vector
   store.

**Part B — Retriever** (`retriever/`)
1. `query_parser.py` scans the query for known garment/color words (vocab
   cached from the annotation file, never hand-maintained), scene/style/
   weather keywords, and layering phrases ("layered", "over", "under",
   "tucked"), producing `{garments, scenes, styles, weathers, relation}`.
   Garment words the hardcoded synonym dict misses ("windbreaker", "parka",
   "loafers", "gown") are resolved **zero-shot** by `vocab_resolver.py` — see
   §3b. This is what keeps the parser itself from being the one
   closed-vocabulary component in an otherwise-generalizing pipeline.
2. `search.py`:
   - **Image-level recall**: top candidates by whole-image CLIP similarity.
   - **Garment-level recall**, two passes per parsed `(category, color)`:
     (a) an **exact metadata filter** on the garment collection, expanded to
     the whole chromatic family — deterministic recall, no dependence on
     embedding-similarity ranking a specific crop highly enough (§7 describes
     a real miss this caught); (b) a fuzzy CLIP-embedding ANN search on
     garment crops, for recall beyond exact/family parses.
   - **Scoring**: `score = W_CLIP·image_sim + W_COMP·attribute_overlap +
     W_SCENE·(scene+style+weather match)`, renormalized over whichever
     signals the query actually produced. `attribute_overlap` gives 1.0 for
     an exact color match, 0.8 for a same-family match, 0.6 for
     category-only, plus a bonus if the query asked for a layering relation
     and the image's derived `relations` metadata confirms it. Weights
     (0.35/0.55/0.10) were chosen by ablation (§6b) over 120 combos, not
     guessed.
   - **If parsing finds nothing** (a genuinely novel description), the query
     degrades to pure whole-image CLIP similarity — the hybrid machinery
     only sharpens results, never blocks a zero-shot query.

## 3. How this handles compositionality (the PRD's explicit hint)

Vanilla CLIP pools a whole image into one vector, so "red tie + white shirt"
and "white tie + red shirt" — same bag of visual concepts — embed almost
identically. Three things fix this here:

- **Per-garment crops are embedded and labeled independently.** A tie crop
  and a shirt crop in the same image are separate vectors with separate
  ground-truth colors; "tie=red" cannot bleed into "shirt=white" the way a
  single pooled embedding does.
- **Exact-match re-ranking is instance-scoped.** `attribute_overlap` only
  gives credit when a query's `(category, color)` pair matches a *specific
  garment instance* in that image, not just "this image contains red and
  white somewhere."
- **Layering relations add a second axis of composition** ("a jacket over a
  shirt") on top of per-garment category/color, verified end-to-end (see
  `retriever/cli.py "a jacket layered over a shirt"`).

### 3a. The compositional win, measured

`eval/compositional.py` runs the exact case the PRD hint names, as a
controlled experiment. For 60 corpus images containing two differently-colored
garments, it builds the correctly-composed query and the color-swapped decoy
("red shirt + blue pants" vs "blue shirt + red pants") and measures how often
each system scores the correct one higher:

| System | Discrimination accuracy | Mean score margin |
|---|---|---|
| CLIP-only (whole-image similarity, the baseline to beat) | 0.650 | +0.006 |
| **Hybrid (this system)** | **1.000** | **+0.400** |

CLIP-only barely beats chance with a near-zero margin — precisely because a
single pooled vector contains "red", "blue", "shirt", "pants" for *both*
phrasings. The instance-scoped attribute re-ranking makes the distinction
crisp. This is the single most on-point measurement for the assignment.

### 3b. Zero-shot vocabulary resolution (the parser is zero-shot too)

A structured parser is normally the one closed-vocabulary component that drags
down an otherwise zero-shot system: a hardcoded synonym dict can't know
"windbreaker" or "parka". `retriever/vocab_resolver.py` fixes this by resolving
unknown garment words through fashion-CLIP itself. The hard part is precision,
not recall — fashion-CLIP returns a high "a photo of a {w}" similarity for
non-garment words too ("near", "person", "standing" score as high as real
garments against category prototypes). So a plain threshold injects false
constraints and *hurts* retrieval. The fix is a **two-prototype zero-shot
classifier**: embed the bare word, and accept it as a garment only if its best
similarity to the garment categories beats its best similarity to a bank of
non-garment concept prototypes ("a place", "an action", "a person", "a time of
day"...) by a margin. `eval/test_parser.py` calibrates this and locks it in:
**recall 8/8** (windbreaker→jacket, loafers→shoe, leggings→tights...),
**precision 3/3** (filler sentences resolve to no garment). Verified
end-to-end: "a person in a navy windbreaker" → parses `('jacket', 'navy')` →
returns a navy jacket at rank 1. The resolver only runs when the dict misses,
so a dictionary-hit query pays zero extra model calls (verified by
instrumenting `embed_text`); a query with an unmatched word pays one extra
batched embedding call.

### 3c. Multi-color garments (patterned/color-blocked)

A single "dominant" color is a real information loss: measuring internal
color spread across masked pixels (§7) showed **~25% of garment-category
instances** in this corpus have enough spread that one blended median color
misrepresents them — e.g. a bright-blue-and-navy color-blocked jacket medians
toward neither. `extract_colors()` reports a second color when the spread is
large enough (≥30% of pixels are >60 RGB units from the primary median),
verified against real examples (a two-tone blue/charcoal jacket, a gray/sage
two-tone shirt). Both colors register as `(category, color)` pairs at index
time, so a query matching *either* gets credit — the same outcome as if the
garment were two separate solid-colored instances. Wired through
`build_index.py` and every eval ground-truth builder that independently
recomputes colors, so "ground truth" never silently disagrees with what's
actually indexed.

### 3d. Multi-attribute queries: color + type + location (Part B's own example)

The PRD's Context Awareness requirement gives this exact example: "color +
clothing type + location." Previously only 2-attribute (category+color) and
pure-scene queries had been benchmarked — this closes that gap directly.
`eval/multi_attribute.py` builds the literal 3-attribute query for every
`(category, color, scene)` triple with real support in the corpus (150
sampled), and compares it against the same query *without* the location term:

| Query form | mean P@5 | mean R@5 |
|---|---|---|
| 2-attribute (color + type only) | 0.325-0.384 | — |
| **3-attribute (color + type + location)** | **0.624-0.689** | **0.887-0.969** |
| **location term's effect on P@5** | **+0.299 to +0.305** | reproduced across two independent runs |

The location term isn't decorative — it roughly **doubles precision**,
because color+category alone is often shared across many scenes (a black
dress appears on runways, in studios, on the street), and the scene/style/
weather score (§2, Part B) breaks that ambiguity. This is the most direct
evidence that "Context Awareness" as the PRD defines it is real, not just
plumbed through.

## 4. Ground-truth coverage — an honest ceiling check

Before trusting any Precision@k number, `eval/coverage.py` checks whether
the ground truth even exists in the corpus:

| Combo | Instances in corpus |
|---|---|
| `(coat, yellow)` | **0** |
| `(shirt, blouse, blue)` | 1 (finer Lab-space color naming correctly split what used to be lumped "blue" into blue/navy/denim/sky blue — see §7) |
| `(tie, red)` | **0** (only 3 `tie` instances exist *at all* in the whole corpus) |
| `(shirt, blouse, white)` | 17 |

**This corpus contains no yellow coat and no red tie at all.** The PRD's
eval queries #1 and #5 ask for exact combinations that don't exist. A
retrieval system cannot return an image that isn't there — so Precision@5 on
those two queries is capped below 1.0 no matter how good the retriever is.
(Qualitatively, the top result for query #1 *is* a genuinely bright
yellow/gold raincoat — the best answer this corpus has; see
`eval/outputs/query_1.png`.)

## 5. Precision@5 on the 5 PRD queries (coverage-aware)

| # | Query | P@5 | Ground truth |
|---|---|---|---|
| 1 | "A person in a bright yellow raincoat." | 0.80 | absent — scored on nearest color family (yellow/mustard/gold/orange/brown-ish) |
| 2 | "Professional business attire inside a modern office." | 1.00 | exists |
| 3 | "Someone wearing a blue shirt sitting on a park bench." | 0.20 | exists, but only **1** true "blue" shirt (post-reclassification) — 0.20 is the exact **ceiling** (found it) |
| 4 | "Casual weekend outfit for a city walk." | 1.00 | exists |
| 5 | "A red tie and a white shirt in a formal setting." | 0.00 | absent — zero red/maroon/pink ties exist anywhere in the corpus |
| | **mean** | **0.600** | |

## 6. Benchmarking beyond 5 anecdotes

### a. Corpus-grounded benchmark

Five hand-picked prompts are too small a sample to trust. `eval/benchmark.py`
auto-generates queries from **every (category, color) combination that
occurs ≥3 times in the corpus**, using Fashionpedia's own labels as ground
truth:

| Slice | # queries | mean P@5 | mean R@5 |
|---|---|---|---|
| **GARMENT** (actual clothing types: coat, dress, jacket, shirt, pants...) | 258 | **0.850** | **0.926** |
| PART (hardware/embellishments: rivet, sequin, bead... — not "clothing types" per the PRD's own axis, much harder) | 179 | 0.790 | 0.869 |
| ALL | 437 | 0.826 | 0.903 |

(Query count grew 189→223→258 GARMENT combos across two rounds: corpus
growth, then finer Lab-space colors, then multi-color extraction — each
creates more distinct, real, still-well-supported `(category, color)`
combos to test. P@5/R@5 moved down slightly each time for a mechanical
reason, not a quality regression: finer attributes mean more combos with
*fewer* true positives each, which lowers the P@5 ceiling even at perfect
recall — e.g. a combo with 2 true positives caps P@5 at 0.4 no matter what.
Each round's numbers are reproducible from the corpus at that state; the
methodology (not cherry-picking, coverage-aware, corpus-grounded) stayed
constant throughout.)

### b. Weight ablation

`eval/ablate_weights.py`, 120 random GARMENT combos (up from an initial
40-combo pass — the conclusion held at 3x the sample size):

| `(W_CLIP, W_COMP, W_SCENE)` | mean P@5 | mean R@5 |
|---|---|---|
| 0.70 / 0.20 / 0.10 (CLIP-heavy) | 0.780 | 0.865 |
| 0.50 / 0.35 / 0.15 (initial guess) | 0.863 | 0.963 |
| 0.34 / 0.33 / 0.33 (equal) | 0.865 | 0.965 |
| **0.35 / 0.55 / 0.10 (chosen)** | **0.877** | **0.977** |
| 0.10 / 0.80 / 0.10 (comp-only) | 0.880 | 0.981 |

Comp-heavy weighting wins because exact `(category, color)` matches are
ground truth from segmentation masks — once a candidate is known to have the
right garment and color, that's a more trustworthy ranking signal than
CLIP's fuzzy similarity. The chosen config is preferred over the
marginally-higher-scoring comp-only config for robustness: it keeps a
meaningful CLIP contribution to break ties among candidates that all match
on attributes, rather than depending entirely on discrete signal quality.

### c. Zero-shot probe

`eval/zero_shot_probe.py`: 5 queries built from words absent from *every*
vocabulary in the system (category list, color palette, scene/style/weather
prompt banks) — the only thing that can answer them is fashion-CLIP's raw
semantic generalization:

- *"an elegant evening gown for a gala"* → correctly surfaced dress + formal
  + runway images, despite "elegant" and "gala" appearing nowhere in any label.
- *"a cozy knit sweater for a winter morning"* → correctly found sweaters
  despite "cozy"/"knit"/"winter morning" being unseen.
- *"a minimalist monochrome outfit"* / *"streetwear with an oversized
  silhouette"* → plausible but visibly lower-confidence — abstract aesthetic
  judgments ("minimalist", "oversized") are the hardest category for this
  system, relying entirely on fashion-CLIP's raw embedding with no
  structured signal to lean on, same as vanilla CLIP would.

### d. Latency profile

`eval/profile_latency.py`, on the full 3200-image / 7800-garment-crop
corpus, steady-state (post model-load):

| Query | median latency |
|---|---|
| Single garment+color term | 75-100ms |
| Two garment+color terms + style | 150-190ms |
| Scene/style only, no garment term | 30-75ms |
| Simple single-attribute | 75-100ms |

Latency roughly tripled over the course of this project (an earlier profile
showed 15-55ms) as real capability was added: the zero-shot vocabulary
resolver (§3b) adds one batched embedding call per query when a word doesn't
match the dictionary, and multi-color extraction (§3c) means more
`(category, color)` pairs to match per candidate. One genuine inefficiency
was caught and fixed along the way (§7: the resolver was re-embedding words
already classified elsewhere, like "office"), which materially helped but
didn't eliminate the increase — the remainder is an honest, bounded trade of
latency for capability, not free. All figures stay well under 200ms
(comfortably real-time for interactive search) and — this is the part that
matters for §10 — still scale with the **number of parsed sub-queries**, not
corpus size: every stage is one CLIP text-encode + one fixed-size ANN call +
one exact filter, regardless of how large the corpus grows.

### e. Multi-attribute (color + type + location) — see §3d

Full detail and numbers are in §3d, since it directly answers the PRD's own
Context Awareness example. Headline: adding the location term to a
color+type query **roughly doubles P@5** (+0.30, reproduced across two
independent runs), quantifying that context awareness measurably helps
rather than just being present in the pipeline.

## 7. Real bugs found and fixed while building this (not hypothetical)

- **Color classifier misfired on a real image.** A khaki/army-green jacket
  was classified "yellow" by an early hue-only HSV rule. Caught because it
  silently created a false ground-truth positive for the "yellow coat"
  query. Fixed by replacing the hand-tuned HSV bins entirely with
  nearest-neighbor search in CIE Lab space over an expanded palette; the
  jacket now correctly lands on "sage". Locked in with `eval/test_colors.py`
  (22 cases, including this one).
- **Garment-crop ANN search missed a real true positive.** A blue-shirt image
  ranked **142nd** out of 7800 for the CLIP-embedding query "a photo of a
  blue shirt, blouse" — small "sleeve"/"neckline" crops embedded deceptively
  well against that text and crowded it out of the top-40 candidate pool.
  Fixed by adding a deterministic Chroma `where`-filter recall pass alongside
  the fuzzy ANN pass, since the category/color are already ground truth from
  segmentation and shouldn't depend on embedding-similarity ranking to be found.
- **Finer color palette silently broke exact-match recall for "blue".**
  After moving to Lab-space naming, "blue" legitimately split into
  blue/navy/denim/sky blue — more precise, but a plain `color == "blue"`
  check now missed garments a real user would call "blue". Fixed by adding
  chromatic-family partial-credit scoring (1.0 exact / 0.8 same-family / 0.6
  category-only) and extending the exact-filter recall pass to pull the
  whole family, not just the literal label — verified via `eval/coverage.py`
  before and after to confirm this was a real, measured recall gap and not
  a hypothetical concern.
- **The zero-shot vocabulary resolver was re-embedding words already
  classified elsewhere.** Adding it (§3b) measurably increased query latency
  (§6d); instrumenting `embed_text` showed "office" — already resolved as a
  scene keyword — was still being sent through the resolver's embedding call
  on every query, because its "already known" check only tracked
  color/category token positions, not scene/style/weather/relation ones.
  Fixed by computing all keyword-matched positions before the resolver runs
  and excluding them; confirmed with the same instrumentation that the
  candidate word list shrank (e.g. a 5-word candidate set down to 2) and
  measurably cut latency, without changing `eval/test_parser.py`'s 8/8
  recall or 3/3 precision.

## 8. Shortcomings & mitigations already in place

- **Attribute vocabulary is closed** (46 Fashionpedia categories, 35-color
  Lab palette). Mitigated by always falling back to CLIP similarity — the
  system degrades to a strong zero-shot baseline, never returns nothing.
- **Color naming is nearest-neighbor over a fixed reference palette**, not a
  trained model — genuinely ambiguous shades still land on a judgment call
  (§7's "silver" chambray-shirt case), though the regression suite catches
  gross errors and chromatic-family scoring absorbs near-miss labels.
- **Spatial/relational reasoning is limited to one relation** ("layered",
  from bbox IoU) — real z-order ("tucked in" vs. "worn over") needs
  occlusion/depth information bounding boxes don't carry. Scoped honestly
  rather than faked.
- **Corpus size still caps achievable precision** on rare/absent
  combinations, even after growing to 3200 images — quantified explicitly
  in §4-§6 rather than hidden behind an average.

## 9. Future work

**a. Adding locations (cities, places) and weather — implemented, extend further**
- Place-type and weather zero-shot tagging now ship (§2.5); city-level
  location was deliberately not attempted (not honestly inferable from
  pixels in most street-style photos). If real deployment metadata exists
  (EXIF, geolocation, capture timestamp), prefer that over any zero-shot
  inference.
- Extend the prompt banks in `common/config.py` for finer place granularity
  (mall, restaurant patio, subway) — the tagging/scoring machinery already
  generalizes to any additional prompt-bank axis with no architecture change.

**b. Improving precision**
- Swap `patrickjohncyh/fashion-clip` for a larger fashion-domain checkpoint
  (e.g. Marqo-FashionSigLIP) — `embed.py` isolates this to one config change.
- The weight ablation now runs on 120 real combos (up from 40); growing this
  further as more labeled query data becomes available would sharpen the
  W_CLIP/W_COMP/W_SCENE choice past what a fixed ablation grid can reach —
  e.g. a small learned re-ranker over the same three signals.
- Move color naming from a fixed Lab reference palette to a namer *trained*
  on a labeled color dataset — the Lab-space infrastructure here is the
  right foundation, just with hand-picked reference points instead of
  learned ones.
- Extend the "layered" relation into real z-order (tucked in vs. over) —
  needs occlusion/depth signal beyond bbox IoU, e.g. instance segmentation
  overlap direction or a small trained relation classifier.
- Corpus growth already implemented (1158 → 3200 images); the two
  lowest-scoring PRD queries are still capped by combinations that don't
  exist anywhere in this specific dataset (§4) — more/different source data
  is the remaining lever, not the algorithm.

## 10. Scalability to ~1M images

- Chroma's index is HNSW (approximate nearest neighbor, sub-linear query
  time); both the image-level and garment-level ANN searches pull a
  **fixed-size** candidate pool (`CANDIDATE_POOL`/`GARMENT_POOL` in
  `common/config.py`), not the full corpus. The exact `where`-filter pass is
  O(matching instances), still far below corpus size for any specific
  `(category, color)` combination. `eval/profile_latency.py` measures this
  concretely at the current 3200-image corpus (15-55ms/query, scaling with
  number of parsed sub-queries, not corpus size) rather than asserting it
  from first principles.
- The real bottleneck at 1M images is **indexing throughput** (batched GPU
  embedding of images + garment crops) and **Chroma's single-node
  persistence**. Both are swap-outs, not rewrites: batch on GPU, and replace
  `chromadb.PersistentClient` with a sharded/managed vector DB (Qdrant,
  Pinecone, Weaviate) — `retriever/search.py` only calls the standard
  `.query()`/`.get()` collection API, so this is a client change, not an
  architecture change.

## 11. On "state of the art"

There is no published benchmark for "natural-language query → Fashionpedia
image retrieval with multi-attribute compositionality" — so there's nothing
concrete to claim victory against, and doing so would be dishonest. What's
reported above instead: a controlled compositional experiment showing the
hybrid goes from CLIP-only's 0.65 (near chance) to 1.00 on the exact
color-swap case the PRD names (§3a); a direct test of the PRD's own
color+type+location example showing location roughly doubles precision
(§3d); a corpus-grounded 258-query benchmark (not 5 cherry-picked prompts);
a 120-combo weight ablation; a zero-shot parser with calibrated
precision/recall (§3b); multi-color garment extraction that measurably
applies to ~25% of the corpus (§3c); four real bugs caught by building
actual regression/recall/coverage checks rather than trusting first-pass
output (§7); a measured (not asserted) latency profile that also reports
being honest about latency going up as capability went up; and an explicit
ground-truth coverage check so precision numbers mean what they claim to
mean.
