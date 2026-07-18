"""Validates the PRD's §2 dataset requirement directly: "The data must
contain variations across three primary axes: Environment, Clothing Types,
Color Theory." This has been true by construction (scene tagging, category
extraction, and color extraction all exist and are exercised throughout
eval/), but never actually reported as evidence -- this closes that gap.

Reads straight from the already-built index (no recomputation, same
philosophy as eval/ground_truth.py).

Run: python -m eval.corpus_composition
"""
from collections import Counter

import chromadb

from common.config import CHROMA_DIR, IMAGE_COLLECTION
from eval.ground_truth import load_indexed_records


def run():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    img_col = client.get_collection(IMAGE_COLLECTION)
    all_meta = img_col.get(include=["metadatas"])["metadatas"]

    print(f"Corpus composition across the PRD's 3 required axes ({len(all_meta)} total images)\n")

    # --- Environment: office interiors, urban streets, parks, home settings ---
    scenes = Counter(m["scene"] for m in all_meta if m.get("scene"))
    print("1. ENVIRONMENT (zero-shot scene tag, all images):")
    for scene, n in scenes.most_common():
        print(f"     {scene:8}  {n:>5}  ({100*n/len(all_meta):.1f}%)")
    required_envs = {"office", "street", "park", "home"}
    present = required_envs & set(scenes)
    print(f"   PRD's named environments present: {sorted(present)} "
          f"({len(present)}/{len(required_envs)})\n")

    # --- Clothing Types: formal, casual, outerwear ---
    styles = Counter(m["style"] for m in all_meta if m.get("style"))
    print("2. CLOTHING TYPES (zero-shot style tag, all images):")
    for style, n in styles.most_common():
        print(f"     {style:8}  {n:>5}  ({100*n/len(all_meta):.1f}%)")
    categories = Counter()
    for _iid, _fn, pairs, _scene in load_indexed_records():
        categories.update(cat for cat, _col in pairs)
    print(f"   distinct garment/part categories present: {len(categories)}")
    print(f"   top 10 by instance count: {[c for c, _ in categories.most_common(10)]}\n")

    # --- Color Theory: wide palette of garment colors ---
    colors = Counter()
    for _iid, _fn, pairs, _scene in load_indexed_records():
        colors.update(col for _cat, col in pairs if col != "unknown")
    print(f"3. COLOR THEORY (extracted garment colors, {len(colors)} distinct colors used):")
    for color, n in colors.most_common():
        print(f"     {color:12}  {n:>5}")


if __name__ == "__main__":
    run()
