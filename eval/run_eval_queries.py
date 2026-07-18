"""Runs the 5 evaluation prompts from the assignment PRD and saves a contact
sheet (query + top-k thumbnails) per prompt to eval/outputs/.

Run: python -m eval.run_eval_queries
"""
import os

import matplotlib.pyplot as plt
from PIL import Image

from retriever.search import search

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
K = 5


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, query in enumerate(EVAL_QUERIES, 1):
        results, parsed = search(query, k=K)

        fig, axes = plt.subplots(1, K, figsize=(4 * K, 5.5))
        sub = (f"garments={parsed['garments']} scenes={parsed['scenes']} styles={parsed['styles']} "
               f"weathers={parsed['weathers']} relation={parsed['relation']}")
        fig.suptitle(f"{query}\n{sub}", fontsize=10, y=0.99)
        for ax, r in zip(axes, results):
            img = Image.open(r["path"]).convert("RGB")
            ax.imshow(img)
            cats = ", ".join(r["categories"][:3]) + ("..." if len(r["categories"]) > 3 else "")
            cols = ", ".join(r["colors"][:3]) + ("..." if len(r["colors"]) > 3 else "")
            ax.set_title(f"score={r['score']:.2f} [{r['scene']}/{r['style']}]\n{cats}\n{cols}", fontsize=8, wrap=True)
            ax.axis("off")

        out_path = os.path.join(OUT_DIR, f"query_{i}.png")
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        print(f"[{i}] {query!r} -> {out_path}")


if __name__ == "__main__":
    run()
