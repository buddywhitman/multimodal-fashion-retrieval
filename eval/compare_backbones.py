"""Direct comparison of CLIP backbones on THIS corpus, not a trust of the
checkpoint author's own reported benchmark numbers (Marqo reports +57% on
their own eval set, which may not transfer to Fashionpedia street/runway
photos or to compositional color-swap discrimination specifically).

Metric: the same color-swap discrimination test as eval/compositional.py
(CLIP-image-embedding vs CLIP-text-embedding only -- no hybrid/structured
attribute matching, so this isolates what the *backbone itself* grounds,
not what our exact-match re-ranking adds on top). Both models are loaded
directly here (not through indexer/embed.py's single cached model) so both
can be compared in one run.

Run: python -m eval.compare_backbones
"""
import random

import numpy as np
import torch
from PIL import Image

from eval.benchmark import GARMENT_CATEGORIES
from eval.ground_truth import primary_color_garments

MAX_PAIRS = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _find_swap_pairs():
    pairs = []
    for image_id, file_name, garments in primary_color_garments(GARMENT_CATEGORIES):
        for i in range(len(garments)):
            for j in range(i + 1, len(garments)):
                (ca, cola), (cb, colb) = garments[i], garments[j]
                if cola != colb:
                    pairs.append(((image_id, file_name), (ca, cola), (cb, colb)))
                    break
            else:
                continue
            break
    return pairs


def _q(cat_a, col_a, cat_b, col_b):
    ha, hb = cat_a.split(",")[0], cat_b.split(",")[0]
    return f"a person wearing a {col_a} {ha} and a {col_b} {hb}"


_PATH_BY_ID = None


def _resolve_path(image_id, file_name):
    # eval/ground_truth.py's image_id comes from Chroma (a string);
    # indexer/dataset.py's is an int -- normalize to string once, cached.
    global _PATH_BY_ID
    if _PATH_BY_ID is None:
        from indexer.dataset import load_dataset
        _PATH_BY_ID = {str(rec.image_id): rec.path for rec in load_dataset()}
    return _PATH_BY_ID.get(str(image_id))


class _TransformersBackend:
    name = "patrickjohncyh/fashion-clip (transformers)"

    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    def embed(self, images=None, texts=None):
        with torch.no_grad():
            if images is not None:
                inputs = self.processor(images=images, return_tensors="pt").to(DEVICE)
                out = self.model.get_image_features(**inputs)
            else:
                inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
                out = self.model.get_text_features(**inputs)
            feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


class _OpenClipBackend:
    name = "Marqo/marqo-fashionCLIP (open_clip)"

    def __init__(self):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionCLIP")
        self.model = self.model.to(DEVICE).eval()
        self.tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionCLIP")

    def embed(self, images=None, texts=None):
        with torch.no_grad():
            if images is not None:
                pixel_values = torch.stack([self.preprocess(im) for im in images]).to(DEVICE)
                feats = self.model.encode_image(pixel_values)
            else:
                tokens = self.tokenizer(texts).to(DEVICE)
                feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


def _evaluate(backend, pairs):
    correct, margins = 0, []
    for (image_id, file_name), (ca, cola), (cb, colb) in pairs:
        path = _resolve_path(image_id, file_name)
        if path is None:
            continue
        img = Image.open(path).convert("RGB")
        emb = backend.embed(images=[img])[0]
        qc, qs = backend.embed(texts=[_q(ca, cola, cb, colb), _q(ca, colb, cb, cola)])
        sim_c, sim_s = float(np.dot(emb, qc)), float(np.dot(emb, qs))
        correct += sim_c > sim_s
        margins.append(sim_c - sim_s)
    n = len(margins)
    return correct / n, float(np.mean(margins)), n


def run():
    random.seed(0)
    pairs = _find_swap_pairs()
    random.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]
    print(f"Backbone comparison: color-swap discrimination over up to {len(pairs)} images\n")

    for backend_cls in (_TransformersBackend, _OpenClipBackend):
        backend = backend_cls()
        acc, margin, n = _evaluate(backend, pairs)
        print(f"  {backend.name}")
        print(f"    discrimination accuracy: {acc:.3f}   mean margin: {margin:+.4f}   (n={n})\n")
        del backend
        torch.cuda.empty_cache() if DEVICE == "cuda" else None


if __name__ == "__main__":
    run()
