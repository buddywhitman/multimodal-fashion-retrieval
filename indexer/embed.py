"""Fashion-CLIP embedding wrapper — the one module that knows the model.

Image and text are projected into the same normalized space, so a text query
can be compared to image (and garment-crop) vectors by cosine similarity.
Uses HuggingFace transformers, so switching backbones is just changing
CLIP_MODEL_NAME.
"""
import numpy as np
import torch

from common.config import CLIP_MODEL_NAME, EMBED_BATCH

_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _as_tensor(out):
    # transformers>=5 returns a ModelOutput wrapper whose pooler_output is the
    # projected (aligned) embedding; older versions returned the tensor directly.
    return out.pooler_output if hasattr(out, "pooler_output") else out


def _load():
    global _model, _processor
    if _model is None:
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device).eval()
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _model, _processor


def embed_images(pil_images):
    """list[PIL.Image] -> np.ndarray [N, D], L2-normalized."""
    model, processor = _load()
    out = []
    for i in range(0, len(pil_images), EMBED_BATCH):
        batch = pil_images[i:i + EMBED_BATCH]
        inputs = processor(images=batch, return_tensors="pt").to(_device)
        with torch.no_grad():
            feats = _as_tensor(model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def embed_text(texts):
    """list[str] -> np.ndarray [N, D], L2-normalized, same space as embed_images."""
    model, processor = _load()
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(_device)
        with torch.no_grad():
            feats = _as_tensor(model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)
