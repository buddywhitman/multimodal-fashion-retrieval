"""Fashion-CLIP embedding wrapper — the one module that knows the model.

Image and text are projected into the same normalized space, so a text query
can be compared to image (and garment-crop) vectors by cosine similarity.

Two loading backends, chosen by CLIP_BACKEND in common/config.py:
  "transformers" -- HuggingFace-native checkpoints (CLIPModel/CLIPProcessor).
  "open_clip"    -- OpenCLIP-native checkpoints loaded via the "hf-hub:"
                     convention. Needed for some fashion-domain checkpoints
                     (e.g. Marqo/marqo-fashionCLIP) that are packaged in
                     OpenCLIP's state-dict format, not HuggingFace's --
                     verified directly: loading Marqo's checkpoint via
                     transformers.CLIPModel produces hundreds of MISSING/
                     UNEXPECTED keys (an untrained model wearing the right
                     name), while open_clip's own loader reads it correctly.
                     "Swapping backbones is a one-line config change" is
                     only true when the new checkpoint's packaging matches;
                     this module now handles both without a rewrite.
"""
import numpy as np
import torch

from common.config import CLIP_BACKEND, CLIP_MODEL_NAME, EMBED_BATCH

_model = None
_processor = None   # transformers backend: CLIPProcessor
_preprocess = None  # open_clip backend: image transform
_tokenizer = None   # open_clip backend: text tokenizer
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _as_tensor(out):
    # transformers>=5 returns a ModelOutput wrapper whose pooler_output is the
    # projected (aligned) embedding; older versions returned the tensor directly.
    return out.pooler_output if hasattr(out, "pooler_output") else out


def _load():
    global _model, _processor, _preprocess, _tokenizer
    if _model is not None:
        return

    if CLIP_BACKEND == "open_clip":
        import open_clip
        _model, _, _preprocess = open_clip.create_model_and_transforms(f"hf-hub:{CLIP_MODEL_NAME}")
        _tokenizer = open_clip.get_tokenizer(f"hf-hub:{CLIP_MODEL_NAME}")
        _model = _model.to(_device).eval()
    else:
        from transformers import CLIPModel, CLIPProcessor
        _model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(_device).eval()
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)


def embed_images(pil_images):
    """list[PIL.Image] -> np.ndarray [N, D], L2-normalized."""
    _load()
    out = []
    for i in range(0, len(pil_images), EMBED_BATCH):
        batch = pil_images[i:i + EMBED_BATCH]
        with torch.no_grad():
            if CLIP_BACKEND == "open_clip":
                pixel_values = torch.stack([_preprocess(im) for im in batch]).to(_device)
                feats = _model.encode_image(pixel_values)
            else:
                inputs = _processor(images=batch, return_tensors="pt").to(_device)
                feats = _as_tensor(_model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)


def embed_text(texts):
    """list[str] -> np.ndarray [N, D], L2-normalized, same space as embed_images."""
    _load()
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        with torch.no_grad():
            if CLIP_BACKEND == "open_clip":
                tokens = _tokenizer(batch).to(_device)
                feats = _model.encode_text(tokens)
            else:
                inputs = _processor(text=batch, return_tensors="pt", padding=True, truncation=True).to(_device)
                feats = _as_tensor(_model.get_text_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0)
