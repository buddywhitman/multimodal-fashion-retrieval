"""Real z-order for layered garments: which one is worn *over* the other.

indexer/relations.py's bbox-IoU "layered" check can tell you two garments in
the same body region overlap, but not which is on top -- that needs actual
depth/occlusion information, which bounding boxes don't carry. This module
adds that, using a monocular depth model (Depth-Anything-V2-Small, loaded via
`transformers` -- no new dependency) rather than pretending bbox geometry can
answer a question it structurally can't.

Verified empirically before relying on it: for these depth maps, higher pixel
value = closer to the camera (checked foreground-subject vs. background
region on a real corpus photo). For a layered pair, the garment whose mask
has the higher median depth value is the one worn on top ("over").

Scoped deliberately: only run the (comparatively expensive, ~74ms/image warm
vs. ~8ms for CLIP embedding) depth model on images that indexer/relations.py
already flagged as having a candidate layered pair via the cheap bbox check
-- not every image, since most don't have overlapping same-region garments
at all.
"""
import numpy as np
import torch
from PIL import Image, ImageDraw

_pipe = None
_device = 0 if torch.cuda.is_available() else -1


def _load():
    global _pipe
    if _pipe is None:
        from transformers import pipeline
        _pipe = pipeline(task="depth-estimation",
                          model="depth-anything/Depth-Anything-V2-Small-hf", device=_device)
    return _pipe


def estimate_depth(image: Image.Image) -> np.ndarray:
    """PIL image -> depth array, same size as the input, uint8. Higher value
    = closer to camera."""
    out = _load()(image)
    return np.array(out["depth"])


def _rasterize(size, segmentation):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation if isinstance(segmentation, list) else []:
        if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0:
            draw.polygon(polygon, fill=255)
    return np.array(mask) > 0


# minimum median-depth gap (out of 0-255) before we trust an ordering --
# below this, the two garments are ambiguously close in depth (e.g. a shirt
# barely peeking from a jacket's collar) and we'd rather report "unknown"
# than guess
MIN_DEPTH_GAP = 4


def order_pair(depth: np.ndarray, seg_a, seg_b):
    """Returns (over_category_index, under_category_index) as 0/1, or None if
    the depths are too close to call confidently."""
    size = (depth.shape[1], depth.shape[0])  # PIL (width, height)
    mask_a = _rasterize(size, seg_a)
    mask_b = _rasterize(size, seg_b)
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return None

    depth_a = np.median(depth[mask_a])
    depth_b = np.median(depth[mask_b])
    if abs(depth_a - depth_b) < MIN_DEPTH_GAP:
        return None
    return (0, 1) if depth_a > depth_b else (1, 0)
