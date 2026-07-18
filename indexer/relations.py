"""Spatial/relational attributes between garment instances in the same image.

Scoped honestly: bounding boxes alone don't reliably tell you z-order (is the
shirt tucked *into* the pants, or is the jacket worn *over* the shirt?) --
that needs actual depth/occlusion reasoning this project doesn't have. What
bbox geometry *does* reliably tell you is co-occurrence in the same body
region: two garments in the same body area (both upperbody, or both
lowerbody) with substantial bbox overlap are being worn together/layered,
regardless of which is "on top". So "layered" is the one relation implemented
here — not a full solve of "tucked in" vs "over", but a real, honestly-scoped
step past "no spatial reasoning at all", and the same (A, B) pair format
extends cleanly to a real z-order signal later if one becomes available
(e.g. from instance depth ordering or occlusion masks).
"""

LAYERABLE_SUPERCATEGORIES = {"upperbody", "lowerbody"}
IOU_THRESHOLD = 0.20


def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def extract_layered_pairs(instances):
    """list[GarmentInstance] -> list[(category_a, category_b)], sorted, deduped."""
    pairs = set()
    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            a, b = instances[i], instances[j]
            if a.supercategory != b.supercategory:
                continue
            if a.supercategory not in LAYERABLE_SUPERCATEGORIES:
                continue
            if a.category == b.category:
                continue
            if _bbox_iou(a.bbox, b.bbox) >= IOU_THRESHOLD:
                pairs.add(tuple(sorted((a.category, b.category))))
    return sorted(pairs)
