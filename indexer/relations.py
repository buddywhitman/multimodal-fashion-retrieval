"""Spatial/relational attributes between garment instances in the same image.

Bounding-box overlap alone can tell you two garments in the same body region
(both upperbody, or both lowerbody) are worn together/layered, but not which
is "on top" -- that needs actual depth/occlusion reasoning bboxes don't carry.
extract_layered_pairs() gives the cheap, undirected version (co-occurrence
only). indexer/depth_relations.py adds real z-order on top, using a monocular
depth model -- run only on the candidate pairs this module finds, since most
images have none and full-corpus depth estimation would be needlessly
expensive (see that module's docstring for the cost tradeoff).
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


def extract_layered_instance_pairs(instances):
    """list[GarmentInstance] -> list[(i, j)] instance-index pairs (i<j) that
    are candidate layered pairs -- same body region, overlapping bboxes,
    different categories. Instance-level (not just category names) so a
    caller can access each instance's segmentation for real z-order via
    indexer/depth_relations.py."""
    pairs = []
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
                pairs.append((i, j))
    return pairs


def extract_layered_pairs(instances):
    """list[GarmentInstance] -> list[(category_a, category_b)], sorted,
    deduped -- the undirected version, for callers that don't need z-order."""
    pairs = set()
    for i, j in extract_layered_instance_pairs(instances):
        pairs.add(tuple(sorted((instances[i].category, instances[j].category))))
    return sorted(pairs)
