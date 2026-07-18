"""Garment color(s) from a segmentation mask.

Fashionpedia's attribute labels describe design details (necklines, closures),
not basic colors, so color is derived here from pixels — this is the "color
theory" axis and what lets retrieval separate "red shirt" from "blue shirt"
precisely, independent of CLIP's fuzzier color grounding.

Two robustness steps over a naive average:
  1. Erode the mask inward so we sample garment interior, not the boundary
     where background/skin bleeds in on loose polygons.
  2. Take the median (outlier-robust) in RGB, then name it in Lab space
     (common/colors.py).

A single "dominant" color is a real information loss for patterned/multi-color
garments -- a measured ~25% of garment-category instances in this corpus have
enough internal color spread that one blended median doesn't represent them
(e.g. a black-and-white striped shirt medians to gray, matching neither "black"
nor "white" queries). extract_colors() detects that spread and returns a
second color when it's large enough to matter, instead of silently discarding
it.

PERFORMANCE: mask rasterization/erosion/pixel extraction all operate on a crop
around the garment's bbox, not the full image. Measured why this matters:
after GPU-accelerating embedding (indexer/embed.py auto-detects CUDA -- ~8ms/
image warm, a real ~10x over CPU), color extraction became the dominant cost
of indexing at ~60ms/instance -- because the original version rasterized and
eroded a mask the *size of the whole image* for every single garment instance
regardless of how small it was, and re-converted the whole image to a numpy
array on every call. Cropping first operates on a region roughly the size of
the garment (often 5-20x fewer pixels than the full image), which is where
that time was actually going.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from common.colors import classify_color

# fraction of masked pixels that must be "far" from the primary color (in RGB
# units) before we bother reporting a second color -- calibrated against the
# corpus: real solid-color garments (shadow/fold gradients only) sit well
# under this; genuinely patterned/color-blocked garments sit well over it.
SECONDARY_COLOR_FRAC = 0.30
SECONDARY_COLOR_RGB_DIST = 60
MIN_PIXELS_FOR_SECONDARY = 80

# padding around a garment's bbox before cropping, so the erosion filter
# doesn't get starved of margin right at the polygon's own edge
CROP_PAD_PX = 4


def _polygon_bounds(segmentation):
    xs, ys = [], []
    for polygon in segmentation if isinstance(segmentation, list) else []:
        if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0:
            xs.extend(polygon[0::2])
            ys.extend(polygon[1::2])
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _masked_pixels(image: Image.Image, segmentation):
    bounds = _polygon_bounds(segmentation)
    if bounds is None:
        return np.empty((0, 3))

    x0, y0, x1, y1 = bounds
    left = max(0, int(x0) - CROP_PAD_PX)
    top = max(0, int(y0) - CROP_PAD_PX)
    right = min(image.width, int(x1) + CROP_PAD_PX + 1)
    bottom = min(image.height, int(y1) + CROP_PAD_PX + 1)
    if right <= left or bottom <= top:
        return np.empty((0, 3))

    crop_size = (right - left, bottom - top)
    mask = Image.new("L", crop_size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in segmentation:
        if isinstance(polygon, list) and len(polygon) >= 6 and len(polygon) % 2 == 0:
            shifted = [(v - left if i % 2 == 0 else v - top) for i, v in enumerate(polygon)]
            draw.polygon(shifted, fill=255)

    # erode ~2px inward to avoid boundary bleed; keep original if erosion empties it
    eroded = mask.filter(ImageFilter.MinFilter(5))
    mask_arr = np.array(eroded)
    if mask_arr.sum() == 0:
        mask_arr = np.array(mask)

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    img_arr = np.array(crop)
    return img_arr[mask_arr > 0]


def dominant_color_name(image: Image.Image, segmentation) -> str:
    """Single primary color name. Kept for callers that only want one label
    (e.g. a garment crop's headline color)."""
    pixels = _masked_pixels(image, segmentation)
    if pixels.shape[0] == 0:
        return "unknown"
    median_rgb = tuple(np.median(pixels, axis=0).astype(int))
    return classify_color(median_rgb)


def extract_colors(image: Image.Image, segmentation) -> list:
    """Primary color, plus a second color if the garment shows real internal
    color spread (patterned/color-blocked). Returns 1 or 2 color names,
    primary first, never duplicated."""
    pixels = _masked_pixels(image, segmentation)
    if pixels.shape[0] == 0:
        return ["unknown"]

    median_rgb = np.median(pixels, axis=0)
    primary = classify_color(tuple(median_rgb.astype(int)))

    if pixels.shape[0] < MIN_PIXELS_FOR_SECONDARY:
        return [primary]

    dists = np.linalg.norm(pixels - median_rgb, axis=1)
    far_mask = dists > SECONDARY_COLOR_RGB_DIST
    frac_far = far_mask.mean()
    if frac_far < SECONDARY_COLOR_FRAC or far_mask.sum() < MIN_PIXELS_FOR_SECONDARY:
        return [primary]

    secondary_rgb = np.median(pixels[far_mask], axis=0)
    secondary = classify_color(tuple(secondary_rgb.astype(int)))
    return [primary, secondary] if secondary != primary else [primary]
