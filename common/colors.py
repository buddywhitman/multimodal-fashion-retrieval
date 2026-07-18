"""Fashion color vocabulary + a perceptual (CIE Lab) color namer.

Two jobs:
  - PALETTE keys are the set of color *words* the query parser recognizes.
  - classify_color() names a garment's dominant color from its median RGB.

Naming is nearest-neighbor in CIE Lab space, not hand-tuned HSV thresholds.
Lab is built to be perceptually uniform (Euclidean distance in Lab tracks how
different two colors *look* to a human, unlike RGB or raw HSV), so this both
removes the hand-tuned hue/saturation/value cutoffs of the earlier version and
gives finer resolution for exactly the shades fashion queries hinge on:
burgundy vs. maroon, khaki vs. olive vs. beige, navy vs. denim blue. Adding a
new distinguishable color is "add one more Lab reference point," not "find
another hue bin boundary that doesn't break the existing ones."
"""

# sRGB reference points for each color name. Chosen to cover the fashion
# color vocabulary people actually query with, including several
# perceptually-close pairs (maroon/burgundy, khaki/olive/beige, navy/denim)
# that a coarse palette collapses.
_PALETTE_RGB = {
    "black": (20, 20, 20), "charcoal": (55, 55, 58), "gray": (128, 128, 128),
    "silver": (192, 192, 196), "white": (245, 245, 245),
    "navy": (25, 35, 90), "blue": (40, 90, 200), "denim": (70, 110, 160),
    "sky blue": (135, 190, 230), "teal": (20, 130, 130), "turquoise": (40, 180, 170),
    "green": (40, 140, 60), "forest green": (20, 90, 40), "mint": (150, 220, 180),
    "olive": (110, 120, 40), "khaki": (170, 160, 110), "sage": (135, 133, 108),
    "yellow": (240, 220, 30), "mustard": (200, 160, 40), "gold": (200, 170, 60),
    "orange": (255, 140, 0), "coral": (255, 120, 90),
    "red": (200, 30, 30), "maroon": (128, 0, 0), "burgundy": (90, 10, 30),
    "brick": (150, 60, 40), "rust": (160, 80, 40),
    "pink": (255, 150, 180), "hot pink": (255, 90, 160), "blush": (230, 180, 180),
    "purple": (120, 50, 150), "lavender": (190, 160, 220),
    "brown": (110, 70, 40), "chocolate": (80, 45, 20),
    "beige": (222, 196, 160), "tan": (200, 170, 120), "cream": (240, 225, 190),
}

PALETTE = set(_PALETTE_RGB.keys())


def _srgb_to_linear(c):
    c = c / 255.0
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def _rgb_to_lab(rgb):
    r, g, b = (_srgb_to_linear(c) for c in rgb)
    # sRGB (D65) -> XYZ
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    # normalize by D65 white point, then XYZ -> Lab
    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return (L, a, bb)


_PALETTE_LAB = {name: _rgb_to_lab(rgb) for name, rgb in _PALETTE_RGB.items()}


def classify_color(rgb):
    """median RGB (0-255 tuple) -> nearest color name in Lab space."""
    target = _rgb_to_lab(rgb)
    best_name, best_dist = None, float("inf")
    for name, lab in _PALETTE_LAB.items():
        dist = sum((t - p) ** 2 for t, p in zip(target, lab))
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name


# query-side synonym folding: words with no dedicated palette entry
COLOR_SYNONYMS = {
    "grey": "gray", "stone": "gray", "cognac": "brown",
}


def canonical_color(word):
    return COLOR_SYNONYMS.get(word, word)


# Chromatic neighborhoods for partial-credit matching. A finer palette (16
# colors -> 35+) means "blue" now legitimately splits into blue/navy/denim/sky
# blue -- a real improvement in label precision, but a user who says "a blue
# shirt" is reasonably satisfied by any of those, not only the exact label
# "blue". Neutrals (black/white/gray/charcoal/silver) are deliberately NOT
# grouped here: those differences are usually the point of the query, unlike
# denim-vs-blue.
COLOR_FAMILIES = {
    "blue": {"blue", "navy", "denim", "sky blue"},
    "red": {"red", "maroon", "burgundy", "brick", "rust"},
    "green": {"green", "forest green", "mint", "olive"},
    "yellow": {"yellow", "mustard", "gold"},
    "pink": {"pink", "hot pink", "blush"},
    "purple": {"purple", "lavender"},
    "brown": {"brown", "chocolate", "tan", "khaki", "sage"},
    "beige": {"beige", "tan", "cream"},
    "teal": {"teal", "turquoise"},
}
_COLOR_TO_FAMILY = {c: fam for fam, members in COLOR_FAMILIES.items() for c in members}


def same_family(color_a, color_b):
    if color_a == color_b:
        return True
    return _COLOR_TO_FAMILY.get(color_a) is not None and _COLOR_TO_FAMILY.get(color_a) == _COLOR_TO_FAMILY.get(color_b)


def family_members(color):
    """All colors in the same chromatic family as `color`, including itself."""
    return COLOR_FAMILIES.get(_COLOR_TO_FAMILY.get(color), {color})
