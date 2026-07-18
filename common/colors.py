"""Fashion color vocabulary + a data-driven color namer.

Two jobs:
  - PALETTE: the color *words* the query parser recognizes.
  - classify_color(): names a garment's dominant color from its median RGB.

Naming is nearest-neighbor in CIE Lab space (perceptually uniform -- distance
tracks how different two colors actually *look*) against a reference table
built from the XKCD color survey (https://xkcd.com/color/rgb.txt, CC0): ~950
crowd-sourced human colors-name judgments, filtered down to ~800 usable
single/two-word names. This replaces an earlier version's 35 hand-picked
reference points with an externally-sourced, human-labeled dataset -- more
of a "trained on real labeled data" namer than one person's color intuition,
and with far finer resolution (bordeaux vs. burgundy vs. maroon vs. wine, not
just "maroon").

Chromatic families (for graded partial-credit matching -- see
retriever/search.py) are computed programmatically from each color's hue/
saturation, not hand-curated: with ~800 reference colors, manually grouping
them isn't tractable, and a hue-wheel bucketing rule generalizes to any size
palette without maintenance.
"""
import colorsys
import json
import os

_XKCD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "raw", "xkcd_colors.txt")
_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "color_palette_cache.json")

# names surviving XKCD's raw list that read as informal/joke names rather
# than usable color words in a fashion context
_INFORMAL_BLOCKLIST = {
    "shit", "puke", "poop", "piss", "fart", "ass", "damn", "hell", "fuck",
    "sex", "crap", "sperm", "vomit", "booger", "nasty", "gross", "ugly",
    "toxic", "windows", "ugh", "diarrhea", "snot",
}

# a handful of query-relevant synonyms XKCD spells differently or not at all
COLOR_SYNONYMS = {
    "grey": "gray", "stone": "stone gray" if False else "gray", "cognac": "brown",
}


def _load_xkcd_palette():
    """Parses data/raw/xkcd_colors.txt into {name: (r,g,b)}, filtered to
    clean single/two-word names. Falls back to a small built-in palette if
    the file isn't present (e.g. a fresh checkout before running the
    download step in README.md)."""
    if not os.path.exists(_XKCD_PATH):
        return {
            "black": (20, 20, 20), "white": (245, 245, 245), "gray": (128, 128, 128),
            "red": (200, 30, 30), "maroon": (128, 0, 0), "orange": (255, 140, 0),
            "yellow": (240, 220, 30), "green": (40, 140, 60), "blue": (40, 90, 200),
            "navy": (20, 30, 90), "purple": (120, 50, 150), "pink": (255, 150, 180),
            "brown": (110, 70, 40), "beige": (222, 196, 160), "teal": (20, 130, 130),
        }
    palette = {}
    with open(_XKCD_PATH, "r", encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, hexcode = parts[0], parts[1]
            words = name.split()
            if len(words) > 2 or any(w in _INFORMAL_BLOCKLIST for w in words):
                continue
            hexcode = hexcode.lstrip("#")
            rgb = tuple(int(hexcode[i:i + 2], 16) for i in (0, 2, 4))
            palette[name] = rgb
    return palette


_PALETTE_RGB = _load_xkcd_palette()
PALETTE = set(_PALETTE_RGB.keys())


def _srgb_to_linear(c):
    c = c / 255.0
    return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92


def _rgb_to_lab(rgb):
    r, g, b = (_srgb_to_linear(c) for c in rgb)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116

    fx, fy, fz = f(x), f(y), f(z)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    bb = 200 * (fy - fz)
    return (L, a, bb)


_PALETTE_LAB = {name: _rgb_to_lab(rgb) for name, rgb in _PALETTE_RGB.items()}
_PALETTE_NAMES = list(_PALETTE_LAB.keys())
_PALETTE_LAB_ARRAY = None  # lazily built numpy array for vectorized nearest-neighbor


def classify_color(rgb):
    """median RGB (0-255 tuple) -> nearest color name in Lab space, from the
    ~800-name XKCD-derived reference palette."""
    global _PALETTE_LAB_ARRAY
    import numpy as np
    if _PALETTE_LAB_ARRAY is None:
        _PALETTE_LAB_ARRAY = np.array([_PALETTE_LAB[n] for n in _PALETTE_NAMES])
    target = np.array(_rgb_to_lab(rgb))
    dists = np.sum((_PALETTE_LAB_ARRAY - target) ** 2, axis=1)
    return _PALETTE_NAMES[int(dists.argmin())]


def canonical_color(word):
    return COLOR_SYNONYMS.get(word, word)


# --- Chromatic families, computed programmatically (not hand-curated) -----
# Bucketed by hue (12 wheel sectors matching common color-word boundaries)
# plus an achromatic bucket for low-saturation/very dark/very light colors.
# With ~800 reference names, hand-grouping isn't maintainable; a hue-wheel
# rule generalizes to any palette size.
_HUE_BUCKETS = [
    (18, "red"), (45, "orange"), (75, "yellow"), (95, "yellow-green"),
    (155, "green"), (195, "teal"), (250, "blue"),
    (285, "purple"), (320, "magenta"), (345, "pink"), (360, "red"),
]


def _hue_bucket(rgb):
    r, g, b = (c / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.12 or v < 0.10:
        return "neutral"
    hue = h * 360.0
    for cutoff, name in _HUE_BUCKETS:
        if hue < cutoff:
            return name
    return "red"


_FAMILY_CACHE = {}


def _family_of(color_name):
    if color_name not in _FAMILY_CACHE:
        rgb = _PALETTE_RGB.get(color_name)
        _FAMILY_CACHE[color_name] = _hue_bucket(rgb) if rgb else None
    return _FAMILY_CACHE[color_name]


def same_family(color_a, color_b):
    if color_a == color_b:
        return True
    fa, fb = _family_of(color_a), _family_of(color_b)
    if fa is None or fa != fb:
        return False
    # neutrals (black/white/gray/...) are never "same family" as each other --
    # those differences are the whole point of the query, unlike blue-vs-azure
    return fa != "neutral"


def family_members(color):
    """All palette colors in the same chromatic family as `color`, including
    itself. Note: with ~800 names this can be a large set (e.g. dozens of
    "blue" shades) -- that's intentional, it's exactly the graded-credit
    scope a query like "a blue shirt" should match."""
    target_family = _family_of(color)
    if target_family is None:
        return {color}
    if target_family == "neutral":
        return {color}  # neutrals (black/white/gray) are NOT grouped -- see below
    return {n for n in _PALETTE_NAMES if _family_of(n) == target_family}
