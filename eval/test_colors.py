"""Regression test for the color namer — the logic the whole
attribute-matching precision story depends on. Not a framework test, just a
runnable self-check.

The namer's reference palette is the XKCD color survey (data/raw/xkcd_colors.txt,
CC0, ~800 names after filtering) rather than a hand-picked list, so exact
output names are specific XKCD vocabulary ("cornflower blue", not just
"blue") — this test locks in the *current correct* mapping as a regression
baseline, and includes the real bug this project hit twice: a muted
olive/khaki color being misclassified into the yellow family.

Run: python -m eval.test_colors
"""
from common.colors import classify_color, family_members, same_family

CASES = [
    # primary / vivid colors -> specific XKCD names, locked in as baseline
    ((255, 255, 0), "bright yellow"), ((200, 30, 30), "scarlet"),
    ((128, 0, 0), "dark red"), ((90, 10, 30), "burgundy"),
    ((20, 30, 90), "navy blue"), ((40, 90, 200), "cornflower blue"),
    ((0, 0, 0), "black"), ((255, 255, 255), "white"), ((128, 128, 128), "medium grey"),
    ((222, 196, 160), "putty"), ((110, 70, 40), "cocoa"), ((40, 140, 60), "darkish green"),
    ((255, 150, 180), "pinky"), ((120, 50, 150), "purply"), ((20, 130, 130), "dark cyan"),
    ((255, 140, 0), "tangerine"),
    # regression: this exact garment (a real army-green/khaki jacket from the
    # corpus) was misclassified "yellow" twice during this project -- once by
    # an early hue-only HSV rule, once by a hue-bucket boundary bug in the
    # family grouping. Locked in here so it can't happen a third time.
    ((129, 129, 104), "brown grey"),
    ((150, 140, 110), "brown grey"),
]

FAMILY_CASES = [
    # (color_a, color_b, expected_same_family)
    ("black", "white", False),          # neutrals never grouped -- the whole point of the query
    ("black", "black", True),
    ("navy blue", "cornflower blue", True),
    ("bright yellow", "mustard", True), # both land in the "yellow" hue bucket
    ("bright yellow", "army green", False),  # yellow vs. yellow-green: the bucket-boundary bug this locks in
]


def run():
    ok = 0
    for rgb, expected in CASES:
        got = classify_color(rgb)
        status = "OK" if got == expected else "MISS"
        if got == expected:
            ok += 1
        print(f"  {status:4} {rgb} -> {got:16} (expected {expected})")

    fam_ok = 0
    print()
    for a, b, expected in FAMILY_CASES:
        got = same_family(a, b)
        status = "OK" if got == expected else "MISS"
        if got == expected:
            fam_ok += 1
        print(f"  {status:4} same_family({a!r}, {b!r}) = {got} (expected {expected})")

    total_ok = ok + fam_ok
    total = len(CASES) + len(FAMILY_CASES)
    print(f"\n{total_ok}/{total} passed")
    assert total_ok == total, "color classifier regression"


if __name__ == "__main__":
    run()
