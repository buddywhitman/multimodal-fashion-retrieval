"""Regression test for the Lab-space color classifier — the logic the whole
attribute-matching precision story depends on. Not a framework test, just a
runnable self-check.

Run: python -m eval.test_colors
"""
from common.colors import classify_color

CASES = [
    # primary / vivid colors
    ((255, 255, 0), "yellow"), ((200, 30, 30), "red"),
    ((20, 30, 90), "navy"), ((40, 90, 200), "blue"),
    ((0, 0, 0), "black"), ((255, 255, 255), "white"), ((128, 128, 128), "gray"),
    ((222, 196, 160), "beige"), ((110, 70, 40), "brown"), ((40, 140, 60), "green"),
    ((255, 150, 180), "pink"), ((120, 50, 150), "purple"),
    ((20, 130, 130), "teal"), ((255, 140, 0), "orange"),
    # perceptually-close pairs the earlier HSV-bin classifier collapsed --
    # this is the whole point of moving to Lab-space nearest neighbor
    ((128, 0, 0), "maroon"), ((90, 10, 30), "burgundy"),
    ((110, 120, 40), "olive"), ((170, 160, 110), "khaki"),
    ((25, 35, 90), "navy"), ((70, 110, 160), "denim"),
    # regression: this exact garment (a real army-green jacket from the
    # corpus) used to be misclassified "yellow" by the old hue-only rule
    ((129, 129, 104), "sage"),
    ((150, 140, 110), "sage"),
]


def run():
    ok = 0
    for rgb, expected in CASES:
        got = classify_color(rgb)
        status = "OK" if got == expected else "MISS"
        if got == expected:
            ok += 1
        print(f"  {status:4} {rgb} -> {got:12} (expected {expected})")
    print(f"\n{ok}/{len(CASES)} passed")
    assert ok == len(CASES), "color classifier regression"


if __name__ == "__main__":
    run()
