"""Command-line entry point for Part B.

    python -m retriever.cli "A red tie and a white shirt in a formal setting" --k 5
"""
import argparse

from retriever.search import search


def main():
    ap = argparse.ArgumentParser(description="Natural-language fashion image search")
    ap.add_argument("query", type=str)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    results, parsed = search(args.query, k=args.k)

    print(f"\nQuery: {args.query!r}")
    print(f"  garments={parsed['garments'] or '(none)'}  scenes={parsed['scenes']}  "
          f"styles={parsed['styles']}  weathers={parsed['weathers']}  relation={parsed['relation']}\n")
    for rank, r in enumerate(results, 1):
        print(f"{rank:>2}. score={r['score']:.3f}  (img={r['image_sim']:.3f} comp={r['comp_score']:.3f} tag={r['tag_score']:.3f})")
        print(f"    {r['file_name']}  [scene={r['scene']}, style={r['style']}, weather={r['weather']}]")
        print(f"    categories={r['categories'][:5]} colors={r['colors'][:5]}")


if __name__ == "__main__":
    main()
