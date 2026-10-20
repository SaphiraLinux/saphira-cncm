#!/usr/bin/env python3
"""Export eval inputs to plain text the C app can read (no pickle in C).
Usage:
  python3 tools/export_data.py sudoku --in data/sudoku/easy --out weights/sudoku-easy
  python3 tools/export_data.py text --in data/xsum --out weights/xsum \
      --split test --cols passages.pickle --refcols reference_summaries.pickle
Writes <out>/inputs.txt (one item per line: ints csv, or raw text) and
<out>/refs.txt (same layout; empty lines for unconditional tasks).
"""
import argparse
import os
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["sudoku", "text"])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--cols", default="")
    ap.add_argument("--refcols", default="")
    args = ap.parse_args()

    if args.kind == "sudoku":
        items = pickle.load(open(os.path.join(args.inp, "puzzles.pickle"), "rb"))
        refs = pickle.load(open(os.path.join(args.inp, "solutions.pickle"), "rb"))
        with open(os.path.join(args.out, "inputs.txt"), "w") as f:
            for ids in items:
                f.write(",".join(str(int(x)) for x in ids) + "\n")
        with open(os.path.join(args.out, "refs.txt"), "w") as f:
            for ids in refs:
                f.write(",".join(str(int(x)) for x in ids) + "\n")
        print(f"sudoku: {len(items)} items")
        return

    items = pickle.load(open(os.path.join(args.inp, args.cols), "rb"))
    refs = pickle.load(open(os.path.join(args.inp, args.refcols), "rb")) if args.refcols else [""] * len(items)
    with open(os.path.join(args.out, "inputs.txt"), "w") as f:
        for it in items:
            if isinstance(it, str):
                f.write(it.replace("\n", " ") + "\n")
            else:
                f.write(",".join(str(int(x)) for x in it) + "\n")
    with open(os.path.join(args.out, "refs.txt"), "w") as f:
        for r in refs:
            f.write((r if isinstance(r, str) else ",".join(str(int(x)) for x in r)).replace("\n", " ") + "\n")
    print(f"text: {len(items)} items")


if __name__ == "__main__":
    main()
