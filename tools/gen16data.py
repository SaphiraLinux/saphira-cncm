#!/usr/bin/env python3
"""Generate 16x16 Sudoku train / held-out datasets (stdlib+numpy only).

Each split (seeded separately) emits:
  puzzles.txt    256-char rows ('.' = hole)
  solutions.txt  256-char full grids (symbols 1-9,A-G)
  prefix.txt     273 comma-separated token ids per row (wrap/sep framing)
  sol271.txt     271 comma-separated token ids per row (cells+seps, VQ input)

Overlap between splits is checked (exact prefix match) and reported.

Usage:
  python3 tools/gen16data.py --out weights/sudoku-16/data \
      --train-n 4096 --train-seed 1601 --held-n 512 --held-seed 1617 --holes 128
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import sudoku16


def gen_split(rng, n, holes):
    puzzles, solutions, prefs, s271 = [], [], [], []
    for _ in range(n):
        full = sudoku16.shuffled_full(rng)
        pz = sudoku16.dig(full, rng, holes)
        fs = sudoku16.board_str(full)
        ps = sudoku16.board_str(pz)
        assert sudoku16.validate(fs)
        assert sudoku16.clues_of(sudoku16.prefix_ids(pz)) == ps
        puzzles.append(ps)
        solutions.append(fs)
        prefs.append(",".join(map(str, sudoku16.prefix_ids(pz))))
        # VQ row: 256 SOLUTION cell ids + 15 sep ids (no wrap, no holes).
        # NOTE: this must come from `full`, never the digged puzzle.
        ids = []
        for r in range(16):
            ids += full[r]
            if r < 15:
                ids.append(17)
        assert len(ids) == 271 and all(1 <= v <= 18 for v in ids)
        s271.append(",".join(map(str, ids)))
    return puzzles, solutions, prefs, s271


def write_split(d, name, data):
    puzzles, solutions, prefs, s271 = data
    for fn, rows in (("puzzles.txt", puzzles), ("solutions.txt", solutions),
                     ("prefix.txt", prefs), ("sol271.txt", s271)):
        p = os.path.join(d, name, fn)
        with open(p, "w") as f:
            f.write("\n".join(rows) + "\n")
    print("%s: %d rows" % (name, len(puzzles)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-n", type=int, default=4096)
    ap.add_argument("--train-seed", type=int, default=1601)
    ap.add_argument("--held-n", type=int, default=512)
    ap.add_argument("--held-seed", type=int, default=1617)
    ap.add_argument("--holes", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "train"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "held"), exist_ok=True)
    tr = gen_split(random.Random(a.train_seed), a.train_n, a.holes)
    he = gen_split(random.Random(a.held_seed), a.held_n, a.holes)
    ts = set(tr[2])
    ov = [i for i, p in enumerate(he[2]) if p in ts]
    print("train=%d held=%d overlap=%d" % (len(tr[0]), len(he[0]), len(ov)))
    assert not ov, ov[:5]
    write_split(a.out, "train", tr)
    write_split(a.out, "held", he)
    # provenance
    with open(os.path.join(a.out, "PROVENANCE.txt"), "w") as f:
        f.write("train: n=%d seed=%d holes=%d\n" % (a.train_n, a.train_seed, a.holes))
        f.write("held:  n=%d seed=%d holes=%d (disjoint, exact-prefix check)\n"
                % (a.held_n, a.held_seed, a.holes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
