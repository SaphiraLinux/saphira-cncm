#!/usr/bin/env python3
"""3x3 (and general-n) magic square data: generate, validate, emit rows.

Normal magic squares: permutation of 1..n^2, every row/col/diag sums to
n*(n^2+1)/2 (3->15, 6->111). Order 3 has exactly 8 (Lo Shu symmetries).

Prefix framing mirrors the 9x9 Sudoku id scheme (values stay 1..9, so
token ids 0..11 are reused): [wrap=11] row [sep=10] row [sep] row [wrap=11]
(P = 9+2+2 = 13). VQ rows: 9 cells + 2 seps, no wrap (T = 11).

Usage:
  python3 tools/magic3.py --selftest
  python3 tools/magic3.py --gen N --seed S --holes H   # prefix rows on stdout
  python3 tools/magic3.py --emit-data DIR --train-n N --held-n M --seed S
"""
import argparse
import itertools
import os
import random
import sys


def magic_sum(n):
    return n * (n * n + 1) // 2


def loshu():
    return [[8, 1, 6], [3, 5, 7], [4, 9, 2]]


def transforms(g):
    """All 8 symmetries of a square (4 rotations x flip)."""
    out = []
    for _ in range(4):
        out.append([row[:] for row in g])
        out.append([row[::-1] for row in g])
        g = [list(r) for r in zip(*g[::-1])]
    seen, uniq = set(), []
    for h in out:
        t = tuple(v for row in h for v in row)
        if t not in seen:
            seen.add(t)
            uniq.append(h)
    return uniq


def validate(grid, n=3):
    """grid: flat list of n*n ints 1..n^2. Normal magic square check."""
    if sorted(grid) != list(range(1, n * n + 1)):
        return False
    s = magic_sum(n)
    for r in range(n):
        if sum(grid[r * n:(r + 1) * n]) != s:
            return False
    for c in range(n):
        if sum(grid[r * n + c] for r in range(n)) != s:
            return False
    if sum(grid[i * n + i] for i in range(n)) != s:
        return False
    if sum(grid[i * n + (n - 1 - i)] for i in range(n)) != s:
        return False
    return True


def board_str(grid):
    return "".join(str(v) for v in grid)


def dig(full, rng, holes):
    idx = list(range(9))
    rng.shuffle(idx)
    pz = list(full)
    for i in idx[:holes]:
        pz[i] = 0
    return pz


def clue_str(pz):
    return "".join(str(v) if v else "." for v in pz)


def prefix_ids(pz):
    wrap, sep = 11, 10
    ids = [wrap]
    for r in range(3):
        ids += pz[r * 3:(r + 1) * 3]
        if r < 2:
            ids.append(sep)
    ids.append(wrap)
    assert len(ids) == 13
    return ids


def sol11(full):
    ids = []
    for r in range(3):
        ids += full[r * 3:(r + 1) * 3]
        if r < 2:
            ids.append(10)
    assert len(ids) == 11
    return ids


def selftest():
    assert magic_sum(3) == 15 and magic_sum(6) == 111
    base = [v for row in loshu() for v in row]
    assert validate(base)
    ts = transforms(loshu())
    assert len(ts) == 8, len(ts)
    assert all(validate([v for row in t for v in row]) for t in ts)
    bad = base[:]
    bad[0], bad[1] = bad[1], bad[0]
    assert not validate(bad)
    dup = base[:]
    dup[0] = dup[1]
    assert not validate(dup)
    rng = random.Random(3)
    pz = dig(base, rng, 4)
    assert clue_str(pz).count(".") == 4
    assert len(prefix_ids(pz)) == 13 and prefix_ids(pz)[0] == 11
    assert len(sol11(base)) == 11
    print("magic3 selftest OK: 8 squares, P=13 T=11 sum=15")


def all_puzzles(sols, holes_list=(3, 4, 5)):
    """Enumerate the full puzzle space: every solution x every hole set."""
    out = []
    for full in sols:
        for h in holes_list:
            for hs in itertools.combinations(range(9), h):
                pz = list(full)
                for i in hs:
                    pz[i] = 0
                out.append((full, pz))
    return out


def gen_data(rng, sols, n, holes):
    puzzles, solutions, prefs, s11 = [], [], [], []
    for _ in range(n):
        full = sols[rng.randrange(len(sols))]
        pz = dig(full, rng, holes)
        puzzles.append(clue_str(pz))
        solutions.append(board_str(full))
        prefs.append(",".join(map(str, prefix_ids(pz))))
        s11.append(",".join(map(str, sol11(full))))
    return puzzles, solutions, prefs, s11


def rows_of(pairs):
    puzzles, solutions, prefs, s11 = [], [], [], []
    for full, pz in pairs:
        puzzles.append(clue_str(pz))
        solutions.append(board_str(full))
        prefs.append(",".join(map(str, prefix_ids(pz))))
        s11.append(",".join(map(str, sol11(full))))
    return puzzles, solutions, prefs, s11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--holes", type=int, default=4)
    ap.add_argument("--emit-data", default="")
    ap.add_argument("--train-n", type=int, default=1024)
    ap.add_argument("--held-n", type=int, default=128)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    sols = [[v for row in t for v in row] for t in transforms(loshu())]
    rng = random.Random(a.seed)
    if a.emit_data:
        os.makedirs(os.path.join(a.emit_data, "train"), exist_ok=True)
        os.makedirs(os.path.join(a.emit_data, "held"), exist_ok=True)
        # Exhaustible space: partition all (solution, hole-set) combos so
        # train/held are disjoint by construction (holes 3..5).
        space = all_puzzles(sols)
        rng.shuffle(space)
        assert len(space) >= a.train_n + a.held_n, len(space)
        tr = rows_of(space[:a.train_n])
        he = rows_of(space[a.train_n:a.train_n + a.held_n])
        st = set(tr[2])
        ov = [i for i, p in enumerate(he[2]) if p in st]
        print("space=%d train=%d held=%d overlap=%d"
              % (len(space), len(tr[0]), len(he[0]), len(ov)))
        assert not ov
        for name, data in (("train", tr), ("held", he)):
            for fn, rows in (("puzzles.txt", data[0]), ("solutions.txt", data[1]),
                             ("prefix.txt", data[2]), ("sol11.txt", data[3])):
                with open(os.path.join(a.emit_data, name, fn), "w") as f:
                    f.write("\n".join(rows) + "\n")
        with open(os.path.join(a.emit_data, "PROVENANCE.txt"), "w") as f:
            f.write("magic-3: 8 Lo Shu solutions, holes=%d\n" % a.holes)
            f.write("train n=%d held n=%d seed=%d (disjoint prefixes)\n"
                    % (a.train_n, a.held_n, a.seed))
        return 0
    for _ in range(a.gen):
        full = sols[rng.randrange(len(sols))]
        print(",".join(map(str, prefix_ids(dig(full, rng, a.holes)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
