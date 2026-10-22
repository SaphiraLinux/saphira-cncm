#!/usr/bin/env python3
"""16x16 (and general N) Sudoku puzzle generation + validation (stdlib only).

Symbols: 1-9,A-G for values 1..16. Prefix row framing mirrors the 9x9
layout: [wrap=18] row0 [sep=17] row1 ... [sep] row15 [wrap=18]
(P = 256 + 15 + 2 = 273). No torch, no checkpoints: safe to run anywhere.

Usage:
  python3 tools/sudoku16.py --selftest        # deterministic checks
  python3 tools/sudoku16.py --gen N --seed S  # print puzzle+solution
"""
import argparse
import random
import sys

N = 16
BOX = 4
SYMS = "123456789ABCDEFG"
SYMS25 = "123456789ABCDEFGHIJKLMNOP"


def val_to_sym(v, alpha=SYMS):
    return alpha[v - 1]


def sym_to_val(c, n=N, alpha=None):
    if c == "." or c == "_":
        return 0
    a = alpha or (SYMS25 if n > 16 else SYMS)
    i = a.find(c)
    v = i + 1 if i >= 0 else -1
    return v if 1 <= v <= n else -1


def base_grid(n=N, box=BOX):
    return [[(box * (r % box) + r // box + c) % n + 1 for c in range(n)]
            for r in range(n)]


def shuffled_full(rng, n=N, box=BOX):
    g = base_grid(n, box)
    # digit permutation
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    g = [[perm[v - 1] for v in row] for row in g]
    # row bands + rows within bands
    bands = list(range(box))
    rng.shuffle(bands)
    rows = []
    for b in bands:
        inner = list(range(box))
        rng.shuffle(inner)
        for i in inner:
            rows.append(g[b * box + i])
    # column stacks + cols within stacks
    stacks = list(range(box))
    rng.shuffle(stacks)
    cols = []
    for s in stacks:
        inner = list(range(box))
        rng.shuffle(inner)
        for i in inner:
            cols.append(s * box + i)
    g = [[row[c] for c in cols] for row in rows]
    if rng.random() < 0.5:
        g = [list(r) for r in zip(*g)]
    return g


def dig(full, rng, holes):
    cells = [(r, c) for r in range(N) for c in range(N)]
    rng.shuffle(cells)
    pz = [row[:] for row in full]
    for r, c in cells[:holes]:
        pz[r][c] = 0
    return pz


def board_str(grid, alpha=SYMS):
    return "".join(val_to_sym(v, alpha) if v else "." for row in grid for v in row)


def validate(s, n=N, box=BOX, alpha=None):
    want = set(range(1, n + 1))
    if len(s) != n * n:
        return False
    vs = []
    for ch in s:
        v = sym_to_val(ch, n, alpha)
        if v < 1:
            return False
        vs.append(v)
    for r in range(n):
        if set(vs[r * n:(r + 1) * n]) != want:
            return False
    for c in range(n):
        if set(vs[r * n + c] for r in range(n)) != want:
            return False
    for br in range(box):
        for bc in range(box):
            cells = [vs[(br * box + r) * n + bc * box + c]
                     for r in range(box) for c in range(box)]
            if set(cells) != want:
                return False
    return True


def prefix_ids(puzzle):
    """puzzle: 16x16 grid of 0..16 -> 273 prefix ids."""
    wrap, sep = N + 2, N + 1
    ids = [wrap]
    for r in range(N):
        ids += puzzle[r]
        if r < N - 1:
            ids.append(sep)
    ids.append(wrap)
    assert len(ids) == N * N + (N - 1) + 2
    return ids


def clues_of(prefix):
    """Mirror of C prefix_clues_sz: ids -> 256-char clue string."""
    wrap, sep = N + 2, N + 1
    assert len(prefix) == N * N + (N - 1) + 2
    assert prefix[0] == wrap and prefix[-1] == wrap
    out = []
    for r in range(N):
        base = 1 + r * (N + 1)
        if r < N - 1:
            assert prefix[base + N] == sep
        for c in range(N):
            v = prefix[base + c]
            assert 0 <= v <= N
            out.append(val_to_sym(v) if v else ".")
    return "".join(out)


def selftest():
    rng = random.Random(16)
    full = shuffled_full(rng)
    fs = board_str(full)
    assert validate(fs), "shuffled grid must validate"
    pz = dig(full, rng, 96)
    ps = board_str(pz)
    assert len(ps) == 256 and validate(fs)
    # clue preservation roundtrip
    ids = prefix_ids(pz)
    assert len(ids) == 273 and ids[0] == 18 and ids[-1] == 18
    assert clues_of(ids) == ps
    assert sym_to_val("G", 16) == 16 and sym_to_val("H", 16) == -1
    assert sym_to_val("9", 9) == 9 and sym_to_val("A", 9) == -1
    # corruptions must fail
    bad = "1" + fs[1:]
    assert not validate(bad), "dup in row 0 must fail"
    print("sudoku16 selftest OK: P=273 text=271 valid=1 roundtrip=1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--seed", type=int, default=16)
    ap.add_argument("--holes", type=int, default=96)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    rng = random.Random(a.seed)
    for i in range(a.gen):
        full = shuffled_full(rng)
        pz = dig(full, rng, a.holes)
        print("puzzle %d: %s" % (i, board_str(pz)))
        print("solution %d: %s" % (i, board_str(full)))
        print("prefix %d: %s" % (i, ",".join(map(str, prefix_ids(pz)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
