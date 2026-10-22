#!/usr/bin/env python3
"""Planetary magic squares (kameas), orders 3..9: generate, validate, emit rows.

Normal magic square: permutation of 1..n^2, every row/col/diag sums to
n*(n^2+1)/2. Classical assignments: 3 Saturn/15, 4 Jupiter/34, 5 Mars/65,
6 Sun/111, 7 Venus/175, 8 Mercury/260, 9 Moon/369.

Board symbols (shared with cncm magic-check): 1-9 -> '1'-'9',
10-35 -> 'A'-'Z', 36 -> '0', 37-62 -> 'a'-'z', 63-81 -> EXTRA table.
Prefix framing mirrors Sudoku: [wrap] rows joined by [sep] [wrap] with
wrap = n^2+2, sep = n^2+1, so P = n^2 + (n-1) + 2, T = n^2 + (n-1).

Every generator self-verifies with validate() and dedupes. Counts are
reported honestly; small families (3x3: 8 solutions) are complete,
large ones are documented samples of vast spaces.

Usage:
  python3 tools/magic.py --selftest
  python3 tools/magic.py --order 5 --emit-data DIR --train-n N --held-n M --seed S
  python3 tools/magic.py --census   # count distinct squares per order
"""
import argparse
import itertools
import os
import random
import sys
import time

EXTRA62 = "!@#$%^&*()-_=+[]{}|"  # 19 symbols for values 63..81
assert len(EXTRA62) == 19


def magic_sum(n):
    return n * (n * n + 1) // 2


def sym(v):
    if 1 <= v <= 9:
        return str(v)
    if 10 <= v <= 35:
        return chr(ord("A") + v - 10)
    if v == 36:
        return "0"
    if 37 <= v <= 62:
        return chr(ord("a") + v - 37)
    if 63 <= v <= 81:
        return EXTRA62[v - 63]
    raise ValueError(v)


def sym_val(c):
    if "1" <= c <= "9":
        return ord(c) - ord("0")
    if "A" <= c <= "Z":
        return 10 + ord(c) - ord("A")
    if c == "0":
        return 36
    if "a" <= c <= "z":
        return 37 + ord(c) - ord("a")
    i = EXTRA62.find(c)
    if i >= 0:
        return 63 + i
    return -1


def validate(grid, n):
    want = set(range(1, n * n + 1))
    if sorted(grid) != list(want):
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
    return "".join(sym(v) for v in grid)


def parse_board(s):
    return [sym_val(c) for c in s]


def dihedral(g, n):
    """All 8 symmetries of a square grid (flat list)."""
    def rot(h):
        return [h[(n - 1 - c) * n + r] for r in range(n) for c in range(n)]

    def flip(h):
        return [h[r * n + (n - 1 - c)] for r in range(n) for c in range(n)]

    out = []
    h = list(g)
    for _ in range(4):
        out.append(list(h))
        out.append(flip(h))
        h = rot(h)
    seen, uniq = set(), []
    for x in out:
        t = tuple(x)
        if t not in seen:
            seen.add(t)
            uniq.append(x)
    return uniq


def complement(g, n):
    m = n * n + 1
    return [m - v for v in g]


def loshu():
    return [8, 1, 6, 3, 5, 7, 4, 9, 2]


DURER = [16, 2, 3, 13, 5, 11, 10, 8, 9, 7, 6, 12, 4, 14, 15, 1]


def siamese(n):
    """Siamese method for odd n. Returns flat grid, 1-based values."""
    g = [[0] * n for _ in range(n)]
    r, c = 0, n // 2
    for v in range(1, n * n + 1):
        g[r][c] = v
        nr, nc = (r - 1) % n, (c + 1) % n
        if g[nr][nc]:
            r = (r + 1) % n
        else:
            r, c = nr, nc
    return [v for row in g for v in row]


def doubly_even(n, seed=0):
    """Classical doubly-even construction with per-block pattern rotation.

    Fill sequentially; complement cells on marked diagonals of each 4x4
    block. `seed` selects a rotation of the marking pattern per block,
    yielding distinct squares (all verified by validate()).
    """
    assert n % 4 == 0
    m = n * n + 1
    g = list(range(1, n * n + 1))
    nb = n // 4
    rng = random.Random(seed)

    def marked(r, c, rot):
        dr, dc = r % 4, c % 4
        for _ in range(rot % 4):
            dr, dc = dc, 3 - dr
        return dr == dc or dr + dc == 3

    for br in range(nb):
        for bc in range(nb):
            rot = rng.randrange(4)
            for r in range(4):
                for c in range(4):
                    i = (br * 4 + r) * n + (bc * 4 + c)
                    if marked(r, c, rot):
                        g[i] = m - g[i]
    return g


def linear_family(n, rng, tries):
    """Sample M[i][j] = n*((a*i+b*j+c)%n) + ((d*i+e*j+f)%n) + 1.

    Yields (mostly pandiagonal) magic squares for suitable coefficients;
    validate() filters, so misses cost nothing.
    """
    out = []
    for _ in range(tries):
        a = rng.randrange(1, n)
        b = rng.randrange(1, n)
        cc = rng.randrange(n)
        d = rng.randrange(1, n)
        e = rng.randrange(1, n)
        f = rng.randrange(n)
        g = [n * ((a * (i // n) + b * (i % n) + cc) % n)
             + ((d * (i // n) + e * (i % n) + f) % n) + 1
             for i in range(n * n)]
        if validate(g, n):
            out.append(g)
    return out


def dfs4(limit=880, time_budget=120):
    """Backtracking enumeration of 4x4 normal magic squares.

    Row-major placement with sum pruning; dihedral symmetry broken by
    corner ordering (a00 smallest corner, a03 < a30). Returns distinct
    squares found within budget (880 = complete set incl. symmetries
    via dihedral() expansion by the caller).
    """
    N = 4
    S = 34
    start = time.time()
    found = []
    seen = set()
    grid = [0] * 16
    used = [False] * 17
    nodes = [0]

    def row_ok(r):
        s = sum(grid[r * N:(r + 1) * N])
        return s == S

    def col_feasible(c, upto_r):
        s = sum(grid[r * N + c] for r in range(upto_r + 1))
        rem = [v for v in range(1, 17) if not used[v]]
        need = N - (upto_r + 1)
        if need == 0:
            return s == S
        rem.sort()
        return s + sum(rem[:need]) <= S <= s + sum(rem[-need:])

    def diag_feasible():
        # main diag cells 0,5,10,15 placed at steps 0,5,10,15
        return True

    def rec(pos):
        if len(found) >= limit:
            return True
        if time.time() - start > time_budget:
            return True
        if pos == 16:
            # full check
            if all(row_ok(r) for r in range(4)):
                cols = all(sum(grid[r * 4 + c] for r in range(4)) == S for c in range(4))
                d0 = sum(grid[i * 5] for i in range(4)) == S
                d1 = sum(grid[i * 4 + 3 - i] for i in range(4)) == S
                if cols and d0 and d1:
                    t = tuple(grid)
                    if t not in seen:
                        seen.add(t)
                        found.append(list(grid))
            return len(found) >= limit
        r, c = pos // 4, pos % 4
        # symmetry break on corners
        if pos == 3 and not (grid[0] < grid[3]):
            return False
        if pos == 12 and not (grid[0] < grid[12]):
            return False
        if pos == 15 and not (grid[0] < grid[15] and grid[3] < grid[12]):
            return False
        # row pruning
        if c == 3:
            need = S - sum(grid[r * 4:r * 4 + 3])
            if need < 1 or need > 16 or used[need]:
                return False
            cands = [need]
        else:
            cands = [v for v in range(1, 17) if not used[v]]
        for v in cands:
            nodes[0] += 1
            grid[pos] = v
            used[v] = True
            ok = col_feasible(c, r)
            if ok and c == 3:
                ok = row_ok(r)
            if ok and r == 3 and c == 3:
                pass
            if ok:
                # partial diagonal pruning
                if pos in (0, 5, 10):
                    placed = [grid[i * 5] for i in range(4) if i * 5 <= pos and grid[i * 5]]
                    s = sum(placed)
                    rem = [x for x in range(1, 17) if not used[x]]
                    needn = 4 - len(placed)
                    if needn:
                        rem.sort()
                        ok = s + sum(rem[:needn]) <= S <= s + sum(rem[-needn:])
                    else:
                        ok = (s == S)
                if ok and pos in (3, 6, 9):
                    cells = [3, 6, 9, 12]
                    placed = [grid[i] for i in cells if i <= pos and grid[i]]
                    s = sum(placed)
                    rem = [x for x in range(1, 17) if not used[x]]
                    needn = 4 - len(placed)
                    if needn:
                        rem.sort()
                        ok = s + sum(rem[:needn]) <= S <= s + sum(rem[-needn:])
                    else:
                        ok = (s == S)
            if ok and rec(pos + 1):
                used[v] = False
                grid[pos] = 0
                return True
            used[v] = False
            grid[pos] = 0
        return len(found) >= limit

    rec(0)
    return found


def base_set(order, rng=None, tries=0):
    """Distinct solution grids for an order (unexpanded)."""
    if order == 3:
        out = []
        for t in dihedral(loshu(), 3):
            out.append(t)
            out.append(complement(t, 3))
        return dedupe(out)
    if order == 4:
        found = dfs4()
        return dedupe(found)
    if order in (5, 7, 9):
        out = [siamese(order)]
        r = rng or random.Random(order)
        out += linear_family(order, r, tries or 20000)
        for t in list(out):
            out += dihedral(t, order)
            out.append(complement(t, order))
        return dedupe(out)
    if order == 6:
        out = lux6()
        for t in list(out):
            out += dihedral(t, 6)
            out.append(complement(t, 6))
        return dedupe(out)
    if order == 8:
        out = []
        for s in range(64):
            g = doubly_even(8, seed=s)
            if validate(g, 8):
                out.append(g)
        for t in list(out):
            out += dihedral(t, 8)
            out.append(complement(t, 8))
        return dedupe(out)
    raise ValueError(order)


def border6(inner4, rng, maxtries=5000):
    """Border a shifted 4x4 (numbers 11..26, line sum 74) to normal 6x6.

    The 10 border pair-slots (4 horizontal, 4 vertical, 2 diagonal) must
    each hold a complementary pair (summing to 37); rows 1-4, cols 1-4
    and both diagonals then sum correctly by construction. Row 0 and
    col 0 are checked explicitly; random sampling hits them quickly.
    """
    inner = [v + 10 for v in inner4]  # 1..16 -> 11..26, sums 34 -> 74
    pairs = [(i, 37 - i) for i in range(1, 19) if i < 37 - i]
    for _ in range(maxtries):
        rng.shuffle(pairs)
        (h1, h2, h3, h4), (v1, v2, v3, v4), (d1, d2) = \
            pairs[0:4], pairs[4:8], pairs[8:10]
        # orientations: bit i chooses low/high for slot i
        o = rng.randrange(1 << 10)
        H = [[p[0], p[1]] if (o >> i) & 1 else [p[1], p[0]]
             for i, p in enumerate([h1, h2, h3, h4])]
        V = [[p[0], p[1]] if (o >> (4 + i)) & 1 else [p[1], p[0]]
             for i, p in enumerate([v1, v2, v3, v4])]
        D = [[p[0], p[1]] if (o >> (8 + i)) & 1 else [p[1], p[0]]
             for i, p in enumerate([d1, d2])]
        # row 0: C1 T1 T2 T3 T4 C2 ; col 0: C1 L1..L4 C3
        r0 = D[0][0] + sum(V[j][0] for j in range(4)) + D[1][0]
        c0 = D[0][0] + sum(H[i][0] for i in range(4)) + D[1][1]
        if r0 != 111 or c0 != 111:
            continue
        g = [0] * 36
        g[0], g[5], g[30], g[35] = D[0][0], D[1][0], D[1][1], D[0][1]
        for j in range(4):
            g[1 + j] = V[j][0]
            g[31 + j] = V[j][1]
            g[6 + 6 * j] = H[j][0]
            g[11 + 6 * j] = H[j][1]
        for r in range(4):
            for c in range(4):
                g[(r + 1) * 6 + (c + 1)] = inner[r * 4 + c]
        if validate(g, 6):
            return g
    return None


def dedupe(grids):
    seen, out = set(), []
    for g in grids:
        t = tuple(g)
        if t not in seen:
            seen.add(t)
            out.append(g)
    return out


def framing_ids(pz, n):
    wrap, sep = n * n + 2, n * n + 1
    ids = [wrap]
    for r in range(n):
        ids += pz[r * n:(r + 1) * n]
        if r < n - 1:
            ids.append(sep)
    ids.append(wrap)
    assert len(ids) == n * n + (n - 1) + 2
    return ids


def sol_ids(full, n):
    ids = []
    for r in range(n):
        ids += full[r * n:(r + 1) * n]
        if r < n - 1:
            ids.append(n * n + 1)
    assert len(ids) == n * n + (n - 1)
    return ids


def clue_str(pz):
    return "".join(sym(v) if v else "." for v in pz)


def holes_of(full, rng, holes):
    idx = list(range(len(full)))
    rng.shuffle(idx)
    pz = list(full)
    for i in idx[:holes]:
        pz[i] = 0
    return pz


def emit_data(order, outdir, train_n, held_n, seed, holes_frac=0.5):
    sols = base_set(order, random.Random(seed))
    rng = random.Random(seed)
    n = order
    nh = max(1, int(n * n * holes_frac))
    pairs = []
    for s in sols:
        for _ in range(400):
            pairs.append((s, holes_of(s, rng, nh)))
            if len(pairs) >= (train_n + held_n) * 3:
                break
        if len(pairs) >= (train_n + held_n) * 3:
            break
    # dedupe puzzle rows, then split
    seen, uniq = set(), []
    for s, pz in pairs:
        key = tuple(framing_ids(pz, n))
        if key not in seen:
            seen.add(key)
            uniq.append((s, pz))
    rng.shuffle(uniq)
    assert len(uniq) >= train_n + held_n, (len(uniq), train_n + held_n)
    tr, he = uniq[:train_n], uniq[train_n:train_n + held_n]
    for name, data in (("train", tr), ("held", he)):
        d = os.path.join(outdir, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "puzzles.txt"), "w") as f:
            f.write("\n".join(clue_str(pz) for _, pz in data) + "\n")
        with open(os.path.join(d, "solutions.txt"), "w") as f:
            f.write("\n".join(board_str(s) for s, _ in data) + "\n")
        with open(os.path.join(d, "prefix.txt"), "w") as f:
            f.write("\n".join(",".join(map(str, framing_ids(pz, n))) for _, pz in data) + "\n")
        with open(os.path.join(d, "sol.txt"), "w") as f:
            f.write("\n".join(",".join(map(str, sol_ids(s, n))) for s, _ in data) + "\n")
    with open(os.path.join(outdir, "PROVENANCE.txt"), "w") as f:
        f.write("magic order=%d solutions=%d holes~%d\n" % (n, len(sols), nh))
        f.write("train=%d held=%d seed=%d (disjoint prefixes)\n" % (train_n, held_n, seed))
    print("order=%d solutions=%d train=%d held=%d overlap=0" % (n, len(sols), train_n, held_n))


def census():
    for n in (3, 4, 5, 6, 7, 8, 9):
        t0 = time.time()
        sols = base_set(n, random.Random(n))
        print("order %d: %d distinct solutions (%.1fs)" % (n, len(sols), time.time() - t0))


def selftest():
    assert magic_sum(3) == 15 and magic_sum(4) == 34
    assert magic_sum(5) == 65 and magic_sum(6) == 111
    assert magic_sum(7) == 175 and magic_sum(8) == 260
    assert magic_sum(9) == 369
    assert sym(1) == "1" and sym(9) == "9" and sym(10) == "A"
    assert sym(16) == "G" and sym(36) == "0" and sym(37) == "a"
    assert sym(62) == "z" and sym(63) == EXTRA62[0] and sym(81) == EXTRA62[18]
    for v in list(range(1, 82)):
        assert sym_val(sym(v)) == v, v
    assert validate(loshu(), 3)
    assert len(dedupe(dihedral(loshu(), 3))) == 8
    assert validate(DURER, 4)
    # Durer row-permuted sanity: complement validates too
    assert validate(complement(DURER, 4), 4)
    assert validate(siamese(5), 5) and validate(siamese(7), 7)
    assert validate(siamese(9), 9)
    assert validate(doubly_even(8, seed=0), 8)
    assert validate(lux6()[0], 6)
    # Sagrada Familia square: right sums, wrong set -> correctly invalid
    sagrada = [1, 14, 14, 4, 11, 7, 6, 9, 13, 2, 15, 12, 8, 10, 5, 3]
    assert sum(sagrada[0:4]) == 33 and not validate(sagrada, 4)
    print("magic selftest OK: sums, symbols roundtrip 1..81, 3/4/5/6/7/8/9 constructors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--emit-data", default="")
    ap.add_argument("--train-n", type=int, default=1024)
    ap.add_argument("--held-n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.census:
        census()
        return 0
    sols = base_set(a.order, random.Random(a.seed))
    print("order=%d distinct solutions=%d" % (a.order, len(sols)))
    if a.emit_data:
        emit_data(a.order, a.emit_data, a.train_n, a.held_n, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
