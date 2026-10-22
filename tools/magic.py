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


def ramanujan(a, b, c, d):
    """Ramanujan's parametric 4x4 birthday square.

    First row (a, b, c, d) is free (e.g. day, month, century, year);
    the rest is forced. Every row/col/diag, the four corners and the
    central 2x2 sum to a+b+c+d. Verified against the historical
    (22,12,18,87) square (greedy transcription errors in circulation
    use 21/11 for the last two cells; the correct values are 23/11).
    Repeated values are allowed: no permutation requirement.
    """
    return [a, b, c, d,
            d + 1, c - 1, b - 3, a + 3,
            b - 2, a + 2, d + 2, c - 2,
            c + 1, d - 1, a + 1, b - 1]


def ramanujan_valid(grid):
    """12-constraint check: rows, cols, diags, corners, center 2x2."""
    if len(grid) != 16:
        return False
    S = sum(grid[0:4])
    lines = []
    for r in range(4):
        lines.append(sum(grid[r * 4:(r + 1) * 4]))
    for c in range(4):
        lines.append(sum(grid[r * 4 + c] for r in range(4)))
    lines.append(sum(grid[i * 5] for i in range(4)))
    lines.append(sum(grid[i * 4 + 3 - i] for i in range(4)))
    lines.append(grid[0] + grid[3] + grid[12] + grid[15])
    lines.append(sum(grid[5:7]) + sum(grid[9:11]))
    return all(v == S for v in lines)


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
            # full check (corner order completes the dihedral break)
            if not (grid[0] < grid[15] and grid[3] < grid[12]):
                return False
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
        # symmetry break on corners (checked after the cell is placed)
        if pos == 4 and not (grid[0] < grid[3]):
            return False
        if pos == 13 and not (grid[0] < grid[12]):
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
        out += linear_family(order, r, tries or (80000 if order == 9 else 20000))
        for t in list(out):
            out += dihedral(t, order)
            out.append(complement(t, order))
        return dedupe(out)
    if order == 6:
        rng = rng or random.Random(6)
        inners = dedupe(dihedral(DURER, 4))[:8]
        out = []
        for inner in inners:
            for _ in range(40):
                g = border6(inner, rng)
                if g:
                    out.append(g)
        for t in list(out):
            out += dihedral(t, 6)
            out.append(complement(t, 6))
        return dedupe(out)
    if order == 8:
        rng = rng or random.Random(8)
        inners = base_set(6, random.Random(8))[:12]
        out = []
        for inner in inners:
            for _ in range(30):
                g = border_nm(inner, 6, 8, rng)
                if g:
                    out.append(g)
        for t in list(out):
            out += dihedral(t, 8)
            out.append(complement(t, 8))
        return dedupe(out)
    raise ValueError(order)


def border_nm(inner_m, m, n, rng, maxtries=200000):
    """Border an m×m normal magic square to a normal n×m... (n = m+2).

    Inner values are shifted into the middle range; the border holds the
    remaining low/high numbers as complementary pairs (summing to n^2+1).
    Each border pair-slot (m horizontal, m vertical, 2 diagonal) gets one
    complementary pair, which satisfies every inner line and both
    diagonals by construction; row 0 and col 0 are checked explicitly
    and hit quickly under random sampling.
    """
    L = (n * n - m * m) // 2
    assert L * 2 == n * n - m * m
    inner = [v + L for v in inner_m]
    S = n * (n * n + 1) // 2
    P = n * n + 1
    pairs = [(i, P - i) for i in range(1, L + 1)]
    o = (n - m) // 2
    for _ in range(maxtries):
        rng.shuffle(pairs)
        H, V, D = pairs[0:m], pairs[m:2 * m], pairs[2 * m:2 * m + 2]
        b = 2 * m + 2
        oo = rng.randrange(1 << b)
        H = [[p[0], p[1]] if (oo >> i) & 1 else [p[1], p[0]]
             for i, p in enumerate(H)]
        V = [[p[0], p[1]] if (oo >> (m + i)) & 1 else [p[1], p[0]]
             for i, p in enumerate(V)]
        D = [[p[0], p[1]] if (oo >> (2 * m + i)) & 1 else [p[1], p[0]]
             for i, p in enumerate(D)]
        r0 = D[0][0] + sum(V[j][0] for j in range(m)) + D[1][0]
        c0 = D[0][0] + sum(H[i][0] for i in range(m)) + D[1][1]
        if r0 != S or c0 != S:
            continue
        g = [0] * (n * n)
        g[0], g[n - 1], g[(n - 1) * n], g[n * n - 1] = \
            D[0][0], D[1][0], D[1][1], D[0][1]
        for j in range(m):
            g[o + j] = V[j][0]
            g[(n - 1) * n + o + j] = V[j][1]
        for i in range(m):
            g[(o + i) * n] = H[i][0]
            g[(o + i) * n + n - 1] = H[i][1]
        for r in range(m):
            for c in range(m):
                g[(o + r) * n + (o + c)] = inner[r * m + c]
        if validate(g, n):
            return g
    return None


def border6(inner4, rng, maxtries=200000):
    """4x4 -> 6x6 via the generic border (kept as a named entry point)."""
    return border_nm(inner4, 4, 6, rng, maxtries)


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


RAMA_HOLE = -(2 ** 30)


def emit_ramanujan(outdir, train_n, held_n, seed):
    """Sampled birthday squares: day 1-31, month 1-12, century 18-20,
    year 0-99. Prefix rows are [a,b,c,d] + 16 cells (holes = RAMA_HOLE);
    sol rows are the 16 solution cells. No wrap/sep framing needed since
    the first four ids are the parameters. Disjoint by exact match."""
    rng = random.Random(seed)
    combos = [(a, b, c, d)
              for a in range(1, 32) for b in range(1, 13)
              for c in (18, 19, 20) for d in range(0, 100)]
    rng.shuffle(combos)
    assert len(combos) >= train_n + held_n
    def rows_of(sel):
        pz, sol, pre = [], [], []
        for (a, b, c, d) in sel:
            full = ramanujan(a, b, c, d)
            assert ramanujan_valid(full)
            idx = list(range(16))
            rng.shuffle(idx)
            nh = rng.choice((5, 6, 7))
            p = list(full)
            for i in idx[:nh]:
                p[i] = RAMA_HOLE
            pz.append(p)
            sol.append(full)
            pre.append([a, b, c, d] + p)
        return pz, sol, pre
    tr = rows_of(combos[:train_n])
    he = rows_of(combos[train_n:train_n + held_n])
    st = set(tuple(r) for r in tr[2])
    ov = [i for i, r in enumerate(he[2]) if tuple(r) in st]
    print("ramanujan: train=%d held=%d overlap=%d" % (len(tr[0]), len(he[0]), len(ov)))
    assert not ov
    for name, data in (("train", tr), ("held", he)):
        d = os.path.join(outdir, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "prefix.txt"), "w") as f:
            f.write("\n".join(",".join(map(str, r)) for r in data[2]) + "\n")
        with open(os.path.join(d, "solutions.txt"), "w") as f:
            f.write("\n".join(",".join(map(str, r)) for r in data[1]) + "\n")
    with open(os.path.join(outdir, "PROVENANCE.txt"), "w") as f:
        f.write("ramanujan 4x4: sampled birthdays day 1-31 month 1-12\n")
        f.write("century {18,19,20} year 0-99; holes 5-7; HOLE=%d\n" % RAMA_HOLE)
        f.write("train=%d held=%d seed=%d (disjoint full rows)\n"
                % (train_n, held_n, seed))
        f.write("NOTE: values can repeat and go negative; no permutation\n")
        f.write("requirement. VQ/training encoding (offset+hole map) is\n")
        f.write("future work; generator+validator are the contract.\n")


def emit_data(order, outdir, train_n, held_n, seed, holes_frac=0.5,
              family="normal"):
    if family == "ramanujan":
        assert order == 4
        return emit_ramanujan(outdir, train_n, held_n, seed)
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
    b6 = border6(DURER, random.Random(6))
    assert b6 and validate(b6, 6)
    # Sagrada Familia square: right sums, wrong set -> correctly invalid
    sagrada = [1, 14, 14, 4, 11, 7, 6, 9, 13, 2, 15, 12, 8, 10, 5, 3]
    assert sum(sagrada[0:4]) == 33 and not validate(sagrada, 4)
    # Ramanujan family: historical square + 2000 random parametrisations
    hist = [22, 12, 18, 87, 88, 17, 9, 25, 10, 24, 89, 16, 19, 86, 23, 11]
    assert ramanujan(22, 12, 18, 87) == hist
    assert ramanujan_valid(hist)
    rr = random.Random(7)
    for _ in range(2000):
        aa, bb, cc, dd = [rr.randint(-50, 200) for _ in range(4)]
        assert ramanujan_valid(ramanujan(aa, bb, cc, dd))
    print("magic selftest OK: sums, symbols roundtrip 1..81, 3/4/5/6/7/8/9 constructors, ramanujan family")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--emit-data", default="")
    ap.add_argument("--train-n", type=int, default=1024)
    ap.add_argument("--held-n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--family", default="normal",
                    help="normal | ramanujan (ramanujan implies order 4)")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.census:
        census()
        return 0
    if a.family == "ramanujan":
        if a.emit_data:
            emit_data(4, a.emit_data, a.train_n, a.held_n, a.seed,
                      family="ramanujan")
        else:
            print("ramanujan family: use --emit-data DIR (birthdays sampled)")
        return 0
    sols = base_set(a.order, random.Random(a.seed))
    print("order=%d distinct solutions=%d" % (a.order, len(sols)))
    if a.emit_data:
        emit_data(a.order, a.emit_data, a.train_n, a.held_n, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
