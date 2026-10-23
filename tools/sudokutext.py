#!/usr/bin/env python3
"""tools/sudokutext.py — 9x9 puzzle/solution text corpus for LM training.

Solutions are valid by construction: a base pattern grid transformed by
validity-preserving shuffles (digit relabel, row/cols within bands/stacks,
band/stack order, optional transpose). Puzzles dig random holes; the
solution is A valid completion (uniqueness not enforced — the eval checks
constraint validity via the deterministic engine, not exact match).

Docs are SINGLE lines (mkqwenrows takes one document per file line):

    Puzzle: <81 chars, . empty> / Solution: <81 digits>

Splits are disjoint by solution grid, audited in-process, with a manifest.
9x9 now; the doc format generalizes to 16x16/25x25 by symbol width.

Usage:
    python3 tools/sudokutext.py --out results/sudoku-text-v1 \\
        --train 20000 --val 1000 --test 2000 --seed 11
"""
import argparse
import hashlib
import json
import os
import random
import sys


def base_grid():
    return [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]


def shuffled_solution(rng):
    g = base_grid()
    perm = list(range(1, 10))
    rng.shuffle(perm)
    g = [[perm[v - 1] for v in row] for row in g]
    # rows within bands, then bands
    rows = []
    bands = [0, 1, 2]
    rng.shuffle(bands)
    for b in bands:
        rr = [0, 1, 2]
        rng.shuffle(rr)
        for r in rr:
            rows.append(g[b * 3 + r])
    g = rows
    # cols within stacks, then stacks
    order = []
    stacks = [0, 1, 2]
    rng.shuffle(stacks)
    for s in stacks:
        cc = [0, 1, 2]
        rng.shuffle(cc)
        for c in cc:
            order.append(s * 3 + c)
    g = [[row[c] for c in order] for row in g]
    if rng.random() < 0.5:
        g = [list(r) for r in zip(*g)]
    return tuple(tuple(r) for r in g)


def dig(rng, sol, lo=35, hi=55):
    cells = [(r, c) for r in range(9) for c in range(9)]
    rng.shuffle(cells)
    k = rng.randint(lo, hi)
    holes = set(cells[:k])
    puz = []
    for r in range(9):
        for c in range(9):
            puz.append("." if (r, c) in holes else str(sol[r][c]))
    return "".join(puz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    used = set()
    splits = {"train": [], "val": [], "test": []}
    need = {"train": args.train, "val": args.val, "test": args.test}
    for sp in ("train", "val", "test"):
        guard = 0
        while len(splits[sp]) < need[sp]:
            sol = shuffled_solution(rng)
            if sol in used:
                guard += 1
                if guard > 10 * need[sp] + 100000:
                    sys.exit("sudokutext: solution space exhausted")
                continue
            used.add(sol)
            flat = "".join(str(v) for row in sol for v in row)
            splits[sp].append((dig(rng, sol), flat))
    assert len(used) == sum(len(v) for v in splits.values())

    manifest = {"seed": args.seed, "counts": {}, "sha256": {}, "files": {}}
    for sp, docs in splits.items():
        p = os.path.join(args.out, sp + ".txt")
        with open(p, "w") as f:
            for puz, sol in docs:
                f.write("Puzzle: %s / Solution: %s\n" % (puz, sol))
        manifest["counts"][sp] = len(docs)
        manifest["sha256"][sp] = hashlib.sha256(open(p, "rb").read()).hexdigest()
        manifest["files"][sp] = p
    tp = os.path.join(args.out, "test_prompts.txt")
    ta = os.path.join(args.out, "test_answers.txt")
    with open(tp, "w") as f, open(ta, "w") as g:
        for puz, sol in splits["test"]:
            f.write("Puzzle: %s / Solution:\n" % puz)
            g.write(sol + "\n")
    manifest["files"]["test_prompts"] = tp
    manifest["files"]["test_answers"] = ta
    manifest["sha256"]["test_prompts"] = hashlib.sha256(open(tp, "rb").read()).hexdigest()
    manifest["sha256"]["test_answers"] = hashlib.sha256(open(ta, "rb").read()).hexdigest()
    mp = os.path.join(args.out, "manifest.json")
    json.dump(manifest, open(mp, "w"), indent=2)
    print("sudokutext: " + json.dumps(manifest["counts"]) + " manifest=" + mp)


if __name__ == "__main__":
    main()
