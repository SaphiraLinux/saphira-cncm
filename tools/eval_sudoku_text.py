#!/usr/bin/env python3
"""tools/eval_sudoku_text.py — query an LM on held-out Sudoku puzzles and
verify completions with the deterministic engine (sudoku-check).

For each prompt ("Puzzle: <81 .1-9> / Solution:") the model generates a
continuation; the harness takes the first 81 [1-9.] cells after
"Solution:", requires all 81 filled 1-9, and checks via the C engine:
  1. valid: full Sudoku constraints hold.
  2. clue_ok: givens match the puzzle.

A response counts as SOLVED only if complete AND valid AND clue_ok.
Reports validity/clue rates. The reference solution is NOT used for
scoring (puzzles may admit several completions) — only logged.

Usage:
    python3 tools/eval_sudoku_text.py --bin ./cncm-cuda --conf ... \\
        --weights ... --asset-dir ... \\
        --prompts results/sudoku-text-v1/test_prompts.txt --n 50 --seed 0
"""
import argparse
import json
import random
import re
import subprocess
import sys

CELL_RE = re.compile(r"[1-9.]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--new-tokens", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine-bin", default=None)
    args = ap.parse_args()
    engine = args.engine_bin or args.bin

    prompts = [l.rstrip("\n") for l in open(args.prompts) if l.strip()]
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(prompts)), min(args.n, len(prompts)))

    stat = {"n": 0, "solved": 0, "complete": 0, "valid": 0, "clue_ok": 0}
    for i in idx:
        p = prompts[i]
        puz = p.split("Puzzle:")[1].split("/")[0].strip()
        r = subprocess.run(
            [args.bin, "gpt2-gen", "--conf", args.conf, "--weights", args.weights,
             "--asset-dir", args.asset_dir, "--prompt", p,
             "--n", str(args.new_tokens), "--temp", "0.0", "--seed", "1"],
            capture_output=True, text=True, timeout=300)
        text = (r.stdout or "") + (r.stderr or "")
        # drop the "ids: ..." diagnostics line: its digits would pollute
        # the cell parser below.
        genlines = [l for l in text.split("\n") if not l.startswith("ids:")]
        tail = "\n".join(genlines).split("Solution:")[-1]
        cells = CELL_RE.findall(tail)[:81]
        stat["n"] += 1
        if len(cells) < 81 or "." in cells:
            continue
        stat["complete"] += 1
        board = "".join(cells)
        e = subprocess.run(
            [engine, "sudoku-check", "--board", board, "--clues", puz],
            capture_output=True, text=True, timeout=60)
        try:
            got = json.loads((e.stdout or "").strip().split("\n")[0])
        except (ValueError, IndexError):
            continue
        if got.get("valid"):
            stat["valid"] += 1
        if got.get("clue_ok"):
            stat["clue_ok"] += 1
        if got.get("valid") and got.get("clue_ok"):
            stat["solved"] += 1
    n = max(stat["n"], 1)
    print("eval_sudoku_text: n=%d solved=%d (%.1f%%) complete=%.1f%% "
          "valid=%.1f%% clue_ok=%.1f%%" % (
              stat["n"], stat["solved"], 100.0 * stat["solved"] / n,
              100.0 * stat["complete"] / n, 100.0 * stat["valid"] / n,
              100.0 * stat["clue_ok"] / n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
