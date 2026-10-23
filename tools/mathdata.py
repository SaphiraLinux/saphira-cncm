#!/usr/bin/env python3
"""tools/mathdata.py — arithmetic reasoning corpus v1 for LM training.

Emits one document per line (docs are later packed to (T+1)-id rows by
mkqwenrows, EOS-joined). Each doc is a problem with a worked step and a
final answer the deterministic engine can verify by integer arithmetic:

    Solve: 47 + 58 = ?
    Step: 47 + 58 = 105
    Answer: 105

Splits are disjoint by (op, a, b) triple — verified in-process, with a
JSONL manifest recording seeds, counts and sha256 of every split file.

Docs are SINGLE lines (mkqwenrows takes one document per file line,
EOS-joined):

    Solve: 47 + 58 = ? / Step: 47 + 58 = 105 / Answer: 105

Ops: add, sub (non-negative results), mul, div (exact), plus two-step
chains (add-add, mul-add) that exercise multi-hop working.

Usage:
    python3 tools/mathdata.py --out results/math-v1 --train 20000 \\
        --val 1000 --test 2000 --seed 7
"""
import argparse
import hashlib
import json
import os
import random
import sys


def gen_add(rng):
    a = rng.randint(0, 999)
    b = rng.randint(0, 999)
    return ("add", a, b, a + b, "%d + %d" % (a, b))


def gen_sub(rng):
    a = rng.randint(0, 999)
    b = rng.randint(0, a)  # non-negative results in v1
    return ("sub", a, b, a - b, "%d - %d" % (a, b))


def gen_mul(rng):
    a = rng.randint(0, 99)
    b = rng.randint(0, 99)
    return ("mul", a, b, a * b, "%d * %d" % (a, b))


def gen_div(rng):
    b = rng.randint(1, 99)
    q = rng.randint(0, 99)
    a = b * q
    return ("div", a, b, q, "%d / %d" % (a, b))


def gen_chain_add(rng):
    a = rng.randint(0, 199)
    b = rng.randint(0, 199)
    c = rng.randint(0, 199)
    return ("chain", a, b, a + b + c, "%d + %d + %d" % (a, b, c))


def gen_muladd(rng):
    a = rng.randint(0, 49)
    b = rng.randint(0, 49)
    c = rng.randint(0, 499)
    return ("muladd", a, b, a * b + c, "%d * %d + %d" % (a, b, c))


GENS = [gen_add, gen_sub, gen_mul, gen_div, gen_chain_add, gen_muladd]


def make_doc(expr, ans):
    return "Solve: %s = ? / Step: %s = %d / Answer: %d" % (expr, expr, ans, ans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=1000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    used = set()
    splits = {"train": [], "val": [], "test": []}
    need = {"train": args.train, "val": args.val, "test": args.test}
    # op chosen uniformly at random per doc: every split sees every op.
    # (A shared round-robin counter would lock each split to 2 of 6 ops.)
    for sp in ("train", "val", "test"):
        guard = 0
        while len(splits[sp]) < need[sp]:
            g = rng.choice(GENS)
            op, a, b, ans, expr = g(rng)
            key = (op, expr)  # full expression: every varying arg counts
            if key in used:
                guard += 1
                if guard > 10 * need[sp] + 100000:
                    sys.exit("mathdata: space exhausted (duplicate rejection runaway)")
                continue
            used.add(key)
            splits[sp].append((make_doc(expr, ans), op, a, b, ans))
    # hard disjointness audit (triples unique across all splits by construction)
    assert len(used) == sum(len(v) for v in splits.values())

    manifest = {"seed": args.seed, "counts": {}, "sha256": {}, "files": {}}
    for sp, docs in splits.items():
        p = os.path.join(args.out, sp + ".txt")
        with open(p, "w") as f:
            for d, op, a, b, ans in docs:
                f.write(d + "\n")
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        manifest["counts"][sp] = len(docs)
        manifest["sha256"][sp] = h
        manifest["files"][sp] = p
    # test prompts/answers for the query eval (problem line in, answer out)
    tp = os.path.join(args.out, "test_prompts.txt")
    ta = os.path.join(args.out, "test_answers.txt")
    with open(tp, "w") as f, open(ta, "w") as g:
        for d, op, a, b, ans in splits["test"]:
            prob = d.split(" / ")[0]  # "Solve: <expr> = ?"
            f.write(prob + " / Step:" + "\n")
            g.write(str(ans) + "\n")
    manifest["files"]["test_prompts"] = tp
    manifest["files"]["test_answers"] = ta
    manifest["sha256"]["test_prompts"] = hashlib.sha256(open(tp, "rb").read()).hexdigest()
    manifest["sha256"]["test_answers"] = hashlib.sha256(open(ta, "rb").read()).hexdigest()
    mp = os.path.join(args.out, "manifest.json")
    json.dump(manifest, open(mp, "w"), indent=2)
    print("mathdata: " + json.dumps(manifest["counts"]) + " manifest=" + mp)


if __name__ == "__main__":
    main()
