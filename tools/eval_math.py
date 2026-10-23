#!/usr/bin/env python3
"""tools/eval_math.py — query a trained LM on held-out math problems and
verify answers with a deterministic engine (integer arithmetic).

For each prompt ("Solve: <expr> = ? / Step:") the model generates a
continuation; the harness extracts "Step: <expr2> = <v>" and "Answer: <a>"
and checks:
  1. step_valid: expr2 evaluates (exact integer arithmetic) to v, AND
     expr2 is the same expression as asked (no problem-switching).
  2. answer_ok: a equals the reference answer.
  3. consistent: a equals v (model's own step agrees with its answer).

A response counts as SOLVED only if all three hold. Reports per-op
accuracy plus the three diagnostic rates.

Usage:
    python3 tools/eval_math.py --bin ./cncm-cuda --conf tools/gpt2-124m.conf \\
        --weights weights/math-gpt2/ckpt.nckpt --asset-dir weights/gpt2 \\
        --prompts results/math-v1/test_prompts.txt \\
        --answers results/math-v1/test_answers.txt --n 100 --seed 0
"""
import argparse
import ast
import operator
import random
import re
import subprocess
import sys


OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.floordiv, ast.Mod: operator.mod}


def arith(s):
    """Exact integer arithmetic on 'a + b' / 'a - b' / 'a * b' / 'a / b' /
    'a + b + c' / 'a * b + c'. Raises on anything else (no floats, no names)."""
    tree = ast.parse(s, mode="eval")
    if any(isinstance(n, (ast.Name, ast.Call, ast.Attribute)) for n in ast.walk(tree)):
        raise ValueError("names not allowed: " + s)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Div):
                if r == 0 or l % r != 0:
                    raise ValueError("inexact division: " + s)
                return l // r
            return OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError("bad node: " + s)

    v = ev(tree)
    if not isinstance(v, int):
        raise ValueError("non-integer: " + s)
    return v


STEP_RE = re.compile(r"Step:\s*(.+?)\s*=\s*(-?\d+)")
ANS_RE = re.compile(r"Answer:\s*(-?\d+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--conf", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--asset-dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prompts = [l.rstrip("\n") for l in open(args.prompts) if l.strip()]
    answers = [l.strip() for l in open(args.answers) if l.strip()]
    assert len(prompts) == len(answers)
    rng = random.Random(args.seed)
    idx = rng.sample(range(len(prompts)), min(args.n, len(prompts)))

    stat = {"n": 0, "solved": 0, "step_valid": 0, "answer_ok": 0,
            "consistent": 0, "parsed": 0}
    by_op = {}
    for i in idx:
        p, ref = prompts[i], int(answers[i])
        op = "chain" if p.count("+") == 2 else (
            "muladd" if ("*" in p and "+" in p) else (
                {"+": "add", "-": "sub", "*": "mul", "/": "div"}[
                    next(c for c in "+-*/" if c in p)]))
        r = subprocess.run(
            [args.bin, "gpt2-gen", "--conf", args.conf, "--weights", args.weights,
             "--asset-dir", args.asset_dir, "--prompt", p,
             "--n", str(args.new_tokens), "--temp", "0.0", "--seed", "1"],
            capture_output=True, text=True, timeout=300)
        text = (r.stdout or "") + (r.stderr or "")
        sm = STEP_RE.search(text)
        am = ANS_RE.search(text)
        asked = p.split("Solve:")[1].split("=")[0].strip() if "Solve:" in p else ""
        row = by_op.setdefault(op, {"n": 0, "solved": 0})
        row["n"] += 1
        stat["n"] += 1
        if not (sm and am):
            continue
        stat["parsed"] += 1
        try:
            sv = arith(sm.group(1).strip())
            step_ok = (sv == int(sm.group(2))) and (sm.group(1).strip() == asked)
        except ValueError:
            step_ok = False
        a = int(am.group(1))
        ans_ok = (a == ref)
        cons = (a == int(sm.group(2)))
        stat["step_valid"] += bool(step_ok)
        stat["answer_ok"] += bool(ans_ok)
        stat["consistent"] += bool(cons)
        if step_ok and ans_ok and cons:
            stat["solved"] += 1
            row["solved"] += 1
    n = max(stat["n"], 1)
    print("eval_math: n=%d solved=%d (%.1f%%) step_valid=%.1f%% answer_ok=%.1f%% "
          "consistent=%.1f%% parsed=%.1f%%" % (
              stat["n"], stat["solved"], 100.0 * stat["solved"] / n,
              100.0 * stat["step_valid"] / n, 100.0 * stat["answer_ok"] / n,
              100.0 * stat["consistent"] / n, 100.0 * stat["parsed"] / n))
    for op in sorted(by_op):
        r = by_op[op]
        print("  %-7s %3d/%-3d %.1f%%" % (op, r["solved"], r["n"],
                                          100.0 * r["solved"] / max(r["n"], 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
