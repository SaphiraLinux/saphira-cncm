#!/usr/bin/env python3
"""Number-theory discovery data: taxicab collisions + LCM table (stdlib only).

Taxicab: all a^3+b^3 collisions for 1<=a<=b<=C (C=100 finds 1729=1^3+12^3
=9^3+10^3 and 4104=2^3+16^3=9^3+15^3 among others). Output: JSONL rows
{"n": N, "pairs": [[a,b],...]} — the search task, not just the answer.

LCM: {"hi": H, "lcm": L} for H=1..12 (2520 at H=10; multiples noted).
Also emits a small verified-facts JSONL for the two headline numbers.

Usage:
  python3 tools/arith.py --selftest
  python3 tools/arith.py --emit-data DIR [--cubes 100]
"""
import argparse
import json
import math
import os
import sys


def taxicab(limit):
    sums = {}
    for a in range(1, limit + 1):
        a3 = a * a * a
        for b in range(a, limit + 1):
            s = a3 + b * b * b
            sums.setdefault(s, []).append([a, b])
    return {s: ps for s, ps in sums.items() if len(ps) > 1}


def lcm(a, b):
    return a // math.gcd(a, b) * b


def lcm_table(hi):
    out, cur = [], 1
    for h in range(1, hi + 1):
        cur = lcm(cur, h)
        out.append({"hi": h, "lcm": cur})
    return out


def selftest():
    tc = taxicab(20)
    assert 1729 in tc and sorted(map(sorted, tc[1729])) == [[1, 12], [9, 10]]
    assert 4104 in tc
    t = lcm_table(10)
    assert t[-1] == {"hi": 10, "lcm": 2520}
    assert 2520 % 9 == 0 and 5040 % 10 == 0
    print("arith selftest OK: 1729, 4104 collisions; LCM(1..10)=2520")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-data", default="")
    ap.add_argument("--cubes", type=int, default=100)
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return 0
    if a.emit_data:
        os.makedirs(os.path.join(a.emit_data, "train"), exist_ok=True)
        tc = taxicab(a.cubes)
        rows = [{"n": s, "pairs": sorted(ps)}
                for s, ps in sorted(tc.items())]
        with open(os.path.join(a.emit_data, "train", "taxicab.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(a.emit_data, "train", "lcm.jsonl"), "w") as f:
            for r in lcm_table(12):
                f.write(json.dumps(r) + "\n")
        facts = [
            {"fact": "1729=1^3+12^3=9^3+10^3",
             "check": "arith-check --op taxicab --a 9 --b 10 --c 1 --d 12"},
            {"fact": "LCM(1..10)=2520 (multiples like 5040 work too)",
             "check": "arith-check --op divisible --n 2520 --by 1,2,3,4,5,6,7,8,9,10"},
        ]
        with open(os.path.join(a.emit_data, "PROVENANCE.txt"), "w") as f:
            f.write("taxicab collisions a<=b<=%d: %d multi-way sums\n"
                    % (a.cubes, len(rows)))
            f.write("lcm table hi=1..12. learned property, not trivia:\n")
            f.write("a model should find pairs for unseen sums and test\n")
            f.write("divisibility for unseen (n, range) inputs.\n")
        with open(os.path.join(a.emit_data, "train", "facts.jsonl"), "w") as f:
            for r in facts:
                f.write(json.dumps(r) + "\n")
        print("taxicab sums: %d, lcm rows: 12" % len(rows))
        return 0
    selftest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
