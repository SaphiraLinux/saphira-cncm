#!/usr/bin/env python3
"""Minimal stdlib SVG plots for CNCM results (no dependencies).

Usage:
  python3 tools/plot_results.py --eval-jsonl results/magic_held256.jsonl \\
      --out results/magic_held_eval.svg --title "magic 3x3 held-out (n=256)"
  python3 tools/plot_results.py --train-log results/logs/train800_resume.log \\
      --out results/ncp16_resume_curve.svg --title "16x16 NCP resume 600+"
"""
import argparse
import json
import re
import sys


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bar_chart(path, title, items):
    """items: [(label, frac)]. 0..1 bars with % labels."""
    W, H, pad = 560, 120 + 34 * len(items), 150
    bar_w = W - pad - 90
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (W, H)]
    lines.append('<rect width="100%%" height="100%%" fill="white"/>')
    lines.append('<text x="20" y="28" font-family="sans-serif" font-size="16" font-weight="bold">%s</text>'
                 % esc(title))
    y = 60
    for label, frac in items:
        pct = 100.0 * frac
        lines.append('<text x="20" y="%d" font-family="sans-serif" font-size="13">%s</text>' % (y + 13, esc(label)))
        lines.append('<rect x="%d" y="%d" width="%d" height="20" fill="#eeeeee"/>' % (pad, y, bar_w))
        lines.append('<rect x="%d" y="%d" width="%.1f" height="20" fill="#1b5ecf"/>' % (pad, y, bar_w * max(0.0, min(1.0, frac))))
        lines.append('<text x="%d" y="%d" font-family="monospace" font-size="13">%.2f%%</text>' % (pad + bar_w + 8, y + 15, pct))
        y += 34
    lines.append('</svg>')
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


def line_chart(path, title, pts, xlabel="step", ylabel="loss"):
    """pts: [(x, y)]. Auto-scaled axes with min/max labels."""
    W, H = 640, 360
    m = dict(l=64, r=20, t=44, b=44)
    iw, ih = W - m["l"] - m["r"], H - m["t"] - m["b"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    pad = (y1 - y0) * 0.08 or 0.05
    y0 -= pad
    y1 += pad

    def X(x):
        return m["l"] + iw * (x - x0) / (x1 - x0)

    def Y(y):
        return m["t"] + ih * (1 - (y - y0) / (y1 - y0))

    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' % (W, H)]
    lines.append('<rect width="100%%" height="100%%" fill="white"/>')
    lines.append('<text x="20" y="26" font-family="sans-serif" font-size="16" font-weight="bold">%s</text>' % esc(title))
    for f in (0.0, 0.5, 1.0):
        yy = m["t"] + ih * f
        vv = y1 - f * (y1 - y0)
        lines.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#dddddd"/>' % (m["l"], yy, W - m["r"], yy))
        lines.append('<text x="%d" y="%d" font-family="monospace" font-size="11" text-anchor="end">%.3f</text>' % (m["l"] - 6, yy + 4, vv))
    for f in (0.0, 0.5, 1.0):
        xx = m["l"] + iw * f
        vv = x0 + f * (x1 - x0)
        lines.append('<text x="%d" y="%d" font-family="monospace" font-size="11" text-anchor="middle">%d</text>' % (xx, H - m["b"] + 18, vv))
    lines.append('<text x="%d" y="%d" font-family="sans-serif" font-size="12" text-anchor="middle">%s</text>' % (m["l"] + iw // 2, H - 8, esc(xlabel)))
    lines.append('<text x="14" y="%d" font-family="sans-serif" font-size="12" text-anchor="middle" transform="rotate(-90 14 %d)">%s</text>' % (m["t"] + ih // 2, m["t"] + ih // 2, esc(ylabel)))
    d = "M" + " L".join("%.1f,%.1f" % (X(x), Y(y)) for x, y in pts)
    lines.append('<path d="%s" fill="none" stroke="#1b5ecf" stroke-width="1.6"/>' % d)
    lines.append('<text x="%d" y="%d" font-family="monospace" font-size="11">n=%d last=%.4f</text>'
                 % (W - m["r"] - 190, m["t"] - 8, len(pts), ys[-1]))
    lines.append('</svg>')
    open(path, "w").write("\n".join(lines) + "\n")
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", default="")
    ap.add_argument("--train-log", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    if a.eval_jsonl:
        rows = [json.loads(l) for l in open(a.eval_jsonl) if l.strip()]
        n = len(rows)
        items = [
            ("exact match", sum(r["exact"] for r in rows) / n),
            ("valid boards", sum(r["valid"] for r in rows) / n),
            ("clues preserved", sum(r["clue_ok"] for r in rows) / n),
            ("mean cell accuracy", sum(r["cell_frac"] for r in rows) / n),
        ]
        bar_chart(a.out, "%s (n=%d)" % (a.title or "held-out eval", n), items)
    elif a.train_log:
        pts = []
        for l in open(a.train_log):
            m = re.search(r"step (\d+) loss ([0-9.]+)", l)
            if m and "gnorm" not in l:
                pts.append((int(m.group(1)), float(m.group(2))))
        if not pts:
            sys.exit("no loss points in %s" % a.train_log)
        line_chart(a.out, a.title or "training loss", pts)
    else:
        sys.exit("need --eval-jsonl or --train-log")
    return 0


if __name__ == "__main__":
    sys.exit(main())
