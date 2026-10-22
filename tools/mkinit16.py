#!/usr/bin/env python3
"""Build a seeded initial NCM1 weights file for 16x16 Sudoku (numpy only).

Reads the key/shape table of the frozen 9x9 reference
(weights/sudoku-easy/model.ncm) and emits the same 200 keys with
16x16-adjusted shapes:

  tok.encoder.wte.weight      (12,256)  -> (19,256)
  tok.encoder.wpe.weight      (89,256)  -> (271,256)
  tok.vq.learned_{down,up}_l1 (256,256,89) -> (256,256,271)
  tok.decoder.output_layer    (12,256)  -> (19,256)
  gen.transformer.wle.weight  (12,512)  -> (19,512)
  everything else: identical shapes (12 VQ levels kept)

Init (seeded, deterministic): 2D/3D weights N(0,0.02), codebook
embeddings uniform(-0.0625,0.0625), biases 0, norm weights 1,
l0_queries N(0,0.02).

Usage:
  python3 tools/mkinit16.py --ref weights/sudoku-easy/model.ncm \
      --out weights/sudoku-16/init16.ncm --seed 16
"""
import argparse
import struct
import sys

import numpy as np


def read_table(path):
    f = open(path, "rb")
    assert f.read(4) == b"NCM1"
    n = struct.unpack("<I", f.read(4))[0]
    out = []
    for _ in range(n):
        nl = struct.unpack("<B", f.read(1))[0]
        name = f.read(nl).decode()
        nd = struct.unpack("<B", f.read(1))[0]
        shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(nd))
        nb = struct.unpack("<Q", f.read(8))[0]
        f.seek(nb, 1)
        out.append((name, shape))
    return out


def map_shape(name, shape):
    if name == "tok.encoder.wte.weight":
        return (19, 256)
    if name == "tok.encoder.wpe.weight":
        return (271, 256)
    if name in ("tok.vq.learned_down_l1.0.weight",
                "tok.vq.learned_up_l1.0.weight"):
        return (256, 256, 271)
    if name == "tok.decoder.output_layer.weight":
        return (19, 256)
    if name == "gen.transformer.wle.weight":
        return (19, 512)
    return shape


def init_tensor(name, shape, rng):
    if name.endswith(".bias"):
        return np.zeros(shape, dtype=np.float32)
    if ".weight" not in name:
        if name == "gen.l0_queries":
            return rng.normal(0, 0.02, size=shape).astype(np.float32)
        raise SystemExit("no init rule for " + name)
    if "embeddings." in name:
        return rng.uniform(-0.0625, 0.0625, size=shape).astype(np.float32)
    if "ln" in name or "norm" in name:
        return np.ones(shape, dtype=np.float32)
    return rng.normal(0, 0.02, size=shape).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=16)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    table = read_table(a.ref)
    assert len(table) == 200, len(table)
    with open(a.out, "wb") as f:
        f.write(b"NCM1")
        f.write(struct.pack("<I", len(table)))
        for name, shape in table:
            ns = map_shape(name, shape)
            t = np.ascontiguousarray(init_tensor(name, ns, rng))
            nb = name.encode()
            f.write(struct.pack("<B", len(nb)))
            f.write(nb)
            f.write(struct.pack("<B", len(t.shape)))
            for d in t.shape:
                f.write(struct.pack("<I", d))
            raw = t.tobytes()
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
    print("wrote", a.out, "tensors:", len(table))


if __name__ == "__main__":
    sys.exit(main())
