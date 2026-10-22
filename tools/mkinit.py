#!/usr/bin/env python3
"""Build a seeded initial NCM1 weights file for any CNCM config (numpy only).

Key inventory mirrors weights/sudoku-easy/model.ncm (verified 200 keys):
tok.encoder.{wte,wpe}, tok.vq.embeddings.{0..L-1}, tok.vq.quant_resi.*,
tok.vq.learned_{down,up}_l1.0, tok.decoder.{ln_f,middle_block.*,output_layer},
gen.l0_queries, gen.transformer.{wte,wle,blocks.0..7,ln_f}, gen.lm_heads.*.

Shapes derive from CLI dims; count of level-indexed keys = len(ladder).

Init (seeded, deterministic): 2D/3D weights N(0,0.02), codebook
embeddings uniform(-0.0625,0.0625), biases 0, norm weights 1.

Usage (16x16):
  python3 tools/mkinit.py --out weights/sudoku-16/init16.ncm --seed 16 \\
      --ladder 1,2,4,7,11,17,27,42,65,101,157,271 \\
      --T 271 --P 273 --vin 19 --vout 19 --vwle 19
Usage (magic 3x3):
  python3 tools/mkinit.py --out weights/magic-3/init.ncm --seed 3 \\
      --ladder 1,2,3,5,11 --T 11 --P 13 --vin 12 --vout 12 --vwle 12
"""
import argparse
import struct
import sys

import numpy as np


def gen_keys(ladder, T, P, vin, vout, vwle):
    L = len(ladder)
    kv = []
    kv.append(("tok.encoder.wte.weight", (vin, 256)))
    kv.append(("tok.encoder.wpe.weight", (T, 256)))
    for i in range(L):
        kv.append(("tok.vq.embeddings.%d.weight" % i, (16, 256)))
    for i in range(L):
        kv.append(("tok.vq.quant_resi.%d.weight" % i, (256, 256, 3)))
        kv.append(("tok.vq.quant_resi.%d.bias" % i, (256,)))
    kv.append(("tok.vq.learned_down_l1.0.weight", (256, 256, T)))
    kv.append(("tok.vq.learned_down_l1.0.bias", (256,)))
    kv.append(("tok.vq.learned_up_l1.0.weight", (256, 256, T)))
    kv.append(("tok.vq.learned_up_l1.0.bias", (256,)))
    kv.append(("tok.decoder.ln_f.weight", (256,)))
    kv.append(("tok.decoder.middle_block.ln_1.weight", (256,)))
    kv.append(("tok.decoder.middle_block.ln_1.bias", (256,)))
    kv.append(("tok.decoder.middle_block.attn.c_attn.weight", (768, 256)))
    kv.append(("tok.decoder.middle_block.attn.c_attn.bias", (768,)))
    kv.append(("tok.decoder.middle_block.attn.c_proj.weight", (256, 256)))
    kv.append(("tok.decoder.middle_block.attn.c_proj.bias", (256,)))
    kv.append(("tok.decoder.middle_block.attn.q_norm.weight", (64,)))
    kv.append(("tok.decoder.middle_block.attn.k_norm.weight", (64,)))
    kv.append(("tok.decoder.middle_block.ln_2.weight", (256,)))
    kv.append(("tok.decoder.middle_block.ln_2.bias", (256,)))
    kv.append(("tok.decoder.middle_block.mlp.c_fc.weight", (1024, 256)))
    kv.append(("tok.decoder.middle_block.mlp.c_fc.bias", (1024,)))
    kv.append(("tok.decoder.middle_block.mlp.c_proj.weight", (256, 1024)))
    kv.append(("tok.decoder.middle_block.mlp.c_proj.bias", (256,)))
    kv.append(("tok.decoder.output_layer.weight", (vout, 256)))
    kv.append(("gen.l0_queries", (1, 1, 256)))
    kv.append(("gen.transformer.wte.weight", (512, 256)))
    kv.append(("gen.transformer.wte.bias", (512,)))
    kv.append(("gen.transformer.wle.weight", (vwle, 512)))
    for b in range(8):
        p = "gen.transformer.blocks.%d." % b
        kv.append((p + "ln_1.weight", (512,)))
        kv.append((p + "ln_1.bias", (512,)))
        kv.append((p + "attn.c_attn.weight", (1536, 512)))
        kv.append((p + "attn.c_attn.bias", (1536,)))
        kv.append((p + "attn.c_proj.weight", (512, 512)))
        kv.append((p + "attn.c_proj.bias", (512,)))
        kv.append((p + "attn.q_norm.weight", (64,)))
        kv.append((p + "attn.k_norm.weight", (64,)))
        kv.append((p + "ln_2.weight", (512,)))
        kv.append((p + "ln_2.bias", (512,)))
        kv.append((p + "mlp.c_fc.weight", (2048, 512)))
        kv.append((p + "mlp.c_fc.bias", (2048,)))
        kv.append((p + "mlp.c_proj.weight", (512, 2048)))
        kv.append((p + "mlp.c_proj.bias", (512,)))
    kv.append(("gen.transformer.ln_f.weight", (512,)))
    kv.append(("gen.transformer.ln_f.bias", (512,)))
    for i in range(L):
        kv.append(("gen.lm_heads.%d.weight" % i, (16, 512)))
        kv.append(("gen.lm_heads.%d.bias" % i, (16,)))
    return kv


def init_tensor(name, shape, rng):
    if name.endswith(".bias"):
        return np.zeros(shape, dtype=np.float32)
    if "embeddings." in name:
        return rng.uniform(-0.0625, 0.0625, size=shape).astype(np.float32)
    if ".weight" not in name:
        if name == "gen.l0_queries":
            return rng.normal(0, 0.02, size=shape).astype(np.float32)
        raise SystemExit("no init rule for " + name)
    for tag in ("ln_1.weight", "ln_2.weight", "ln_f.weight",
                "q_norm.weight", "k_norm.weight"):
        if name.endswith(tag):
            return np.ones(shape, dtype=np.float32)
    return rng.normal(0, 0.02, size=shape).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--P", type=int, required=True)
    ap.add_argument("--vin", type=int, required=True)
    ap.add_argument("--vout", type=int, required=True)
    ap.add_argument("--vwle", type=int, required=True)
    a = ap.parse_args()
    ladder = [int(v) for v in a.ladder.split(",")]
    assert ladder[-1] == a.T, (ladder[-1], a.T)
    rng = np.random.default_rng(a.seed)
    kv = gen_keys(ladder, a.T, a.P, a.vin, a.vout, a.vwle)
    with open(a.out, "wb") as f:
        f.write(b"NCM1")
        f.write(struct.pack("<I", len(kv)))
        for name, shape in kv:
            t = np.ascontiguousarray(init_tensor(name, shape, rng))
            nb = name.encode()
            f.write(struct.pack("<B", len(nb)))
            f.write(nb)
            f.write(struct.pack("<B", len(t.shape)))
            for d in t.shape:
                f.write(struct.pack("<I", d))
            raw = t.tobytes()
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)
    print("wrote", a.out, "tensors:", len(kv))


if __name__ == "__main__":
    sys.exit(main())
