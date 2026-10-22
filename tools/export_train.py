#!/usr/bin/env python3
"""Export training rows for cncm: prefix ids + VQ codes per row (.nctrain).
Row layout: [P prefix ids][total codes, per-board level-major].
Codes come from the FROZEN tokenizer (release checkpoint or your own).

Usage:
  python3 tools/export_train.py --task sudoku-easy --n 512 --out weights/sudoku-easy/train.nctrain
  (tasks: sudoku-easy/medium/hard use puzzles.pickle; others need a corpus)
"""
import argparse
import os
import struct
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holes", type=int, default=96,
                    help="sudoku-16 only: cells to dig per puzzle")
    args = ap.parse_args()

    if args.task == "sudoku-16":
        # Pure-python path (stdlib only): solved grids + digs.
        # VQ encoding into .nctrain rows happens in the documented
        # 16x16 training flow (TRAINING.md) once a 16x16 tokenizer
        # checkpoint exists; --dry-run only prints prefix rows.
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        import sudoku16
        import random as pyrandom
        rng = pyrandom.Random(args.seed)
        if args.out == "-":
            for i in range(args.n):
                full = sudoku16.shuffled_full(rng)
                pz = sudoku16.dig(full, rng, args.holes)
                print(",".join(map(str, sudoku16.prefix_ids(pz))))
            return
        raise SystemExit("sudoku-16 export to .nctrain needs a 16x16 VQ "
                         "checkpoint (see TRAINING.md); use --out - for rows")

    sys.path.insert(0, "/scratch/ncm")
    os.chdir("/scratch/ncm")
    import torch
    from omegaconf import OmegaConf
    from tokenization.vqvae_tokenizer import VQVAETokenizer
    CFG = OmegaConf.load(f"config/eval/{args.task}.yaml")
    SUB = {"sudoku-easy": "sudoku/easy", "sudoku-medium": "sudoku/medium",
           "sudoku-hard": "sudoku/hard", "lm1b-small": "lm1b/small",
           "tinystories": "tinystories"}
    CK = f"/scratch/ncm/artifacts/{SUB.get(args.task, args.task)}"
    CFG.tokenizer.vqvae.checkpoint_path = os.path.join(CK, CFG.checkpoints.tokenizer)
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if CFG.checkpoints.get(name):
            CFG.tokenizer.pre_tokenizer[key] = os.path.join(CK, CFG.checkpoints[name])
    torch.manual_seed(args.seed)
    tk = VQVAETokenizer.from_config(CFG.tokenizer)
    tk.vqvae = tk.vqvae.eval()

    import pickle
    P = int(CFG.model.prefix_len)
    T = int(CFG.tokenizer.vqvae.context_length)
    if args.task.startswith("sudoku"):
        items = pickle.load(open(f"/scratch/ncm/data/{SUB[args.task]}/puzzles.pickle", "rb"))
        rows = []
        for ids in items[:args.n]:
            ids = [int(v) for v in ids]
            pre = (ids[:P] + [0] * P)[:P]
            chunk = (ids[:T] + [0] * T)[:T]
            with torch.no_grad():
                idx = tk.vqvae.encode(torch.tensor([chunk])).tolist()[0]
            rows.append(pre + idx)
    else:
        raise SystemExit("corpus export for non-sudoku tasks: wire your text corpus here")

    total = len(rows[0])
    print(f"{len(rows)} rows x {total} (P={P}, codes={total - P})")
    with open(args.out, "wb") as f:
        f.write(b"NCMT")
        f.write(struct.pack("<Q", len(rows)))
        f.write(struct.pack("<Q", total))
        import numpy as np
        np.array(rows, dtype=np.int32).tofile(f)
    print("OK", args.out)


if __name__ == "__main__":
    main()
