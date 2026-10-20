#!/usr/bin/env python3
"""One-time exporter: released .pt checkpoints -> dependency-free .ncm + .conf.

The C app never touches torch/pickle/zip. Run once per task on a machine
with torch installed:

    python3 tools/export.py --config /scratch/ncm/config/eval/sudoku-easy.yaml \\
        --checkpoint-dir /scratch/ncm/artifacts/sudoku/easy --out cncm-weights/sudoku-easy

Outputs:
    <out>/model.ncm   weights (magic "NCM1", name/shape/raw-f32 records)
    <out>/model.conf  hyper-parameters as key=value (subset of the yaml)
    <out>/bpe_vocab.txt, bpe_merges.txt   (only for bpe pre-tokenizers)

Pretrained HF tokenizers (bert/gpt2/qwen) are fetched from the HF hub cache
by transformers at export time and re-emitted as plain vocab/merges text so
the C app reads them without transformers. WordPiece (bert) emits
wordpiece.txt; byte-BPE (gpt2/qwen) emits bpe_vocab.txt + bpe_merges.txt.
"""
import argparse
import os
import struct

import torch
from omegaconf import OmegaConf


SKIP_SUBSTRINGS = ("ema_cluster_size_", "ema_w_", "codebook_hit_",
                   "attn_mask_for_masking", "lvl_1L", "rope_cos", "rope_sin",
                   "flex_lvl_full", "_fd_step", "_km_done", "wp_prefix", "wp_1L")


def should_skip(name):
    return any(s in name for s in SKIP_SUBSTRINGS)


def write_ncm(path, named_tensors):
    with open(path, "wb") as f:
        f.write(b"NCM1")
        f.write(struct.pack("<I", len(named_tensors)))
        for name, t in named_tensors:
            t = t.detach().to(torch.float32).contiguous().cpu()
            nb = name.encode("utf-8")
            f.write(struct.pack("<B", len(nb)))
            f.write(nb)
            f.write(struct.pack("<B", len(t.shape)))
            for d in t.shape:
                f.write(struct.pack("<I", d))
            raw = t.numpy().tobytes()
            f.write(struct.pack("<Q", len(raw)))
            f.write(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    os.makedirs(args.out, exist_ok=True)

    gen = torch.load(os.path.join(args.checkpoint_dir, cfg.checkpoints.generator),
                     map_location="cpu", weights_only=False)
    tok = torch.load(os.path.join(args.checkpoint_dir, cfg.checkpoints.tokenizer),
                     map_location="cpu", weights_only=False)

    recs = []
    for k, v in tok["model"].items():
        if isinstance(v, torch.Tensor) and not should_skip(k):
            recs.append(("tok." + k, v))
    for k, v in gen["model"].items():
        if isinstance(v, torch.Tensor) and not should_skip(k):
            recs.append(("gen." + k, v))
    inf = gen.get("inference", {}) or {}
    for k, v in inf.items():
        if isinstance(v, torch.Tensor):
            recs.append(("inference." + k, v))
    write_ncm(os.path.join(args.out, "model.ncm"), recs)
    print(f"wrote {len(recs)} tensors")

    m, t = cfg.model, cfg.tokenizer
    g = cfg.get("generation", {})
    L = {}

    def put(k, v):
        if v is None:
            v = -1
        L[k] = v

    put("task", cfg.task)
    for k in ("n_layer", "n_head", "n_embd", "context_length", "prefix_len",
              "raw_token_prefix", "prefix_pad_token_id", "predict_level0",
              "query_level0", "l0_autoregressive", "use_positional_encoding",
              "use_level_encoding", "use_rope", "rope_base", "use_qk_norm",
              "use_flex_attention", "dropout", "bias", "n_kv_heads",
              "contextual_prefix", "mlp_ratio", "use_swiglu", "use_geglu",
              "use_relu2", "parallel_block", "normformer", "use_layerscale",
              "layerscale_init", "use_rezero", "tie_heads_to_codebook",
              "tie_heads_learn_scale", "tie_heads_trainable", "tie_heads_bias",
              "shared_output_head", "shared_head_per_level_bias",
              "untied_head_bias"):
        if k in m:
            put("model." + k, m[k])
    put("model.attn_pattern", m.get("attn_pattern", "block_diagonal"))
    put("model.vocab_size", ",".join(str(x) for x in m.vocab_size))
    ctx = m.get("ctx_encoder", {}) or {}
    put("model.ctx_enabled", int(bool(ctx.get("enabled", False))))
    put("model.ctx_n_layer", int(ctx.get("n_layer", 0)))
    put("model.ctx_vocab", int(ctx.get("vocab_size", 0)))
    puc = m.get("post_upsample_conv", {}) or {}
    put("model.puc_enabled", int(bool(puc.get("enabled", False))))
    put("model.puc_kernel", int(puc.get("kernel_size", 3)))

    v = t.vqvae
    put("vq.embed_dim", v.embed_dim)
    put("vq.in_vocab", v.in_vocab_size)
    put("vq.context_length", v.context_length)
    put("vq.n_layers", v.n_layers)
    put("vq.compression", int(v.get("compression_factor", 1)))
    q = v.vector_quantizer_config
    lv = list(q.levels.manual_levels)
    put("vq.levels", ",".join(str(x) for x in lv))
    put("vq.vocab", ",".join(str(x) for x in v.vqvae_vocab_size))
    qr = q.get("quant_resi", {}) or {}
    put("vq.qr_enabled", int(bool(qr.get("enabled", True))))
    put("vq.qr_ratio", float(qr.get("ratio", 0.5)))
    put("vq.qr_share", int(qr.get("share_mode", 0)))
    put("vq.l1_sampling", int(bool(q.get("learned_l1_sampling", False))))
    put("vq.l1_cascade", int(bool(q.get("learned_l1_cascade", False))))
    put("vq.l1_base", int(q.get("learned_l1_cascade_base", 4)))
    put("vq.l1_norm", int(bool(q.get("learned_l1_cascade_norm", False))))
    put("vq.l1_norm_final", int(bool(q.get("learned_l1_cascade_norm_final", False))))
    put("vq.l1_norm", int(bool(q.get("learned_l1_cascade_norm", False))))
    put("vq.l1_norm_final", int(bool(q.get("learned_l1_cascade_norm_final", False))))
    put("vq.all_sampling", int(bool(q.get("learned_all_sampling", False))))
    put("vq.pre_quant_gn", int(v.get("pre_quant_groupnorm") or 0))
    put("vq.qk_norm", int(bool(v.get("use_qk_norm", False))))
    put("vq.enc_layers", int(v.get("enc_layers", 0)))
    put("vq.enc_window", int(v.get("enc_attn_window", 0)))
    put("vq.enc_convs", int(v.get("enc_conv_layers", 0)))
    put("vq.enc_conv_kernel", int(v.get("enc_conv_kernel", 3)))
    put("vq.enc_positional", int(bool(v.get("enc_positional", True))))
    put("vq.dec_layers", int(v.get("dec_layers", 1)))
    put("vq.dec_positional", int(bool(v.get("dec_positional", False))))

    pt = t.pre_tokenizer
    put("tok.kind", pt.name)
    put("tok.model_id", pt.get("model_id", ""))
    specials = dict(pt.get("special_tokens", {}) or {})
    put("tok.eos", specials.get("eos_token", ""))
    put("tok.pad", specials.get("pad_token", ""))
    # resolved integer ids (needed by generate for CFG null prefix etc.)
    if pt.name == "sudoku":
        put("tok.eos_id", -1)
        put("tok.pad_id", -1)
    elif pt.name == "bpe":
        vals = list(specials.values())
        eos_s, pad_s = specials.get("eos_token"), specials.get("pad_token")
        put("tok.eos_id", vals.index(eos_s) if eos_s in vals else -1)
        put("tok.pad_id", vals.index(pad_s) if pad_s in vals else -1)

    for k, ck in (("temperature", "temperature"), ("top_k", "top_k"),
                  ("top_p", "top_p"), ("cfg_scale", "cfg_scale"),
                  ("frequency_penalty", "freq"), ("frequency_deficit", "mu")):
        if k in g:
            put("gen." + ck, g[k])
    adj = str(g.get("adjacency_penalty", "") or "")
    put("gen.adj", adj)

    ckpts = dict(cfg.checkpoints)
    for k in ("bpe_vocab", "bpe_merges"):
        if ckpts.get(k):
            put("tok." + k, ckpts[k])

    with open(os.path.join(args.out, "model.conf"), "w") as f:
        for k in sorted(L):
            f.write(f"{k}={L[k]}\n")
    print("wrote model.conf")

    # ---- tokenizer assets as plain text ----
    if pt.name == "bpe":
        import pickle
        vf = os.path.join(args.checkpoint_dir, ckpts["bpe_vocab"])
        mf = os.path.join(args.checkpoint_dir, ckpts["bpe_merges"])
        vocab = pickle.load(open(vf, "rb"))      # id -> bytes
        merges = pickle.load(open(mf, "rb"))     # list of (bytes, bytes)
        with open(os.path.join(args.out, "bpe_vocab.txt"), "w") as f:
            for i in range(len(vocab)):
                f.write(f"{i}\t{vocab[i].hex()}\n")
        with open(os.path.join(args.out, "bpe_merges.txt"), "w") as f:
            for a, b in merges:
                f.write(f"{a.hex()} {b.hex()}\n")
        print(f"wrote bpe assets ({len(vocab)} vocab, {len(merges)} merges)")
    if pt.name == "hf":
        mid = pt.model_id
        from transformers import AutoTokenizer
        hz = AutoTokenizer.from_pretrained(mid)
        eos_s, pad_s = specials.get("eos_token"), specials.get("pad_token")
        if eos_s:
            hz.eos_token = eos_s
        if pad_s:
            if hz.convert_tokens_to_ids(pad_s) in (None, hz.unk_token_id, hz.eos_token_id) \
                    and pad_s != hz.eos_token:
                hz.add_special_tokens({"pad_token": pad_s})
            else:
                hz.pad_token = pad_s
        elif hz.pad_token is None:
            hz.pad_token = hz.eos_token
        put("tok.eos_id", hz.eos_token_id if hz.eos_token_id is not None else -1)
        put("tok.pad_id", hz.pad_token_id if hz.pad_token_id is not None else -1)
        if "bert" in mid.lower():
            vv = hz.get_vocab()  # token -> id
            inv = sorted((i, s) for s, i in vv.items())
            assert [i for i, _ in inv] == list(range(len(inv))), "non-contiguous wordpiece ids"
            with open(os.path.join(args.out, "wordpiece.txt"), "w") as f:
                for _, s in inv:
                    f.write(f"{s}\n")  # line number == id
            print(f"wrote wordpiece.txt ({len(inv)} entries)")
        else:
            # byte-BPE (gpt2/qwen): merges as hex of UTF-8 bytes, one per line;
            # vocab as id<TAB>token lines. C decodes UTF-8 text to bytes and
            # runs the same byte-BPE loop as the custom BPE path.
            vv = hz.get_vocab()
            inv = sorted((i, s) for s, i in vv.items())
            with open(os.path.join(args.out, "bpe_vocab.txt"), "w") as f:
                for i, s in inv:
                    f.write(f"{i}\t{s.encode('utf-8').hex()}\n")
            ranks = sorted(hz.bpe_ranks.items(), key=lambda kv: kv[1])
            with open(os.path.join(args.out, "bpe_merges.txt"), "w") as f:
                for (a, b), _ in ranks:
                    ah = a.hex() if isinstance(a, bytes) else a.encode("utf-8").hex()
                    bh = b.hex() if isinstance(b, bytes) else b.encode("utf-8").hex()
                    f.write(f"{ah} {bh}\n")
            print(f"wrote hf byte-bpe assets ({len(inv)} vocab, {len(ranks)} merges)")
    print("OK", args.out)


if __name__ == "__main__":
    main()
