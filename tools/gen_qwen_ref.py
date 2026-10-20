#!/usr/bin/env python3
"""One-time Qwen2.5-0.5B reference fixture generation (offline only).
Uses transformers+torch; output consumed by C tests. NOT a runtime dep.
"""
import json
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT = "/scratch/cncm/tests/fixtures"
os.makedirs(OUT, exist_ok=True)

tk = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B",
                                         torch_dtype=torch.float32).eval()

texts = {
    "s1": "Hello, my dog is cute",
    "s2": "The quick brown fox jumps over the lazy dog.",
    "num": "The price is 12345 dollars and 678 cents.",
    "uni": "café naïve — “quoted”",
    "punct": "She said, \"Hello... world!\" — and left.",
}
ref = {"texts": {}, "logits_files": {}, "generations": {}, "ppl": {}}
with torch.no_grad():
    for name, t in texts.items():
        ids = tk.encode(t)
        lg = m(torch.tensor([ids])).logits[0].float().numpy()
        fn = f"qwen_logits_{name}.f32"
        lg.astype(np.float32).tofile(os.path.join(OUT, fn))
        ref["texts"][name] = {"text": t, "ids": ids}
        ref["logits_files"][name] = {"file": fn, "T": lg.shape[0], "V": lg.shape[1]}
        print(name, ids, lg.shape)
    for name, prompt in [("gen1", "Hello, my dog is cute and"),
                        ("gen2", "In the beginning")]:
        ids = tk.encode(prompt)
        out = m.generate(torch.tensor([ids]), max_new_tokens=20, do_sample=False)
        gen = out[0].tolist()[len(ids):]
        ref["generations"][name] = {"prompt": prompt, "ids": gen,
                                    "text": tk.decode(gen)}
        print(name, gen, repr(tk.decode(gen)))
    ppl_text = ("The Washington Post reported that the committee would meet on "
                "Tuesday to discuss the new proposal for high-speed rail.")
    ids = tk.encode(ppl_text)
    x = torch.tensor([ids])
    lg = m(x).logits[0]
    nll = torch.nn.functional.cross_entropy(lg[:-1], x[0][1:],
                                            reduction="mean").item()
    ref["ppl"] = {"text": ppl_text, "ids": ids, "nll": nll,
                  "ppl": float(np.exp(nll))}
    print("ppl ids:", ids, "nll:", nll)

with open(os.path.join(OUT, "qwen_ref.json"), "w") as f:
    json.dump(ref, f, indent=1)

# C-friendly files (hex text \t ids)
def hx(s):
    return s.encode("utf-8").hex()

with open(os.path.join(OUT, "qwen_tok.txt"), "w") as f:
    for name in texts:
        t = ref["texts"][name]
        f.write(f"{name}\t{hx(t['text'])}\t{' '.join(map(str, t['ids']))}\n")
with open(os.path.join(OUT, "qwen_gen.txt"), "w") as f:
    for name in ["gen1", "gen2"]:
        g = ref["generations"][name]
        f.write(f"{name}\t{hx(g['prompt'])}\t{' '.join(map(str, g['ids']))}\n")
p = ref["ppl"]
with open(os.path.join(OUT, "qwen_ppl.txt"), "w") as f:
    f.write(f"{hx(p['text'])}\t{p['nll']:.10f}\t{p['ppl']:.10f}\n")
print("OK", OUT)
