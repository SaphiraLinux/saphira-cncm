#!/usr/bin/env python3
"""One-time GPT-2 reference fixture generation (offline verification only).
Uses transformers+torch; output is consumed by C tests. NOT a runtime dep.
"""
import json
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tk = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
m = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()

OUT = "/scratch/cncm/tests/fixtures"
import os
os.makedirs(OUT, exist_ok=True)

texts = {
    "s1": "Hello, my dog is cute",
    "s2": "The quick brown fox jumps over the lazy dog.",
    "uni": "café naïve façade résumé",
    "punct": "She said, \"Hello... world!\" — and left.",
}

ref = {"texts": {}, "logits_files": {}, "generations": {}, "ppl": {}}
with torch.no_grad():
    for name, t in texts.items():
        ids = tk.encode(t)
        ref["texts"][name] = {"text": t, "ids": ids}
        lg = m(torch.tensor([ids])).logits[0].float().numpy()
        fn = f"gpt2_logits_{name}.f32"
        lg.astype(np.float32).tofile(os.path.join(OUT, fn))
        ref["logits_files"][name] = {"file": fn, "T": lg.shape[0], "V": lg.shape[1]}
        print(name, "ids:", ids, "logits:", lg.shape)

    # greedy generations (do_sample=False)
    for name, prompt in [("gen1", "Hello, my dog is cute and"),
                        ("gen2", "In the beginning")]:
        ids = tk.encode(prompt)
        out = m.generate(torch.tensor([ids]), max_new_tokens=20, do_sample=False,
                         pad_token_id=50256)
        gen = out[0].tolist()[len(ids):]
        ref["generations"][name] = {"prompt": prompt, "ids": gen,
                                    "text": tk.decode(gen)}
        print(name, gen, repr(tk.decode(gen)))

    # perplexity: mean NLL over a fixed passage (shifted, no reduction tricks)
    ppl_text = ("The Washington Post reported that the committee would meet on "
                "Tuesday to discuss the new proposal for high-speed rail.")
    ids = tk.encode(ppl_text)
    x = torch.tensor([ids])
    lg = m(x).logits[0]
    shift_lg, shift_lb = lg[:-1], x[0][1:]
    nll = torch.nn.functional.cross_entropy(shift_lg, shift_lb, reduction="mean")
    ref["ppl"] = {"text": ppl_text, "ids": ids, "nll": float(nll),
                  "ppl": float(torch.exp(nll))}
    print("ppl ids:", ids, "nll:", float(nll), "ppl:", float(torch.exp(nll)))

with open(os.path.join(OUT, "gpt2_ref.json"), "w") as f:
    json.dump(ref, f, indent=1)
print("OK", OUT)
