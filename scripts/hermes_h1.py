"""B: load the smallest Hermes (4-bit) + verify H1 non-destruction. Local-only.

Loads NousResearch/Hermes-3-Llama-3.2-3B in 4-bit, injects the reservoir at a mid
layer, and checks H1 the memory-frugal way (one model copy): the injected forward with
the readout zeroed must equal the same model with the injection hook removed. Reports
VRAM and writes results/hermes_h1.json.

    python scripts/hermes_h1.py
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from reservoir.inject import ReservoirInjectedLM  # noqa: E402

MODEL = "NousResearch/Hermes-3-Llama-3.2-3B"
PROMPT = ("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
          "<|im_start|>assistant\n")


def main():
    print(f"loading {MODEL} in 4-bit…", flush=True)
    lm = ReservoirInjectedLM(MODEL, load_in_4bit=True, n_reservoir=512, seed=0)
    print(f"loaded: {len(lm.blocks)} layers, d_model={lm.d_model}, "
          f"injection layer={lm.layer}", flush=True)

    lm.reset_reservoir()
    inj = lm.logits(PROMPT).detach().float().cpu().numpy()   # W_out = 0
    lm.remove_hook()
    base = lm.logits(PROMPT).detach().float().cpu().numpy()  # no injection
    max_diff = float(np.max(np.abs(inj - base)))
    h1 = bool(np.allclose(inj, base, atol=1e-4))
    vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"H1 (zeroed readout == base): {h1}   max|diff|={max_diff:.2e}   "
          f"peak VRAM={vram:.2f} GB", flush=True)
    os.makedirs("results", exist_ok=True)
    with open("results/hermes_h1.json", "w") as f:
        json.dump({"model": MODEL, "n_layers": len(lm.blocks), "d_model": lm.d_model,
                   "injection_layer": lm.layer, "h1_pass": h1,
                   "max_abs_diff": max_diff, "peak_vram_gb": round(vram, 2),
                   "quantization": "nf4 4-bit"}, f, indent=2)
    print("wrote results/hermes_h1.json", flush=True)


if __name__ == "__main__":
    main()
