"""Interactive console for a downloaded reservoir agent.

Loads a published reservoir-agent model and runs a REPL where reservoir state **persists
across turns** — each line you type is a forward pass that ticks the reservoir, and the
carried state influences later passes. That cross-pass statefulness is the point of the
model type, so the loop deliberately does NOT reset state between turns.

``resolve_load_dir`` (pure, unit-tested) picks the right directory to load: a single-model
repo loads itself; a batch population repo loads its recommended seed subdir. The model
loading + generation need the ``models`` extra and are exercised at runtime.
"""
from __future__ import annotations

import json
import os


def resolve_load_dir(path: str) -> str:
    """Given a downloaded repo dir, return the directory to hand to ``load_reservoir_model``.

    - A **single model** repo has ``config.json`` at the root -> load the root.
    - A **batch** repo has ``batch_manifest.json`` -> load the recommended ``seed_<best>``.
    Raises ``FileNotFoundError`` if neither is present or the recommended seed is missing.
    """
    manifest = os.path.join(path, "batch_manifest.json")
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            best = json.load(f)["best"]["seed"]
        seed_dir = os.path.join(path, f"seed_{best}")
        if not os.path.exists(os.path.join(seed_dir, "config.json")):
            raise FileNotFoundError(
                f"batch recommends seed_{best} but {seed_dir}/config.json is missing")
        return seed_dir
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    raise FileNotFoundError(f"{path} is neither a model nor a batch (no config.json / "
                            f"batch_manifest.json)")


def download_and_resolve(repo_id: str) -> str:
    """Download a model/batch repo from the Hub and resolve the dir to load. Network-gated."""
    from huggingface_hub import snapshot_download
    return resolve_load_dir(snapshot_download(repo_id))


class ReservoirConsole:
    """A stateful REPL over a loaded reservoir agent. State persists across turns."""

    def __init__(self, lm, *, max_new_tokens: int = 40):
        self.lm = lm
        self.max_new_tokens = max_new_tokens
        self.lm.reset_state()                 # fresh state at session start only

    def step(self, text: str) -> str:
        """One pass: feed text (reservoir ticks), return the model's greedy continuation.
        State is NOT reset — it carries into the next call."""
        torch = self.lm.torch
        tok = self.lm.tokenizer
        ids = tok(text, return_tensors="pt").to(self.lm.device)
        # a forward pass through the reservoir-injected model ticks the reservoir state
        with torch.no_grad():
            out = self.lm.model.generate(
                **ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id)
        new = out[0][ids["input_ids"].shape[1]:]
        return tok.decode(new, skip_special_tokens=True).strip()

    def repl(self, input_fn=input, print_fn=print):
        print_fn("reservoir-agent console — reservoir state persists across turns. "
                 "Ctrl-C or empty line to exit.")
        while True:
            try:
                line = input_fn("you> ")
            except (EOFError, KeyboardInterrupt):
                break
            if not line.strip():
                break
            print_fn("agent> " + self.step(line))


def run(repo_id: str, *, max_new_tokens: int = 40):
    """Download + load + REPL. Network + torch gated."""
    from reservoir.persist import load_reservoir_model
    load_dir = download_and_resolve(repo_id)
    lm = load_reservoir_model(load_dir)
    ReservoirConsole(lm, max_new_tokens=max_new_tokens).repl()
