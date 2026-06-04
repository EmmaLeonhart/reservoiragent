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


def recall_demo_session(lm, keys, *, print_fn=print):
    """Print the guided cross-pass recall demonstration for a loaded reservoir agent.

    Runs the (training-free) recall eval and prints, per key, the stateful prediction
    (reservoir carried across passes) vs the baseline (state wiped) with a hit/miss mark,
    then a recall-accuracy summary. Returns the eval records."""
    from reservoir.crosspass import eval_recall, recall_accuracy

    records = eval_recall(lm, keys)
    id2word = {tok_id: word for word, tok_id in keys}
    print_fn("")
    print_fn("Cross-pass recall demo — does the reservoir carry a wiped secret word "
             "across forward passes?")
    print_fn("  pass 1: 'The secret word is X.'  ->  context wiped  ->  "
             "pass 2: 'The secret word was ___'")
    print_fn("")
    for r in records:
        sp = id2word.get(r["stateful_pred"], f"<{r['stateful_pred']}>")
        bp = id2word.get(r["baseline_pred"], f"<{r['baseline_pred']}>")
        sm = "✓" if r["stateful_ok"] else "✗"
        bm = "✓" if r["baseline_ok"] else "✗"
        print_fn(f"  secret word -> {r['word']:<8}  reservoir (stateful): {sp:<8} {sm}"
                 f"   stateless baseline: {bp:<8} {bm}")
    n = len(records)
    print_fn("")
    print_fn(f"  recall accuracy   stateful: {recall_accuracy(records, 'stateful'):.0%}"
             f"   baseline: {recall_accuracy(records, 'baseline'):.0%}"
             f"   chance: {1.0 / max(n, 1):.0%}")
    print_fn("")
    return records


def generate_stateful(lm, input_ids, attention_mask, max_new_tokens):
    """Greedy-decode from a kv-prefix reservoir model using ``forward_logits`` (which
    applies the trained reservoir->prefix write-path and ticks the state every pass).
    Returns the list of generated token ids; stops early at eos."""
    torch = lm.torch
    ids, mask = input_ids, attention_mask
    out = []
    for _ in range(max_new_tokens):
        logits = lm.forward_logits(ids, mask)
        next_id = int(logits[0, -1].argmax().item())
        out.append(next_id)
        if next_id == lm.tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=lm.device)], dim=1)
        mask = torch.cat(
            [mask, torch.ones((1, 1), dtype=mask.dtype, device=lm.device)], dim=1)
    return out


class ReservoirConsole:
    """A stateful REPL over a loaded reservoir agent. State persists across turns."""

    def __init__(self, lm, *, max_new_tokens: int = 40):
        self.lm = lm
        self.max_new_tokens = max_new_tokens
        self.lm.reset_state()                 # fresh state at session start only

    def step(self, text: str) -> str:
        """One turn: greedy-decode the model's continuation via the reservoir write-path.
        Reservoir state is NOT reset — it carries into the next call."""
        tok = self.lm.tokenizer
        enc = tok(text, return_tensors="pt").to(self.lm.device)
        ids = generate_stateful(self.lm, enc["input_ids"], enc["attention_mask"],
                                self.max_new_tokens)
        return tok.decode(ids, skip_special_tokens=True).strip()

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
