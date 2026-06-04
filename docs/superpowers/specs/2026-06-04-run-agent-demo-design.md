# Design: `run_agent.bat` — local "does it actually work?" demo launcher

**Date:** 2026-06-04
**Status:** approved (design), pending spec review

## Problem

The project is feature-complete enough to write up, but there is no one-command way
to *watch the shipped model work*. We want to point a launcher at the most recent top
Hugging Face model, prove the headline result (cross-pass recall) on the downloaded
weights, and then let the user poke at it interactively.

## Scope & positioning

`run_agent.bat` is a **repo-local developer / debugging convenience**, NOT a
distributable. The distributable already exists: `reservoir-agent-installer.exe`
(built by `installer/build_exe.py` via `.github/workflows/build-installer.yml`,
linked from the GitHub Pages site). That `.exe` is responsible for portability
(bootstrap pip install, venv, any-machine Python). `run_agent.bat` is the opposite:
it assumes *this* machine and exists so the author can quickly verify a model works.

Because it is local-only, the `.bat` may hardcode the known-good local interpreter
path rather than trying to be portable.

## What the user picked (brainstorming)

**Guided recall demo, then REPL.** On launch: pick the most recent top model, run a
scripted cross-pass recall demonstration (feed a secret word, wipe the context, ask
for it back — the stateful reservoir recalls it, a state-wiped baseline cannot), then
drop into the stateful REPL.

## Two findings that shaped the design

1. **The recall demo is a reuse, not new ML.** `crosspass.run_cross_pass_kv` already
   contains the exact eval: a `forward_pair(key_word, reset_between)` that runs
   pass 1 `"The secret word is {X}."`, wipes context (and, for the baseline, the
   reservoir state), runs pass 2 `"The secret word was"`, and checks the argmax
   against the key token. The demo factors this out and runs it on the *loaded*
   artifact — no retraining.

2. **A real bug in the existing REPL.** `installer/console.py`'s `step()` generates
   with `lm.model.generate(...)`. That fires the read hook (reservoir *ticks*), but
   the trained **write path** — reservoir → prefix pseudo-tokens — lives only inside
   `forward_logits`, which `generate` does not use. So the REPL accumulates state but
   never feeds it back; the free-form half is effectively vanilla-LoRA output with a
   dangling reservoir. The fix: generate via a greedy loop over `forward_logits`, so
   the prefix is injected every pass and the REPL is genuinely stateful.

## Components

### 1. `run_agent.bat` (repo root)
Thin launcher. Resolves a working Python in order: `%RESERVOIR_PYTHON%`, then the
known-good local path `C:\Users\Immanuelle\AppData\Local\Programs\Python\Python313\python.exe`
(if it exists), then `py -3`, then `python`. `cd /d "%~dp0"`. Runs
`python -m reservoir.installer` with the default (no-menu, demo+REPL) flow. Forwards
any extra args (`%*`) so `run_agent.bat --menu`, `--demo-only`, `--repo-id ...` work.

### 2. `eval_recall(lm, keys)` — in `src/reservoir/crosspass.py`
Factored out of `run_cross_pass_kv` (DRY — the training loop's eval calls it too).
Pure of training; runs only forward passes on an already-built/loaded `lm`. Returns a
list of per-key records:
```
{"word": str, "tok_id": int,
 "stateful_pred": int, "stateful_ok": bool,
 "baseline_pred": int, "baseline_ok": bool}
```
plus a small summary helper (`recall_accuracy(records, which)`). `keys` come from the
existing `_single_token_keys(tokenizer, n_keys)`; `n_keys` is read from the loaded
model's saved config meta when present (the training `result` is saved via
`save_reservoir_model(extra_meta=...)`), default 6.

### 3. Demo + REPL flow — in `installer/console.py` and `installer/menu.py`
- `recall_demo_session(lm, print_fn=print)` — builds the keys, calls `eval_recall`,
  and prints the guided demo: per-key `secret word -> X` with `reservoir (stateful): X ✓`
  vs `stateless baseline: <pred> ✗`, then a one-line accuracy summary
  (stateful vs baseline vs chance `1/n_keys`).
- `generate_stateful(lm, ids, max_new_tokens)` — greedy decode using
  `lm.forward_logits` token-by-token (applies the prefix write-path every step, ticks
  state). `ReservoirConsole.step()` switches to this.
- `console.run(repo_id, *, demo=True, repl=True)` — download + resolve + load, then
  optionally `recall_demo_session`, then optionally `ReservoirConsole(...).repl()`.
- `menu.main`:
  - **Default** (what the `.bat` invokes): auto-pick `default_model()`, no prompt,
    `run(repo_id, demo=True, repl=True)`.
  - `--menu`: show `_format_menu` + `choose_repo(input(...))` as today.
  - `--demo-only`: `run(..., demo=True, repl=False)`.
  - `--no-demo`: `run(..., demo=False, repl=True)`.
  - Existing `--repo-id`, `--no-hf`, `--list` preserved.

## Data flow

```
run_agent.bat
  -> python -m reservoir.installer            (menu.main, default flow)
       -> registry.default_model()            most-recent recommended (HF + bundled)
       -> console.run(repo_id, demo, repl)
            -> download_and_resolve(repo_id)   snapshot_download + resolve_load_dir
            -> load_reservoir_model(dir)       TorchReservoirPrefixInjectedLM
            -> recall_demo_session(lm)         eval_recall -> printed ✓/✗ table
            -> ReservoirConsole(lm).repl()     generate_stateful per turn (state persists)
```

## Error handling
- No models / empty registry: `default_model` returns `None` → print a clear message
  and exit non-zero (don't crash). Bundled fallback means this only happens with
  `--no-hf` against an empty bundled list, which won't occur.
- Missing torch / `models` extra at load: let the existing import error surface with a
  hint to `pip install -e .[models]` (matches current behaviour; the `.exe` handles
  install, the `.bat` assumes a dev env).
- `eval_recall` on a model whose config lacks `n_keys`: default to 6.
- `.bat` finds no usable Python: echo the tried paths and exit 1.

## Testing (pure parts only — matches existing gating)
- `eval_recall` output shape + ✓/✗ correctness against a **fake lm** (a stub exposing
  `tokenizer`, `forward_logits`, `reset_state`, `torch`) — no real model.
- `recall_accuracy` summary math.
- `menu.main` flag routing: default → `run(demo=True, repl=True)`; `--demo-only` →
  `repl=False`; `--no-demo` → `demo=False`; `--menu` calls the chooser. Patch `run`
  with a spy so no download/torch.
- `choose_repo` / `resolve_load_dir` existing tests stay green.
- Torch/network paths (`generate_stateful`, `download_and_resolve`, real REPL) remain
  runtime-gated, not in CI — as they are now.

## Out of scope (YAGNI)
- Portability of the `.bat` (that is the `.exe`'s job).
- Streaming generation, sampling controls, multi-turn tool calls in the REPL.
- Any change to the GitHub Pages site or the installer `.exe` itself.
- Retraining or publishing new models.
