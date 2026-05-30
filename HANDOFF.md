# HANDOFF — resume guide (as of 2026-05-30, commit `a7d9be4`)

Single-page state dump so a fresh session can pick up. Full detail lives in
`devlog.md` (chronological), `FINDINGS.md` (results), `todo.md` (open threads),
`literature/REVIEW.md` (grounding). Site: <https://reservoir.emmaleonhart.com/>
(also `https://emmaleonhart.github.io/reservoiragent/`).

## What the project is
A fixed, randomly-initialized **reservoir** (echo state network) injected into a
**pretrained transformer's** attention so its state persists **across forward passes**
(a real time axis). Only a readout (+ light LoRA) is trained; the reservoir + lower
layers are fixed. Verified **novel** against the searched prior art (REVIEW.md §4).

## What WORKS (demonstrated, tested, committed)
- **H1 non-destruction** — injection generalized to GPT-2 *and* Llama (`_arch.py`);
  zeroed readout → byte-identical logits. Verified on **GPT-2 and Hermes-3-Llama-3.2-3B**
  (4-bit, `scripts/hermes_h1.py`, max|diff|=0, 2.35 GB VRAM).
- **H2 dynamics** — echo-state boundary at **ρ ≈ 1** on synthetic *and* real GPT-2
  activations; input-scaling sweet spot **~0.08–0.24** (real activations over-saturate at
  unit scale). `run.py sweep` / `sweep-real` / `sweep-scaling`.
- **H3 memory** — trained readout recovers input ~18 steps back; stateless baseline = 0.
  `run.py h3`.
- **Cross-pass recall (the core claim)** — additive injection FAILS (reservoir ignored,
  chance); **content-addressable KV-prefix injection gives 100% recall vs 0.17 baseline**
  on GPT-2. `run.py crosspass --mode kv`. (`kv_live.py`.)
- **Trained silence policy** — gate on reservoir state F1≈0.96 vs stateless 0.34 (can't
  be selectively silent). `run.py silence`. (`silence.py`.)
- **Real GPT-2 LoRA fine-tune** on GPU. `run.py finetune`. N-seed selection + a (negative)
  pre-selection-proxy result. `run.py nseed-select`.
- **Always-alive runtime** (`runtime.py`) + **Hermes-format harness core** (`hermes_harness.py`,
  ChatML + tool-call loop). 54 tests pass; CI + Pages green.

## OPEN THREADS (the real next work — in `todo.md`)
1. **Transfer cross-pass recall to Hermes 3B.** 100% on GPT-2; **chance on Hermes** across
   3 attempts (4-bit ×2, bf16+higher-LR). Loss plateaus ~2.8 regardless of quantization.
   A gradient diagnostic proves it's **NOT a bug** (state updates; ∇ flows to W_res+LoRA) —
   a **bootstrapping/scale** difficulty (prefix diluted through 28 layers vs GPT-2's shallow
   stack). **Try next:** a **curriculum** (key in-context first, anneal out), many more
   steps, stronger prefix coupling (inject at multiple layers / larger n_prefix), or
   unfreeze more of the model.
2. **Full Nous Hermes harness fork** (`HERMES_HARNESS_REMAINING`): streaming + exact
   scaffolding; fuse the unprompted/idle pass + silence gate into the loop; regression vs
   vanilla Hermes (a GPU run); multi-tool routing.
3. KV-append into live attention on the Llama path (`GPT2_INTEGRATION_BLOCKER`).

## The conceptual frame (user's, documented in FINDINGS "## D")
Default = **respond** (decayed/empty reservoir ≈ base prior); **silence** attaches to an
**active/novel** reservoir state (the natural handle to fine-tune "still processing" onto);
the **echo state property** empties the reservoir over time → the agent **reverts to
baseline responding**. Teaching a pretrained model this new behavioural axis is aggressive
**brain surgery** — genuinely hard (the same wall the Hermes recall transfer hit).

## How to run / resume
- Tests: `PYTHONPATH=src python -m pytest` (torch/peft tests skip without the `models` extra;
  Hermes runs are local GPU only). Install: `pip install -e ".[dev]"` (+ `.[models]` for torch).
- Hardware used: RTX 4070 (8.6 GB), bitsandbytes 4-bit works on Windows; Hermes 3B cached.
- Workflow: `queue.md` (now-work) → `devlog.md` (done) → `todo.md` (horizons). Crons are
  session-local and were **cancelled at handoff** — recreate them (CLAUDE.md "Autonomous
  productivity loop") if resuming autonomous work.
