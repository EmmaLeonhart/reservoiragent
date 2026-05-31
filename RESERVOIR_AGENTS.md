# Reservoir Agents — a new model type (foundational doc)

This document records *what a reservoir agent is* and *how we must build and preserve
them*. It is foundational: the rest of the project (training, publishing, the installer)
serves it. Grounding: `data_lake/transcripts/so-because-reservoir-computer-claude.md`
(esp. lines 529–549) and `reservoir-state-injection-in-transformer-architecture-claude.md`.

## A reservoir agent is not a transformer

We did not modify a transformer. We made a **new kind of AI model**. Starting from a
pretrained transformer LLM, a fixed randomly-initialized **reservoir** is brain-surgeried
in — injected as content-addressable, attended state (`kv_live.py`: reservoir-derived
prefix tokens the model attends to at every layer) whose state is **carried across forward
passes**. The result is **fundamentally different in nature**: it behaves more like an
**RNN than a transformer**, it has a genuine **time axis** (state causally downstream of
every past pass), and the reservoir is now a *fundamental part of the model*, not a
detachable add-on. It inherits the transformer's pretrained world knowledge + cheap
fine-tuning AND reservoir computing's rich nonlinear temporal dynamics, while shedding the
worst of each (transformers' statelessness; reservoirs' train-from-scratch cost).

The published GPT-2 cross-pass models are, to our knowledge, **the first reservoir agents
to exist.** "Reservoir agent" is the **canonical name of the model type** — every model we
ship is tagged `reservoir-agent` on the Hub.

## Why we build in batches — and why we keep the bad ones

Reservoir computing's known flaw: performance is **stochastic in the random reservoir**.
The same architecture with two different seeds can give wildly different performance, and
there is no gradient pulling a bad seed toward a good solution (the recurrent weights are
fixed). This is why reservoir computing stayed niche.

The fix (the project's core method): **reservoir selection via fine-tuning.** Generate a
**population of N reservoirs** (different seeds), brain-surgery each into its own copy of
the base model, **fine-tune all of them**, and keep the one that empirically performs best.
This sidesteps the theoretical problem — we let fine-tuning tell us which reservoir is
good, rather than needing to know in advance.

**We preserve the entire population, not just the winner.** The suboptimal models are
**signal**: the only way to discover *what makes a reservoir good* (which spectral /
connectivity / dynamics properties survive selection) is to keep the ones that lost and
compare. Over many batches this builds empirical knowledge that should eventually let us
**design** good reservoirs instead of sampling randomly. Throwing away the bad seeds throws
away exactly the data that answers the central open question. (Early signal already: the
seed-0 GPT-2 model needed ~2× the training steps of its baseline to converge — convergence
difficulty is itself a per-seed signal worth recording.)

### Preservation mandate (non-negotiable)
- **Every model in a batch is saved** (`persist.save_reservoir_model`) — good and bad.
- Each carries its **benchmark score + reservoir properties** (seed, spectral radius,
  participation ratio, memory-capacity proxy, convergence steps/loss) as analysis signal.
- The **best is privileged** (`recommended`) and is the installer's default; the rest stay
  listed and installable. A **batch manifest** links the population to its privileged best.
- Versioning privileges the best while keeping the population (e.g. the recommended model
  is the headline version; siblings are retained as the batch).
- Everything is tagged `reservoir-agent` so the registry / installer auto-discover it.

## Scope reality (kept accurate)
- The full per-seed *fine-tune-the-whole-LLM* batch selection is the method above; the
  current published GPT-2 model is a **single seed (seed=0)**, not yet a batch-selected
  winner. Building the batch pipeline + running batches of increasing size is the active
  work (`queue.md`). Named plainly so we don't overclaim.
