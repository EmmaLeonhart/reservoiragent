# Novelty re-check — searched prior art (2024–2026 sweep)

Closes the `todo.md` §D item "Citation-checked novelty follow-up". This is a
**searched-prior-art** statement, not a proof of absolute novelty; the space
moves fast, so re-run before any hard novelty claim in a submitted paper.

## The claim being positioned

A **fixed, randomly-initialized reservoir** (echo state network) injected as a
sidecar into a **pretrained** transformer's mid-layer attention, such that its
state **persists across separate forward passes** (a real time axis), with **only
a readout (+ light LoRA) trained** and the reservoir + lower layers frozen.

## Sweep result (web, 2026-05)

No work was found occupying that specific combination. The nearest neighbours,
and how they differ on the load-bearing axes (fixed-vs-trained; within-sequence
recurrence vs across-separate-passes; sidecar-into-pretrained vs trained-as-such):

1. **Reservoir Transformers** (Shen et al., 2021) — interleaves *frozen random*
   layers into the forward stack. Frozen, yes; but it is a layer in a single
   forward pass, with **no cross-pass / between-call state axis**.
2. **Echo State Transformer** (Berté et al., arXiv 2507.02917, 2025) — parallel
   reservoirs as a fixed-size *working memory* with attention over the reservoir
   units. Within-sequence working memory; the reservoir dynamics are **learned
   (adaptive leak rate)**, and it is trained as an architecture, not injected
   into a pretrained model.
3. **FreezeTST / "Frozen in Time"** (arXiv 2508.18130, 2025) — frozen random
   reservoir blocks interleaved with trainable transformer layers for time
   series. Frozen-random + trainable-query, but again **within the forward
   stack**, no persistence across separate inference calls.
4. **Reservoir Computing as a Language Model** (Köster & Uchida, arXiv
   2507.15779, 2025) — replaces self-attention with fixed reservoirs in an LM.
   Reservoir *as the backbone*, not a sidecar carrying state between passes of a
   pretrained transformer.
5. **Titans: Learning to Memorize at Test Time** (Behrouz et al., arXiv
   2501.00663, 2025) and the associative-memory / MIRAS follow-on — the closest
   *new* idea: a long-term memory module that **persists and updates at test
   time**. Differs on the decisive axis: Titans' memory is **trained / online-
   optimized during the forward pass**, whereas this project's reservoir is
   **fixed and random** (surprise/optimization-free); only a readout is trained.
   Titans is also an architecture trained end-to-end, not a fixed reservoir
   grafted onto a *pretrained* transformer's attention.

Also re-confirmed as non-overlapping: Mamba / S4 / RWKV (trained within-sequence
recurrent state, not a fixed random reservoir across separate passes);
Transformer-XL / Compressive Transformer (segment recurrence within one rollout).

## Verdict

The scoped novelty claim **stands** against the searched 2021–2026 art. The
distinctive quadruple — *fixed random reservoir × pretrained transformer ×
persistence across separate forward passes × only a readout trained* — is not
occupied by any found work. The claim is now positioned explicitly against the
test-time-memorization line (Titans) rather than against 2021 work alone.

## Sources

- Reservoir Transformers — https://openreview.net/pdf?id=5FRJWsiLRmA
- Echo State Transformer — https://arxiv.org/abs/2507.02917
- Frozen in Time (FreezeTST) — https://arxiv.org/abs/2508.18130
- Reservoir Computing as a Language Model — https://arxiv.org/abs/2507.15779
- Titans: Learning to Memorize at Test Time — https://arxiv.org/abs/2501.00663
