# data_lake — supplied material & preserved originating context

This project began from three Claude conversations + a planning doc the user
distilled from them. Those conversations are the **originating context** for the
Reservoir Agent idea and are preserved here.

## What's committed (publishable)

- **`reservoir_agent_plan.md`** — the user-authored research & engineering plan.
  The clean, canonical spec of the architecture, training pipeline, benchmarks,
  open questions, and build order. Start here.
- **`start of the convo.md`** — the user's originating prompt: the pasted message
  that opened the first chat and kicked off the whole Reservoir Agent idea.
- **`Screenshot 2026-05-29 *.png`** — user-supplied clean, light-themed screenshots
  of the two canonical diagrams (the per-forward-pass architecture + ensemble
  training, and the always-alive runtime). These are the diagrams the user wants
  shown; published copies live at `docs/diagram-architecture.png` and
  `docs/diagram-runtime.png`, and the social-preview card `docs/og-preview.png` is
  built from the architecture one.
- **`transcripts/`** — clean Markdown transcripts of the three originating Claude
  chats (User/Claude turns only). Extracted from the raw HTML exports below.
  - `reservoir-state-injection-in-transformer-architecture-claude.md` (longest; the core design discussion)
  - `so-because-reservoir-computer-claude.md`
  - `transformer-diagram-misconceptions-claude.md`
- **`diagrams/`** — the architecture diagrams recovered from the chats:
  - `*.svg` — vector diagrams (per-pass injection, always-alive runtime, joint
    attention layer with reservoir nodes, a standard-transformer baseline).
    **Note:** these SVGs were authored for Claude's dark UI and reference CSS
    custom properties for their fills, so they render with black box-fills when
    opened standalone. All text labels and structure are intact; they will be
    re-themed (or redrawn cleanly) for the published `docs/` report.
  - `*.webp` — rendered raster previews of the diagram artifacts (as-shown).
- **`extract_chat_context.py`** — the (re-runnable) extractor that produced
  `transcripts/` and `diagrams/` from the raw exports. Run:
  `python data_lake/extract_chat_context.py`.

## What's kept local but NOT committed (see `.gitignore`)

- **`*.html`** (the raw Claude chat exports) and their **`*_files/`** asset
  folders. Two reasons they are gitignored rather than published:
  1. **Privacy** — each export embeds a "Recents" sidebar listing many of the
     user's *unrelated* private conversation titles. The distilled `transcripts/`
     exclude that sidebar entirely.
  2. **Weight** — each `_files/` folder is ~10 MB of web chrome (analytics,
     tracking pixels, a 7.6 MB JS bundle) with no project value.

  They remain on disk so the originals can still be browsed locally and re-mined
  (e.g. if more diagrams or the initial pasted attachment need recovering).
