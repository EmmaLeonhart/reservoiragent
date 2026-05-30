# reservoiragent — Devlog

**This file is where "done" lives.** `queue.md` is delete-only: when a queue
item is finished, the item is **deleted from `queue.md`** and a dated entry
is **appended here**, in the same commit as the work, then pushed. Never
tick a box in place — a checked box left in `queue.md` is the failure mode
this file exists to prevent.

Also record releases (tag + a one-line note), notable milestones, and
anything else worth a chronological trail. Newest entries at the bottom.

This is the **same convention as the cleanvibe repo's own `devlog.md`** —
every cleanvibe-scaffolded project gets one for the same reason.

See `CLAUDE.md` § "Workflow Rules" and `queue.md`'s preamble.

---

## 2026-05-29 — Project scaffolded

Scaffolded with `cleanvibe new` (cleanvibe v1.13.0). Future entries
land here as queue items get deleted.

## 2026-05-29 — Bootstrap 1: autonomous loop started

Started the three session-local crons (`durable:false`): work-loop `3 * * * *`
(de9f5f85), auto-flush `15 * * * *` (42d311a4), status-report `42 * * * *`
(ab624872). Mirrored the bootstrap items into the task tool.

## 2026-05-29 — Bootstrap 2: data_lake triage + originating-context preservation

The project's originating context was three Claude chat HTML exports + a
user-authored `reservoir_agent_plan.md`. Preserved the durable, publishable parts
and protected privacy:

- `data_lake/extract_chat_context.py` distils each export into clean Markdown
  **transcripts** (`data_lake/transcripts/`, 24/6/32 turns) and recovers the
  **architecture diagrams** as SVG (`data_lake/diagrams/`: per-pass injection,
  always-alive runtime, joint attention layer + a transformer baseline) plus
  WEBP raster previews.
- The raw `*.html` exports + `*_files/` chrome are **gitignored** (kept local):
  they embed a private "Recents" sidebar of unrelated conversations and ~10 MB
  of web chrome each. Transcripts verified to contain zero sidebar leakage.
- Recovered SVGs use Claude's dark-theme CSS vars, so they render black-boxed
  standalone (labels/structure intact); they'll be re-themed for the `docs/`
  report. `data_lake/README.md` documents provenance.
