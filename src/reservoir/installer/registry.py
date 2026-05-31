"""The reservoir-agent model registry — what the installer can install.

Every model the project ships is a *reservoir agent*: a pretrained transformer with a
fixed random reservoir brain-surgeried in (attended, cross-pass-stateful — RNN-like), not
a vanilla transformer. Models are produced in **batches of N seeds**; per the design
(``data_lake/transcripts/so-because-reservoir-computer-claude.md``), we **keep all N** —
the best is *recommended*, but the worse seeds are retained as signal for learning which
reservoir properties survive selection. So the registry lists every model and privileges
the recommended one as the default; it never drops the "bad" ones.

The live list is queried from the Hugging Face Hub (by author + tag); a bundled list is
the offline fallback so the installer always has something to show. The selection logic
(sort / default / merge) is pure and unit-tested.
"""
from __future__ import annotations

HF_AUTHOR = "EmmaLeonhart"
HF_TAG = "reservoir-agent"

# Offline fallback — also the seed of the list before many models are published. Keep the
# recommended (privileged-best) flag accurate as batches land.
BUNDLED_MODELS = [
    {
        "repo_id": "EmmaLeonhart/reservoir-agent-gpt2-crosspass",
        "title": "GPT-2 cross-pass reservoir",
        "last_modified": "2026-05-30T00:00:00",
        "recommended": True,
        "note": "Verified 100% cross-context recall (vs 17% chance). Single seed (seed=0).",
    },
]


def sort_by_recency(models: list[dict]) -> list[dict]:
    """Newest first. Models with no ``last_modified`` sort last (never crash)."""
    return sorted(models, key=lambda m: m.get("last_modified") or "", reverse=True)


def default_model(models: list[dict]) -> dict | None:
    """The installer's default pick: the most recent *recommended* (privileged-best)
    model if any are flagged, else simply the most recent. ``None`` for an empty list."""
    if not models:
        return None
    recommended = [m for m in models if m.get("recommended")]
    pool = recommended if recommended else models
    return sort_by_recency(pool)[0]


def merge_models(hf_models: list[dict], bundled: list[dict]) -> list[dict]:
    """Union by ``repo_id``; live HF metadata wins over the bundled entry, but bundled
    fields (title/note/recommended) fill gaps the HF record doesn't carry."""
    by_id: dict[str, dict] = {}
    for m in bundled:
        by_id[m["repo_id"]] = dict(m)
    for m in hf_models:
        merged = {**by_id.get(m["repo_id"], {}), **m}
        by_id[m["repo_id"]] = merged
    return list(by_id.values())


REPO_PREFIX = f"{HF_AUTHOR}/reservoir-agent"


def _is_reservoir_agent(info) -> bool:
    """A repo is one of ours if it follows the ``reservoir-agent-*`` naming convention or
    carries the ``reservoir-agent`` tag (the author also hosts unrelated models)."""
    tags = set(getattr(info, "tags", None) or [])
    return info.id.startswith(REPO_PREFIX) or HF_TAG in tags


def _query_hf() -> list[dict]:
    """Live query of the Hub for the project's reservoir-agent models. Network-gated.

    The author hosts unrelated repos too, so filter by naming convention / tag rather than
    listing everything. ``last_modified`` may be absent in the bulk listing — the bundled
    merge backfills it, and missing timestamps sort last (never crash)."""
    from huggingface_hub import HfApi
    api = HfApi()
    out = []
    for info in api.list_models(author=HF_AUTHOR):
        if not _is_reservoir_agent(info):
            continue
        lm = getattr(info, "last_modified", None)
        out.append({"repo_id": info.id,
                    "last_modified": lm.isoformat() if lm is not None else None})
    return out


def list_models(*, use_hf: bool = True) -> list[dict]:
    """All installable reservoir-agent models, newest first. Falls back to the bundled
    list if the Hub query fails or is disabled — never raises, always returns something."""
    hf_models: list[dict] = []
    if use_hf:
        try:
            hf_models = _query_hf()
        except Exception:
            hf_models = []
    return sort_by_recency(merge_models(hf_models, BUNDLED_MODELS))
