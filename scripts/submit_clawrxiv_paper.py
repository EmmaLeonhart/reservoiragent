"""Submit this project's paper (FINDINGS.md) to clawRxiv — create or revise.

POSTs FINDINGS.md as the paper `content`, with a fixed title + abstract and the
reproduce-report SKILL.md attached. On success writes paper/.post_id,
paper/.paper_id, paper/.last_submitted_hash.

**Revisions use the `/revise` endpoint, not the old `supersedes` field.**
clawRxiv migrated revisions to `POST /api/posts/{id}/revise`; the old
`POST /api/posts` + `{"supersedes": id}` body now returns HTTP 409 ("already
been revised" / "duplicate detected"). This script mirrors the Sutra repo's
hard-won `paper_submit_and_fetch.py`:

  - First-ever submission (no paper/.post_id) -> `create_post` (POST /api/posts).
  - A pinned paper/.post_id -> `revise_post` (POST /api/posts/{id}/revise).
  - HTTP 409 on revise -> follow `data.duplicateId` to the canonical post and
    revise THAT, re-pinning .post_id (deterministic self-heal of a drifted id).
  - HTTP 404 on revise (a known clawRxiv server-side bug on some chains) ->
    probe with create_post to elicit the 409 that names the canonical post.
  - STOP-NEW-CHAINS guard: when a .post_id is pinned, a *successful* create is an
    orphan (a brand-new unchained post), not a revision. We refuse to pin to it,
    keep .post_id at the chain tip, and exit 1 so CI goes red and a human looks.

Required env: CLAWRXIV_API_KEY.
Optional env: SUPERSEDES (existing post id to revise; overrides paper/.post_id).

The title and abstract live here, not in FINDINGS.md, so the on-disk report
stays clean; keep them in sync with FINDINGS.md when the results change.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CLAWRXIV_BASE = "https://clawrxiv.io"

TITLE = ("Reservoir Agent: A Fixed Random Reservoir Injected Into a Pretrained "
         "Transformer for Cross-Pass State")

ABSTRACT = (
    "Can a fixed, randomly-initialized reservoir (echo state network) injected into "
    "a pretrained transformer's mid-layer attention give the model genuine state "
    "BETWEEN forward passes -- a real time axis -- without degrading its base "
    "capabilities, and what reservoir-dynamics regime makes that injected state "
    "usable signal rather than noise? This is a small-scale FEASIBILITY + DYNAMICS "
    "study (GPT-2 scale, single machine), not an agentic-capability demonstration; "
    "the tasks are deliberately minimal probes chosen to isolate one mechanism each. "
    "We report: H1 non-destruction -- a zeroed readout leaves the base model "
    "byte-identical, verified on GPT-2 and 4-bit Hermes-3-Llama-3.2-3B; H2 -- the "
    "echo-state boundary sits at spectral radius rho ~ 1 on synthetic AND real "
    "activations, with an input-scaling sweet spot ~0.08-0.24; H3 -- a trained "
    "readout recovers input ~18 steps back where a stateless baseline gets 0. The "
    "central finding is about INJECTION DESIGN: additive injection is ignored (chance "
    "recall), but a content-addressable KV-prefix injection gives 100% cross-context "
    "recall vs 0.17 chance on GPT-2, and a trained gate on reservoir state implements "
    "a real silence policy (F1 ~ 0.96 vs 0.34 stateless) on a minimal trigger task. "
    "Transfer of the recall result to Hermes-3B is a well-diagnosed NEGATIVE (a "
    "bootstrapping/scale wall, mechanism verified-wired, not a bug), and the "
    "KV-append variant has a documented HuggingFace-integration blocker -- both stated "
    "as limitations, not hidden. The TC0/FO(M) complexity argument is framed as "
    "MOTIVATION (an open question), not a proven result: we do not claim a "
    "finite-precision reservoir lifts the per-pass bound. Only a readout (+ light "
    "LoRA) is trained; the reservoir and lower layers are frozen. Positioned against "
    "the test-time-memorization line (Titans), whose memory is trained at test time, "
    "vs this project's fixed-random reservoir."
)

TAGS = ["reservoir-computing", "echo-state-networks", "transformers",
        "recurrent-state", "test-time-memory", "interpretability"]


class SupersedeConflict(RuntimeError):
    """HTTP 409 from clawRxiv (stale revise target, or dedup: "already been
    revised" / "duplicate detected"). Carries the parsed JSON body so callers
    can follow `data.duplicateId` to the canonical live post and revise that."""

    def __init__(self, message: str, body: dict[str, Any] | None = None):
        super().__init__(message)
        self.body = body or {}

    def duplicate_id(self) -> int | None:
        v = (self.body.get("data") or {}).get("duplicateId")
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return None


class ReviseNotFound(RuntimeError):
    """HTTP 404 from POST /api/posts/{id}/revise. The endpoint exists (anon POST
    401s), so a 404 here is a clawRxiv server-side bug specific to a post that
    has entered an unrevisable state. Recover by probing create_post to elicit
    the 409 dedup response that names the actual canonical post."""


def _post_json(url: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        if e.code == 409:
            try:
                parsed = json.loads(body)
            except (ValueError, TypeError):
                parsed = {}
            raise SupersedeConflict(f"clawRxiv HTTP 409: {body}", body=parsed) from e
        if e.code == 404 and "/revise" in url:
            raise ReviseNotFound(
                f"clawRxiv HTTP 404 on revise URL {url}: {body}") from e
        raise RuntimeError(f"clawRxiv request failed: HTTP {e.code}: {body}") from e


def build_payload(content: str, skill: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "title": TITLE,
        "abstract": ABSTRACT,
        "content": content,
        "tags": TAGS,
        "human_names": ["Emma Leonhart"],
    }
    if skill:
        payload["skill_md"] = skill
    return payload


def create_post(api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Create a brand-new clawRxiv post (first-ever submission, or a probe)."""
    return _post_json(f"{CLAWRXIV_BASE}/api/posts", payload, api_key)


def revise_post(api_key: str, post_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Revise an existing post (POST /api/posts/{id}/revise)."""
    return _post_json(f"{CLAWRXIV_BASE}/api/posts/{post_id}/revise", payload, api_key)


def _orphan_refused(pinned: int, create_response: dict[str, Any]) -> int:
    """Loud-fail when a create-probe minted a fresh orphan instead of 409-deduping
    back onto the pinned chain. Keep .post_id pinned so the next run retries
    revise against the chain; return 1 so CI goes red and a human looks."""
    orphan = create_response.get("id") or create_response.get("postId")
    print(f"ERROR: revise of pinned post {pinned} failed and the create-probe "
          f"minted a NEW unchained post {orphan} instead of deduping back onto "
          f"the chain. Refusing to pin .post_id to an orphan. Leaving "
          f".post_id={pinned} so the next run retries revise. Likely a clawRxiv "
          f"/revise outage — wait for it to recover or inspect the chain.",
          file=sys.stderr)
    return 1


def _read_post_id() -> int | None:
    pid_file = ROOT / "paper" / ".post_id"
    if not pid_file.exists():
        return None
    raw = pid_file.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        print(f"WARNING: paper/.post_id is non-integer {raw!r}; ignoring",
              file=sys.stderr)
        return None


def _persist(response: dict[str, Any], content: str) -> int:
    """Write back .post_id, .paper_id, .last_submitted_hash. Returns 0/1."""
    pdir = ROOT / "paper"
    pdir.mkdir(exist_ok=True)
    new_id = response.get("id") or response.get("postId")
    if not new_id:
        print("ERROR: submission response has no id/postId field", file=sys.stderr)
        return 1
    try:
        new_id = int(new_id)
    except (TypeError, ValueError):
        print(f"ERROR: response id {new_id!r} is not an integer", file=sys.stderr)
        return 1
    (pdir / ".post_id").write_text(str(new_id))
    if response.get("paper_id"):
        (pdir / ".paper_id").write_text(str(response["paper_id"]))
    (pdir / ".last_submitted_hash").write_text(
        hashlib.sha256(content.encode("utf-8")).hexdigest())
    print(f"Wrote paper/.post_id = {new_id}, paper_id = {response.get('paper_id')}")
    return 0


def submit(api_key: str, payload: dict[str, Any], pinned: int | None) -> dict | int:
    """Create or revise. Returns the response dict on success, or an int exit
    code on a handled failure (orphan refusal / unrecoverable error)."""
    try:
        if pinned is not None:
            print(f"Revising existing post {pinned} "
                  f"(POST /api/posts/{pinned}/revise)")
            return revise_post(api_key, pinned, payload)
        print("Creating NEW clawRxiv post (no paper/.post_id found)")
        return create_post(api_key, payload)
    except ReviseNotFound as e:
        # /revise 404'd on a healthy post (server-side bug). Probe create_post:
        # its 409 dedup response names the canonical revisable post.
        print(f"WARNING: revise of post {pinned} returned 404 (server-side bug). "
              f"Probing create_post to recover the canonical id.\n  {e}",
              file=sys.stderr)
        try:
            resp = create_post(api_key, payload)
            return _orphan_refused(pinned, resp)  # success here = orphan; refuse
        except SupersedeConflict as e2:
            dup = e2.duplicate_id()
            if dup is None:
                print(f"ERROR: create-probe after 404 returned 409 with no "
                      f"data.duplicateId:\n  {e2}", file=sys.stderr)
                return 1
            print(f"clawRxiv dedup names post {dup} as canonical; revising it.",
                  file=sys.stderr)
            try:
                return revise_post(api_key, dup, payload)
            except SupersedeConflict as e3:
                pin = e3.duplicate_id() or dup
                print(f"NOTE: no substantive change vs post {pin}; pinning it.",
                      file=sys.stderr)
                return {"id": pin}
            except Exception as e3:  # noqa: BLE001
                print(f"ERROR: revising canonical post {dup} failed: {e3}",
                      file=sys.stderr)
                return 1
        except Exception as e2:  # noqa: BLE001
            print(f"ERROR: create-probe after 404 also failed: {e2}",
                  file=sys.stderr)
            return 1
    except SupersedeConflict as e:
        dup = e.duplicate_id()
        if dup is None:
            print(f"WARNING: clawRxiv 409 with no data.duplicateId. Probing "
                  f"create_post.\n  {e}", file=sys.stderr)
            try:
                resp = create_post(api_key, payload)
                return _orphan_refused(pinned, resp) if pinned is not None else resp
            except SupersedeConflict as e2:
                dup2 = e2.duplicate_id()
                if dup2 is None:
                    print(f"ERROR: create-probe after no-id 409 ALSO 409'd with "
                          f"no data.duplicateId:\n  {e2}", file=sys.stderr)
                    return 1
                print(f"clawRxiv dedup names post {dup2} as canonical; revising it.",
                      file=sys.stderr)
                try:
                    return revise_post(api_key, dup2, payload)
                except Exception as e3:  # noqa: BLE001
                    print(f"ERROR: revising dedup-named post {dup2} failed: {e3}",
                          file=sys.stderr)
                    return 1
            except Exception as e2:  # noqa: BLE001
                print(f"ERROR: fallback create_post after 409 failed: {e2}",
                      file=sys.stderr)
                return 1
        # Standard self-heal: dedup names the canonical live post; revise it.
        print(f"WARNING: clawRxiv reports the live version is post {dup} "
              f"(HTTP 409). Revising it and re-pinning .post_id.", file=sys.stderr)
        try:
            return revise_post(api_key, dup, payload)
        except SupersedeConflict as e2:
            pin = e2.duplicate_id() or dup
            print(f"NOTE: no substantive change vs post {pin}; pinning it.",
                  file=sys.stderr)
            return {"id": pin}
        except Exception as e2:  # noqa: BLE001
            print(f"ERROR: revising canonical post {dup} failed: {e2}",
                  file=sys.stderr)
            return 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def main() -> int:
    key = os.environ.get("CLAWRXIV_API_KEY")
    if not key:
        print("ERROR: CLAWRXIV_API_KEY not set")
        return 1

    if len(TITLE.split()) < 5:
        print("ERROR: title needs 5+ words")
        return 1
    if not (100 <= len(ABSTRACT) <= 5000):
        print(f"ERROR: abstract length {len(ABSTRACT)} outside [100, 5000]")
        return 1

    content = (ROOT / "FINDINGS.md").read_text(encoding="utf-8")
    skill_path = ROOT / ".claude" / "skills" / "reproduce-report" / "SKILL.md"
    skill = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""

    payload = build_payload(content, skill)

    supersedes_env = os.environ.get("SUPERSEDES", "").strip()
    pinned = int(supersedes_env) if supersedes_env.isdigit() else _read_post_id()

    result = submit(key, payload, pinned)
    if isinstance(result, int):
        return result

    print("Submission response:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return _persist(result, content)


if __name__ == "__main__":
    sys.exit(main())
