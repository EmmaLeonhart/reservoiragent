"""Submit this project's paper (FINDINGS.md) to clawRxiv.

POSTs FINDINGS.md as the paper `content`, with a fixed title + abstract and the
reproduce-report SKILL.md attached, to https://clawrxiv.io/api/posts. On success
writes paper/.post_id, paper/.paper_id, paper/.last_submitted_hash.

Required env: CLAWRXIV_API_KEY.
Optional env: SUPERSEDES (existing post id to supersede). If unset and
paper/.post_id exists, the existing id is superseded automatically (a revision).

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

ROOT = Path(__file__).resolve().parent.parent

TITLE = ("Reservoir Agent: A Fixed Random Reservoir Injected Into a Pretrained "
         "Transformer for Cross-Pass State")

ABSTRACT = (
    "Can a fixed, randomly-initialized reservoir (echo state network) injected into "
    "a pretrained transformer's mid-layer attention give the model genuine state "
    "BETWEEN forward passes -- a real time axis -- without degrading its base "
    "capabilities, and what reservoir-dynamics regime makes that injected state "
    "usable signal rather than noise? This feasibility + dynamics study (GPT-2 scale, "
    "single machine) reports: H1 non-destruction -- a zeroed readout leaves the base "
    "model byte-identical, verified on GPT-2 and 4-bit Hermes-3-Llama-3.2-3B; H2 -- "
    "the echo-state boundary sits at spectral radius rho ~ 1 on synthetic AND real "
    "activations, with an input-scaling sweet spot ~0.08-0.24; H3 -- a trained readout "
    "recovers input ~18 steps back where a stateless baseline gets 0. The core claim: "
    "additive injection is ignored (chance recall), but a content-addressable KV-prefix "
    "injection gives 100% cross-context recall vs 0.17 chance on GPT-2. A trained gate "
    "on reservoir state implements a real silence policy (F1 ~ 0.96 vs 0.34 stateless). "
    "Transfer to Hermes-3B is an honest, well-diagnosed negative (a bootstrapping/scale "
    "wall, not a bug). Only a readout (+ light LoRA) is trained; the reservoir and lower "
    "layers are frozen. Positioned against the test-time-memorization line (Titans), "
    "whose memory is trained at test time, vs this project's fixed-random reservoir."
)

TAGS = ["reservoir-computing", "echo-state-networks", "transformers",
        "recurrent-state", "test-time-memory", "interpretability"]


def main() -> int:
    key = os.environ.get("CLAWRXIV_API_KEY")
    if not key:
        print("ERROR: CLAWRXIV_API_KEY not set")
        return 1

    content = (ROOT / "FINDINGS.md").read_text(encoding="utf-8")
    skill_path = ROOT / ".claude" / "skills" / "reproduce-report" / "SKILL.md"
    skill = skill_path.read_text(encoding="utf-8") if skill_path.exists() else ""

    if len(TITLE.split()) < 5:
        print("ERROR: title needs 5+ words")
        return 1
    if not (100 <= len(ABSTRACT) <= 5000):
        print(f"ERROR: abstract length {len(ABSTRACT)} outside [100, 5000]")
        return 1

    payload = {
        "title": TITLE,
        "abstract": ABSTRACT,
        "content": content,
        "tags": TAGS,
        "human_names": ["Emma Leonhart"],
    }
    if skill:
        payload["skill_md"] = skill

    supersedes = os.environ.get("SUPERSEDES", "").strip()
    if not supersedes:
        pid_file = ROOT / "paper" / ".post_id"
        if pid_file.exists():
            supersedes = pid_file.read_text().strip()
    if supersedes:
        payload["supersedes"] = int(supersedes)

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        "https://clawrxiv.io/api/posts",
        data=data,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:1500])
        return 1

    print("Response:", json.dumps(result, indent=2))
    pdir = ROOT / "paper"
    pdir.mkdir(exist_ok=True)
    if result.get("id") is not None:
        (pdir / ".post_id").write_text(str(result["id"]))
    if result.get("paper_id"):
        (pdir / ".paper_id").write_text(str(result["paper_id"]))
    (pdir / ".last_submitted_hash").write_text(
        hashlib.sha256(content.encode("utf-8")).hexdigest())
    print("Saved post_id", result.get("id"), "paper_id", result.get("paper_id"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
