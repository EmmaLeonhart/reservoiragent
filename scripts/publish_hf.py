"""Publish a saved reservoir-model artifact to the Hugging Face Hub.

Usage:
    python scripts/publish_hf.py --artifact-dir artifacts/gpt2-crosspass-reservoir \
        --repo-id EmmaLeonhart/reservoir-agent-gpt2-crosspass [--private]

Requires huggingface_hub and a token with write access (huggingface-cli login, or the
HF_TOKEN env var). The artifact directory must contain config.json + reservoir_readout.npz
+ adapter/ (produced by `scripts/run.py crosspass --save <dir>`), and ideally a README.md
model card.
"""
from __future__ import annotations

import argparse
import os
import sys


def _validate_single(d):
    required = ["config.json", "reservoir_readout.npz", "adapter"]
    return [f for f in required if not os.path.exists(os.path.join(d, f))]


def _prepare_batch_card(batch_dir, repo_id):
    """Read the batch manifest, write a generated README.md card into the batch dir, and
    return the validated upload dir. Raises on a malformed batch dir."""
    import json
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from reservoir.batch import build_batch_card

    man_path = os.path.join(batch_dir, "batch_manifest.json")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"no batch_manifest.json in {batch_dir}")
    with open(man_path, encoding="utf-8") as f:
        manifest = json.load(f)
    # every seed dir in the manifest must be a complete artifact
    for p in manifest["population"]:
        seed_dir = os.path.join(batch_dir, f"seed_{p['seed']}")
        miss = _validate_single(seed_dir)
        if miss:
            raise FileNotFoundError(f"seed_{p['seed']} missing {miss}")
    card = build_batch_card(manifest, repo_id=repo_id)
    with open(os.path.join(batch_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    return batch_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--artifact-dir", help="a single saved model directory to upload")
    g.add_argument("--batch-dir", help="a batch directory (seed_*/ + batch_manifest.json) "
                                       "to upload as one population repo")
    ap.add_argument("--repo-id", required=True, help="e.g. EmmaLeonhart/reservoir-agent-gpt2-crosspass")
    ap.add_argument("--private", action="store_true", help="create the repo private (default public)")
    ap.add_argument("--dry-run", action="store_true", help="validate locally, do not upload")
    args = ap.parse_args(argv)

    if args.batch_dir:
        if not os.path.isdir(args.batch_dir):
            print(f"error: no such directory {args.batch_dir}", file=sys.stderr)
            return 1
        try:
            upload_dir = _prepare_batch_card(args.batch_dir, args.repo_id)
        except (FileNotFoundError, KeyError) as e:
            print(f"error: malformed batch dir: {e}", file=sys.stderr)
            return 1
        commit_msg = "Upload reservoir-agent batch population"
    else:
        upload_dir = args.artifact_dir
        if not os.path.isdir(upload_dir):
            print(f"error: no such directory {upload_dir}", file=sys.stderr)
            return 1
        missing = _validate_single(upload_dir)
        if missing:
            print(f"error: artifact missing {missing} in {upload_dir}", file=sys.stderr)
            return 1
        if not os.path.exists(os.path.join(upload_dir, "README.md")):
            print("warning: no README.md model card in the artifact dir", file=sys.stderr)
        commit_msg = "Upload reservoir-agent cross-pass artifact"

    if args.dry_run:
        print(f"[dry-run] would upload {upload_dir} -> "
              f"hf.co/{args.repo_id} (private={args.private})")
        return 0

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    who = api.whoami()
    print(f"authenticated as {who.get('name')}")
    create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)
    url = api.upload_folder(folder_path=upload_dir, repo_id=args.repo_id,
                            repo_type="model", commit_message=commit_msg)
    print(f"uploaded -> https://huggingface.co/{args.repo_id}")
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
