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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-dir", required=True, help="saved model directory to upload")
    ap.add_argument("--repo-id", required=True, help="e.g. EmmaLeonhart/reservoir-agent-gpt2-crosspass")
    ap.add_argument("--private", action="store_true", help="create the repo private (default public)")
    ap.add_argument("--dry-run", action="store_true", help="validate locally, do not upload")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.artifact_dir):
        print(f"error: no such directory {args.artifact_dir}", file=sys.stderr)
        return 1
    required = ["config.json", "reservoir_readout.npz", "adapter"]
    missing = [f for f in required if not os.path.exists(os.path.join(args.artifact_dir, f))]
    if missing:
        print(f"error: artifact missing {missing} in {args.artifact_dir}", file=sys.stderr)
        return 1
    if not os.path.exists(os.path.join(args.artifact_dir, "README.md")):
        print("warning: no README.md model card in the artifact dir", file=sys.stderr)

    if args.dry_run:
        print(f"[dry-run] would upload {args.artifact_dir} -> "
              f"hf.co/{args.repo_id} (private={args.private})")
        return 0

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    who = api.whoami()
    print(f"authenticated as {who.get('name')}")
    create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)
    url = api.upload_folder(folder_path=args.artifact_dir, repo_id=args.repo_id,
                            repo_type="model",
                            commit_message="Upload reservoir-agent cross-pass artifact")
    print(f"uploaded -> https://huggingface.co/{args.repo_id}")
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
