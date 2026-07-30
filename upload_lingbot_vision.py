#!/usr/bin/env python
"""Upload converted LingBot-Vision payloads to the staging repos.

Requires all four variants to be present and valid: every payload is revalidated against
its manifest (bytes, identity, config semantics, parity acceptance, provenance) via
``lingbot_common.validate_payload`` before the first network write. Produce the payloads
with ``convert_lingbot_vision.py`` first; check the uploaded repos afterward with
``verify_lingbot_vision.py``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from lingbot_common import PRETRAINED_TAG, SPECS, STAGING_OWNER, save_dir_for


def push_variant(variant: str, save_dir: Path) -> str:
    """Upload a validated save directory to its staging repo.

    Args:
      variant (str): One of small, base, large, giant.
      save_dir (Path): Directory holding the validated payload and manifest.

    Returns:
      str: The staging repo id.
    """
    repo_id = f"{STAGING_OWNER}/{SPECS[variant]['arch']}.{PRETRAINED_TAG}"
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=str(save_dir), commit_message="Add converted LingBot-Vision weights")
    print(f"[{variant}] pushed to https://huggingface.co/{repo_id}")
    return repo_id


def main() -> None:
    """Validate all four payloads, then upload them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "converted")
    args = parser.parse_args()

    from lingbot_common import validate_payload

    save_dirs = {}
    for variant in SPECS:
        save_dir = save_dir_for(variant, args.output_dir)
        validate_payload(variant, save_dir)
        save_dirs[variant] = save_dir
        print(f"[{variant}] payload revalidated: {save_dir}")
    for variant, save_dir in save_dirs.items():
        push_variant(variant, save_dir)


if __name__ == "__main__":
    main()
