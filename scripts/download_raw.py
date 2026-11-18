"""Download the pinned raw release; never silently use a moving main revision."""

import argparse
import json
from pathlib import Path

from raw_release import ROOT, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = json.loads((ROOT / "data-release.json").read_text())
    if not release.get("revision"):
        raise RuntimeError(
            "The raw upload has not been published/pinned yet; confirm HF visibility first"
        )
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=release["repo_id"],
        repo_type="dataset",
        revision=release["revision"],
        local_dir=args.output,
    )
    verify(args.output.resolve())
    print(f"Verified {release['episodes']} episodes at {release['revision']}")


if __name__ == "__main__":
    main()
