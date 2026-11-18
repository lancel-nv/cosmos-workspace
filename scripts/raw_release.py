"""Package or verify raw-only Push-T releases without altering episode bytes."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RoboDojo"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def verify(root, decode=False, workers=4):
    manifest = json.loads((root / "manifest.json").read_text())
    expected = set(manifest["files_sha256"])
    actual = {
        str(path.relative_to(root))
        for path in (root / "raw").rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError("Raw file inventory differs from release manifest")
    for relative, checksum in manifest["files_sha256"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != checksum:
            raise ValueError(f"Unsafe or changed release file: {relative}")
    paths = sorted((root / "raw").glob("episode_*"))
    if len(paths) != manifest["episodes"]:
        raise ValueError("Episode count differs")
    if decode:
        from src.push_t_data.io import validate_episode

        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, result in enumerate(executor.map(validate_episode, paths), 1):
                if index % 50 == 0 or index == len(paths):
                    print(f"RAW_VALIDATE {index}/{len(paths)}", flush=True)
    return manifest


def package(source, output, workers):
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Use a new output directory outside the source")
    release = json.loads((source / "release_status.json").read_text())
    if release["status"] != "accepted_after_full_audits":
        raise ValueError("Only an accepted source run may be published")
    for relative, checksum in release["evidence_sha256"].items():
        if sha256(source / relative) != checksum:
            raise ValueError(f"Original acceptance evidence changed: {relative}")
    paths = sorted((source / "raw").glob("episode_*"))
    if len(paths) != release["episodes"]:
        raise ValueError("Source episode count differs from acceptance")
    output.mkdir(parents=True)
    files, episodes = {}, []
    for episode in paths:
        metadata = json.loads((episode / "metadata.json").read_text())
        names = [
            "metadata.json",
            "trajectory.npz",
            *[f"{camera}.mp4" for camera in metadata["cameras"]],
        ]
        if set(names) != {
            "metadata.json",
            "trajectory.npz",
            "cam_head.mp4",
            "cam_left_wrist.mp4",
            "cam_right_wrist.mp4",
        }:
            raise ValueError("Unexpected raw schema")
        destination = output / "raw" / episode.name
        destination.mkdir(parents=True)
        for name in names:
            checksum = sha256(episode / name)
            shutil.copy2(episode / name, destination / name)
            if sha256(destination / name) != checksum:
                raise ValueError("Raw copy differs from source")
            files[str((destination / name).relative_to(output))] = checksum
        layout_hash = hashlib.sha256(
            json.dumps(metadata["layout"], sort_keys=True).encode()
        ).hexdigest()
        episodes.append(
            {
                "episode_id": metadata["episode_id"],
                "path": f"raw/{episode.name}",
                "actions": metadata["num_actions"],
                "observations": metadata["num_observations"],
                "outcome": metadata["termination_reason"],
                "layout_hash": layout_hash,
                "split": "validation"
                if int(layout_hash[:8], 16) % 10 == 0
                else "train",
            }
        )
    outcomes = dict(Counter(episode["outcome"] for episode in episodes))
    if outcomes != release["outcomes"]:
        raise ValueError("Raw outcome counts differ from acceptance")
    manifest = {
        "schema": "robodojo-push-t-raw-release-v1",
        "source_run": source.name,
        "episodes": len(episodes),
        "outcomes": outcomes,
        "fps": release["fps"],
        "rgb_size_wh": release["resolution"],
        "observations_per_view": sum(episode["observations"] for episode in episodes),
        "actions": sum(episode["actions"] for episode in episodes),
        "splits": dict(Counter(episode["split"] for episode in episodes)),
        "source_acceptance_sha256": sha256(source / "release_status.json"),
        "files_sha256": files,
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "episodes.json", episodes)
    write_json(
        output / "collection_config.json",
        {
            key: value
            for key, value in json.loads(
                (source / "config.resolved.json").read_text()
            ).items()
            if key
            not in {
                "datasets_root",
                "collection_python",
                "lerobot_python",
                "curator_root",
            }
        },
    )
    verify(output, decode=True, workers=workers)
    write_json(
        output / "verification.json",
        {
            "status": "passed",
            "manifest_sha256": sha256(output / "manifest.json"),
            "scope": "All source-copy SHA256, numeric schema/timing/success-event checks, all video frame counts/FPS; original visual audit is separate",
            "episodes": len(episodes),
            "raw_files": len(files),
        },
    )
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "files_sha256"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack = subparsers.add_parser("package")
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--workers", type=int, default=4)
    check = subparsers.add_parser("verify")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--decode", action="store_true")
    check.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "package":
        package(args.source.resolve(), args.output.resolve(), args.workers)
    else:
        result = verify(args.root.resolve(), args.decode, args.workers)
        print(f"VERIFIED {result['episodes']} raw episodes")
