"""Print or execute the raw -> LeRobot -> Curator -> single-view Bimanual workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

from raw_release import ROOT, sha256, verify, write_json


def commands(args):
    work = args.work
    robodojo = ROOT / "RoboDojo"
    framework = ROOT / "cosmos-framework"
    staging = work.parent / (work.name + "-staging")
    short = work.parent / (work.name + "-head-short")
    final = work.parent / (work.name + "-head-var593")
    curator_config = work / "curator.yaml"
    return [
        (
            "export",
            robodojo,
            [
                str(args.lerobot_python),
                "-m",
                "src.push_t_data.export_lerobot",
                "--input",
                str(args.raw / "raw"),
                "--output",
                str(work / "lerobot"),
            ],
        ),
        (
            "curate",
            ROOT / "cosmos-curator",
            [
                str(args.curator_python),
                "-m",
                "cosmos_curator.pipelines.pipeline_runtime",
                str(curator_config),
            ],
        ),
        (
            "augment",
            robodojo,
            [
                str(args.curator_python),
                "-m",
                "src.push_t_data.augment_curator",
                "--lance",
                str(work / "curator/lance/clips.lance"),
                "--lerobot",
                str(work / "lerobot"),
                "--output",
                str(work / "curator_complete"),
            ],
        ),
        (
            "staging",
            robodojo,
            [
                str(args.curator_python),
                "scripts/prepare_push_t_framework.py",
                "--run-root",
                str(work),
                "--output",
                str(staging),
            ],
        ),
        (
            "bundle",
            framework,
            [
                str(args.framework_python),
                "tools/prepare_robodojo_bimanual.py",
                "--staging",
                str(staging),
                "--output",
                str(short),
                "--tokenizer",
                str(args.tokenizer),
                "--robot-assets",
                str(args.robot_assets),
            ],
        ),
        (
            "audit-short",
            framework,
            [
                str(args.framework_python),
                "tools/audit_robodojo_bimanual.py",
                "--root",
                str(short),
            ],
        ),
        (
            "windows",
            framework,
            [
                str(args.framework_python),
                "tools/rewindow_robodojo_bimanual.py",
                "--source",
                str(short),
                "--output",
                str(final),
                "--max-actions",
                "592",
            ],
        ),
        (
            "audit-final",
            framework,
            [
                str(args.framework_python),
                "tools/audit_robodojo_bimanual.py",
                "--root",
                str(final),
            ],
        ),
    ]


def processing_receipt(args, manifest):
    work = args.work
    source_meta = {
        int(path.parent.name.removeprefix("episode_")): json.loads(path.read_text())
        for path in (args.raw / "raw").glob("episode_*/metadata.json")
    }
    rows = [
        json.loads(line)
        for line in (work / "curator_complete/manifest.jsonl").read_text().splitlines()
    ]
    provenance = [
        json.loads(line)
        for line in (work / "lerobot/meta/provenance.jsonl").read_text().splitlines()
    ]
    if len(provenance) != manifest["episodes"] or len(rows) != 3 * manifest["episodes"]:
        raise ValueError("Episode or camera count changed during processing")
    if {row["episode_id"] for row in provenance} != set(source_meta):
        raise ValueError("Export source IDs differ")
    for row in provenance:
        original = source_meta[row["episode_id"]]
        if any(row[key] != value for key, value in original.items()):
            raise ValueError("Raw provenance changed during processing")
        if (
            Path(row["raw_path"]).resolve()
            != args.raw / "raw" / f"episode_{row['episode_id']:06d}"
        ):
            raise ValueError("Unexpected raw source path")
    evidence = {
        str(path.relative_to(work)): sha256(path)
        for folder in ("lerobot", "curator", "curator_complete")
        for path in sorted((work / folder).rglob("*"))
        if path.is_file()
    }
    write_json(
        work / "processing_receipt.json",
        {
            "status": "raw_to_curator_verified",
            "raw_manifest_sha256": sha256(args.raw / "manifest.json"),
            "episodes": manifest["episodes"],
            "evidence_sha256": evidence,
            "scope": "Verified input bytes and export provenance/counts; staging and native audits check temporal/action contracts next",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--lerobot-python", type=Path, required=True)
    parser.add_argument("--curator-python", type=Path, required=True)
    parser.add_argument("--framework-python", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--robot-assets", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            expanded = value.expanduser()
            setattr(
                args,
                name,
                expanded.absolute() if name.endswith("_python") else expanded.resolve(),
            )
    plan = commands(args)
    for stage, cwd, command in plan:
        print(
            f"{stage}: (cd {shlex.quote(str(cwd))} && {shlex.join(command)})",
            flush=True,
        )
    if not args.execute:
        return
    if args.work.exists() or args.work.is_relative_to(args.raw):
        raise ValueError("Use a new work directory outside the downloaded raw release")
    for suffix in ("-staging", "-head-short", "-head-var593"):
        if args.work.with_name(args.work.name + suffix).exists():
            raise ValueError("Derived output already exists; use a fresh work name")
    manifest = verify(args.raw)
    args.work.mkdir(parents=True)
    config = {
        "schema_version": 1,
        "kind": "robot-action-split",
        "input": {
            "uris": [str(args.work / "lerobot")],
            "source_dataset": "robodojo_push_t",
        },
        "split": {
            "min_duration_s": 0.01,
            "max_duration_s": None,
            "skip_labels": [],
            "skip_label_prefixes": [],
            "skip_label_substrings": [],
            "max_keep_per_description": max(1000, manifest["episodes"]),
        },
        "output": {
            "media_root": str(args.work / "curator"),
            "action_format": "pickle",
            "video_bitrate": "8M",
        },
        "execution": {"discovery_workers": 2, "storage_profile": "default"},
    }
    write_json(args.work / "curator.yaml", config)
    environment = {
        **os.environ,
        "COSMOS_INTERNAL": "0",
        "COSMOS_TRAINING": "1",
        "PYTHONNOUSERSITE": "1",
    }
    for stage, cwd, command in plan:
        environment["PYTHONPATH"] = str(cwd)
        log = args.work / f"{stage}.log"
        with log.open("x") as stream:
            subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if stage == "augment":
            processing_receipt(args, manifest)
        print(f"COMPLETE {stage}: {log}", flush=True)


if __name__ == "__main__":
    main()
