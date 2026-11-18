"""Build variable-length, stride-one windows from an accepted native bundle."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from cosmos_framework.scripts.validate_bimanual_data import validate_bundle


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def windows_for(actions, maximum):
    if maximum < 16 or maximum % 16 or actions < 16:
        raise ValueError("Need a positive multiple-of-16 action cap and at least 16 actions")
    length = min(maximum, actions // 16 * 16)
    return [(start, start + length) for start in range(actions - length + 1)]


def rewindow(source, output, maximum):
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Use a new output directory outside the accepted input")
    readiness = json.loads((source / "readiness.json").read_text())
    if readiness["status"] != "native_bimanual_data_preflight_passed" or not readiness["data_ready_for_sft"]:
        raise ValueError("Source bundle has not passed native validation")
    if digest(source / "export_receipt.json") != readiness["export_receipt_sha256"]:
        raise ValueError("Source export receipt changed")
    output.mkdir(parents=True)
    summaries = {}
    for split in ("train", "validation"):
        original, bundle = source / split, output / split
        receipt = json.loads((original / "receipts/dataset-receipt.json").read_text())
        for relative, expected in receipt["files_sha256"].items():
            if digest(original / relative) != expected:
                raise ValueError(f"Source bytes changed: {relative}")
        bundle.mkdir()
        for directory in ("assets", "resources", "receipts"):
            shutil.copytree(original / directory, bundle / directory)
        templates = {}
        for record in json.loads((original / "samples.json").read_text()):
            templates.setdefault(record["request"]["episode_index"], record)
        records = []
        for episode, template in sorted(templates.items()):
            actions = template["request"]["frame_count"] - 1
            for start, end in windows_for(actions, maximum):
                record = copy.deepcopy(template)
                record["identity"] = (
                    f"push_t_var593:episode_{record['provenance']['source_episode_id']:06d}:{start:06d}-{end:06d}"
                )
                request = record["request"]
                request["source_frame_indices"] = list(range(start, end + 1))
                request["location"]["local_start_frame"] = start
                request["location"]["video_frame_count"] = end - start + 1
                request["draw_count"] = len(records)
                request["global_index"] = len(records)
                record["provenance"]["contains_terminal_transition"] = end == actions
                records.append(record)
        fingerprint = hashlib.sha256(json.dumps([record["identity"] for record in records]).encode()).hexdigest()
        for record in records:
            record["request"]["dataset_length"] = len(records)
            record["request"]["index_space_fingerprint"] = fingerprint
        manifest = json.loads((original / "manifest.json").read_text())
        manifest["sample_count"] = len(records)
        manifest["max_window_actions"] = maximum
        manifest["window_policy"] = (
            "stride-one all legal starts; cap then floor to 16 actions; no frame padding; preserve all episodes"
        )
        write_json(bundle / "samples.json", records)
        write_json(bundle / "manifest.json", manifest)
        evidence = {
            "split": split,
            "source_bundle": str(source),
            "source_bundle_receipt_sha256": digest(original / "receipts/dataset-receipt.json"),
            "source_readiness_sha256": digest(source / "readiness.json"),
            "episodes": len(templates),
            "windows": len(records),
            "outcomes": dict(Counter(record["provenance"]["episode_outcome"] for record in templates.values())),
            "rgb_frame_histogram": dict(Counter(len(record["request"]["source_frame_indices"]) for record in records)),
            "static_preflight": validate_bundle(bundle),
            "materialized_preflight": "not_run",
            "training_started": False,
            "exporter_sha256": digest(__file__),
            "files_sha256": {
                str(path.relative_to(bundle)): digest(path)
                for path in sorted(bundle.rglob("*"))
                if path.is_file() and path != bundle / "receipts/dataset-receipt.json"
            },
        }
        write_json(bundle / "receipts/dataset-receipt.json", evidence)
        summaries[split] = {key: evidence[key] for key in ("episodes", "windows", "outcomes", "rgb_frame_histogram")}
        print(json.dumps({split: summaries[split]}), flush=True)
    write_json(
        output / "export_receipt.json", {"source": str(source), "max_window_actions": maximum, "splits": summaries}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, default=592)
    args = parser.parse_args()
    rewindow(args.source.resolve(), args.output.resolve(), args.max_actions)
