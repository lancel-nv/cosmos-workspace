from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

import av
import lance
import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.push_t_data.workflow import read_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def windows_for(actions, length):
    require(length > 0 and actions >= length, "Episode too short; refusing silent dropping or padding")
    starts = list(range(0, actions - length + 1, length))
    if starts[-1] != actions - length:
        starts.append(actions - length)
    return [(start, start + length) for start in starts]


def decode_check(path, observations, fps, full):
    count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        require(float(stream.average_rate) == fps, f"Wrong FPS: {path}")
        for frame in container.decode(stream):
            require(frame.pts is not None, f"Missing PTS: {path}")
            pts = float(frame.pts * frame.time_base)
            if count == 0:
                origin = pts
            require(abs(pts - origin - count / fps) < 1e-6, f"Irregular PTS: {path}")
            require((frame.width, frame.height) == (640, 480), f"Wrong dimensions: {path}")
            if count in (0, observations // 2, observations - 1):
                rgb = frame.to_ndarray(format="rgb24")
                require(rgb.shape == (480, 640, 3), f"Wrong RGB shape: {path}")
            count += 1
            if not full:
                break
    require(count > 0, f"Empty video: {path}")
    require(not full or count == observations, f"Wrong decoded frame count: {path}")
    return {"first_pts_seconds": origin, "decoded_frames": count, "full_decode": full}


def prepare(root, output, length):
    require(not output.exists(), "Output exists; use a new output directory, never overwrite an accepted run")
    require(not output.is_relative_to(root), "Output must be outside the frozen source run")
    release = read_json(root / "processing_receipt.json")
    require(release["status"] == "raw_to_curator_verified", "Processing has not been verified")
    for name, expected in release["evidence_sha256"].items():
        require(sha256(root / name) == expected, f"Release evidence changed: {name}")

    source = lance.dataset(str(root / "curator/lance/clips.lance"))
    source_rows = source.to_table().to_pylist()
    indexed = {row["clip_id"]: row for row in source_rows}
    rows = [json.loads(line) for line in (root / "curator_complete/manifest.jsonl").read_text().splitlines()]
    require(len(indexed) == len(source_rows) == len(rows), "Duplicate or missing Lance clips")
    require({row["clip_id"] for row in rows} == set(indexed), "Manifest/Lance clip IDs differ")
    provenance = [json.loads(line) for line in (root / "lerobot/meta/provenance.jsonl").read_text().splitlines()]
    origins = {item["lerobot_episode_index"]: item for item in provenance}
    splits = read_json(root / "lerobot/meta/grouped_splits.json")
    split_by_episode = {}
    for split, indices in splits.items():
        for index in indices:
            require(index not in split_by_episode, "Episode appears in multiple splits")
            split_by_episode[index] = split
    require(set(origins) == set(split_by_episode), "Split coverage differs from provenance")
    layout_splits = defaultdict(set)
    groups = defaultdict(list)
    for row in rows:
        original = indexed[row["clip_id"]]
        for key in ("span_group_id", "episode_index", "view_name", "frame_start", "frame_end", "native_fps"):
            require(row[key] == original[key], f"Lance/manifest mismatch: {key}")
        require(not original["error_stage"] and not original["error_message"], "Curator error row")
        groups[row["episode_index"]].append(row)
    require(set(groups) == set(origins), "Episode coverage differs")
    metadata = {index: read_json(Path(origin["raw_path"]) / "metadata.json") for index, origin in origins.items()}
    buckets = defaultdict(list)
    for index, origin in origins.items():
        require(origin["split"] == split_by_episode[index], "Split provenance mismatch")
        layout_splits[origin["layout_hash"]].add(origin["split"])
        buckets[(origin["split"], metadata[index]["termination_reason"])].append(index)
    require(all(len(value) == 1 for value in layout_splits.values()), "Layout leakage between splits")
    sampled = {index for bucket in buckets.values() for index in (min(bucket), max(bucket))}

    output.mkdir(parents=True)
    records, decode_reports, npz_hashes = [], [], {}
    episode_counts, outcome_counts = Counter(), Counter()
    views = {f"observation.images.{name}" for name in ("cam_head", "cam_left_wrist", "cam_right_wrist")}
    for index, group in sorted(groups.items()):
        origin, meta = origins[index], metadata[index]
        require(len(group) == 3 and {row["view_name"] for row in group} == views, "Missing/duplicate camera")
        require(len({row["transition_npz"] for row in group}) == 1, "Views have different numeric data")
        require(len({row["span_group_id"] for row in group}) == 1, "Views have different span identity")
        npz_path = Path(group[0]["transition_npz"])
        npz_hashes[str(npz_path)] = sha256(npz_path)
        actions = meta["num_actions"]
        observations = actions + 1
        fps = meta["fps"]
        with np.load(npz_path, allow_pickle=False) as data:
            require(len(data.files) == 23, "Unexpected numeric schema")
            for key in data.files:
                require(len(data[key]) == observations, f"Length mismatch: {key}")
                require(np.isfinite(data[key]).all(), f"Nonfinite numeric data: {key}")
            require(data["action"].shape == (observations, 16), "Wrong EEF command shape")
            require(data["observation.state"].shape == (observations, 18), "Wrong measured state shape")
            require(np.array_equal(data["frame_index"], np.arange(observations)), "Frame indices differ")
            require(np.allclose(data["timestamp"], np.arange(observations) / fps, atol=2e-6, rtol=0), "Bad time")
            require(np.all(data["source.episode_id"] == meta["episode_id"]), "Wrong source episode")
            require(np.array_equal(data["action_valid"].reshape(-1), np.arange(observations) < actions), "Bad mask")
            done = data["next.done"].reshape(-1)
            require(np.flatnonzero(done).tolist() == [actions - 1], "Wrong transition termination")
            require(np.array_equal(done, (data["next.terminated"] | data["next.truncated"]).reshape(-1)), "Bad done")
            require(bool(data["next.success"][actions - 1]) == meta["success"], "Wrong success")
            cameras = []
            for row in sorted(group, key=lambda item: item["view_name"]):
                require(row["observations"] == observations and row["valid_actions"] == actions, "Wrong counts")
                require(row["frame_start"] == 0 and row["frame_end"] == observations, "Not a whole episode")
                require(row["native_fps"] == fps and row["prompt"] == meta["prompt"], "FPS/prompt mismatch")
                video = Path(row["clip_uri"])
                check = decode_check(video, observations, fps, index in sampled)
                decode_reports.append({"clip_id": row["clip_id"], "episode_index": index, **check})
                cameras.append(
                    {
                        "view_name": row["view_name"],
                        "clip_id": row["clip_id"],
                        "video_path": str(video),
                        "first_pts_seconds": check["first_pts_seconds"],
                    }
                )
            for start, end in windows_for(actions, length):
                records.append(
                    {
                        "identity": f"{root.name}:episode_{meta['episode_id']:06d}:{start:06d}-{end:06d}",
                        "split": origin["split"],
                        "layout_hash": origin["layout_hash"],
                        "layout_index": origin["layout_index"],
                        "source_episode_id": meta["episode_id"],
                        "episode_index": index,
                        "span_group_id": group[0]["span_group_id"],
                        "episode_success": meta["success"],
                        "episode_outcome": meta["termination_reason"],
                        "prompt": meta["prompt"],
                        "fps": fps,
                        "transition_npz": str(npz_path),
                        "action_start": start,
                        "action_end_exclusive": end,
                        "observation_start": start,
                        "observation_end_exclusive": end + 1,
                        "contains_terminal_transition": bool(done[start:end].any()),
                        "video_views": cameras,
                    }
                )
        episode_counts[origin["split"]] += 1
        outcome_counts[meta["termination_reason"]] += 1
        if (len(npz_hashes) % 50) == 0:
            print(f"Validated {len(npz_hashes)}/{len(groups)} episodes", flush=True)

    require(len({record["identity"] for record in records}) == len(records), "Duplicate windows")
    require(
        sum(record["contains_terminal_transition"] for record in records) == len(groups), "Lost terminal transition"
    )
    table = pa.Table.from_pylist(records)
    table = table.replace_schema_metadata(
        {b"format": b"push-t-framework-staging-windows-v1", b"framework_ready": b"false"}
    )
    lance.write_dataset(table, str(output / "windows.lance"), mode="create")
    reread = lance.dataset(str(output / "windows.lance")).to_table()
    require(reread.to_pylist() == records, "Lance output roundtrip differs")
    with (output / "windows.jsonl").open("x") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "decode_checks.json").write_text(json.dumps(decode_reports, indent=2) + "\n")
    (output / "numeric_sha256.json").write_text(json.dumps(npz_hashes, indent=2) + "\n")
    receipt = {
        "schema": "push-t-framework-staging-v1",
        "completed_at": datetime.now(UTC).isoformat(),
        "status": "validated_staging_index",
        "framework_ready": False,
        "target_model": "Cosmos3-Nano-Sim-Bimanual",
        "source_root": str(root),
        "source_fingerprint": release["raw_manifest_sha256"],
        "verified_frozen_stages": ["export", "curate", "augment"],
        "source_lance_version": source.version,
        "source_lance_rows": len(rows),
        "source_release_sha256": sha256(root / "processing_receipt.json"),
        "source_receipt_file": "processing_receipt.json",
        "source_evidence_sha256": release["evidence_sha256"],
        "script_sha256": sha256(Path(__file__)),
        "episodes": len(groups),
        "outcomes": dict(outcome_counts),
        "episodes_by_split": dict(episode_counts),
        "windows": len(records),
        "windows_by_split": dict(Counter(record["split"] for record in records)),
        "window_actions": length,
        "window_observations": length + 1,
        "window_policy": "stride=action length; append one overlapping full tail window; no padding",
        "index_semantics": "half-open slices; action[start:end], observations[start:end+1]",
        "action_semantics": "Unmodified commanded EEF world/link6 targets, not measured future poses",
        "numeric_columns": 23,
        "layout_disjoint": True,
        "video_first_frame_checks": len(decode_reports),
        "full_decode_episodes": sorted(sampled),
        "full_decode_clips": sum(item["full_decode"] for item in decode_reports),
        "full_decoded_frames": sum(item["decoded_frames"] for item in decode_reports if item["full_decode"]),
        "portability": "Reference-only absolute paths; retain frozen source. Not a portable training bundle.",
        "remaining": [
            "Implement/register ARX-X5 commanded-EEF adapter with calibrated link/tool/gripper semantics",
            "Validate unified-v1 59-to-64 action mapping, masks, 16-step anchoring and pinned normalization",
            "Finalize model-native video window/view policy; staging windows are provisional",
            "Materialize portable native local bundle with tokenizer/resources and receipts",
            "Pass native static and materialize preflights before a short training/rollout check",
        ],
    }
    for name, expected in release["evidence_sha256"].items():
        require(sha256(root / name) == expected, f"Processing evidence changed: {name}")
    receipt["output_sha256"] = {
        name: sha256(output / name)
        for name in (
            "windows.jsonl",
            "decode_checks.json",
            "numeric_sha256.json",
        )
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="Read-only Curator-to-Framework staging; not a native bundle exporter")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-actions", type=int, default=32)
    args = parser.parse_args()
    require(args.window_actions > 0, "Window length must be positive")
    prepare(args.run_root.resolve(), args.output.resolve(), args.window_actions)


if __name__ == "__main__":
    main()
