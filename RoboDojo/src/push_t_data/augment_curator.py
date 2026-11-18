import argparse
import json
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlparse

import lance
import numpy as np
import pyarrow.parquet as parquet


def main():
    parser = argparse.ArgumentParser(description="Preserve complete transition fields beside Curator video clips")
    parser.add_argument("--lance", type=Path, required=True)
    parser.add_argument("--lerobot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--expected-episodes", type=int)
    args = parser.parse_args()
    mappings = [tuple(value.split("=", 1)) for value in args.path_map]

    def local_path(uri):
        parsed = urlparse(uri)
        if parsed.scheme not in ("", "file"):
            raise ValueError(f"Only local paths are supported: {uri}")
        path = unquote(parsed.path)
        for old, new in mappings:
            if path == old or path.startswith(old.rstrip("/") + "/"):
                path = new.rstrip("/") + path[len(old.rstrip("/")) :]
                break
        return Path(path)

    features = json.loads((args.lerobot / "meta/info.json").read_text())["features"]
    columns = [name for name, spec in features.items() if spec["dtype"] not in ("video", "image", "string")]
    rows = lance.dataset(str(args.lance)).to_table().to_pylist()
    if not rows or any(row["error_stage"] for row in rows):
        raise ValueError("Curator has no clips or has failed clips; do not publish a partial training manifest")
    expected_views = {name for name, spec in features.items() if spec["dtype"] == "video"}
    groups = {}
    for row in rows:
        groups.setdefault(row["span_group_id"], []).append(row)
    if args.expected_episodes is not None and len(groups) != args.expected_episodes:
        raise ValueError("Curator dropped episodes or split whole episodes unexpectedly")
    if len({row["episode_index"] for row in rows}) != len(groups):
        raise ValueError("Expected exactly one span group per episode")
    for group, views in groups.items():
        if len(views) != len(expected_views) or {row["view_name"] for row in views} != expected_views:
            raise ValueError(f"Missing or duplicate camera views: {group}")
        spans = {(row["episode_index"], row["frame_start"], row["frame_end"]) for row in views}
        if len(spans) != 1:
            raise ValueError(f"Camera spans are not aligned: {group}")
    args.output.mkdir(parents=True, exist_ok=False)
    payloads = {}
    manifest = []
    for row in rows:
        group = row["span_group_id"]
        frame_count = row["frame_end"] - row["frame_start"]
        if group not in payloads:
            table = parquet.read_table(
                local_path(row["data_parquet_uri"]),
                columns=columns,
                filters=[
                    ("episode_index", "=", row["episode_index"]),
                    ("frame_index", ">=", row["frame_start"]),
                    ("frame_index", "<", row["frame_end"]),
                ],
            ).sort_by("frame_index")
            arrays = {name: np.asarray(table[name].to_pylist(), dtype=features[name]["dtype"]) for name in columns}
            if len(arrays["frame_index"]) != frame_count:
                raise ValueError(f"Clip/state mismatch: {group}")
            validity = arrays["action_valid"].reshape(-1)
            if not validity[:-1].all() or validity[-1]:
                raise ValueError("Expected whole episodes including a single final observation")
            done = arrays["next.done"].reshape(-1)
            if done.sum() != 1 or not done[-2]:
                raise ValueError("Transition termination was lost or shifted")
            for name, value in arrays.items():
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite field: {name}")
            payload = args.output / f"{group}.npz"
            np.savez_compressed(payload, **arrays)
            payloads[group] = payload
        video = local_path(row["clip_uri"])
        probe = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                str(video),
            ],
            text=True,
        )
        if int(probe.strip()) != frame_count:
            raise ValueError(f"Curator video/state length mismatch: {video}")
        manifest.append(
            {
                **row,
                "clip_uri": str(video.resolve()),
                "transition_npz": str(payloads[group].resolve()),
                "prompt": row["task_name"],
                "observations": frame_count,
                "valid_actions": frame_count - 1,
            }
        )
    temporary = args.output / "manifest.jsonl.partial"
    temporary.write_text("".join(json.dumps(row) + "\n" for row in manifest))
    temporary.rename(args.output / "manifest.jsonl")
    (args.output / "transition_contract.json").write_bytes(
        (args.lerobot / "meta/transition_contract.json").read_bytes()
    )
    print(json.dumps({"clips": len(manifest), "episodes": len(payloads), "preserved_fields": columns}))


if __name__ == "__main__":
    main()
