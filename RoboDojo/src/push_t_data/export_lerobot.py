from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil

import av
import numpy as np

from src.push_t_data.io import validate_episode


def numeric_frame(arrays, metadata, index):
    count = metadata["num_actions"]
    valid = index < count
    poses = arrays["observation.eef_pose"][index].reshape(2, 7)
    grippers = arrays["observation.gripper_position"][index].reshape(2, 2)
    actual = np.concatenate([np.r_[pose, gripper] for pose, gripper in zip(poses, grippers, strict=True)])
    action = np.zeros(16, dtype=np.float32)
    if valid:
        commands = arrays["transition.action_eef_pose"][index].reshape(2, 7)
        commanded_grippers = arrays["transition.action_gripper"][index]
        action = np.concatenate(
            [np.r_[pose, gripper] for pose, gripper in zip(commands, commanded_grippers, strict=True)]
        ).astype(np.float32)
    result = {
        "observation.state": actual.astype(np.float32),
        "observation.states.joint_position": arrays["observation.joint_position"][index].astype(np.float32),
        "observation.states.joint_velocity": arrays["observation.joint_velocity"][index].astype(np.float32),
        "observation.states.end_effector": arrays["observation.eef_pose"][index].astype(np.float32),
        "observation.object_pose": arrays["observation.object_pose"][index].astype(np.float32),
        "observation.target_pose": arrays["observation.target_pose"][index].astype(np.float32),
        "observation.physics_time": np.asarray([arrays["observation.physics_time"][index]], dtype=np.float64),
        "action": action,
        "action_valid": np.asarray([valid], dtype=np.bool_),
        "observation.is_terminal": np.asarray([not valid], dtype=np.bool_),
        "episode.success": np.asarray([metadata["success"]], dtype=np.bool_),
        "source.episode_id": np.asarray([metadata["episode_id"]], dtype=np.int64),
    }
    for source, target in (
        ("terminated", "next.terminated"),
        ("truncated", "next.truncated"),
        ("success", "next.success"),
    ):
        result[target] = np.asarray([bool(arrays[f"transition.{source}"][index]) if valid else False])
    result["next.done"] = result["next.terminated"] | result["next.truncated"]
    for source, target in (
        ("applied_joint_targets", "action.applied_joint_targets"),
        ("ik_success", "action.ik_success"),
    ):
        values = arrays[f"transition.{source}"]
        value = values[index] if valid else np.zeros_like(values[0])
        result[target] = value.reshape(-1).astype(np.bool_ if source == "ik_success" else np.float32)
    return result


def export(args):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    paths = sorted(args.input.glob("episode_*"))
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError("No completed episodes to export")
    if args.output.exists():
        raise FileExistsError("Use a new output directory; existing exports are never overwritten")
    first_meta = json.loads((paths[0] / "metadata.json").read_text())
    with np.load(paths[0] / "trajectory.npz", allow_pickle=False) as arrays:
        example = numeric_frame(arrays, first_meta, 0)
    features = {name: {"dtype": str(value.dtype), "shape": value.shape} for name, value in example.items()}
    for camera in first_meta["cameras"]:
        with av.open(str(paths[0] / f"{camera}.mp4")) as container:
            stream = container.streams.video[0]
            features[f"observation.images.{camera}"] = {
                "dtype": "video",
                "shape": (stream.height, stream.width, 3),
                "names": ["height", "width", "channels"],
            }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output,
        fps=first_meta["fps"],
        robot_type="robodojo_dual_arx_x5",
        features=features,
        use_videos=True,
        image_writer_threads=6,
        vcodec="h264",
        video_backend="pyav",
    )
    provenance = []
    splits = {"train": [], "validation": []}
    for episode_index, path in enumerate(paths):
        validate_episode(path)
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata["cameras"] != first_meta["cameras"] or metadata["fps"] != first_meta["fps"]:
            raise ValueError(f"Inconsistent camera/FPS contract: {path}")
        with np.load(path / "trajectory.npz", allow_pickle=False) as arrays, ExitStack() as stack:
            decoders = {
                camera: iter(stack.enter_context(av.open(str(path / f"{camera}.mp4"))).decode(video=0))
                for camera in metadata["cameras"]
            }
            for index in range(metadata["num_observations"]):
                frame = numeric_frame(arrays, metadata, index)
                frame["task"] = metadata["prompt"]
                for camera, decoder in decoders.items():
                    frame[f"observation.images.{camera}"] = next(decoder).to_ndarray(format="rgb24")
                dataset.add_frame(frame)
            dataset.save_episode()
        layout_hash = hashlib.sha256(json.dumps(metadata["layout"], sort_keys=True).encode()).hexdigest()
        split = "validation" if int(layout_hash[:8], 16) % 10 == 0 else "train"
        splits[split].append(episode_index)
        provenance.append(
            {
                "lerobot_episode_index": episode_index,
                "raw_path": str(path.resolve()),
                "layout_hash": layout_hash,
                "split": split,
                **metadata,
            }
        )
        print(f"EXPORTED {episode_index + 1}/{len(paths)}", flush=True)
    dataset.finalize()
    dataset.stop_image_writer()
    destination = args.output / "meta" / "provenance.jsonl"
    destination.write_text("".join(json.dumps(row) + "\n" for row in provenance))
    (args.output / "meta" / "grouped_splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    (args.output / "meta" / "transition_contract.json").write_text(
        json.dumps(
            {
                "observations_per_episode": "T+1",
                "valid_actions_per_episode": "T",
                "terminal_row": "action_valid=false; action and next.* are padding, never train this action",
                "transition": "row[t] observation and action -> row[t+1] observation; next.* describes this transition",
                "prompt": "task_index refers to meta/tasks.parquet; full prompt also in provenance.jsonl",
                "action": "left xyz,qw,qx,qy,qz,open_fraction; right xyz,qw,qx,qy,qz,open_fraction",
                "gripper_action": "0 means closed, 1 means open (normalized open fraction)",
                "state": "left xyz,qw,qx,qy,qz,measured_finger1,measured_finger2; then right, units m/rad",
                "frame": "environment-local world coordinates, EEF link6, NOT fingertip; quaternion wxyz",
                "split": "group by full initialization layout hash, not frame or action seed",
                "curator_warning": (
                    "stock robot-action-split drops custom termination/mask columns; retain provenance/raw source"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    if (args.input / "quality_report.json").exists():
        shutil.copy2(args.input / "quality_report.json", args.output / "meta" / "raw_quality_report.json")
    loaded = LeRobotDataset(args.repo_id, root=args.output, video_backend="pyav")
    if loaded.num_episodes != len(paths):
        raise ValueError("Export episode count mismatch")
    for episode_index, metadata in enumerate(provenance):
        offset = sum(row["num_observations"] for row in provenance[:episode_index])
        first = loaded[offset]
        last = loaded[offset + metadata["num_observations"] - 1]
        if not bool(first["action_valid"].item()) or bool(last["action_valid"].item()):
            raise ValueError("Roundtrip action validity mismatch")
    print(f"LEROBOT_ROUNDTRIP_OK episodes={loaded.num_episodes} frames={len(loaded)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Export genuine LeRobot v3 with terminal observations")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default="local/robodojo-push-t-500")
    parser.add_argument("--limit", type=int)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
