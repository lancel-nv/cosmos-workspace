from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from src.push_t_data.render_capture import validate_capture_metadata
from src.push_t_data.render_quality import validate_reset_render
from src.push_t_data.success_events import validate_success_event
from src.push_t_data.visual_quality import check_head_exposure
from utils.save_file import VideoStreamWriter


class EpisodeWriter:
    def __init__(self, root: Path, episode_id: int, metadata: dict):
        self.destination = root / f"episode_{episode_id:06d}"
        self.staging = root / f".episode_{episode_id:06d}.partial"
        if self.destination.exists() or self.staging.exists():
            raise FileExistsError(self.destination)
        self.staging.mkdir(parents=True)
        self.metadata = metadata
        self.observations = []
        self.transitions = []
        self.videos = {}

    def observe(self, state: dict, images: dict):
        if len(images) != 3:
            raise ValueError(f"Expected three RGB views, received {list(images)}")
        exposure = check_head_exposure(images["cam_head"])
        summary = self.metadata.setdefault(
            "exposure_guard",
            {"frames_checked": 0, "max_table_white_fraction": 0.0, "threshold": 0.10},
        )
        summary["frames_checked"] += 1
        summary["max_table_white_fraction"] = max(summary["max_table_white_fraction"], exposure["table_white_fraction"])
        for name, frame in images.items():
            frame = np.asarray(frame)
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Invalid RGB frame: {name}: {frame.shape}, {frame.dtype}")
            if float((frame.max(axis=-1) < 5).mean()) >= 0.95:
                raise ValueError(f"Nearly black RGB frame: {name}")
            if name not in self.videos:
                if self.observations:
                    raise ValueError("Camera set changed during episode")
                self.videos[name] = VideoStreamWriter(
                    str(self.staging / f"{name}.mp4"), *frame.shape, fps=self.metadata["fps"]
                )
            self.videos[name].append(frame)
        if set(images) != set(self.videos):
            raise ValueError("Missing camera")
        self.observations.append({key: np.asarray(value).copy() for key, value in state.items()})

    def transition(self, action: dict):
        if len(self.observations) != len(self.transitions) + 1:
            raise ValueError("Action must follow its source observation")
        self.transitions.append({key: np.asarray(value).copy() for key, value in action.items()})

    def finish(self, success: bool, reason: str):
        if not self.transitions or len(self.observations) != len(self.transitions) + 1:
            raise ValueError("Require T actions and T+1 observations")
        for writer in self.videos.values():
            writer.close(announce=False)
        arrays = {}
        for prefix, rows in (("observation", self.observations), ("transition", self.transitions)):
            for key in rows[0]:
                values = np.stack([row[key] for row in rows])
                if np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite {prefix}.{key}")
                arrays[f"{prefix}.{key}"] = values
        np.savez_compressed(self.staging / "trajectory.npz", **arrays)
        self.metadata.update(
            success=bool(success),
            termination_reason=reason,
            num_actions=len(self.transitions),
            num_observations=len(self.observations),
            cameras=sorted(self.videos),
        )
        (self.staging / "metadata.json").write_text(json.dumps(self.metadata, indent=2) + "\n")
        validate_episode(self.staging)
        self.staging.rename(self.destination)
        return self.metadata

    def close(self):
        for writer in self.videos.values():
            writer.close(announce=False)


def validate_episode(path: Path):
    metadata = json.loads((path / "metadata.json").read_text())
    if "render_capture.py" in metadata.get("collector_source_hashes", {}) or "render_capture" in metadata:
        validate_capture_metadata(metadata)
    if type(metadata["success"]) is not bool:
        raise ValueError(f"Success must be a boolean: {path}")
    if not isinstance(metadata.get("prompt"), str) or not metadata["prompt"].strip():
        raise ValueError(f"Missing overall prompt: {path}")
    if len(metadata["cameras"]) != 3 or len(set(metadata["cameras"])) != 3:
        raise ValueError(f"Require exactly three distinct cameras: {path}")
    with np.load(path / "trajectory.npz", allow_pickle=False) as arrays:
        required_shapes = {
            "observation.eef_pose": (14,),
            "observation.gripper_position": (4,),
            "observation.joint_position": (16,),
            "observation.joint_velocity": (16,),
            "observation.object_pose": (7,),
            "observation.target_pose": (7,),
            "transition.action_eef_pose": (14,),
            "transition.action_gripper": (2,),
            "transition.applied_joint_targets": (10, 16),
            "transition.ik_success": (2,),
            "transition.terminated": (),
            "transition.truncated": (),
            "transition.success": (),
        }
        for name, shape in required_shapes.items():
            if arrays[name].shape[1:] != shape:
                raise ValueError(f"Unexpected {name} shape: {path}")
        count = metadata["num_actions"]
        if "render_quality.py" in metadata.get("collector_source_hashes", {}) or "reset_render" in metadata:
            validate_reset_render(metadata, arrays["observation.physics_time"][0])
        if count <= 0 or metadata["num_observations"] != count + 1:
            raise ValueError(f"Invalid episode lengths: {path}")
        if "scene_resources" in metadata:
            resources = metadata["scene_resources"]
            exposure = metadata.get("exposure_guard", {})
            if resources.get("room_count") != 1 or sorted(
                (light["type"], light["intensity"]) for light in resources.get("lights", [])
            ) != [("DomeLight", 1000.0), ("RectLight", 15000.0)]:
                raise ValueError(f"Invalid fixed-scene resources: {path}")
            if (
                exposure.get("frames_checked") != count + 1
                or exposure.get("threshold") != 0.10
                or not 0 <= exposure.get("max_table_white_fraction", float("nan")) <= 0.10
            ):
                raise ValueError(f"Missing or invalid per-frame exposure guard: {path}")
        for name in arrays.files:
            values = arrays[name]
            expected = count + 1 if name.startswith("observation.") else count
            if len(values) != expected or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name}: {path}")
        times = arrays["observation.time"]
        if not np.allclose(np.diff(times), 1 / metadata["fps"], atol=1e-8):
            raise ValueError(f"Irregular control timestamps: {path}")
        if not np.allclose(np.diff(arrays["observation.physics_time"]), 1 / metadata["fps"], atol=1e-6):
            raise ValueError(f"Physics/control clock mismatch: {path}")
        terminated = arrays["transition.terminated"].astype(bool)
        truncated = arrays["transition.truncated"].astype(bool)
        for field in ("terminated", "truncated", "success", "ik_success"):
            if arrays[f"transition.{field}"].dtype != np.bool_:
                raise ValueError(f"Expected boolean transition.{field}: {path}")
        if np.any(terminated & truncated) or np.any((terminated | truncated)[:-1]):
            raise ValueError(f"Invalid termination flags: {path}")
        if not (terminated | truncated)[-1]:
            raise ValueError(f"Missing final termination: {path}")
        if metadata["success"] and not terminated[-1]:
            raise ValueError(f"Success must terminate: {path}")
        success = arrays["transition.success"].astype(bool)
        if np.any(success[:-1]) or bool(success[-1]) != bool(metadata["success"]):
            raise ValueError(f"Inconsistent success flags: {path}")
        expected_reason = "success" if metadata["success"] else "time_limit" if truncated[-1] else "task_failure"
        if metadata["termination_reason"] != expected_reason:
            raise ValueError(f"Termination reason disagrees with transition flags: {path}")
        if "success_semantics" in metadata:
            validate_success_event(arrays, metadata)
        for arm in range(2):
            quaternions = arrays["observation.eef_pose"][:, arm * 7 + 3 : arm * 7 + 7]
            if not np.allclose(np.linalg.norm(quaternions, axis=-1), 1, atol=1e-3):
                raise ValueError(f"Invalid EEF quaternion: {path}")
        object_motion = np.linalg.norm(
            arrays["observation.object_pose"][:, :3] - arrays["observation.object_pose"][0, :3], axis=1
        ).max()
        eef_positions = arrays["observation.eef_pose"].reshape(count + 1, 2, 7)[..., :3]
        eef_motion = np.linalg.norm(eef_positions - eef_positions[0], axis=-1).max(axis=0)
        ik_success_rate = arrays["transition.ik_success"].astype(bool).mean(axis=0)
    for camera in metadata["cameras"]:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames,avg_frame_rate",
                "-of",
                "json",
                str(path / f"{camera}.mp4"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        if int(stream["nb_read_frames"]) != count + 1 or numerator / denominator != metadata["fps"]:
            raise ValueError(f"RGB/state count or FPS mismatch: {path}/{camera}")
    return {
        "episode": path.name,
        "success": metadata["success"],
        "actions": count,
        "object_displacement_max_m": float(object_motion),
        "eef_displacement_max_m_per_arm": eef_motion.tolist(),
        "ik_success_rate_per_arm": ik_success_rate.tolist(),
        "termination_reason": metadata["termination_reason"],
    }
