"""Command-conditioned ARX-X5 windows, materialized only from local bundle assets."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
import torch.nn.functional as functional
from scipy.spatial.transform import Rotation

from cosmos_framework.data.generator.action.lance.lance_action_dataset import (
    BaseLanceActionDataset,
    LanceActionSampleRequest,
    require_action_caption,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_domain_id
from cosmos_framework.data.generator.action.utils.unified_action_schema import dual_arm_pose_mapping
from cosmos_framework.data.generator.local_simulation.core import LocalByteRangeBackend

ADAPTER_TARGET = "cosmos_framework.data.generator.action.robodojo_eef.RoboDojoEEFLocalDataset"
CONTRACT = "robodojo-arx-x5-commanded-link6-v1"
VIEW_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def commanded_eef_actions(initial_state: np.ndarray, commands: np.ndarray) -> np.ndarray:
    """Anchor to initial measured link6, then prior commands; never future measured states."""
    initial_state = np.asarray(initial_state, dtype=np.float64)
    commands = np.asarray(commands, dtype=np.float64)
    if initial_state.shape != (18,) or commands.ndim != 2 or commands.shape[1] != 16 or len(commands) == 0:
        raise ValueError("Expected measured initial state [18] and nonempty commanded actions [T,16]")
    if not np.isfinite(initial_state).all() or not np.isfinite(commands).all():
        raise ValueError("Nonfinite EEF source values")
    arms = []
    for arm in range(2):
        command = commands[:, arm * 8 : arm * 8 + 8]
        poses = np.concatenate((initial_state[arm * 9 : arm * 9 + 7][None], command[:, :7]))
        quaternions = poses[:, 3:7]
        if not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-5, rtol=0):
            raise ValueError("Source quaternions must already be unit wxyz rotations")
        rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]]).as_matrix()
        anchors = (np.arange(len(commands)) // 16) * 16
        inverse = rotations[anchors].transpose(0, 2, 1)
        translations = np.einsum("tij,tj->ti", inverse, poses[1:, :3] - poses[anchors, :3])
        relative = inverse @ rotations[1:]
        rot6d = relative[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        grip = command[:, 7]
        if np.any((grip < 0) | (grip > 1)):
            raise ValueError("Source gripper fraction is outside [0,1]")
        effective_openness = np.clip(-0.01 + 0.054 * grip, 0, 0.044) / 0.044
        arms.append(np.concatenate((translations, rot6d, effective_openness[:, None]), axis=1))
    return np.concatenate(arms, axis=1).astype(np.float32)


def decode_window(payload: bytes, indices: tuple[int, ...], fps: float, first_pts: float) -> torch.Tensor:
    """Decode the exact contiguous source frames using the measured PTS origin."""
    if not indices or indices != tuple(range(indices[0], indices[-1] + 1)) or indices[0] < 0:
        raise ValueError("Only contiguous nonnegative source-frame windows are supported")
    frames = []
    with av.open(io.BytesIO(payload)) as container:
        stream = container.streams.video[0]
        if abs(float(stream.average_rate) - fps) > 1e-6:
            raise ValueError("Video FPS differs from request")
        container.seek(int((first_pts + indices[0] / fps) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Video frame has no PTS")
            position = (float(frame.pts * frame.time_base) - first_pts) * fps
            index = round(position)
            if abs(position - index) > 1e-5:
                raise ValueError("Video PTS is off the source frame grid")
            if index < indices[0]:
                continue
            if index > indices[-1]:
                break
            if index != indices[len(frames)]:
                raise ValueError("Missing or repeated video frame")
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) == len(indices):
                break
    if len(frames) != len(indices):
        raise ValueError("Decoded video does not cover requested T+1 frames")
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


class RoboDojoEEFLocalDataset(BaseLanceActionDataset):
    """Forward-dynamics-only adapter; distinct from RoboDojo's 14D joint domain."""

    pose_convention = "backward_chunk_anchored_16f"

    def __init__(self, calibration_path: str, creds_path: str = "", viewpoint: str = "concat_view") -> None:
        del creds_path
        if viewpoint not in ("ego_view", "concat_view"):
            raise ValueError("Supported ARX viewpoints are ego_view and concat_view")
        self.viewpoint = viewpoint
        self.view_names = VIEW_NAMES[:1] if viewpoint == "ego_view" else VIEW_NAMES
        calibration = json.loads(Path(calibration_path).read_text())
        if calibration.get("contract") != CONTRACT or calibration.get("wrist_frame") != "link6":
            raise ValueError("Wrong RoboDojo EEF calibration contract")
        if calibration.get("link6_to_wrist") != np.eye(4).tolist():
            raise ValueError("This version explicitly defines wrist=link6, not a tool-tip frame")
        if calibration.get("source_command_joint_endpoints_m") != [-0.01, 0.044]:
            raise ValueError("Wrong gripper command endpoints")
        if calibration.get("physical_joint_limits_m") != [0.0, 0.044]:
            raise ValueError("Wrong physical gripper limits")
        self.fully_open_gripper_width_m = float(calibration["fully_open_gripper_width_m"])
        if not np.isfinite(self.fully_open_gripper_width_m) or self.fully_open_gripper_width_m <= 0:
            raise ValueError("Invalid calibrated gripper width")

    def get_unified_action_mapping(self):
        return dual_arm_pose_mapping(order="left_right")

    def materialize_local_sample(
        self,
        request: LanceActionSampleRequest,
        source_rows: dict[str, dict[str, Any]],
        backend: LocalByteRangeBackend,
    ) -> dict[str, Any]:
        if set(source_rows) != set(self.view_names) or set(request.row_ids_by_view) != set(self.view_names):
            raise ValueError(f"Expected exactly the declared ARX camera views: {self.view_names}")
        indices = request.source_frame_indices
        if len(indices) < 5 or (len(indices) - 1) % 16:
            raise ValueError("ARX windows require a multiple of 16 actions and T+1 observations")
        if indices != tuple(range(indices[0], indices[-1] + 1)) or indices[0] < 0:
            raise ValueError("Only contiguous nonnegative windows are supported")
        if request.source_fps != 25 or request.target_fps != 25:
            raise ValueError("This ARX contract preserves the original 25 Hz actions and video")
        if request.location.local_start_frame != indices[0]:
            raise ValueError("Request anchor and frame indices disagree")
        if len({row["numeric_uri"] for row in source_rows.values()}) != 1:
            raise ValueError("Views do not reference the same numeric episode")
        first = source_rows[VIEW_NAMES[0]]
        with np.load(io.BytesIO(backend.get(first["numeric_uri"])), allow_pickle=False) as data:
            start, end = indices[0], indices[-1]
            if end >= len(data["action"]) or len(data["action"]) != request.frame_count:
                raise ValueError("Window exceeds numeric episode")
            if not data["action_valid"][start:end].all():
                raise ValueError("Window contains an invalid/padded action")
            if not np.all(data["episode_index"] == request.episode_index):
                raise ValueError("Numeric episode identity differs from request")
            action = commanded_eef_actions(data["observation.state"][start], data["action"][start:end])
        videos = []
        for name in self.view_names:
            row = source_rows[name]
            if row["episode_index"] != request.episode_index or row["row_id"] != request.row_ids_by_view[name]:
                raise ValueError("Camera source identity differs from request")
            videos.append(decode_window(backend.get(row["video_uri"]), indices, 25, row["first_pts_seconds"]))
        head = videos[0]
        if self.viewpoint == "concat_view":
            if any(view.shape != head.shape for view in videos[1:]):
                raise ValueError("Camera shapes differ")
            height, width = head.shape[-2:]
            bottom = torch.cat(
                [
                    functional.interpolate(
                        view.float(), size=(height // 2, width // 2), mode="bilinear", align_corners=False
                    )
                    .round()
                    .to(torch.uint8)
                    for view in videos[1:]
                ],
                dim=-1,
            )
            head = torch.cat((head, bottom), dim=-2)
        video = head.permute(1, 0, 2, 3).contiguous()
        sample = {
            "ai_caption": require_action_caption(request.caption, dataset_name=CONTRACT),
            "video": video,
            "action": torch.from_numpy(action),
            "conditioning_fps": torch.tensor(25, dtype=torch.float32),
            "mode": "forward_dynamics",
            "domain_id": torch.tensor(get_domain_id("robodojo_arx_x5_eef"), dtype=torch.long),
            "viewpoint": self.viewpoint,
        }
        if self.viewpoint == "concat_view":
            sample["additional_view_description"] = (
                "The top row is the head camera. The bottom row shows the left wrist camera on the left "
                "and the right wrist camera on the right."
            )
        return sample
