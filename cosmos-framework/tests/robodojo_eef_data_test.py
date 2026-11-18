from __future__ import annotations

import io
import json
from fractions import Fraction

import av
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from cosmos_framework.data.generator.action import robodojo_eef
from cosmos_framework.data.generator.action.lance.lance_action_dataset import LanceActionSampleRequest
from cosmos_framework.data.generator.action.robodojo_eef import (
    ADAPTER_TARGET,
    CONTRACT,
    RoboDojoEEFLocalDataset,
    commanded_eef_actions,
    decode_window,
)
from cosmos_framework.data.generator.action.utils.domain_utils import get_action_dim, get_domain_id
from cosmos_framework.data.generator.lance.action_window_index import ActionWindowLocation
from cosmos_framework.data.generator.local_simulation.bimanual_config import _validate_dataset_target
from cosmos_framework.data.generator.local_simulation.dataset import _model_constructors

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


def source_arrays(count=32):
    state = np.zeros(18)
    state[[3, 12]] = 1
    commands = np.zeros((count, 16))
    commands[:, [3, 11]] = 1
    return state, commands


def test_command_anchoring_and_arm_order():
    state, commands = source_arrays()
    state[0] = 2
    state[9] = -3
    commands[:, 0] = 2 + np.arange(1, 33) * 0.01
    commands[:, 8] = -3 + np.arange(1, 33) * 0.02
    result = commanded_eef_actions(state, commands)
    np.testing.assert_allclose(result[[0, 15, 16, 31], 0], [0.01, 0.16, 0.01, 0.16], atol=1e-6)
    np.testing.assert_allclose(result[[0, 15, 16, 31], 10], [0.02, 0.32, 0.02, 0.32], atol=1e-6)
    mapping = RoboDojoEEFLocalDataset.get_unified_action_mapping(None)
    unified, valid = mapping.to_unified(torch.from_numpy(result))
    assert valid.sum().item() == 20
    torch.testing.assert_close(unified[:, 9:18], torch.from_numpy(result[:, 10:19]))
    torch.testing.assert_close(unified[:, 34:43], torch.from_numpy(result[:, :9]))
    assert mapping.valid_mask().nonzero().flatten().tolist() == list(range(9, 18)) + [33] + list(range(34, 43)) + [58]


def test_rot6d_columns_and_rotated_anchor():
    state, commands = source_arrays(16)
    quaternion = Rotation.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    state[3:7] = quaternion
    commands[:, 3:7] = quaternion
    commands[:, 0] = 1
    result = commanded_eef_actions(state, commands)
    np.testing.assert_allclose(result[0, :3], [0, -1, 0], atol=1e-6)
    np.testing.assert_allclose(result[0, 3:9], [1, 0, 0, 0, 1, 0], atol=1e-6)
    state[3:7] = [1, 0, 0, 0]
    result = commanded_eef_actions(state, commands)
    np.testing.assert_allclose(result[0, 3:9], [0, 1, 0, -1, 0, 0], atol=1e-6)


def test_gripper_command_deadzone_is_physical_not_assumed_linear():
    state, commands = source_arrays(4)
    commands[:, 7] = [0, 0.1, 0.5, 1]
    result = commanded_eef_actions(state, commands)
    np.testing.assert_allclose(result[:, 9], [0, 0, 0.017 / 0.044, 1], atol=1e-6)
    assert (result[:, 19] == 0).all()


@pytest.mark.parametrize("corruption", ["nan", "nonunit", "gripper"])
def test_bad_numeric_source_rejected(corruption):
    state, commands = source_arrays()
    if corruption == "nan":
        commands[1, 0] = np.nan
    elif corruption == "nonunit":
        commands[1, 3] = 2
    else:
        commands[1, 7] = 1.1
    with pytest.raises(ValueError):
        commanded_eef_actions(state, commands)


def test_no_joint_domain_relabeling_and_closed_constructor_registry():
    assert get_domain_id("robodojo") == 17
    assert get_action_dim("robodojo") == 14
    assert get_domain_id("robodojo_arx_x5_eef") == 33
    assert get_action_dim("robodojo_arx_x5_eef") == 20
    assert _model_constructors("bimanual")[ADAPTER_TARGET] is RoboDojoEEFLocalDataset
    _validate_dataset_target(ADAPTER_TARGET)
    with pytest.raises(ValueError):
        _validate_dataset_target("untrusted.RoboDojoEEFLocalDataset")


def test_calibration_contract(tmp_path):
    calibration = {
        "contract": CONTRACT,
        "wrist_frame": "link6",
        "link6_to_wrist": np.eye(4).tolist(),
        "source_command_joint_endpoints_m": [-0.01, 0.044],
        "physical_joint_limits_m": [0, 0.044],
        "fully_open_gripper_width_m": 0.0888,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration))
    assert RoboDojoEEFLocalDataset(str(path)).fully_open_gripper_width_m == 0.0888
    calibration["link6_to_wrist"][0][3] = 0.145
    path.write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="wrist=link6"):
        RoboDojoEEFLocalDataset(str(path))


def test_pts_origin_and_seek_do_not_drop_first_frame():
    payload = io.BytesIO()
    with av.open(payload, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(40):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 5, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index + 1, Fraction(1, 25)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    encoded = payload.getvalue()
    with av.open(io.BytesIO(encoded)) as container:
        decoded = list(container.decode(video=0))
    origin = float(decoded[0].pts * decoded[0].time_base)
    assert abs(origin - 0.04) < 1e-6
    for start in (0, 7, 23):
        actual = decode_window(encoded, tuple(range(start, start + 17)), 25, origin)
        expected = torch.from_numpy(
            np.stack([frame.to_ndarray(format="rgb24") for frame in decoded[start : start + 17]])
        )
        torch.testing.assert_close(actual, expected.permute(0, 3, 1, 2))
    with pytest.raises(ValueError, match="cover"):
        decode_window(encoded, tuple(range(30, 50)), 25, origin)
    with pytest.raises(ValueError, match="contiguous"):
        decode_window(encoded, (0, 2, 4), 25, origin)


@pytest.mark.parametrize("viewpoint", ["concat_view", "ego_view"])
def test_materializer_does_not_condition_on_future_measured_state(monkeypatch, viewpoint):
    initial, commands = source_arrays(48)
    commands[:, 0] = np.arange(48) * 0.01
    state = np.tile(initial, (49, 1))
    values = {
        "observation.state": state,
        "action": np.concatenate((commands, np.zeros((1, 16)))),
        "action_valid": np.r_[np.ones(48, dtype=bool), False],
        "episode_index": np.zeros(49, dtype=int),
    }

    reads = []

    class MemoryBackend:
        def get(self, name):
            reads.append(name)
            if name != "numeric":
                return b"unused"
            payload = io.BytesIO()
            np.savez(payload, **values)
            return payload.getvalue()

    monkeypatch.setattr(
        robodojo_eef,
        "decode_window",
        lambda payload, indices, fps, origin: torch.zeros(
            len(indices),
            3,
            24,
            32,
            dtype=torch.uint8,
        ),
    )
    rows = {
        name: {
            "numeric_uri": "numeric",
            "video_uri": name,
            "episode_index": 0,
            "row_id": index,
            "first_pts_seconds": 0.04,
        }
        for index, name in enumerate(
            robodojo_eef.VIEW_NAMES[:1] if viewpoint == "ego_view" else robodojo_eef.VIEW_NAMES
        )
    }
    request = LanceActionSampleRequest(
        draw_count=0,
        global_index=0,
        aux_seed=42,
        location=ActionWindowLocation(0, 0, 7, 33),
        source_frame_indices=tuple(range(7, 40)),
        row_ids_by_view={name: row["row_id"] for name, row in rows.items()},
        source_version=2,
        source_index_version=1,
        dataset_length=1,
        index_space_fingerprint="test",
        frame_count=49,
        source_fps=25,
        target_fps=25,
        shard_id="train",
        episode_index=0,
        caption="Push the T.",
    )
    adapter = object.__new__(RoboDojoEEFLocalDataset)
    adapter.viewpoint = viewpoint
    adapter.view_names = tuple(rows)
    before = adapter.materialize_local_sample(request, rows, MemoryBackend())
    values["observation.state"][8:] = 1000
    after = adapter.materialize_local_sample(request, rows, MemoryBackend())
    torch.testing.assert_close(before["action"], after["action"])
    assert after["video"].shape == (3, 33, 24 if viewpoint == "ego_view" else 36, 32)
    assert after["viewpoint"] == viewpoint
    if viewpoint == "ego_view":
        assert set(reads) == {"numeric", "cam_head"}
        assert "additional_view_description" not in after
    values["action_valid"][10] = False
    with pytest.raises(ValueError, match="invalid/padded"):
        adapter.materialize_local_sample(request, rows, MemoryBackend())
