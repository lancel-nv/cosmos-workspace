from __future__ import annotations

import numpy as np

SEMANTICS = "official_latched_physics_substep_v1"


def capture_success_event(env, action_index, substep_index):
    manager = env.robot_manager
    layout = env.scene_manager.layout_manager
    parser = env.reward_manager.func_parser
    block_name = layout.get_instance_name(env_idx=0, label="t")
    return {
        "action_index": action_index,
        "substep_index": substep_index,
        "physics_time": env.sim.sim.current_time,
        "official_reward": float(env.reward_manager.get_reward()[0]),
        "eef_pose": np.concatenate([manager.get_real_endpose(robot, [0])[0] for robot in manager.robot_list]).tolist(),
        "object_pose": np.concatenate(layout.get_instance_pose(0, label="t")).tolist(),
        "target_pose": np.concatenate(layout.get_instance_pose(0, label="target_t")).tolist(),
        "reference_eef_pose": np.concatenate(
            [parser.robot_origin_endpose[0][robot.arm_name] for robot in manager.robot_list]
        ).tolist(),
        "reference_object_pose": parser.pre_state[0][block_name]["pose"].tolist(),
    }


def angle_degrees(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left / np.linalg.norm(left, axis=-1, keepdims=True)
    right = right / np.linalg.norm(right, axis=-1, keepdims=True)
    return 2 * np.rad2deg(np.arccos(np.clip(abs(np.sum(left * right, axis=-1)), 0, 1)))


def validate_success_event(raw, metadata):
    event = metadata.get("success_event")
    if metadata.get("success_semantics") != SEMANTICS:
        raise ValueError("Missing explicit success-event semantics")
    if event is None:
        if metadata["success"]:
            raise ValueError("Successful episode has no physical success-event evidence")
        return
    count = metadata["num_actions"]
    action_index = event["action_index"]
    substep_index = event["substep_index"]
    if type(action_index) is not int or action_index != count - 1:
        raise ValueError("Success event must occur inside the final action")
    if type(substep_index) is not int or not 0 <= substep_index < metadata["substeps"]:
        raise ValueError("Invalid success-event substep")
    expected_time = raw["observation.physics_time"][action_index] + (substep_index + 1) * metadata["physics_dt"]
    if not np.isfinite(event["physics_time"]) or abs(event["physics_time"] - expected_time) > 1e-6:
        raise ValueError("Success event does not align with its physical substep")
    arrays = {}
    for name, size in (
        ("eef_pose", 14),
        ("reference_eef_pose", 14),
        ("object_pose", 7),
        ("target_pose", 7),
        ("reference_object_pose", 7),
    ):
        values = np.asarray(event[name], dtype=np.float64)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid success-event {name}")
        poses = values.reshape(-1, 7)
        if not np.allclose(np.linalg.norm(poses[:, 3:], axis=-1), 1, atol=1e-3):
            raise ValueError(f"Invalid success-event quaternion: {name}")
        arrays[name] = poses
    reference = arrays["reference_eef_pose"]
    initial = raw["observation.eef_pose"][0].reshape(2, 7)
    if (
        np.max(abs(reference[:, :3] - initial[:, :3])) > 1e-5
        or np.max(angle_degrees(reference[:, 3:], initial[:, 3:])) > 0.001
    ):
        raise ValueError("Success-event home reference differs from the initial observation")
    block = arrays["object_pose"][0]
    target = arrays["target_pose"][0]
    if not np.allclose(target, raw["observation.target_pose"][action_index], atol=1e-5):
        raise ValueError("Success-event target differs from the recorded target")
    origin_block = arrays["reference_object_pose"][0]
    if not np.allclose(origin_block, raw["observation.object_pose"][0], atol=1e-5):
        raise ValueError("Success-event object reference differs from the initial observation")
    poses = arrays["eef_pose"]
    if not (
        event["official_reward"] == 1.0
        and np.linalg.norm(block[:2] - target[:2]) < 0.007
        and angle_degrees(block[3:], target[3:]) < 7
        and np.max(abs(poses[:, :3] - reference[:, :3])) <= 0.15
        and np.max(angle_degrees(poses[:, 3:], reference[:, 3:])) <= 20
        and block[2] - origin_block[2] <= 0.01
    ):
        raise ValueError("Physical success event violates the unchanged official criteria")
    if metadata["success"] and np.max(raw["observation.object_pose"][:, 2] - origin_block[2]) > 0.01:
        raise ValueError("Successful episode contains a recorded forbidden lift")
