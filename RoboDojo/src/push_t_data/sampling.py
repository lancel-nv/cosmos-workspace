import math
import re


def validate_policy_calibration(calibration, priority_indices):
    if not isinstance(calibration, dict):
        raise ValueError("policy_calibration must be an object")
    numeric = {"speed_m_s", "lateral_bias_m", "speed_jitter_m_s", "bias_jitter_m"}
    for index, preset in calibration.items():
        if index not in {str(value) for value in priority_indices}:
            raise ValueError("Calibration must reference a configured priority layout")
        if not isinstance(preset, dict) or set(preset) != numeric | {"source_episode", "source_trajectory_sha256"}:
            raise ValueError("Invalid calibration preset keys")
        if any(type(preset[key]) not in (int, float) or not math.isfinite(preset[key]) for key in numeric):
            raise ValueError("Calibration parameters must be finite numbers")
        if preset["speed_jitter_m_s"] <= 0 or preset["bias_jitter_m"] <= 0:
            raise ValueError("Calibration requires nonzero seeded parameter variation")
        if (
            not 0
            < preset["speed_m_s"] - preset["speed_jitter_m_s"]
            < preset["speed_m_s"] + preset["speed_jitter_m_s"]
            <= 0.5
        ):
            raise ValueError("Calibrated speed range must remain in (0, 0.5] m/s")
        if abs(preset["lateral_bias_m"]) + preset["bias_jitter_m"] > 0.005:
            raise ValueError("Calibrated lateral bias must stay within 5 mm")
        if not isinstance(preset["source_episode"], str) or not preset["source_episode"]:
            raise ValueError("Calibration needs a source episode")
        checksum = preset["source_trajectory_sha256"]
        if not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum):
            raise ValueError("Calibration needs the source trajectory SHA256")


def calibrated_parameters(layout_index, seed, calibration):
    preset = calibration.get(str(layout_index))
    if preset is None:
        return None
    import numpy as np

    rng = np.random.default_rng(seed)
    return (
        preset["speed_m_s"] + float(rng.uniform(-preset["speed_jitter_m_s"], preset["speed_jitter_m_s"])),
        preset["lateral_bias_m"] + float(rng.uniform(-preset["bias_jitter_m"], preset["bias_jitter_m"])),
    )


def validate_priority_indices(indices, layout_count=None):
    if not isinstance(indices, list) or any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("priority_layout_indices must be a list of nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("priority_layout_indices must not contain duplicates")
    if layout_count is not None and (layout_count <= 0 or any(index >= layout_count for index in indices)):
        raise ValueError("Priority layout index is outside the public layout catalog")


def select_layout_index(episode, start, layout_count, priority_indices):
    validate_priority_indices(priority_indices, layout_count)
    if episode < start or start < 0:
        raise ValueError("Episode must not precede the configured start")
    if not priority_indices:
        return episode % layout_count
    cycle, slot = divmod(episode - start, 3)
    if slot < 2:
        return priority_indices[(cycle * 2 + slot) % len(priority_indices)]
    return cycle % layout_count
