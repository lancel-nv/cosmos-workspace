from __future__ import annotations

import numpy as np

from src.push_t_data.render_capture import render_capture

RESET_RENDER_SEMANTICS = "static_three_view_convergence_v1"
MIN_RESET_RENDERS = 16
MAX_RESET_RENDERS = 96
STABLE_RESET_PAIRS = 3
MAX_MEAN_RGB_DELTA = 0.10
MAX_CHANGED_PIXEL_FRACTION = 0.001
CAMERAS = {"cam_head", "cam_left_wrist", "cam_right_wrist"}


def rgb_images(observation):
    images = {name: np.asarray(camera["color"]).copy() for name, camera in observation["vision"].items()}
    if set(images) != CAMERAS:
        raise ValueError("Reset convergence requires all three cameras")
    for name, pixels in images.items():
        if pixels.shape != (480, 640, 3) or pixels.dtype != np.uint8:
            raise ValueError(f"Invalid reset RGB: {name}")
    return images


def settle_reset_observation(env, observation):
    physics_time = env.sim.sim.current_time
    previous = rgb_images(observation)
    stable_pairs = 0
    history = []
    for render_pass in range(2, MAX_RESET_RENDERS + 1):
        render_capture(env)
        if env.sim.sim.current_time != physics_time:
            raise RuntimeError("Reset rendering advanced physics")
        observation = env.obs_manager.get_obs(env_idx_list=[0])[0]
        current = rgb_images(observation)
        metrics = {}
        for name in sorted(CAMERAS):
            delta = np.abs(current[name].astype(np.int16) - previous[name].astype(np.int16))
            metrics[name] = {
                "mean_absolute_delta": float(delta.mean()),
                "changed_pixel_fraction": float((delta.max(axis=-1) > 20).mean()),
            }
        stable = all(
            values["mean_absolute_delta"] <= MAX_MEAN_RGB_DELTA
            and values["changed_pixel_fraction"] <= MAX_CHANGED_PIXEL_FRACTION
            for values in metrics.values()
        )
        stable_pairs = stable_pairs + 1 if stable and render_pass >= MIN_RESET_RENDERS else 0
        history.append(metrics)
        if stable_pairs >= STABLE_RESET_PAIRS:
            report = {
                "semantics": RESET_RENDER_SEMANTICS,
                "render_passes": render_pass,
                "physics_time_before": physics_time,
                "physics_time_after": env.sim.sim.current_time,
                "stable_pairs": stable_pairs,
                "final_pairs": history[-STABLE_RESET_PAIRS:],
            }
            return observation, report
        previous = current
    raise RuntimeError(f"Reset RGB did not converge in {MAX_RESET_RENDERS} renders: {history[-1]}")


def validate_reset_render(metadata, first_physics_time):
    report = metadata.get("reset_render", {})
    if report.get("semantics") != RESET_RENDER_SEMANTICS:
        raise ValueError("Missing reset-render convergence evidence")
    passes = report.get("render_passes")
    if type(passes) is not int or not MIN_RESET_RENDERS + STABLE_RESET_PAIRS - 1 <= passes <= MAX_RESET_RENDERS:
        raise ValueError("Invalid reset-render count")
    if report.get("stable_pairs") != STABLE_RESET_PAIRS or len(report.get("final_pairs", [])) != STABLE_RESET_PAIRS:
        raise ValueError("Missing consecutive reset-render convergence pairs")
    if (
        report.get("physics_time_before") != first_physics_time
        or report.get("physics_time_after") != first_physics_time
    ):
        raise ValueError("Reset-render physics timestamp mismatch")
    for metrics in report["final_pairs"]:
        if set(metrics) != CAMERAS:
            raise ValueError("Missing reset-render camera")
        for values in metrics.values():
            if not (
                0 <= values.get("mean_absolute_delta", float("nan")) <= MAX_MEAN_RGB_DELTA
                and 0 <= values.get("changed_pixel_fraction", float("nan")) <= MAX_CHANGED_PIXEL_FRACTION
            ):
                raise ValueError("Invalid reset-render convergence metrics")
