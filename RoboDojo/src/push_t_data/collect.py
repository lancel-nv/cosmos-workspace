from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback


def arguments():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Collect dense RoboDojo push_T transitions locally")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--min-successes", type=int, default=0)
    parser.add_argument("--min-failures", type=int, default=0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--priority-layout-indices", type=int, nargs="+", default=[])
    parser.add_argument("--policy-calibration", type=json.loads, default={})
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.episodes <= 0 or args.max_steps <= 0 or args.start < 0:
        parser.error("episodes/max-steps must be positive and start nonnegative")
    args.max_episodes = args.max_episodes if args.max_episodes is not None else args.episodes
    if min(args.min_successes, args.min_failures) < 0:
        parser.error("outcome minima must be nonnegative")
    if args.max_episodes < max(args.episodes, args.min_successes + args.min_failures):
        parser.error("max-episodes is too small for the requested outcome quotas")
    args.headless = True
    args.enable_cameras = True
    args.fast_shutdown = int(version("isaacsim").split(".")[0]) >= 6
    if args.fast_shutdown and not args.experience:
        import isaacsim

        args.experience = str(Path(isaacsim.__file__).resolve().parent / "apps/isaacsim.exp.base.python.kit")
    args.multi_gpu = False
    return args


def create_environment(app):
    from src.push_t_data.sim_compat import enable_legacy_physx_imports

    enable_legacy_physx_imports()
    from omegaconf import OmegaConf

    from env.observation_manager.obs_manager import ObsManager
    from src.push_t_data.render_capture import RENDER_CONFIG
    from task.RoboDojo.tasks.push_T import push_T
    from utils.pipeline_utils import process_config, process_randomization

    base = OmegaConf.load("env_cfg/arx_x5.yml")
    config = OmegaConf.create(
        {
            **{name: OmegaConf.load(f"env_cfg/{name}/{filename}.yml") for name, filename in base.config.items()},
            "eval_cfg": base,
            "task_env": OmegaConf.load("task/RoboDojo/config/push_T.yml"),
        }
    )
    config.sim.scene.num_envs = 1
    config.sim.seed = [0]
    config = process_randomization(config)
    config, _ = process_config(config, "push_T")
    config.camera.default_frequency = 25
    config.camera.enable_tiled = False
    config.sim.render = deepcopy(RENDER_CONFIG)

    class CollectorEnvironment(push_T):
        def __init__(self):
            super().__init__(config, app)
            self.success = [True]
            self.end_flag = [False]
            self.obs_manager = ObsManager(deepcopy(base.observation), 1, self.dt, "push_T", {}, [0])
            self.renderer_bootstrapped = False

        def _post_setup_scene(self, sim):
            super()._post_setup_scene(sim)
            self.obs_manager.initialize(self)

        def reset_layout(self, layout, seed):
            if not self.renderer_bootstrapped:
                self._reset_layout_once(layout, seed)
                for _bootstrap_render in range(16):
                    self.render()
                self.renderer_bootstrapped = True
                print("RENDER_BOOTSTRAP discarded initialization; resetting same layout and seed", flush=True)
            self._reset_layout_once(layout, seed)

        def _reset_layout_once(self, layout, seed):
            from src.push_t_data.visual_quality import scene_resources

            self.collection_action_index = 0
            self.collection_success_event = None
            self.success = [True]
            self.end_flag = [False]
            self.scene_manager.layout_manager.set_saved_layout(0, deepcopy(layout))
            super().reset(seed=[seed])
            if len(self.capture_manager.tiled_cameras) != 3 or len(self.capture_manager.tiled_render_products) != 3:
                raise RuntimeError("Expected exactly three live camera render products after reset")
            print("CAMERA_RESOURCES views=3 render_products=3", flush=True)
            self.scene_resource_report = scene_resources(self.scene_manager)
            print("SCENE_RESOURCES " + json.dumps(self.scene_resource_report), flush=True)
            self.scene_manager.apply_saved_poses(env_idx_list=[0])
            stable, unstable = self.scene_manager.layout_manager.check_layout_stability(self)
            if not stable or unstable:
                raise RuntimeError(f"Unstable initialization: {unstable}")
            for warmup_step in range(200):
                self.sim_step(render=warmup_step < 199)
            self.obs_manager.desc_manager.update_env_seeds([seed])
            self.obs_manager.reset()
            self.robot_manager.set_origin_endpose()
            self.robot_manager.set_robot_init_state()
            self.reward_manager.init_state()
            self.run_reward()

    return CollectorEnvironment()


def observe(env, frame_index):
    import numpy as np

    from env.camera_manager.capture.camera_view import CameraBufferNotReady
    from env.camera_manager.capture.render_sync import dispatcher_time, zero_delay_active
    from src.push_t_data.render_capture import render_capture

    if not zero_delay_active():
        raise RuntimeError("Zero-delay rendering settings are not active")
    before_dispatch = dispatcher_time()
    before_physics = env.sim.sim.current_time
    for _ in range(env.obs_manager.capture_render_max_passes):
        render_capture(env)
        after_dispatch = dispatcher_time()
        if after_dispatch is not None and after_dispatch != before_dispatch:
            try:
                observation = env.obs_manager.get_obs(env_idx_list=[0])[0]
            except CameraBufferNotReady:
                continue
            break
    else:
        raise RuntimeError("Camera dispatcher did not deliver a fresh frame")
    if env.sim.sim.current_time != before_physics:
        raise RuntimeError("Rendering advanced physics")
    if frame_index == 0:
        from src.push_t_data.render_quality import settle_reset_observation

        observation, env.reset_render_report = settle_reset_observation(env, observation)
    manager = env.robot_manager
    eef, gripper, joints, velocities = [], [], [], []
    for index, robot in enumerate(manager.robot_list):
        eef.extend(manager.get_real_endpose(robot, [0])[0])
        gripper.extend(manager.get_end_effector_real_val(robot, [0])[0])
        articulation = manager.robot_key[index]
        indices = robot.arm_joint_indices + robot.gripper_joint_indices
        joints.extend(articulation.data.joint_pos[0, indices].cpu().numpy())
        velocities.extend(articulation.data.joint_vel[0, indices].cpu().numpy())
    layout = env.scene_manager.layout_manager
    object_pose = np.concatenate(layout.get_instance_pose(0, label="t"))
    target_pose = np.concatenate(layout.get_instance_pose(0, label="target_t"))
    return {
        "time": frame_index / 25,
        "physics_time": before_physics,
        "camera_dispatch_time": after_dispatch,
        "eef_pose": eef,
        "gripper_position": gripper,
        "joint_position": joints,
        "joint_velocity": velocities,
        "object_pose": object_pose,
        "target_pose": target_pose,
    }, {name: camera["color"] for name, camera in observation["vision"].items()}


def apply_action(env, poses):
    import numpy as np

    from env.robot_manager.control_manager import MetaControl
    from src.push_t_data.success_events import capture_success_event

    manager = env.robot_manager
    control = {}
    ik_success = []
    for robot, pose in zip(manager.robot_list, poses, strict=True):
        solution = manager.solve_ik(pose, 0, robot)
        valid = solution["status"] == "Success"
        ik_success.append(valid)
        current = manager.get_joint(robot, [0])[0]
        target = solution["joint_value"] if valid else current
        target = current + np.clip(np.asarray(target) - current, -0.06, 0.06)
        control[manager.process_name(robot.arm_name)] = {"position": target, "velocity": np.zeros(len(target))}
        control[manager.process_name(robot.gripper_name)] = {"position": [robot.gripper_scale[0]]}
    applied = []
    for substep_index in range(10):
        env.step([MetaControl(control)])
        targets = []
        for index, robot in enumerate(manager.robot_list):
            indices = robot.arm_joint_indices + robot.gripper_joint_indices
            targets.extend(manager.robot_key[index].data.joint_pos_target[0, indices].cpu().numpy())
        applied.append(targets)
        env.sim_step(render=False)
        env.reward_manager.step(env_idx_list=[0])
        if env.collection_success_event is None and env.success[0] and not env.reward_manager.check_list[0]:
            event = capture_success_event(env, env.collection_action_index, substep_index)
            if event["official_reward"] == 1.0:
                env.collection_success_event = event
    env.collection_action_index += 1
    return np.asarray(applied), np.asarray(ik_success, dtype=bool)


def collect(args, app):
    import numpy as np

    from src.push_t_data.io import EpisodeWriter, validate_episode
    from src.push_t_data.outcomes import OutcomeQuota
    from src.push_t_data.policy import PushController
    from src.push_t_data.render_capture import capture_metadata
    from src.push_t_data.sampling import (
        calibrated_parameters,
        select_layout_index,
        validate_policy_calibration,
        validate_priority_indices,
    )
    from src.push_t_data.success_events import SEMANTICS

    layouts = sorted(Path("Assets/Eval_Layout/RoboDojo/arx_x5").glob("*/push_T_[0-9]*.json"))
    if not layouts:
        raise FileNotFoundError("Download public push_T layouts first")
    validate_priority_indices(args.priority_layout_indices, len(layouts))
    validate_policy_calibration(args.policy_calibration, args.priority_layout_indices)
    sampling = {
        "mode": "two_priority_one_uniform" if args.priority_layout_indices else "uniform_cycle",
        "priority_layout_indices": args.priority_layout_indices,
        "policy_calibration": args.policy_calibration,
        "catalog_sha256": hashlib.sha256(
            json.dumps(
                [(str(path), hashlib.sha256(path.read_bytes()).hexdigest()) for path in layouts], sort_keys=True
            ).encode()
        ).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    collector_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    from src.push_t_data.provenance import dependency_provenance

    submodules, submodule_changes = dependency_provenance()
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    runtime_versions = {
        "python": platform.python_version(),
        **{package: version(package) for package in ("isaacsim", "isaaclab", "torch", "numpy", "warp-lang")},
    }
    quota = OutcomeQuota(args.episodes, args.min_successes, args.min_failures)
    completed = sorted(args.output.glob("episode_*"))
    expected_names = [f"episode_{index:06d}" for index in range(args.start, args.start + len(completed))]
    if [path.name for path in completed] != expected_names or len(completed) > args.max_episodes:
        raise ValueError("Resume requires a contiguous prefix of completed episode IDs within max-episodes")
    successes = 0
    for episode in range(args.start, args.start + len(completed)):
        destination = args.output / f"episode_{episode:06d}"
        validate_episode(destination)
        metadata = json.loads((destination / "metadata.json").read_text())
        successes += bool(metadata["success"])
        expected = {
            "episode_id": episode,
            "seed": args.seed + episode,
            "max_steps": args.max_steps,
            "collector_sha256": collector_hash,
            "repository_commit": revision,
            "submodule_commits": submodules,
            "submodule_changes_sha256": submodule_changes,
            "collector_source_hashes": source_hashes,
            "runtime_versions": runtime_versions,
            "layout": json.loads(
                layouts[
                    select_layout_index(episode, args.start, len(layouts), args.priority_layout_indices)
                ].read_text()
            ),
            "layout_sampling": sampling,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"Resume mismatch in {key}: {destination}; use a new run directory")
    total = len(completed)
    if quota.met(total, successes):
        print("COLLECTION_ALREADY_COMPLETE", flush=True)
        return
    if total == args.max_episodes:
        quota.require(total, successes)
    env = create_environment(app)
    if not np.isclose(env.dt * 10, 1 / 25):
        raise ValueError("Collector requires 250 Hz physics and 25 Hz controls")
    try:
        for episode in range(args.start + total, args.start + args.max_episodes):
            episode_started = time.monotonic()
            seed = args.seed + episode
            layout_index = select_layout_index(episode, args.start, len(layouts), args.priority_layout_indices)
            layout_path = layouts[layout_index]
            layout = json.loads(layout_path.read_text())
            env.reset_layout(layout, seed)
            state, images = observe(env, 0)
            controller = PushController(state, seed, args.max_steps)
            parameters = calibrated_parameters(layout_index, seed, args.policy_calibration)
            if parameters is not None:
                controller.speed, controller.lateral_bias = parameters
            metadata = {
                "schema": "robodojo-push-t-transitions-v1",
                "episode_id": episode,
                "seed": seed,
                "layout_path": str(layout_path),
                "layout": layout,
                "layout_index": layout_index,
                "layout_sampling": sampling,
                "fps": 25,
                "scene_resources": env.scene_resource_report,
                "reset_render": env.reset_render_report,
                "render_capture": capture_metadata(),
                "success_semantics": SEMANTICS,
                "physics_dt": env.dt,
                "substeps": 10,
                "prompt": env.gen_instruction(0)[0],
                "repository_commit": revision,
                "collector_sha256": collector_hash,
                "submodule_commits": submodules,
                "submodule_changes_sha256": submodule_changes,
                "collector_source_hashes": source_hashes,
                "runtime_versions": runtime_versions,
                "assets_revision": "1127bda94f172d757386945af3eb3a6754d1a4e6",
                "policy": controller.name,
                "eef_frame": "environment_world_link6",
                "eef_order": "left_xyz_qwqxqyqz,right_xyz_qwqxqyqz",
                "length_unit": "m",
                "action_semantics": (
                    "absolute commanded EEF pose, closed grippers; joint slew limited 0.06 rad/control step"
                ),
                "state_semantics": "measured simulation state, NOT requested targets",
                "controller_speed_m_s": controller.speed,
                "controller_lateral_bias_m": controller.lateral_bias,
                "controller_calibration": args.policy_calibration.get(str(layout_index)),
                "max_steps": args.max_steps,
                "retention": "all_success_and_failure",
                "collection_quota": vars(quota),
                "joint_names": [
                    [articulation.joint_names[index] for index in robot.arm_joint_indices + robot.gripper_joint_indices]
                    for articulation, robot in zip(
                        env.robot_manager.robot_key, env.robot_manager.robot_list, strict=True
                    )
                ],
            }
            writer = EpisodeWriter(args.output, episode, metadata)
            try:
                writer.observe(state, images)
                for step in range(args.max_steps):
                    command = controller.action(state, step)
                    joint_targets, ik_success = apply_action(env, command)
                    success = bool(env.reward_manager.get_reward()[0])
                    failure = not env.success[0]
                    terminated = success or failure
                    truncated = step + 1 == args.max_steps and not terminated
                    writer.transition(
                        {
                            "action_eef_pose": command.reshape(-1),
                            "action_gripper": np.zeros(2),
                            "applied_joint_targets": joint_targets,
                            "ik_success": ik_success,
                            "terminated": terminated,
                            "truncated": truncated,
                            "success": success,
                        }
                    )
                    state, images = observe(env, step + 1)
                    writer.observe(state, images)
                    if (step + 1) % 50 == 0:
                        print(
                            "EPISODE_PROGRESS "
                            + json.dumps(
                                {
                                    "episode_id": episode,
                                    "actions": step + 1,
                                    "simulation_seconds": (step + 1) / 25,
                                    "wall_seconds": round(time.monotonic() - episode_started, 2),
                                }
                            ),
                            flush=True,
                        )
                    if terminated or truncated:
                        reason = "success" if success else "task_failure" if failure else "time_limit"
                        writer.metadata["collection_wall_seconds"] = round(time.monotonic() - episode_started, 2)
                        writer.metadata["success_event"] = env.collection_success_event
                        result = writer.finish(success, reason)
                        total += 1
                        successes += int(success)
                        print(
                            "EPISODE_COMPLETE "
                            + json.dumps(
                                {
                                    key: result[key]
                                    for key in ("episode_id", "success", "termination_reason", "num_actions")
                                }
                            ),
                            flush=True,
                        )
                        print(
                            "OUTCOME_PROGRESS "
                            + json.dumps(
                                {
                                    "episodes": total,
                                    "success": successes,
                                    "failure": total - successes,
                                    "quota_met": quota.met(total, successes),
                                }
                            ),
                            flush=True,
                        )
                        break
            finally:
                writer.close()
            if quota.met(total, successes):
                break
        quota.require(total, successes)
    finally:
        env.close()


def main():
    from isaaclab.app import AppLauncher

    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    args = arguments()
    add_zero_delay_kit_args(args)
    launcher = AppLauncher(args)
    failed = True
    try:
        collect(args, launcher.app)
        failed = False
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if args.fast_shutdown:
            launcher.app.close(exit_code=int(failed))
        else:
            launcher.app.close()


if __name__ == "__main__":
    main()
