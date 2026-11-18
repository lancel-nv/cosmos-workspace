import numpy as np
import transforms3d as transforms


class PushController:
    name = "privileged_feedback_push_v3"

    def __init__(self, observation, seed, max_steps):
        self.rng = np.random.default_rng(seed)
        self.home = np.asarray(observation["eef_pose"], dtype=float).reshape(2, 7).copy()
        self.command = self.home.copy()
        self.max_steps = max_steps
        self.speed = float(self.rng.uniform(0.175, 0.185))
        self.lateral_bias = float(self.rng.uniform(-0.0005, 0.0005))
        self.phase = "plan"
        self.arm = 1
        self.queue = []
        self.dwell = 0
        self.push_steps = 0
        self.tracking_failures = 0

    def return_home(self, poses):
        above = poses[self.arm].copy()
        if np.linalg.norm(above[:2] - self.home[self.arm, :2]) > 0.38:
            above[:2] = above[:2] * 0.75 + self.home[self.arm, :2] * 0.25
        above[2] = 1.02
        self.queue = [above, self.home[self.arm].copy()]
        self.phase = "home"
        self.dwell = 0
        self.tracking_failures = 0

    def action(self, observation, step):
        poses = np.asarray(observation["eef_pose"], dtype=float).reshape(2, 7)
        block = np.asarray(observation["object_pose"], dtype=float)
        target = np.asarray(observation["target_pose"], dtype=float)
        distance = np.linalg.norm(target[:2] - block[:2])
        angle = 2 * np.arccos(np.clip(abs(np.dot(block[3:], target[3:])), 0, 1))
        block_yaw = transforms.euler.quat2euler(block[3:])[2]
        target_yaw = transforms.euler.quat2euler(target[3:])[2]
        yaw_error = (target_yaw - block_yaw + np.pi) % (2 * np.pi) - np.pi
        if (step >= self.max_steps - 50 or (distance < 0.004 and angle < np.deg2rad(5))) and self.phase != "home":
            self.return_home(poses)
        if self.phase == "plan":
            if not self.queue:
                self.arm = int(block[0] >= 0) if step == 0 else self.arm
            direction = target[:2] - block[:2]
            direction /= max(np.linalg.norm(direction), 1e-8)
            self.direction = direction
            perpendicular = np.array([-direction[1], direction[0]])
            center = block[:2] - 0.01 * np.array([np.cos(block_yaw), np.sin(block_yaw)])
            offset = self.lateral_bias - np.clip(yaw_error * 0.025, -0.02, 0.02)
            behind = center - direction * 0.065 + perpendicular * offset
            quaternion = transforms.euler.euler2quat(0, np.pi / 2, np.arctan2(direction[1], direction[0]) % np.pi)
            above = np.r_[behind, 1.00, quaternion]
            contact = np.r_[behind, 0.9302, quaternion]
            lift = poses[self.arm].copy()
            lift[2] = 1.00
            self.queue = [lift, above, contact]
            self.phase = "approach"
            self.dwell = 0
        if self.queue:
            goal = self.queue[0]
            delta = goal[:3] - self.command[self.arm, :3]
            norm = np.linalg.norm(delta)
            self.command[self.arm, :3] += delta * min(1, self.speed / 25 / max(norm, 1e-9))
            current = self.command[self.arm, 3:]
            goal_quat = goal[3:] * (1 if np.dot(current, goal[3:]) >= 0 else -1)
            rotation = 2 * np.arccos(np.clip(np.dot(current, goal_quat), 0, 1))
            ratio = min(1, 0.10 / max(rotation, 1e-9))
            quaternion = current * (1 - ratio) + goal_quat * ratio
            self.command[self.arm, 3:] = quaternion / np.linalg.norm(quaternion)
            position_error = np.linalg.norm(poses[self.arm, :3] - goal[:3])
            orientation_close = abs(np.dot(poses[self.arm, 3:], goal[3:])) > np.cos(np.deg2rad(5) / 2)
            self.dwell = self.dwell + 1 if position_error < 0.004 and orientation_close else 0
            not_tracking = norm < 0.001 and (position_error > 0.02 or not orientation_close)
            self.tracking_failures = self.tracking_failures + 1 if not_tracking else 0
            if self.tracking_failures >= 30 and self.phase != "home":
                self.return_home(poses)
            elif self.dwell >= 3:
                self.queue.pop(0)
                self.dwell = 0
                self.tracking_failures = 0
                if not self.queue and self.phase == "approach":
                    self.phase = "push"
                    self.push_steps = 0
        elif self.phase == "push":
            remaining = target[:2] - block[:2]
            projection = np.dot(remaining, self.direction)
            if projection < 0.003 or self.push_steps >= 110:
                self.phase = "plan"
            else:
                speed = min(0.08, max(0.018, distance * 0.8))
                self.command[self.arm, :2] += self.direction * speed / 25
                self.push_steps += 1
        return self.command.copy()
