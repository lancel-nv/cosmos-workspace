from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC, datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys

from src.push_t_data.outcomes import OutcomeQuota
from src.push_t_data.provenance import dependency_provenance
from src.push_t_data.sampling import validate_policy_calibration, validate_priority_indices

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("smoke", "collect", "validate", "export", "curate", "augment")


def timestamp():
    return datetime.now(UTC).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def resolve_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path):
    config = read_json(path)
    defaults = read_json(ROOT / "configs/push_t_collection.json")
    optional = {"priority_layout_indices", "policy_calibration"}
    if set(config) - set(defaults) or set(defaults) - set(config) - optional or config["schema_version"] != 1:
        raise ValueError("Config keys/schema differ from configs/push_t_collection.json")
    validate_priority_indices(config.get("priority_layout_indices", []))
    validate_policy_calibration(config.get("policy_calibration", {}), config.get("priority_layout_indices", []))
    for key in ("episodes", "max_steps", "smoke_episodes", "max_episodes"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "start", "min_successes", "min_failures"):
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["max_steps"] <= 100:
        raise ValueError("Use max_steps > 100 to allow pushing and the final 50-step return-home window")
    if config["max_episodes"] < max(config["episodes"], config["min_successes"] + config["min_failures"]):
        raise ValueError("max_episodes cannot satisfy the requested outcome quotas")
    for section in ("quality", "curator"):
        if set(config[section]) != set(defaults[section]):
            raise ValueError(f"Unknown or missing {section} keys")
    for key, value in config["quality"].items():
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Invalid quality threshold: {key}")
    for key in ("datasets_root", "collection_python", "lerobot_python", "curator_root"):
        if key.endswith("_python"):
            config[key] = os.path.abspath(ROOT / Path(config[key]).expanduser())
        else:
            config[key] = str(resolve_path(config[key]))
    return config


class Workflow:
    def __init__(self, config, name):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
            raise ValueError("Run name must use 1-80 ASCII letters, digits, underscores or hyphens")
        self.config = config
        self.name = name
        self.root = Path(config["datasets_root"]) / name
        self.curator = Path(config["curator_root"])
        self.sim_python = config["collection_python"]
        self.export_python = config["lerobot_python"]
        self.curator_python = str(self.curator / ".pixi/envs/default/bin/python")
        files = sorted((ROOT / "src/push_t_data").glob("*.py"))
        files.extend([ROOT / "configs/push_t_curator.yaml", ROOT / "scripts/push_t_pipeline.sh"])
        code = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        changes = subprocess.check_output(["git", "diff", "HEAD", "--", "env", "task", "utils", "env_cfg"], cwd=ROOT)
        submodules, submodule_changes = dependency_provenance()
        self.snapshot = {
            "config": config,
            "run": name,
            "repository_commit": revision,
            "native_changes_sha256": hashlib.sha256(changes).hexdigest(),
            "submodules": submodules,
            "submodule_changes_sha256": submodule_changes,
            "source_hashes": code,
        }
        self.fingerprint = digest(self.snapshot)

    @contextmanager
    def locked(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".workflow.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process owns this run; do not launch duplicate collectors") from error
            snapshot_path = self.root / "run.json"
            if snapshot_path.exists():
                if read_json(snapshot_path)["fingerprint"] != self.fingerprint:
                    raise RuntimeError("Config/code changed for this run. Restore them or choose a NEW --run name")
            else:
                write_json(snapshot_path, {**self.snapshot, "fingerprint": self.fingerprint, "created_at": timestamp()})
                write_json(self.root / "config.resolved.json", self.config)
            (self.root / "logs").mkdir(exist_ok=True)
            yield

    def state(self):
        path = self.root / "state.json"
        return read_json(path) if path.exists() else {}

    def status(self):
        episodes = sorted((self.root / "raw").glob("episode_*"))
        outcomes = {"success": 0, "task_failure": 0, "time_limit": 0}
        invalid = []
        actions = 0
        video_seconds = 0
        for path in episodes:
            try:
                metadata = read_json(path / "metadata.json")
                reason = metadata["termination_reason"]
                count = metadata["num_actions"]
                if (
                    reason not in outcomes
                    or type(metadata["success"]) is not bool
                    or metadata["success"] != (reason == "success")
                    or type(count) is not int
                    or count <= 0
                    or metadata["num_observations"] != count + 1
                    or metadata["fps"] != 25
                ):
                    raise ValueError("Invalid completion metadata")
            except (OSError, ValueError, KeyError, TypeError) as error:
                invalid.append({"episode": path.name, "error": str(error)})
                continue
            outcomes[reason] += 1
            actions += count
            video_seconds += (count + 1) / 25
        total = sum(outcomes.values())
        successes = outcomes["success"]
        failures = total - successes
        quota = OutcomeQuota(self.config["episodes"], self.config["min_successes"], self.config["min_failures"])
        return {
            "root": str(self.root),
            "stages": self.state(),
            "completed_raw_episodes": total,
            "partial_raw_episodes": len(list((self.root / "raw").glob(".episode_*.partial"))),
            "progress_source": "metadata_only; run validate for full NPZ/video validation",
            "invalid_episode_metadata": invalid,
            "outcomes": {**outcomes, "failure": failures},
            "quota": vars(quota),
            "quota_met": not invalid and quota.met(total, successes),
            "remaining_minima": {
                "episodes": max(0, quota.min_episodes - total),
                "successes": max(0, quota.min_successes - successes),
                "failures": max(0, quota.min_failures - failures),
            },
            "actions": actions,
            "fps": 25,
            "video_seconds_per_view": round(video_seconds, 2),
            "video_seconds_all_three_views": round(video_seconds * 3, 2),
        }

    def record(self, stage, status, **extra):
        state = self.state()
        state[stage] = {"status": status, "updated_at": timestamp(), **extra}
        write_json(self.root / "state.json", state)

    def require(self, stage):
        if self.state().get(stage, {}).get("status") != "complete":
            raise RuntimeError(f"Run stage '{stage}' first")
        self.check_signature(stage)

    def signature(self, stage):
        folder = (
            self.root
            / {
                "smoke": "smoke",
                "collect": "raw",
                "validate": "raw",
                "export": "lerobot",
                "curate": "curator",
                "augment": "curator_complete",
            }[stage]
        )
        files = []
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name != "quality_report.json":
                stat = path.stat()
                files.append((str(path.relative_to(folder)), stat.st_size, stat.st_mtime_ns))
        return digest(files)

    def check_signature(self, stage):
        if not self.artifact(stage).exists() or self.state()[stage].get("signature") != self.signature(stage):
            raise RuntimeError(f"Completed {stage} output changed or is missing; refusing stale reuse")

    def environment(self, python):
        environment = os.environ.copy()
        for key in (
            "__EGL_VENDOR_LIBRARY_DIRS",
            "__EGL_VENDOR_LIBRARY_FILENAMES",
            "CUDA_VISIBLE_DEVICES",
            "PYTHONHOME",
        ):
            environment.pop(key, None)
        environment.update(
            OMNI_KIT_ACCEPT_EULA="YES",
            PYTHONNOUSERSITE="1",
            PYTHONPATH=f"{ROOT}:{ROOT / 'XPolicyLab'}",
            PATH=f"{Path(python).parent}:{environment.get('PATH', '')}",
        )
        if Path("/usr/share/vulkan/icd.d/nvidia_icd.json").exists():
            environment["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
        return environment

    def run_command(self, command, log):
        print(f"COMMAND {shlex.join(command)}\nLOG {log}", flush=True)
        with log.open("a") as output:
            output.write(f"{timestamp()} {shlex.join(command)}\n")
            output.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=self.environment(command[0]),
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                result = process.wait()
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            if result:
                raise RuntimeError(f"Command exited {result}; inspect {log}")

    def commands(self, stage):
        config = self.config
        if stage in ("smoke", "collect"):
            destination = self.root / ("smoke" if stage == "smoke" else "raw")
            count = config["smoke_episodes"] if stage == "smoke" else config["episodes"]
            return [
                [
                    self.sim_python,
                    "-u",
                    "-m",
                    "src.push_t_data.collect",
                    "--output",
                    str(destination),
                    "--episodes",
                    str(count),
                    "--start",
                    str(config["start"]),
                    "--seed",
                    str(config["seed"]),
                    "--max-steps",
                    str(config["max_steps"]),
                    *(
                        ["--policy-calibration", json.dumps(config["policy_calibration"], sort_keys=True)]
                        if config.get("policy_calibration")
                        else []
                    ),
                    *(
                        ["--priority-layout-indices", *map(str, config["priority_layout_indices"])]
                        if config.get("priority_layout_indices")
                        else []
                    ),
                    *(
                        []
                        if stage == "smoke"
                        else [
                            "--min-successes",
                            str(config["min_successes"]),
                            "--min-failures",
                            str(config["min_failures"]),
                            "--max-episodes",
                            str(config["max_episodes"]),
                        ]
                    ),
                ],
                self.validation_command(destination, smoke=stage == "smoke"),
            ]
        if stage == "validate":
            return [self.validation_command(self.root / "raw")]
        if stage == "export":
            return [
                [
                    self.export_python,
                    "-u",
                    "-m",
                    "src.push_t_data.export_lerobot",
                    "--input",
                    str(self.root / "raw"),
                    "--output",
                    str(self.root / "lerobot"),
                    "--repo-id",
                    f"local/{self.name}",
                ]
            ]
        if stage == "curate":
            container_run = f"/datasets/{self.name}"
            return [
                [
                    "bash",
                    "-c",
                    'source "$1/.pixi/activate-local.sh"; shift; cosmos-curator "$@"',
                    "_",
                    str(self.curator),
                    "local",
                    "launch",
                    "--image-name",
                    config["curator"]["image_name"],
                    "--image-tag",
                    config["curator"]["image_tag"],
                    "--curator-path",
                    str(self.curator),
                    "--pixi-path",
                    str(self.curator),
                    "--extra-volumes",
                    f"{config['datasets_root']}:/datasets,{ROOT}:/robodojo:ro",
                    "--",
                    "pixi",
                    "run",
                    "--as-is",
                    "-e",
                    "default",
                    "python",
                    "-m",
                    "cosmos_curator.pipelines.pipeline_runtime",
                    "/robodojo/configs/push_t_curator.yaml",
                    "--set",
                    "input.uris=" + json.dumps([container_run + "/lerobot"]),
                    "--set",
                    f"output.media_root={container_run}/curator",
                    "--set",
                    f"split.max_keep_per_description={max(self.episode_count(), 1000)}",
                    "--set",
                    f"output.video_bitrate={config['curator']['video_bitrate']}",
                ],
                [
                    self.curator_python,
                    "-c",
                    "import sys, lance; from collections import Counter; "
                    "rows=lance.dataset(sys.argv[1]).to_table().to_pylist(); "
                    "counts=Counter(row['span_group_id'] for row in rows); "
                    "assert len(counts)==int(sys.argv[2]) and all(value==3 for value in counts.values()); "
                    "assert all(not row['error_stage'] for row in rows); print('CURATOR_CLIPS_OK', len(rows))",
                    str(self.root / "curator/lance/clips.lance"),
                    str(self.episode_count()),
                ],
            ]
        if stage == "augment":
            return [
                [
                    self.curator_python,
                    "-m",
                    "src.push_t_data.augment_curator",
                    "--lance",
                    str(self.root / "curator/lance/clips.lance"),
                    "--lerobot",
                    str(self.root / "lerobot"),
                    "--output",
                    str(self.root / "curator_complete"),
                    "--expected-episodes",
                    str(self.episode_count()),
                    "--path-map",
                    f"/datasets={config['datasets_root']}",
                ]
            ]
        raise ValueError(stage)

    def episode_count(self):
        report = self.root / "raw/quality_report.json"
        return read_json(report)["episodes"] if report.exists() else self.config["episodes"]

    def validation_command(self, destination, smoke=False):
        command = [self.sim_python, "-m", "src.push_t_data.validate", str(destination)]
        if smoke:
            return command + ["--expected", str(self.config["smoke_episodes"])]
        return command + [
            "--min-episodes",
            str(self.config["episodes"]),
            "--min-successes",
            str(self.config["min_successes"]),
            "--min-failures",
            str(self.config["min_failures"]),
        ]

    def artifact(self, stage):
        return (
            self.root
            / {
                "smoke": "smoke/quality_report.json",
                "collect": "raw/quality_report.json",
                "validate": "raw/quality_report.json",
                "export": "lerobot/meta/transition_contract.json",
                "curate": "curator/lance/clips.lance",
                "augment": "curator_complete/manifest.jsonl",
            }[stage]
        )

    def check_episodes(self, folder, count):
        actual = sorted(path.name for path in folder.glob("episode_*"))
        expected = [f"episode_{index:06d}" for index in range(self.config["start"], self.config["start"] + count)]
        if actual != expected:
            raise RuntimeError(f"Episode IDs do not match requested range: {folder}")
        if list(folder.glob(".episode_*.partial")):
            raise RuntimeError(f"Incomplete episodes remain in {folder}; inspect then use quarantine")

    def approve(self, reviewed):
        self.require("smoke")
        if not reviewed:
            raise RuntimeError("Watch the three RGB views and verify contact/state alignment, then approve --reviewed")
        report = read_json(self.artifact("smoke"))
        if self.config["min_successes"] > 0 and report["success"] == 0:
            raise RuntimeError("No successful smoke episode; demonstrate genuine task success before batch collection")
        quality = self.config["quality"]
        moving = sum(
            row["object_displacement_max_m"] > quality["min_object_displacement_m"] for row in report["details"]
        )
        if moving / report["episodes"] < quality["min_moving_episode_fraction"]:
            raise RuntimeError("Too few smoke episodes move the object; improve the policy before collecting")
        if any(
            min(row["ik_success_rate_per_arm"]) < quality["min_ik_success_rate_per_arm"] for row in report["details"]
        ):
            raise RuntimeError("Smoke IK rate below threshold; inspect reachability")
        write_json(
            self.root / "smoke_approval.json",
            {
                "fingerprint": self.fingerprint,
                "report_sha256": digest(report),
                "reviewed_at": timestamp(),
                "success": report["success"],
                "failure": report["failure"],
            },
        )
        print("SMOKE_APPROVED; successes AND failures will be retained")

    def check_approval(self):
        self.require("smoke")
        path = self.root / "smoke_approval.json"
        if not path.exists():
            raise RuntimeError("Run smoke, inspect videos, then approve --reviewed before collecting")
        approval = read_json(path)
        if approval["fingerprint"] != self.fingerprint or approval["report_sha256"] != digest(
            read_json(self.artifact("smoke"))
        ):
            raise RuntimeError("Smoke approval is stale")

    def execute(self, stage):
        if stage == "collect":
            self.check_approval()
        prerequisite = {"export": "validate", "curate": "export", "augment": "curate"}.get(stage)
        if prerequisite:
            self.require(prerequisite)
        input_signature = self.state()[prerequisite]["signature"] if prerequisite else None
        if stage != "validate" and self.state().get(stage, {}).get("status") == "complete":
            self.check_signature(stage)
            if self.state()[stage].get("input_signature") != input_signature:
                raise RuntimeError(f"Inputs to completed {stage} changed; use a new run instead of stale output")
            print(f"SKIP completed {stage}")
            return
        folder = {"export": "lerobot", "curate": "curator", "augment": "curator_complete"}.get(stage)
        if folder and (self.root / folder).exists():
            raise RuntimeError(f"Incomplete {folder} exists; inspect logs, then quarantine --stage {stage}")
        log = self.root / "logs" / f"{stage}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.log"
        self.record(stage, "running", log=str(log))
        try:
            for command in self.commands(stage):
                self.run_command(command, log)
            if not self.artifact(stage).exists():
                raise RuntimeError(f"Command returned without producing {self.artifact(stage)}")
            if stage in ("smoke", "collect", "validate"):
                count = self.config["smoke_episodes"] if stage == "smoke" else self.episode_count()
                self.check_episodes(self.root / ("smoke" if stage == "smoke" else "raw"), count)
                if stage != "smoke":
                    report = read_json(self.artifact(stage))
                    OutcomeQuota(
                        self.config["episodes"], self.config["min_successes"], self.config["min_failures"]
                    ).require(count, report["success"])
            self.record(
                stage,
                "complete",
                log=str(log),
                artifact=str(self.artifact(stage)),
                signature=self.signature(stage),
                input_signature=input_signature,
            )
            print(f"COMPLETE {stage}", flush=True)
        except BaseException as error:
            self.record(stage, "failed", log=str(log), error=str(error))
            raise

    def quarantine(self, stage):
        if self.state().get(stage, {}).get("status") == "complete":
            raise RuntimeError("Refusing to quarantine a completed stage")
        if stage in ("smoke", "collect"):
            folder = self.root / ("smoke" if stage == "smoke" else "raw")
            paths = sorted(folder.glob(".episode_*.partial"))
        else:
            folder = {"export": "lerobot", "curate": "curator", "augment": "curator_complete"}.get(stage)
            if folder is None:
                raise ValueError("No quarantine operation for this stage")
            paths = [self.root / folder] if (self.root / folder).exists() else []
        destination = self.root / "quarantine" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
        for path in paths:
            destination.mkdir(parents=True, exist_ok=True)
            path.rename(destination / path.name)
        self.record(stage, "pending", quarantined=[str(destination / path.name) for path in paths])
        print(f"Quarantined {len(paths)} incomplete artifacts; no files deleted")

    def doctor(self):
        checks = []

        def probe(name, command):
            try:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=self.environment(command[0]),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True,
                )
                checks.append({"check": name, "ok": True, "detail": result.stdout.strip()})
                return result.stdout.strip()
            except (OSError, subprocess.SubprocessError) as error:
                checks.append({"check": name, "ok": False, "detail": str(error)})
                return ""

        sim = probe(
            "simulation packages",
            [
                self.sim_python,
                "-c",
                "import json; from importlib.metadata import version; "
                "names=['isaacsim','isaaclab','torch','numpy','nvidia-curobo']; "
                "print(json.dumps({name:version(name) for name in names}))",
            ],
        )
        probe(
            "LeRobot", [self.export_python, "-c", "from importlib.metadata import version; print(version('lerobot'))"]
        )
        probe("Curator", [self.curator_python, "-c", "import lance, pyarrow; print('OK')"])
        driver = probe("GPU", ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"])
        probe(
            "CUDA",
            [self.sim_python, "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count())"],
        )
        probe(
            "Docker image",
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                f"{self.config['curator']['image_name']}:{self.config['curator']['image_tag']}",
            ],
        )
        for executable in ("ffmpeg", "ffprobe", "pixi"):
            checks.append({"check": executable, "ok": shutil.which(executable) is not None})
        for relative in (
            "Assets/Robots/x5/curobo.yml",
            "Assets/Eval_Layout/RoboDojo/arx_x5/0/push_T_0.json",
            "XPolicyLab/client_server/ws/model_client.py",
        ):
            checks.append({"check": relative, "ok": (ROOT / relative).is_file()})
        if sim and json.loads(sim)["isaacsim"].startswith("5.1") and "595.91.07" in driver:
            checks.append(
                {
                    "check": "native RTX compatibility",
                    "ok": False,
                    "detail": "Known Isaac Sim 5.1 RTX segfault on this driver; use the tested RoboDojo6 environment",
                }
            )
        print(json.dumps(checks, indent=2, ensure_ascii=False))
        if not all(check["ok"] for check in checks):
            raise RuntimeError("Preflight failed; fix the reported environment issues before smoke/collection")


def main():
    parser = argparse.ArgumentParser(description="Repeatable local push_T collection and conversion pipeline")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/push_t_collection.json")
    parser.add_argument("--run", required=True)
    parser.add_argument("command", choices=("plan", "doctor", "status", "approve", "all", "quarantine", *STAGES))
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--stage", choices=STAGES)
    args = parser.parse_args()
    workflow = Workflow(load_config(args.config), args.run)
    if args.command == "plan":
        print(
            json.dumps(
                {
                    "root": str(workflow.root),
                    "fingerprint": workflow.fingerprint,
                    "commands": {stage: workflow.commands(stage) for stage in STAGES},
                },
                indent=2,
            )
        )
    elif args.command == "doctor":
        workflow.doctor()
    elif args.command == "status":
        print(json.dumps(workflow.status(), indent=2, ensure_ascii=False))
    else:
        with workflow.locked():
            if args.command == "approve":
                workflow.approve(args.reviewed)
            elif args.command == "quarantine":
                if args.stage is None:
                    parser.error("quarantine requires --stage")
                workflow.quarantine(args.stage)
            elif args.command == "all":
                workflow.check_approval()
                workflow.doctor()
                for stage in STAGES[1:]:
                    workflow.execute(stage)
            else:
                workflow.execute(args.command)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
