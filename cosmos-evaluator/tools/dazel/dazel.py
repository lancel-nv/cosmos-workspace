# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dazel: A tool to execute Bazel commands in a Docker container."""

from __future__ import annotations

import collections
import dataclasses
import functools
import getpass
import hashlib
import json
import logging
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import types
from typing import Any, ClassVar

# Python 3.10-compatible substitute for typing.Self (PEP 673, 3.11+). Lets `from_config` correctly type subclass returns
# so `Subclass.from_config()` is inferred as `Subclass`.
# _DI = TypeVar("_DI", bound="DockerInstance")


DAZEL_RC_FILE = ".dazelrc"
DAZEL_RUN_FILE = ".dazel_run"
BAZEL_MODULE_FILE = "MODULE.bazel"

DEFAULT_INSTANCE_NAME = "dazel"
DEFAULT_IMAGE_NAME = "dazel"
DEFAULT_RUN_COMMAND = "/bin/bash"
DEFAULT_DOCKER_COMMAND = "docker"
DEFAULT_LOCAL_DOCKERFILE = "Dockerfile.dazel"
DEFAULT_REMOTE_REPOSITORY = "dazel"
DEFAULT_DIRECTORY = os.getcwd()
DEFAULT_COMMAND = "/usr/bin/bazel"
DEFAULT_VOLUMES: list[str] = []
DEFAULT_PORTS: list[str] = []
DEFAULT_ENV_VARS: list[str] = []
DEFAULT_GPUS: list[str] = []
DEFAULT_PLATFORM = ""
DEFAULT_SHM_SIZE = ""
DEFAULT_NETWORK = "dazel"
DEFAULT_RUN_DEPS: list[Any] = []
DEFAULT_DOCKER_COMPOSE_FILE = ""
DEFAULT_DOCKER_COMPOSE_COMMAND = "docker-compose"
DEFAULT_DOCKER_COMPOSE_PROJECT_NAME = "dazel"
DEFAULT_DOCKER_COMPOSE_SERVICES = ""
DEFAULT_USER = ""
DEFAULT_DOCKER_BUILD_ARGS = ""
DEFAULT_IMAGE_TAG = ""
DEFAULT_DAZEL_FILES_WATCH: list[str] = []

DEFAULT_DELEGATED_VOLUME = True
DEFAULT_BAZEL_USER_OUTPUT_ROOT = os.path.expanduser(f"~/.cache/bazel/_bazel_{getpass.getuser()}")
TEMP_BAZEL_OUTPUT_USER_ROOT = f"/var/bazel/workspace/_bazel_{getpass.getuser()}"
DEFAULT_BAZEL_USER_OUTPUT_PATHS: list[str] = ["external", "action_cache", "execroot"]
DEFAULT_BAZEL_RC_FILE = ""
DEFAULT_DOCKER_RUN_PRIVILEGED = False
DEFAULT_WORKSPACE_HEX = False

DOCKER_SPECIAL_NETWORK_NAMES: list[str] = ["host", "bridge", "none"]

# Docker engine container/object name spec: must start with [a-zA-Z0-9] and may contain [a-zA-Z0-9_.-]. Used to validate
# DAZEL_RUN_DEPS entry names before they flow into `docker exec`/`docker inspect`/`docker logs` argv slots, so a
# malformed config produces a clear error up-front instead of an opaque subprocess failure.
DOCKER_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

# Default timeout for short Docker plumbing calls (ps, inspect, network ls, image ls). Long-running operations (build,
# pull, container start, run-deps, send_command) pass timeout=None explicitly so the user controls duration.
DEFAULT_DOCKER_TIMEOUT_S = 30

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("dazel")


@functools.cache
def _command_exists(cmd: str) -> bool:
    """Cached `shutil.which` - called several times per invocation per command."""
    return shutil.which(cmd) is not None


@functools.cache
def _find_workspace_directory(start_dir: str) -> str:
    """Walk up from `start_dir` until MODULE.bazel is found. Cached on the start path."""
    directory = os.path.realpath(start_dir)
    root_dir = os.path.join(os.path.splitdrive(os.getcwd())[0], os.sep)
    while directory and directory != root_dir and not os.path.exists(os.path.join(directory, BAZEL_MODULE_FILE)):
        directory = os.path.dirname(directory)
    if not os.path.exists(os.path.join(directory, BAZEL_MODULE_FILE)):
        raise FileNotFoundError(f"ERROR: No {BAZEL_MODULE_FILE} file found!")
    return directory


def _coerce_iterable(value: str | collections.abc.Iterable[str] | None, what: str) -> list[str]:
    """Accept None / comma-separated str / any iterable; return list[str].

    Centralizes the "DAZEL_X must be comma-separated string or iterable" parse + error message that used to be
    duplicated four times across `_add_volumes` / `_add_ports` / `_add_env_vars` / `_add_gpus`. Empty values (None, "",
    []) collapse to an empty list.
    """
    if not value:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, collections.abc.Iterable):
        return [str(v) for v in value]
    raise RuntimeError(f"{what} must be a comma-separated string or python iterable of strings")


def _docker_repeated_flag(flag: str, items: collections.abc.Iterable[str]) -> list[str]:
    """Render a list as repeated docker flags, e.g. -v A -v B -v C."""
    out: list[str] = []
    for v in items:
        out += [flag, v]
    return out


def _shell_repeated_flag(flag: str, items: collections.abc.Iterable[str]) -> str:
    """Render a list as a quoted shell string of repeated flags, e.g. '-v "A" -v "B"'.

    Used only by `send_command`, which intentionally goes through `shell=True` so it can append `> /dev/null 2>&1` for
    quiet mode. Every other docker invocation builds an argv list via `_docker_repeated_flag` and runs with
    `shell=False` to eliminate quoting/injection risk.
    """
    items = list(items)
    if not items:
        return ""
    return f'{flag} "%s"' % f'" {flag} "'.join(items)


def _validate_docker_name(name: str) -> str:
    """Validate a Docker container name against the engine's allowed character set."""
    if not DOCKER_NAME_RE.match(name):
        raise RuntimeError(
            f"DAZEL_RUN_DEPS entry name {name!r} is not a valid Docker name (must match [a-zA-Z0-9][a-zA-Z0-9_.-]*)"
        )
    return name


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ReadyProbe:
    """Optional readiness probe spec for a DAZEL_RUN_DEPS sidecar.

    All fields optional: the probe short-circuits when ``cmd`` is empty.

    Probe spec (mirrored in `_wait_for_ready_probe`):
        cmd:                command to ``docker exec`` inside the container; rc 0 means ready
        timeout_s:          total budget before giving up (default 240s - sized to cover a first-time
                            apt-install + initdb + dump restore in the local-db case).
        interval_s:         poll interval (default 1s)
        progress_every_s:   how often to print "still waiting" progress (default 5s)
        description:        human-readable noun phrase included in progress lines
    """

    cmd: tuple[str, ...] = ()
    timeout_s: float = 240.0
    interval_s: float = 1.0
    progress_every_s: float = 5.0
    description: str = "ready"

    @classmethod
    def from_raw(cls, dep_name: str, raw: dict[str, Any] | None) -> ReadyProbe:
        """Build a ReadyProbe from a raw dict (as it appears in a `.dazelrc` entry)."""
        raw = dict(raw or {})
        cmd_raw = raw.get("cmd") or ()
        # Only validate when the probe is actually configured. cmd=None or cmd=[] are valid "no probe" signals that
        # `_wait_for_ready_probe` short-circuits on.
        if cmd_raw and (not isinstance(cmd_raw, list) or not all(isinstance(a, str) for a in cmd_raw)):
            raise RuntimeError(f"DAZEL_RUN_DEPS entry {dep_name!r} ready_probe.cmd must be a list of strings")
        return cls(
            cmd=tuple(cmd_raw),
            timeout_s=float(raw.get("timeout_s", 240)),
            interval_s=float(raw.get("interval_s", 1)),
            progress_every_s=float(raw.get("progress_every_s", 5)),
            description=str(raw.get("description", "ready")),
        )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class RunDep:
    """A normalized DAZEL_RUN_DEPS entry.

    Two source formats are supported by :meth:`parse` for back-compat:

    * Legacy string form: ``"repo/image:tag"`` or ``"repo/image:tag::container_name"`` (no env vars, no volumes).
    * Dict form: ``{"image": ..., "name": ..., "env": {...}, "volumes": [...], "ready_probe": {...}}``.
      ``name`` is optional; ``env`` is a dict of env-var name -> value;
      ``volumes`` is a list of ``"src:dest"`` strings (named volumes or host paths).
      ``-p`` (port) plumbing is intentionally not exposed because under
      ``DAZEL_NETWORK = "host"`` (the default) Docker silently ignores ``-p``.
    """

    image: str
    name: str
    env: dict[str, str] = dataclasses.field(default_factory=dict)
    volumes: list[str] = dataclasses.field(default_factory=list)
    ready_probe: ReadyProbe = dataclasses.field(default_factory=ReadyProbe)

    @property
    def image_name_and_tag(self) -> tuple[str, str]:
        """Split repo/image:tag into (image, tag), defaulting tag to 'latest'."""
        if ":" in self.image:
            name, tag = self.image.rsplit(":", 1)
            return name, tag
        return self.image, "latest"

    @classmethod
    def parse(cls, network: str, raw: dict[str, Any] | str) -> RunDep:
        """Normalize a dict-or-string raw entry into a typed RunDep."""

        def _default_name(image: str) -> str:
            return f"{network}_{image.replace('/', '_').replace(':', '_')}"

        if isinstance(raw, dict):
            if "image" not in raw:
                raise RuntimeError("DAZEL_RUN_DEPS dict entries require an 'image' key")
            image = raw["image"]
            name = _validate_docker_name(raw.get("name") or _default_name(image))
            return cls(
                image=image,
                name=name,
                env=dict(raw.get("env") or {}),
                volumes=list(raw.get("volumes") or []),
                ready_probe=ReadyProbe.from_raw(name, raw.get("ready_probe")),
            )
        if isinstance(raw, str):
            if "::" in raw:
                image, name = raw.split("::", 1)
            else:
                image, name = raw, _default_name(raw)
            return cls(image=image, name=_validate_docker_name(name))
        raise RuntimeError(f"DAZEL_RUN_DEPS entries must be dict or string, got {type(raw).__name__}")


@dataclasses.dataclass(kw_only=True, slots=True)
class DockerInstance:
    """Manages communication and runs commands on associated docker container.

    A DockerInstance can build the image for the container if necessary, run it, set it up through configuration
    variables, and pass on commands to it. It streams the output directly and blocks until the command finishes.

    Construct via :meth:`from_config` (which reads `.dazelrc` + `user.dazelrc` + DAZEL_* environment) or directly with
    keyword arguments. The `raw_*` fields hold the user-supplied "comma-separated string OR iterable" inputs;
    `__post_init__` parses them into the typed list / RunDep storage that the rest of the class consumes.
    """

    instance_name: str = DEFAULT_INSTANCE_NAME
    image_name: str = DEFAULT_IMAGE_NAME
    run_command: str | None = DEFAULT_RUN_COMMAND
    docker_command: str = DEFAULT_DOCKER_COMMAND
    dockerfile: str = DEFAULT_LOCAL_DOCKERFILE
    repository: str | None = DEFAULT_REMOTE_REPOSITORY
    directory: str = DEFAULT_DIRECTORY
    command: str | None = DEFAULT_COMMAND
    network: str = DEFAULT_NETWORK
    docker_compose_file: str = DEFAULT_DOCKER_COMPOSE_FILE
    docker_compose_command: str = DEFAULT_DOCKER_COMPOSE_COMMAND
    docker_compose_project_name: str = DEFAULT_DOCKER_COMPOSE_PROJECT_NAME
    bazel_user_output_root: str = DEFAULT_BAZEL_USER_OUTPUT_ROOT
    bazel_rc_file: str = DEFAULT_BAZEL_RC_FILE
    docker_run_privileged: bool = DEFAULT_DOCKER_RUN_PRIVILEGED
    dazel_run_file: str = DAZEL_RUN_FILE
    workspace_hex: bool = DEFAULT_WORKSPACE_HEX
    delegated_volume: bool = DEFAULT_DELEGATED_VOLUME
    user: str = DEFAULT_USER
    docker_build_args: str = DEFAULT_DOCKER_BUILD_ARGS
    shm_size: str = DEFAULT_SHM_SIZE
    platform: str = DEFAULT_PLATFORM
    image_tag: str = DEFAULT_IMAGE_TAG
    files_watch: list[str] = dataclasses.field(default_factory=list)
    # Filesystem path of the linter wrapper invoked by `dazel lint` / `dazel fix` and the Bazel label of the formatter
    # run during `dazel fix`. Each workspace's .dazelrc supplies these (and a consumer .dazelrc that exec()s another can
    # override either via the standard last-write-wins merge). Default to empty so unset configurations surface as
    # actionable errors at lint/fix time rather than silently invoking the wrong path.
    linter_script: str = ""
    formatter_target: str = ""

    # Raw inputs that get parsed in __post_init__. `_coerce_iterable` handles str | Iterable[str] | None; `RunDep.parse`
    # handles dict | str entries.
    raw_volumes: str | collections.abc.Iterable[str] | None = None
    raw_ports: str | collections.abc.Iterable[str] | None = None
    raw_env_vars: str | collections.abc.Iterable[str] | None = None
    raw_gpus: str | collections.abc.Iterable[str] | None = None
    raw_run_deps: str | collections.abc.Iterable[Any] | None = None
    raw_docker_compose_services: str | collections.abc.Iterable[str] | None = None

    # Derived fields - populated by __post_init__, never set by the caller.
    workspace_hex_digest: str = dataclasses.field(default="", init=False)
    bazel_output_base: str = dataclasses.field(default="", init=False)
    delegated_volume_flag: str = dataclasses.field(default="", init=False)
    remote_directory: str = dataclasses.field(default="", init=False)
    volumes: list[str] = dataclasses.field(default_factory=list, init=False)
    ports: list[str] = dataclasses.field(default_factory=list, init=False)
    env_vars: list[str] = dataclasses.field(default_factory=list, init=False)
    gpus: list[str] = dataclasses.field(default_factory=list, init=False)
    run_deps: list[RunDep] = dataclasses.field(default_factory=list, init=False)
    compose_services: list[str] = dataclasses.field(default_factory=list, init=False)

    # DAZEL_* config key -> dataclass field name (single source of truth).
    _CONFIG_KEY_MAP: ClassVar[dict[str, str]] = {
        "DAZEL_INSTANCE_NAME": "instance_name",
        "DAZEL_IMAGE_NAME": "image_name",
        "DAZEL_RUN_COMMAND": "run_command",
        "DAZEL_DOCKER_COMMAND": "docker_command",
        "DAZEL_DOCKERFILE": "dockerfile",
        "DAZEL_REPOSITORY": "repository",
        "DAZEL_DIRECTORY": "directory",
        "DAZEL_COMMAND": "command",
        "DAZEL_VOLUMES": "raw_volumes",
        "DAZEL_PORTS": "raw_ports",
        "DAZEL_ENV_VARS": "raw_env_vars",
        "DAZEL_GPUS": "raw_gpus",
        "DAZEL_PLATFORM": "platform",
        "DAZEL_SHM_SIZE": "shm_size",
        "DAZEL_NETWORK": "network",
        "DAZEL_RUN_DEPS": "raw_run_deps",
        "DAZEL_DOCKER_COMPOSE_FILE": "docker_compose_file",
        "DAZEL_DOCKER_COMPOSE_COMMAND": "docker_compose_command",
        "DAZEL_DOCKER_COMPOSE_PROJECT_NAME": "docker_compose_project_name",
        "DAZEL_DOCKER_COMPOSE_SERVICES": "raw_docker_compose_services",
        "DAZEL_BAZEL_RC_FILE": "bazel_rc_file",
        "DAZEL_BAZEL_USER_OUTPUT_ROOT": "bazel_user_output_root",
        "DAZEL_DOCKER_RUN_PRIVILEGED": "docker_run_privileged",
        "DAZEL_RUN_FILE": "dazel_run_file",
        "DAZEL_WORKSPACE_HEX": "workspace_hex",
        "DAZEL_DELEGATED_VOLUME": "delegated_volume",
        "DAZEL_USER": "user",
        "DAZEL_DOCKER_BUILD_ARGS": "docker_build_args",
        "DAZEL_IMAGE_TAG": "image_tag",
        "DAZEL_FILES_WATCH": "files_watch",
        "DAZEL_LINTER_SCRIPT": "linter_script",
        "DAZEL_FORMATTER_TARGET": "formatter_target",
    }

    def __post_init__(self) -> None:
        real_directory = os.path.realpath(self.directory)
        self.delegated_volume_flag = ":delegated" if self.delegated_volume else ""
        self.remote_directory = self._get_remote_directory(real_directory, add_drive=True)

        if self.workspace_hex:
            self.workspace_hex_digest = hashlib.md5(real_directory.encode("ascii")).hexdigest()
            self.instance_name = f"{self.instance_name}_{self.workspace_hex_digest}"
            self.docker_compose_project_name = f"{self.docker_compose_project_name}{self.workspace_hex_digest}"
            if os.path.exists(self.dockerfile):
                self.image_name = f"{self.image_name}_{self.workspace_hex_digest}"

        if not self.image_tag:
            # Hash the files that may affect the result of the "docker build" command, and use that as the image tag.
            # The image tag will end up as part of the "docker run" command, and hence anything that could possibly
            # change the result of the "docker build" command will be detected and a new image will be built.
            self.image_tag = self._hash_files(sorted(self.files_watch))

        if self.docker_compose_file and not self._is_predefined_network():
            self.network = f"{self.docker_compose_project_name}_{self.network}"

        self._add_volumes(self.raw_volumes)
        self.ports = _coerce_iterable(self.raw_ports, "DAZEL_PORTS")
        self.env_vars = _coerce_iterable(self.raw_env_vars, "DAZEL_ENV_VARS")
        self.gpus = _coerce_iterable(self.raw_gpus, "DAZEL_GPUS")
        self.compose_services = _coerce_iterable(self.raw_docker_compose_services, "DAZEL_DOCKER_COMPOSE_SERVICES")
        self._build_run_deps()

    @staticmethod
    def _hash_files(files: list[str]) -> str:
        """Compute a hash of the contents of files.

        Note that the order of the files affects the hash.

        Args:
            files: List of paths to files to hash.

        Returns:
            Hash computed from the tracked files.
        """
        m = hashlib.md5()
        for filename in files:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    m.update(f.read())

        return m.hexdigest()

    @classmethod
    def from_config(cls: type[DockerInstance]) -> DockerInstance:
        """Build from .dazelrc + user.dazelrc + DAZEL_* environment, last-write-wins."""
        config = cls._config_from_files()
        config.update(cls._config_from_environment())
        kwargs = {field: config[key] for key, field in cls._CONFIG_KEY_MAP.items() if key in config}
        return cls(**kwargs)

    def send_command(self, args: list[str], quiet: bool = False) -> int:
        term_size = shutil.get_terminal_size()

        user_flag = f"--user={self.user}" if self.user else ""
        tty_flag = "-t" if sys.stdout.isatty() else ""
        priv_flag = "--privileged" if self.docker_run_privileged else ""
        term_env = os.environ.get("TERM", "")
        # `shell=True` is unavoidable here because of the `> /dev/null 2>&1` quiet-mode redirect, so render env vars to
        # a shell-quoted string just for this caller.
        env_vars_shell = _shell_repeated_flag("-e", self.env_vars)
        docker_exec_command = (
            f"{self.docker_command} exec -i -e COLUMNS={term_size.columns} -e LINES={term_size.lines} "
            f"-e TERM={term_env} {env_vars_shell} {tty_flag} {priv_flag} {user_flag} {self.instance_name}"
        )

        if not self.user:
            output_args = (
                f"--output_user_root={TEMP_BAZEL_OUTPUT_USER_ROOT} --output_base={self.bazel_output_base}"
                if self.command and self.bazel_output_base
                else ""
            )
        else:
            output_args = (
                f"--output_user_root={self.bazel_user_output_root}"
                if self.command and self.bazel_user_output_root
                else ""
            )

        bazelrc_flag = f"--bazelrc={self.bazel_rc_file}" if self.bazel_rc_file and self.command else ""
        joined_args = '"' + '" "'.join(args) + '"'
        command = f"{docker_exec_command} {self.command} {bazelrc_flag} {output_args} {joined_args}"

        if quiet:
            command += " > /dev/null 2>&1"

        rc = self._run_with_cancel_forwarding(command, shell=True)

        if sys.platform == "win32":
            self._fix_win_symlink(docker_exec_command)
        return rc

    def _run_with_cancel_forwarding(self, args: list[str] | str, **popen_kwargs: Any) -> int:
        """Run a subprocess while forwarding SIGINT/SIGTERM to the in-container bazel server.

        Without this the bazel server keeps running in the persistent dazel container after the local client dies
        (orphaned actions, leaked GPU memory, cancels that take >30s to actually stop work). On cancel we issue a
        `bazel shutdown` against the container so the in-flight action gets cleanly cancelled, then let the local
        subprocess exit naturally.

        Returns the exit code, with signal-killed children mapped to 128+N (shell convention).
        """
        proc = subprocess.Popen(args, **popen_kwargs)
        cancel_sent = False

        def _forward_cancel(_signum: int, _frame: types.FrameType | None) -> None:
            nonlocal cancel_sent
            if cancel_sent:
                return
            cancel_sent = True
            try:
                subprocess.run(
                    [self.docker_command, "exec", self.instance_name, "bazel", "shutdown"],
                    timeout=10,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.SubprocessError:
                pass

        prev_int = signal.signal(signal.SIGINT, _forward_cancel)
        prev_term = signal.signal(signal.SIGTERM, _forward_cancel) if sys.platform != "win32" else None
        try:
            rc = proc.wait()
        finally:
            signal.signal(signal.SIGINT, prev_int)
            if sys.platform != "win32":
                signal.signal(signal.SIGTERM, prev_term)

        if rc < 0:
            rc = 128 - rc
        return rc

    def _fix_win_symlink(self, docker_exec_command: str) -> None:
        p = pathlib.Path(self.directory)
        for path in list(p.glob("bazel-*")):
            command = f"{docker_exec_command} realpath {path.name}"
            try:
                output = self._run_command(command).strip()
            except subprocess.CalledProcessError:
                logger.info("INFO: Skipping fixing symlink, it already exists.")
            else:
                drive = pathlib.PureWindowsPath(path.drive)
                local_directory = drive.joinpath(pathlib.PurePosixPath(output))
                path.unlink()
                path.symlink_to(local_directory, target_is_directory=True)

    def start(self) -> int:
        """Starts the dazel docker container."""
        rc = 0

        # Verify that the docker executable exists.
        if not self._docker_exists():
            logger.error("ERROR: Docker executable could not be found!")
            return 1

        # Build or pull the relevant dazel image.
        if os.path.exists(self.dockerfile):
            rc = self._build()
        else:
            rc = self._pull()
            # If we have the image, don't stop everything just because we couldn't pull.
            if rc and self._image_exists():
                rc = 0
        if rc:
            return rc

        # If given a docker-compose file, start the services needed to run.
        if self.docker_compose_file and self._docker_compose_exists():
            rc = self._start_compose_services()
        else:
            # If not through docker-compose, run the various dependencies as necessary ourselves.

            # Setup the network if necessary.
            if not self._network_exists() and not self._is_predefined_network():
                logger.info("Creating network: '%s'", self.network)
                rc = self._start_network()
            if rc:
                return rc

            # Setup run dependencies if necessary.
            rc = self._start_run_deps() or 0
        if rc:
            return rc

        # Run the container itself.
        return self._run_container()

    def is_running(self) -> bool:
        """Checks if the container is currently running."""
        result = self._run_capture(
            [self.docker_command, "ps", "--no-trunc", "--filter", f"name=^{self.instance_name}$"]
        )
        is_running = self._string_exists(self.instance_name, result.stdout)

        # Check if the image tag of the running container differs
        if is_running:
            result = self._run_capture(
                [self.docker_command, "inspect", "--format={{.Config.Image}}", self.instance_name]
            )
            image_tag = result.stdout.split(":")[-1].strip()
            is_running = image_tag == self.image_tag

        # Treat the container as stale if any bind mount the current config expects is missing from it. We can't change
        # the mounts of a running container, so anything missing means the existing instance was started against an
        # older config and must be torn down and recreated.
        if is_running and self.volumes:
            result = self._run_capture(
                [self.docker_command, "inspect", "--format={{json .HostConfig.Binds}}", self.instance_name]
            )
            try:
                binds = json.loads(result.stdout.strip()) or []
            except json.JSONDecodeError:
                # Unexpected non-JSON output from `docker inspect` (e.g. the container disappeared between the `ps`
                # check above and this call, or the daemon hiccupped). Fall back to an empty bind list so the loop below
                # classifies the container as stale and we recreate it cleanly rather than crash.
                logger.warning(
                    "docker inspect returned non-JSON Binds for %s: %r; treating as stale",
                    self.instance_name,
                    result.stdout,
                )
                binds = []
            for vol in self.volumes:
                # Take the `host:container` prefix and match it anchored at the start of each existing bind, accepting
                # either an exact equality or `host:container:<docker-appended-options>` (e.g. `:rw`/`:ro` for
                # default-mode binds). Plain `host_container in b` would falsely match when two binds share a path
                # prefix - e.g. `/cache:/cache` is a substring of `/cache:/cache-other:rw`.
                host_container = ":".join(vol.split(":")[:2])
                if not any(b == host_container or b.startswith(host_container + ":") for b in binds):
                    is_running = False
                    break

        # If we have a network, make sure the running container is using the correct network (if not we need to create a
        # new container on the correct network).
        # Note: with proper naming conventions this shouldn't happen much.
        if self.network and is_running:
            result = self._run_capture(
                [self.docker_command, "inspect", "--format={{.NetworkSettings.Networks}}", self.instance_name]
            )
            is_running = self.network in result.stdout.strip()

        # Treat a container created without `--gpus` as stale when this invocation asks for one. Devices can only be
        # attached at `docker run` time and none of the checks above look at them, so a container left behind by a
        # `DAZEL_GPUS="" dazel ...` invocation would otherwise be handed to one that needs a GPU, where every CUDA call
        # inside fails with cudaErrorNoDevice.
        #
        # This keys on what the creating invocation asked for rather than probing real GPU access, which diverges by
        # host: where the nvidia runtime is the daemon default, the CUDA image's NVIDIA_VISIBLE_DEVICES grants the GPU
        # with no DeviceRequests at all, so a probe would accept there what it rejects under a runc default. Such hosts
        # pay one recreation and then settle, and no per-invocation `docker exec` is needed.
        #
        # The converse is deliberately allowed: a container holding GPUs serves a GPU-less command fine, and tearing it
        # down would make alternating GPU and GPU-less invocations recreate it every time. Devices that were requested
        # but have stopped working - usually a host driver upgrade leaving NVML broken inside long-lived containers -
        # are caught by _nvidia_container_runtime_check in tools/env_setup.sh.
        if is_running and self.gpus:
            result = self._run_capture(
                [self.docker_command, "inspect", "--format={{json .HostConfig.DeviceRequests}}", self.instance_name]
            )
            try:
                device_requests = json.loads(result.stdout.strip()) or []
            except json.JSONDecodeError:
                # Same reasoning as the Binds fallback above: unparseable output means we can't confirm the container is
                # usable, so classify it as stale and recreate rather than crash.
                logger.warning(
                    "docker inspect returned non-JSON DeviceRequests for %s: %r; treating as stale",
                    self.instance_name,
                    result.stdout,
                )
                device_requests = []
            is_running = bool(device_requests)
        return is_running

    def _run_silent_command(self, command: str, ignore_output: bool = False) -> int:
        """Run a shell-string command. Used for the few sites that genuinely need shell=True
        (compose pipelines, build/pull commands with embedded redirection, send_command's quiet-mode `> /dev/null 2>&1`
        suffix). Prefer `_run_argv` for argv-list invocations.
        """
        if ignore_output:
            return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        return subprocess.call(command, stdout=sys.stderr, shell=True)

    def _run_argv(
        self,
        argv: list[str],
        ignore_output: bool = False,
        timeout: float | None = None,
    ) -> int:
        """Run an argv-list command with shell=False. Default `timeout=None` because callers are
        long-running operations (container start, stop, rm); pass a numeric timeout for plumbing.
        """
        argv = self._docker_argv(argv)
        if ignore_output:
            return subprocess.call(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)
        return subprocess.call(argv, stdout=sys.stderr, timeout=timeout)

    def _run_command(self, command: str, timeout: float | None = DEFAULT_DOCKER_TIMEOUT_S) -> str:
        """Run a shell-string command and return its stdout. Default 30s timeout."""
        return subprocess.check_output(command, shell=True, timeout=timeout, text=True)

    def _run_capture(
        self,
        argv: list[str],
        timeout: float | None = DEFAULT_DOCKER_TIMEOUT_S,
    ) -> subprocess.CompletedProcess[str]:
        """Run an argv-list command and capture stdout/stderr. Default 30s timeout.

        Used for `docker inspect`, `docker ps`, `docker image ls`, `docker network ls`, etc. - plumbing calls where we
        need to read the output and want to fail fast on a hung daemon.
        """
        return subprocess.run(
            self._docker_argv(argv),
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout,
        )

    def _string_exists(self, string: str, output: str) -> bool:
        regex = re.compile(r"\b(?=\w)" + re.escape(string) + r"\b(?!\w)")
        return any(regex.findall(output))

    def _image_exists(self) -> bool:
        """Checks if the dazel image exists in the local repository."""
        # Empty `repository` means "image lives locally with no registry prefix"; concatenating an empty repo would
        # produce "/image" which Docker rejects as an invalid reference. Mirror the gating that `_run_container` does.
        repo_prefix = f"{self.repository}/" if self.repository else ""
        result = self._run_capture([self.docker_command, "image", "ls", f"{repo_prefix}{self.image_name}"])
        return len(result.stdout.splitlines()) > 1

    def _get_build_command(self) -> list[str]:
        """Returns the argv for `docker build`.

        Raises:
            RuntimeError: if the Dockerfile does not exist.
        """
        if not os.path.exists(self.dockerfile):
            raise RuntimeError("No Dockerfile specified to dazel image build command.")

        argv: list[str] = [self.docker_command, "build"]
        # docker_build_args is a free-form string from .dazelrc. Use shlex so that shell-style quoting (single/double
        # quotes) is honored - e.g. `--build-arg groups='a:1 b:2'` stays a single argument instead of being shredded
        # into separate positional args, which would otherwise be passed to `docker buildx build` as extra build
        # contexts and rejected with "requires 1 argument".
        if self.docker_build_args:
            argv += shlex.split(self.docker_build_args)
        # Empty `repository` means "build locally with no registry prefix"; concatenating an empty repo would produce
        # "/image:tag" which Docker rejects as an invalid reference. Mirror the gating that `_run_container` does.
        repo_prefix = f"{self.repository}/" if self.repository else ""
        argv += [
            "--tag",
            f"{repo_prefix}{self.image_name}:{self.image_tag}",
            "--file",
            self.dockerfile,
            self.directory,
        ]
        return argv

    def _build(self) -> int:
        """Builds the dazel image from the local dockerfile."""
        return self._run_argv(self._get_build_command())

    def _pull(self) -> int:
        """Pulls the relevant image from the dockerhub repository."""
        if not self.repository:
            raise RuntimeError("No repository to pull the dazel image from.")

        return self._run_argv([self.docker_command, "pull", f"{self.repository}/{self.image_name}"])

    def _is_predefined_network(self) -> bool:
        """Checks if the network is one of the default existing docker network types"""
        return self.network in DOCKER_SPECIAL_NETWORK_NAMES

    def _network_exists(self) -> bool:
        """Checks if the network we need to use exists."""
        result = self._run_capture([self.docker_command, "network", "ls"])
        return self._string_exists(self.network, result.stdout)

    def _start_network(self) -> int:
        """Starts the docker network the container will use."""
        if not self.network:
            return 0

        return self._run_argv([self.docker_command, "network", "create", self.network])

    def _start_run_deps(self) -> int:
        """Start each `RunDep` sidecar that isn't already running, then wait for its ready probe.

        Idempotent: a dep whose container is already running and reports the expected image is left alone; one running a
        different image is recreated from the configured image. Composes only the network with the parent container (no
        workspace bind, no bazel cache, no `-w`); each dep can carry its own dep-specific env+volumes.
        """
        for dep in self.run_deps:
            if not self._dep_is_running(dep):
                logger.info("Starting run dependency: '%s' (name: '%s')", dep.image, dep.name)
                rc = self._run_dep_container(dep)
                if rc:
                    return rc
            rc = self._wait_for_ready_probe(dep)
            if rc:
                return rc
        return 0

    def _run_dep_container(self, dep: RunDep) -> int:
        """`docker run -id` for a single :class:`RunDep`. Best-effort cleans up any stale name."""
        self._run_argv([self.docker_command, "stop", dep.name], ignore_output=True)
        self._run_argv([self.docker_command, "rm", dep.name], ignore_output=True)

        argv: list[str] = [self.docker_command, "run", "-id", f"--name={dep.name}"]
        if self.network:
            argv.append(f"--net={self.network}")
        argv += _docker_repeated_flag("-e", (f"{k}={v}" for k, v in dep.env.items()))
        argv += _docker_repeated_flag("-v", dep.volumes)
        argv.append(dep.image)
        return self._run_argv(argv)

    def _dep_is_running(self, dep: RunDep) -> bool:
        """Idempotency check - is `dep.name` running, started from `dep.image`?

        Both have to match. A container carrying the name but a different image warns and returns False, leaving the
        caller to recreate it. `docker ps` reports an image ID in place of a tag once that tag no longer resolves to
        the running image, which compares unequal for the same effect.
        """
        result = self._run_capture(
            [
                self.docker_command,
                "ps",
                "--no-trunc",
                "--filter",
                f"name=^{dep.name}$",
                "--format",
                "{{.Names}}\t{{.Image}}",
            ]
        )
        for line in result.stdout.splitlines():
            name, _, image = line.partition("\t")
            if name != dep.name:
                continue
            if image == dep.image:
                return True
            logger.warning(
                "Run dependency '%s' is running image '%s' but '%s' is expected; recreating it "
                "(volume data is preserved).",
                dep.name,
                image,
                dep.image,
            )
            return False
        return False

    def _wait_for_ready_probe(self, dep: RunDep) -> int:
        """Block until `dep.ready_probe.cmd` succeeds, the container dies, or we time out.

        See :class:`ReadyProbe` for the supported probe knobs. If `cmd` is empty the probe short-circuits immediately
        (return 0).

        Designed to be silent on the warm path (probe succeeds on first try -> no log spam) and chatty on the cold path
        (so a developer staring at a quiet terminal during a ~90s first start gets a heartbeat instead of wondering
        whether dazel is hung).
        """
        probe = dep.ready_probe
        if not probe.cmd:
            return 0

        start = time.monotonic()
        last_progress = start
        first_attempt = True
        while True:
            elapsed = time.monotonic() - start

            status = self._run_capture([self.docker_command, "inspect", "--format={{.State.Status}}", dep.name])
            container_status = status.stdout.strip().strip("'")
            if status.returncode != 0 or container_status != "running":
                logger.error(
                    "Run dependency '%s' (%s) is not running (state=%s) before ready probe succeeded.",
                    dep.name,
                    dep.image,
                    container_status or "missing",
                )
                self._dump_run_dep_logs(dep.name)
                return 1

            probe_run = self._run_capture([self.docker_command, "exec", dep.name, *probe.cmd])
            if probe_run.returncode == 0:
                if not first_attempt:
                    logger.info("Run dependency '%s' ready after %.1fs.", dep.name, elapsed)
                return 0

            if first_attempt:
                logger.info(
                    "Waiting for run dependency '%s' (%s, up to %.0fs)...",
                    dep.name,
                    probe.description,
                    probe.timeout_s,
                )
                first_attempt = False

            if elapsed >= probe.timeout_s:
                logger.error(
                    "Timed out after %.0fs waiting for run dependency '%s' (%s).",
                    probe.timeout_s,
                    dep.name,
                    probe.description,
                )
                self._dump_run_dep_logs(dep.name)
                return 1

            now = time.monotonic()
            if now - last_progress >= probe.progress_every_s:
                last_progress = now
                logger.info(
                    "  ... still waiting on '%s' (%.0fs elapsed; container=%s; probe rc=%d)",
                    dep.name,
                    elapsed,
                    container_status,
                    probe_run.returncode,
                )

            time.sleep(probe.interval_s)

    def _dump_run_dep_logs(self, container_name: str, n: int = 40) -> None:
        """Tail the run-dep container logs to stderr so failures surface in the terminal."""
        result = self._run_capture([self.docker_command, "logs", "--tail", str(n), container_name])
        sys.stderr.write(f"--- last {n} log lines from run dependency '{container_name}' ---\n")
        if result.stdout:
            sys.stderr.write(result.stdout)
            if not result.stdout.endswith("\n"):
                sys.stderr.write("\n")
        if result.stderr:
            sys.stderr.write(result.stderr)
            if not result.stderr.endswith("\n"):
                sys.stderr.write("\n")
        sys.stderr.write(f"--- end of '{container_name}' logs ---\n")

    def _start_compose_services(self) -> int:
        """Starts the docker-compose services."""
        if not self.docker_compose_file:
            return 0

        env_prefix = f"COMPOSE_PROJECT_NAME={self.docker_compose_project_name}"
        compose_base = f"{self.docker_compose_command} -f {self.docker_compose_file}"
        services = " ".join(self.compose_services)
        command = (
            f"{env_prefix} {compose_base} pull --ignore-pull-failures {services}"
            f" && {env_prefix} {compose_base} build {services}"
            f" && {env_prefix} {compose_base} up --force-recreate -d {services}"
        )
        return self._run_silent_command(command)

    def _run_container(self) -> int:
        """Runs the container itself."""
        logger.info("Starting docker container '%s'...", self.instance_name)

        # Best-effort cleanup of any prior container with this name. Both stop+rm are expected to fail on a clean host
        # (no container to stop, nothing to remove), so ignore_output=True hides the noise.
        self._run_argv([self.docker_command, "stop", self.instance_name], ignore_output=True)
        self._run_argv([self.docker_command, "rm", self.instance_name], ignore_output=True)

        argv: list[str] = [self.docker_command, "run", "-id", f"--name={self.instance_name}"]
        if self.platform:
            argv.append(f"--platform={self.platform}")
        if self.docker_run_privileged:
            argv.append("--privileged")
        if self.user:
            argv.append(f"--user={self.user}")
        if self.remote_directory:
            argv += ["-w", self.remote_directory]
        argv += _docker_repeated_flag("-v", self.volumes)
        argv += _docker_repeated_flag("-p", self.ports)
        argv += _docker_repeated_flag("-e", self.env_vars)
        if self.gpus:
            argv += ["--gpus", ",".join(self.gpus)]
        if self.shm_size:
            argv.append(f"--shm-size={self.shm_size}")
        if self.network:
            argv.append(f"--net={self.network}")
        repo_prefix = f"{self.repository}/" if self.repository else ""
        argv.append(f"{repo_prefix}{self.image_name}:{self.image_tag}")
        if self.run_command:
            argv.append(self.run_command)

        # Long-running operation - container detaches via -d but the engine itself can be slow under load, so let the
        # user control duration.
        rc = self._run_argv(argv)
        if rc:
            return rc

        # Touch the dazel run file to change the timestamp.
        if self.dazel_run_file:
            pathlib.Path(self.dazel_run_file).write_text(self.instance_name + "\n")
            logger.info("Done.")

        return rc

    def _add_volumes(self, raw_volumes: str | collections.abc.Iterable[str] | None) -> None:
        """Compute the docker `-v` bind list, including the bazel cache binds we need anyway.

        Stores the raw "src:dest[:option]" specs as `self.volumes` (a list[str]) - rendered to argv flags by
        `_run_container` and to a shell string by `send_command` only on demand.
        """
        vol_list = _coerce_iterable(raw_volumes, "DAZEL_VOLUMES")

        # Find the real source and output directories.
        real_directory = os.path.realpath(self.directory)
        vol_list.append(f"{real_directory}:{self.remote_directory}")

        # If the user hasn't explicitly set a DAZEL_BAZEL_USER_OUTPUT_ROOT for bazel, set it from the output directory
        # so that we get the build results on the host.
        real_bazelout = os.path.realpath(os.path.join(self.directory, "bazel-out", ".."))

        if not self.bazel_user_output_root and "/_bazel" in real_bazelout:
            parts = real_bazelout.split("/_bazel")
            first_part = parts[0]
            second_part = "/_bazel" + parts[1].split("/")[0]
            self.bazel_user_output_root = first_part + second_part

        # Add the bazel user output directory if it exists, or the real bazelout directory if it does not
        if self.bazel_user_output_root:
            bazel_output_base = os.path.realpath(os.path.join(self.bazel_user_output_root, self.workspace_hex_digest))
            self.bazel_output_base = self._get_remote_directory(bazel_output_base)

            # Add the default bazel user output paths only if delegated volume is enabled
            if self.delegated_volume_flag:
                user_output_paths = DEFAULT_BAZEL_USER_OUTPUT_PATHS + [os.path.basename(real_directory)]
                for user_output_path in user_output_paths:
                    real_user_output_path = os.path.realpath(os.path.join(bazel_output_base, user_output_path))
                    pathlib.Path(real_user_output_path).mkdir(parents=True, exist_ok=True)
                    remote = self._get_remote_directory(real_user_output_path)
                    vol_list.append(f"{real_user_output_path}:{remote}{self.delegated_volume_flag}")
        elif real_bazelout:
            vol_list.append(f"{real_bazelout}:{real_bazelout}{self.delegated_volume_flag}")
            self.bazel_output_base = real_bazelout

        # Make sure the path exists on the host.
        if self.bazel_user_output_root:
            pathlib.Path(self.bazel_user_output_root).mkdir(parents=True, exist_ok=True)

        self.volumes = vol_list

    def _get_remote_directory(self, local_directory: str, add_drive: bool = False) -> str:
        remote_directory = local_directory
        if sys.platform == "win32":
            win_path = os.path.splitdrive(local_directory)[1]
            if add_drive:
                drive = os.path.splitdrive(local_directory)[0].strip(":")
                win_path = f"/{drive}/{win_path}"
            remote_directory = str(pathlib.PureWindowsPath(win_path).as_posix())
        return remote_directory

    def _build_run_deps(self) -> None:
        """Parse `raw_run_deps` into typed `RunDep` objects.

        See :class:`RunDep` for the supported entry formats. When `raw_run_deps` is a string we split on commas
        (legacy); when it's an iterable we walk it directly.
        """
        if not self.raw_run_deps:
            self.run_deps = []
            return

        if isinstance(self.raw_run_deps, str):
            entries: list[Any] = [rd.strip() for rd in self.raw_run_deps.split(",") if rd.strip()]
        elif isinstance(self.raw_run_deps, collections.abc.Iterable):
            entries = list(self.raw_run_deps)
        else:
            raise TypeError("DAZEL_RUN_DEPS must be comma-separated string or python iterable")

        self.run_deps = [RunDep.parse(self.network, rd) for rd in entries]

    def _docker_exists(self) -> bool:
        """Checks if the basic docker executable exists."""
        return _command_exists(self.docker_command)

    def _docker_compose_exists(self) -> bool:
        """Checks if the docker-compose executable exists."""
        return _command_exists(self.docker_compose_command)

    def _docker_argv(self, argv: list[Any]) -> list[str]:
        """Normalize argv elements to str so subprocess.run can execute the binary directly.

        Returns a list of strings with no shell parsing of arguments - container names, cmd_args, image tags, etc.
        cannot be interpreted as shell metacharacters.
        """
        return [str(a) for a in argv]

    @classmethod
    def _config_from_files(cls) -> dict[str, Any]:
        """Creates a configuration from a base .dazelrc and overriding user.dazelrc file (if exists)."""
        directory = cls.find_workspace_directory()
        local_dazelrc_path = os.path.join(directory, DAZEL_RC_FILE)
        dazelrc_path = os.environ.get("DAZEL_RC_FILE", local_dazelrc_path)

        if not os.path.exists(dazelrc_path):
            return {"DAZEL_DIRECTORY": os.environ.get("DAZEL_DIRECTORY", directory)}

        config: dict[str, Any] = {}
        with open(dazelrc_path, "r") as dazelrc:
            exec(dazelrc.read(), config)  # noqa: S102 - load-bearing user-config-as-Python
        config["DAZEL_DIRECTORY"] = os.environ.get("DAZEL_DIRECTORY", directory)

        user_dazelrc_path = os.path.join(directory, "user.dazelrc")
        if os.path.exists(user_dazelrc_path):
            with open(user_dazelrc_path, "r") as user_dazelrc:
                exec(user_dazelrc.read(), config)  # noqa: S102 - load-bearing user-config-as-Python

        return config

    @classmethod
    def _config_from_environment(cls) -> dict[str, str]:
        """Creates a configuration from environment variables."""
        return {name: value for (name, value) in os.environ.items() if name.startswith("DAZEL_")}

    @classmethod
    def find_workspace_directory(cls) -> str:
        """Find the workspace directory.

        This is done by traversing the directory structure from the given dazel directory until we find the WORKSPACE
        file. Cached per start directory because both this and `_config_from_files` call it on every dazel invocation.
        """
        return _find_workspace_directory(os.environ.get("DAZEL_DIRECTORY", DEFAULT_DIRECTORY))


def _update_apt_lockfiles(di: DockerInstance) -> int:
    """Regenerate apt package lockfiles to keep them in sync with upstream repositories.

    Scans MODULE.bazel for apt.install() declarations and regenerates each lockfile unconditionally. This ensures
    lockfiles stay current even when upstream package versions are removed or superseded (e.g., security patches
    replacing older .deb URLs).

    Args:
        di: DockerInstance object

    Returns:
        Exit code (0 on success)
    """
    workspace_dir = di.find_workspace_directory()
    module_bazel_path = os.path.join(workspace_dir, BAZEL_MODULE_FILE)

    if not os.path.exists(module_bazel_path):
        return 0

    with open(module_bazel_path, "r") as f:
        content = f.read()

    apt_install_pattern = re.compile(r"apt\.install\s*\((.*?)\)", re.DOTALL)
    name_pattern = re.compile(r'name\s*=\s*"([^"]+)"')

    for match in apt_install_pattern.finditer(content):
        block = match.group(1)

        name_match = name_pattern.search(block)
        if not name_match:
            continue

        repo_name = name_match.group(1)

        logger.info("Regenerating lockfile for @%s...", repo_name)
        rc = di.send_command(["run", f"@{repo_name}//:lock"], quiet=True)
        if rc:
            logger.error("ERROR: Failed to regenerate lockfile for @%s with exit code %d", repo_name, rc)
            return rc
        logger.info("Lockfile for @%s regenerated successfully!", repo_name)

    return 0


def _update_requirements_txt(di: DockerInstance) -> int:
    """Update requirements.txt if it's out of sync with requirements.in.

    Uses the :requirements.test target to check if requirements.txt needs updating.

    Args:
        di: DockerInstance object

    Returns:
        Exit code (0 on success or if no update needed)
    """
    # Test if requirements.txt is in sync - suppress output since we expect it may fail
    logger.info("Checking if requirements.txt is in sync...")
    test_result = di.send_command(["test", "//:requirements.test", "--test_output=errors"])

    if test_result == 0:
        # Already in sync, nothing to do
        return 0

    logger.info("requirements.txt is out of sync, updating...")

    rc = di.send_command(["run", "//:requirements.update"])
    if rc:
        logger.error("ERROR: Failed to update requirements.txt with exit code %d", rc)
        return rc

    logger.info("requirements.txt updated successfully!")
    return 0


def _update_gazelle_manifest(di: DockerInstance) -> int:
    """Update the Gazelle pip manifest (gazelle_python.yaml) if it's out of sync with requirements.txt.

    Uses the :gazelle_python_manifest.test target to check whether the manifest needs regenerating. Must run
    after _update_requirements_txt: the manifest's integrity hash is derived from requirements.txt, so the
    requirements file must be in its final state first.

    Args:
        di: DockerInstance object

    Returns:
        Exit code (0 on success or if no update needed)
    """
    # Test if the manifest is in sync - suppress output since we expect it may fail
    logger.info("Checking if gazelle python manifest is in sync...")
    test_result = di.send_command(["test", "//:gazelle_python_manifest.test", "--test_output=errors"])

    if test_result == 0:
        # Already in sync, nothing to do
        return 0

    logger.info("gazelle python manifest is out of sync, updating...")

    rc = di.send_command(["run", "//:gazelle_python_manifest.update"])
    if rc:
        logger.error("ERROR: Failed to update gazelle python manifest with exit code %d", rc)
        return rc

    logger.info("gazelle python manifest updated successfully!")
    return 0


def _run_linter(di: DockerInstance, args: list[str], fix_mode: bool = False) -> int:
    """Run the linter script with the given arguments.

    Args:
        di: DockerInstance object
        args: Command arguments after lint/fix command (targets and flags)
        fix_mode: If True, runs with --apply-patches flag

    Returns:
        Exit code from linter script
    """
    targets_str = " ".join(args) if args else "default targets (//...)"
    logger.info("Running linter on targets: %s", targets_str)

    workspace_dir = di.find_workspace_directory()

    linter_script = di.linter_script.strip()
    if not linter_script:
        logger.error(
            "ERROR: DAZEL_LINTER_SCRIPT is not set; cannot run linter. Configure DAZEL_LINTER_SCRIPT in this "
            "workspace's .dazelrc (or override via env-var) to point at the linter wrapper you want dazel to invoke."
        )
        return 1

    cmd_args = [linter_script]
    if fix_mode:
        cmd_args.append("--apply-patches")
    cmd_args.extend(args)

    try:
        rc = di._run_with_cancel_forwarding(cmd_args, cwd=workspace_dir)
    except (subprocess.SubprocessError, OSError) as e:
        logger.error("ERROR: Failed to run linter script: %s", e)
        return 1

    # Exit code 1 is the conventional "linter ran successfully but found violations" signal; the linter script
    # has already printed a colored summary block explaining what remains, and the caller (dazel lint / dazel fix)
    # prints its own status message right after. Logging an additional `ERROR: Linter failed` line here would (a)
    # mislead the user into thinking the linter itself crashed and (b) make the combined summary scroll past the
    # final visible line. We still propagate the non-zero rc to the caller so the invoking build fails. Genuine
    # linter failures (Bazel build errors, missing tools, etc.) come back with a different exit code and do still
    # get the explicit ERROR log.
    if rc and rc != 1:
        logger.error("ERROR: Linter failed with exit code %d", rc)

    return rc


def main() -> int:
    # Read the configuration either from .dazelrc or from the environment.
    di = DockerInstance.from_config()

    # If there is no .dazel_run file, or it is too old, start the DockerInstance.
    if (
        not os.path.exists(di.dazel_run_file)
        or not di.is_running()
        or (os.path.exists(di.dockerfile) and os.path.getctime(di.dockerfile) > os.path.getctime(di.dazel_run_file))
    ):
        rc = di.start()
        if rc:
            return rc

    # Ensure DAZEL_RUN_DEPS sidecars are up on every dazel invocation, not just cold start. start() (and thus
    # _start_run_deps) is gated above on the build container being absent / stale; without this second pass, a developer
    # who already has the build container running before adding a new run-dep (or before stopping a sidecar) would never
    # see the dep launched. _start_run_deps is idempotent: each entry is no-op'd if its container is already running, so
    # the warm-path overhead is one `docker ps` per dep.
    rc = di._start_run_deps()
    if rc:
        return rc

    # Check if this is a "lint" command
    if len(sys.argv) > 1 and sys.argv[1] == "lint":
        rc = _run_linter(di, sys.argv[2:])
        if rc == 0:
            logger.info("Lint command completed successfully!")
        elif rc == 1:
            # Mirror the fix path's "with remaining violations" wording so the logs are consistent and so the
            # combined summary block above isn't followed by a silent non-zero exit. rc is still propagated so the
            # invoking build fails as expected.
            logger.error("Lint command completed with violations. Please fix them and re-run.")
        return rc

    # Check if this is a "fix" command
    if len(sys.argv) > 1 and sys.argv[1] == "fix":
        # Update requirements.txt if out of sync with requirements.in
        rc = _update_requirements_txt(di)
        if rc:
            return rc

        # Refresh the Gazelle pip manifest after requirements.txt (its integrity hash derives from requirements.txt)
        rc = _update_gazelle_manifest(di)
        if rc:
            return rc

        # Regenerate apt package lockfiles to stay in sync with upstream repositories
        rc = _update_apt_lockfiles(di)
        if rc:
            return rc

        # Run the linter with fix mode to apply patches
        rc = _run_linter(di, sys.argv[2:], fix_mode=True)
        if rc:
            return rc

        # Run the formatter
        logger.info("Running formatter...")
        formatter_target = di.formatter_target.strip()
        if not formatter_target:
            logger.error(
                "ERROR: DAZEL_FORMATTER_TARGET is not set; cannot run formatter. Configure DAZEL_FORMATTER_TARGET "
                "in this workspace's .dazelrc (or override via env-var) to point at the Bazel formatter target."
            )
            return 1
        rc = di.send_command(["run", formatter_target])
        if rc:
            logger.error("ERROR: Formatter failed with exit code %d", rc)
            return rc

        # Rerun the linter to check if there are any remaining violations
        rc = _run_linter(di, sys.argv[2:])
        if rc:
            logger.error("Fix command completed with remaining violations. Please fix them manually.")
            return rc

        logger.info("Fix command completed successfully! No remaining violations found.")
        return 0

    # Forward the command line arguments to the container.
    return di.send_command(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
