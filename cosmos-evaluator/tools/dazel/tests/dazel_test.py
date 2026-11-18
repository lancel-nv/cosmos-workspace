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

"""Unit tests for tools/dazel/dazel.py.

Covers the pure helpers / dataclass parsing layer directly, and the subprocess / signal / argv-assembly / orchestration
layers via mocks at the lowest stable boundary (typically `subprocess.run` / `subprocess.call` / `subprocess.Popen`
or the `_run_argv` / `_run_capture` instance methods, depending on what the test is asserting).
"""

from __future__ import annotations

import io

# The module configures `logging.basicConfig(level=INFO)` at import time. Tests exercise many error paths that
# intentionally log to stderr - silence those so the test output stays readable.
import logging
import os
import signal
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

from tools.dazel import dazel

logging.getLogger("dazel").setLevel(logging.CRITICAL)


def _touch(path: str, content: str = "") -> None:
    """Write `content` to `path`, ensuring parent dirs exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def _make_instance(tmpdir: str, **overrides: Any) -> dazel.DockerInstance:
    """Construct a DockerInstance pointed at `tmpdir`.

    `__post_init__` does real filesystem work (mkdir on the bazel cache dirs,
    realpath on the workspace dir). Tests need to point both at a tempdir to
    avoid touching the developer's real `~/.cache/bazel` tree.
    """
    workspace = os.path.join(tmpdir, "workspace")
    cache = os.path.join(tmpdir, "cache")
    os.makedirs(workspace, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    kwargs: dict[str, Any] = {
        "directory": workspace,
        "bazel_user_output_root": cache,
        # `delegated_volume=False` skips the inner per-output-path mkdir loop that we don't want to exercise here.
        "delegated_volume": False,
        "dockerfile": os.path.join(tmpdir, "Dockerfile"),
        "image_tag": "test-tag",  # skip _hash_files in __post_init__
    }
    kwargs.update(overrides)
    return dazel.DockerInstance(**kwargs)


class CoerceIterableTest(unittest.TestCase):
    def test_none_returns_empty(self) -> None:
        self.assertEqual(dazel._coerce_iterable(None, "X"), [])

    def test_empty_string_returns_empty(self) -> None:
        self.assertEqual(dazel._coerce_iterable("", "X"), [])

    def test_empty_list_returns_empty(self) -> None:
        self.assertEqual(dazel._coerce_iterable([], "X"), [])

    def test_comma_separated_string(self) -> None:
        self.assertEqual(dazel._coerce_iterable("a,b,c", "X"), ["a", "b", "c"])

    def test_string_strips_whitespace_and_drops_empties(self) -> None:
        self.assertEqual(dazel._coerce_iterable(" a , , b ,c", "X"), ["a", "b", "c"])

    def test_list_input_is_stringified(self) -> None:
        # `_coerce_iterable` declares `Iterable[str]` but defensively wraps each element in `str()`; pin that behavior.
        # Mixed-type list intentionally violates the hint.
        self.assertEqual(
            dazel._coerce_iterable([1, 2, "three"], "X"),  # type: ignore[list-item]
            ["1", "2", "three"],
        )

    def test_tuple_input(self) -> None:
        self.assertEqual(dazel._coerce_iterable(("a", "b"), "X"), ["a", "b"])

    def test_non_iterable_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "MY_VAR"):
            dazel._coerce_iterable(123, "MY_VAR")  # type: ignore[arg-type]


class DockerRepeatedFlagTest(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(dazel._docker_repeated_flag("-v", []), [])

    def test_single(self) -> None:
        self.assertEqual(dazel._docker_repeated_flag("-v", ["a:b"]), ["-v", "a:b"])

    def test_multiple(self) -> None:
        self.assertEqual(
            dazel._docker_repeated_flag("-e", ["A=1", "B=2", "C=3"]),
            ["-e", "A=1", "-e", "B=2", "-e", "C=3"],
        )

    def test_consumes_generator(self) -> None:
        self.assertEqual(
            dazel._docker_repeated_flag("-v", (x for x in ["a", "b"])),
            ["-v", "a", "-v", "b"],
        )


class ShellRepeatedFlagTest(unittest.TestCase):
    def test_empty_returns_empty_string(self) -> None:
        self.assertEqual(dazel._shell_repeated_flag("-e", []), "")

    def test_single_quoted(self) -> None:
        self.assertEqual(dazel._shell_repeated_flag("-e", ["A=1"]), '-e "A=1"')

    def test_multiple_quoted(self) -> None:
        self.assertEqual(
            dazel._shell_repeated_flag("-e", ["A=1", "B=2"]),
            '-e "A=1" -e "B=2"',
        )


class ValidateDockerNameTest(unittest.TestCase):
    def test_simple_alpha(self) -> None:
        self.assertEqual(dazel._validate_docker_name("cosmos_eval"), "cosmos_eval")

    def test_with_dot_and_hyphen(self) -> None:
        self.assertEqual(dazel._validate_docker_name("a.b-c_d"), "a.b-c_d")

    def test_starts_with_digit(self) -> None:
        self.assertEqual(dazel._validate_docker_name("0abc"), "0abc")

    def test_starts_with_hyphen_invalid(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not a valid Docker name"):
            dazel._validate_docker_name("-bad")

    def test_contains_slash_invalid(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not a valid Docker name"):
            dazel._validate_docker_name("foo/bar")

    def test_empty_invalid(self) -> None:
        with self.assertRaises(RuntimeError):
            dazel._validate_docker_name("")


class ReadyProbeTest(unittest.TestCase):
    def test_from_raw_none_yields_empty_probe(self) -> None:
        probe = dazel.ReadyProbe.from_raw("dep", None)
        self.assertEqual(probe.cmd, ())
        self.assertEqual(probe.timeout_s, 240.0)
        self.assertEqual(probe.interval_s, 1.0)
        self.assertEqual(probe.progress_every_s, 5.0)
        self.assertEqual(probe.description, "ready")

    def test_from_raw_empty_dict_yields_defaults(self) -> None:
        probe = dazel.ReadyProbe.from_raw("dep", {})
        self.assertEqual(probe.cmd, ())
        self.assertEqual(probe.timeout_s, 240.0)

    def test_from_raw_full(self) -> None:
        probe = dazel.ReadyProbe.from_raw(
            "dep",
            {
                "cmd": ["pg_isready", "-q"],
                "timeout_s": 30,
                "interval_s": 0.5,
                "progress_every_s": 2,
                "description": "postgres ready",
            },
        )
        self.assertEqual(probe.cmd, ("pg_isready", "-q"))
        self.assertEqual(probe.timeout_s, 30.0)
        self.assertEqual(probe.interval_s, 0.5)
        self.assertEqual(probe.progress_every_s, 2.0)
        self.assertEqual(probe.description, "postgres ready")

    def test_from_raw_invalid_cmd_type(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ready_probe.cmd"):
            dazel.ReadyProbe.from_raw("dep", {"cmd": "pg_isready -q"})

    def test_from_raw_invalid_cmd_element_type(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ready_probe.cmd"):
            dazel.ReadyProbe.from_raw("dep", {"cmd": ["pg_isready", 1]})

    def test_from_raw_empty_cmd_list_is_no_probe(self) -> None:
        # An empty list is the "no probe" signal that _wait_for_ready_probe
        # short-circuits on; it must NOT be rejected as invalid.
        probe = dazel.ReadyProbe.from_raw("dep", {"cmd": []})
        self.assertEqual(probe.cmd, ())


class RunDepTest(unittest.TestCase):
    def test_legacy_string_form_default_name(self) -> None:
        dep = dazel.RunDep.parse("hostnet", "repo/postgres:14")
        self.assertEqual(dep.image, "repo/postgres:14")
        # Default: replace / and : with _ and prefix with network name.
        self.assertEqual(dep.name, "hostnet_repo_postgres_14")
        self.assertEqual(dep.env, {})
        self.assertEqual(dep.volumes, [])
        self.assertEqual(dep.ready_probe.cmd, ())

    def test_legacy_string_form_explicit_name(self) -> None:
        dep = dazel.RunDep.parse("net", "repo/postgres:14::my_db")
        self.assertEqual(dep.image, "repo/postgres:14")
        self.assertEqual(dep.name, "my_db")

    def test_dict_form_full(self) -> None:
        dep = dazel.RunDep.parse(
            "net",
            {
                "image": "postgres:14",
                "name": "my_db",
                "env": {"POSTGRES_PASSWORD": "x"},
                "volumes": ["my-data:/var/lib/postgres"],
                "ready_probe": {"cmd": ["pg_isready"], "timeout_s": 10},
            },
        )
        self.assertEqual(dep.image, "postgres:14")
        self.assertEqual(dep.name, "my_db")
        self.assertEqual(dep.env, {"POSTGRES_PASSWORD": "x"})
        self.assertEqual(dep.volumes, ["my-data:/var/lib/postgres"])
        self.assertEqual(dep.ready_probe.cmd, ("pg_isready",))
        self.assertEqual(dep.ready_probe.timeout_s, 10.0)

    def test_dict_form_minimal_uses_default_name(self) -> None:
        dep = dazel.RunDep.parse("net", {"image": "postgres:14"})
        self.assertEqual(dep.name, "net_postgres_14")
        self.assertEqual(dep.env, {})

    def test_dict_form_missing_image(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "'image' key"):
            dazel.RunDep.parse("net", {"name": "x"})

    def test_dict_form_invalid_name(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not a valid Docker name"):
            dazel.RunDep.parse("net", {"image": "postgres:14", "name": "bad/name"})

    def test_invalid_type(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must be dict or string"):
            dazel.RunDep.parse("net", 42)  # type: ignore[arg-type]

    def test_image_name_and_tag_with_tag(self) -> None:
        dep = dazel.RunDep.parse("net", {"image": "repo/postgres:14"})
        self.assertEqual(dep.image_name_and_tag, ("repo/postgres", "14"))

    def test_image_name_and_tag_without_tag(self) -> None:
        dep = dazel.RunDep.parse("net", {"image": "postgres"})
        self.assertEqual(dep.image_name_and_tag, ("postgres", "latest"))


class HashFilesTest(unittest.TestCase):
    def test_empty_list(self) -> None:
        # md5 of an empty stream is the well-known constant.
        self.assertEqual(dazel.DockerInstance._hash_files([]), "d41d8cd98f00b204e9800998ecf8427e")

    def test_missing_files_skipped(self) -> None:
        # Non-existent paths are silently skipped (not an error).
        self.assertEqual(
            dazel.DockerInstance._hash_files(["/does/not/exist/a", "/does/not/exist/b"]),
            "d41d8cd98f00b204e9800998ecf8427e",
        )

    def test_content_changes_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "f1")
            _touch(f1, "abc")
            h1 = dazel.DockerInstance._hash_files([f1])
            _touch(f1, "abcd")
            h2 = dazel.DockerInstance._hash_files([f1])
            self.assertNotEqual(h1, h2)

    def test_order_affects_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            a = os.path.join(tmpdir, "a")
            b = os.path.join(tmpdir, "b")
            _touch(a, "AAA")
            _touch(b, "BBB")
            self.assertNotEqual(
                dazel.DockerInstance._hash_files([a, b]),
                dazel.DockerInstance._hash_files([b, a]),
            )


class DockerInstanceConstructionTest(unittest.TestCase):
    def test_defaults_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            self.assertEqual(di.instance_name, "dazel")
            self.assertEqual(di.image_name, "dazel")
            self.assertEqual(di.docker_command, "docker")
            self.assertEqual(di.network, "dazel")
            self.assertEqual(di.run_deps, [])
            self.assertEqual(di.image_tag, "test-tag")
            # __post_init__ always appends the workspace bind to volumes.
            workspace = os.path.realpath(os.path.join(tmpdir, "workspace"))
            self.assertIn(f"{workspace}:{workspace}", di.volumes)

    def test_workspace_hex_mangles_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, workspace_hex=True)
            self.assertNotEqual(di.instance_name, "dazel")
            self.assertTrue(di.instance_name.startswith("dazel_"))
            self.assertEqual(len(di.workspace_hex_digest), 32)  # md5 hex

    def test_raw_volumes_string_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, raw_volumes="/a:/A,/b:/B")
            self.assertIn("/a:/A", di.volumes)
            self.assertIn("/b:/B", di.volumes)

    def test_raw_env_vars_iterable_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, raw_env_vars=["FOO=1", "BAR=2"])
            self.assertEqual(di.env_vars, ["FOO=1", "BAR=2"])

    def test_run_deps_list_form_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(
                tmpdir,
                raw_run_deps=[
                    {"image": "postgres:14", "name": "db"},
                    "redis:7::cache",
                ],
            )
            self.assertEqual(len(di.run_deps), 2)
            self.assertEqual(di.run_deps[0].name, "db")
            self.assertEqual(di.run_deps[1].name, "cache")

    def test_run_deps_string_form_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, raw_run_deps="postgres:14::db, redis:7::cache")
            self.assertEqual([d.name for d in di.run_deps], ["db", "cache"])

    def test_image_tag_derived_from_files_watch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            watched = os.path.join(tmpdir, "watched.txt")
            _touch(watched, "v1")
            # Don't pass image_tag override so __post_init__ falls back to hashing.
            di = _make_instance(tmpdir, image_tag="", files_watch=[watched])
            tag_v1 = di.image_tag
            self.assertNotEqual(tag_v1, "")

            _touch(watched, "v2")
            di2 = _make_instance(tmpdir, image_tag="", files_watch=[watched])
            self.assertNotEqual(tag_v1, di2.image_tag)


class DockerInstancePureHelpersTest(unittest.TestCase):
    def test_string_exists_word_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            self.assertTrue(di._string_exists("foo", "alpha foo bravo"))
            self.assertTrue(di._string_exists("foo-bar", "x foo-bar y"))
            # Word-boundary check rejects substring matches inside another word.
            self.assertFalse(di._string_exists("foo", "foobar"))

    def test_is_predefined_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertTrue(_make_instance(tmpdir, network="host")._is_predefined_network())
            self.assertTrue(_make_instance(tmpdir, network="bridge")._is_predefined_network())
            self.assertTrue(_make_instance(tmpdir, network="none")._is_predefined_network())
            self.assertFalse(_make_instance(tmpdir, network="custom")._is_predefined_network())

    def test_docker_argv_stringifies(self) -> None:
        # `_docker_argv` is purely defensive: it wraps every element in `str()` so
        # subprocess never sees a non-string and raises a `TypeError`. No current
        # caller in `dazel.py` passes `None` (callers either use string literals,
        # f-strings, or already-typed `str` fields), but the helper is robust to it
        # by design: a stray `None` becomes the literal "None" and surfaces as a
        # clear docker-side error (`docker: 'None' is not a docker command`) rather
        # than a stack trace from inside `subprocess`. This test pins that contract.
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            argv = di._docker_argv(["docker", "run", 0, 1.5, None])
            self.assertEqual(argv, ["docker", "run", "0", "1.5", "None"])


class GetBuildCommandTest(unittest.TestCase):
    """Regression tests for `_get_build_command`.

    The free-form ``DAZEL_DOCKER_BUILD_ARGS`` string is split via ``shlex``;
    callers rely on shell-style quoting being honored so that ``--build-arg
    groups='a:1 b:2'`` stays a single argv element. The previous naive
    ``str.split()`` shredded such values into separate tokens, which were
    then forwarded to ``docker buildx build`` as extra positional args and
    rejected with ``'docker buildx build' requires 1 argument``. This test
    pins that contract.
    """

    def test_raises_when_dockerfile_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)  # no Dockerfile created
            with self.assertRaisesRegex(RuntimeError, "No Dockerfile"):
                di._get_build_command()

    def test_no_build_args_yields_minimal_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(tmpdir, dockerfile=dockerfile)
            argv = di._get_build_command()
            self.assertEqual(argv[:2], ["docker", "build"])
            self.assertIn("--tag", argv)
            self.assertIn("--file", argv)
            self.assertEqual(argv[-1], di.directory)
            # No --build-arg should appear without docker_build_args set.
            self.assertNotIn("--build-arg", argv)

    def test_quoted_value_with_spaces_stays_single_token(self) -> None:
        """Regression: multi-word quoted values used to be shredded.

        Triggered by users whose ``id -G`` returns multiple supplementary
        groups, producing ``--build-arg groups='a:1 b:2 c:3'`` which got
        forwarded to ``docker buildx build`` as multiple positional args.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            build_args = (
                "--build-arg user_id=1000 "
                "--build-arg user_name=alice "
                "--build-arg groups='domain-users:74840 sudo:27 docker:998' "
                "--secret id=netrc,src=/root/.netrc "
                "--label 'build_date=2026-05-06' "
                "--label 'git_commit_sha=deadbeef'"
            )
            di = _make_instance(tmpdir, dockerfile=dockerfile, docker_build_args=build_args)
            argv = di._get_build_command()

            # The multi-word groups value MUST stay a single argv element.
            self.assertIn(
                "groups=domain-users:74840 sudo:27 docker:998",
                argv,
                f"groups arg was shredded; argv={argv!r}",
            )
            # And its sibling tokens must NOT leak as standalone args.
            self.assertNotIn("sudo:27", argv)
            self.assertNotIn("docker:998", argv)
            self.assertNotIn("docker:998'", argv)

            # Cosmetic single-quotes around label values get stripped by shlex.
            self.assertIn("build_date=2026-05-06", argv)
            self.assertIn("git_commit_sha=deadbeef", argv)
            self.assertNotIn("'build_date=2026-05-06'", argv)

            # The build context (final positional) is still last and is the ONLY trailing positional - this is what
            # `docker buildx build` validates as "exactly 1 argument".
            self.assertEqual(argv[-1], di.directory)

            # All non-flag args before the final context should be values of preceding flags - i.e. there are no stray
            # positionals.
            stray_positionals = _stray_positionals(argv[2:-1])
            self.assertEqual(stray_positionals, [], f"stray positional args: {stray_positionals!r}")

    def test_unquoted_args_split_normally(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(
                tmpdir,
                dockerfile=dockerfile,
                docker_build_args="--build-arg A=1 --build-arg B=2",
            )
            argv = di._get_build_command()
            self.assertEqual(argv.count("--build-arg"), 2)
            self.assertIn("A=1", argv)
            self.assertIn("B=2", argv)

    def test_tag_includes_repo_prefix_when_repository_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(
                tmpdir,
                dockerfile=dockerfile,
                repository="registry.example.com:5000/team/project",
                image_name="build",
                image_tag="abc123",
            )
            argv = di._get_build_command()
            tag = argv[argv.index("--tag") + 1]
            self.assertEqual(tag, "registry.example.com:5000/team/project/build:abc123")

    def test_tag_omits_repo_prefix_when_repository_empty(self) -> None:
        """Regression: empty repository must not produce a leading-slash tag like '/build:hash'.

        Triggered by `dazel build //...` from within a standalone Cosmos Evaluator build, where `.dazelrc` deliberately
        leaves `DAZEL_REPOSITORY=""` to mean "build locally with no remote registry". The previous formatting produced
        `--tag /build:<hash>`, which Docker rejects as `invalid reference format`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(
                tmpdir,
                dockerfile=dockerfile,
                repository="",
                image_name="build",
                image_tag="abc123",
            )
            argv = di._get_build_command()
            tag = argv[argv.index("--tag") + 1]
            self.assertEqual(tag, "build:abc123")
            self.assertFalse(tag.startswith("/"), f"leading-slash leaked into tag: {tag!r}")


def _stray_positionals(tokens: list[str]) -> list[str]:
    """Return tokens that aren't a flag and aren't the value of a preceding flag.

    Used by GetBuildCommandTest to confirm that the only positional argument
    `docker build` ends up with is the build context. A "flag" here is any
    token starting with ``-``. ``--secret`` and ``--label`` take a single value.
    """
    flags_with_value = {"--build-arg", "--tag", "-t", "--file", "-f", "--secret", "--label", "--platform"}
    out: list[str] = []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in flags_with_value:
            skip_next = True
            continue
        if tok.startswith("-"):
            continue
        out.append(tok)
    return out


class ConfigFromEnvironmentTest(unittest.TestCase):
    def test_only_dazel_prefixed_vars_returned(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "DAZEL_INSTANCE_NAME": "x",
                "DAZEL_NETWORK": "host",
                "OTHER": "ignored",
            },
            clear=True,
        ):
            cfg = dazel.DockerInstance._config_from_environment()
            self.assertEqual(cfg, {"DAZEL_INSTANCE_NAME": "x", "DAZEL_NETWORK": "host"})


class FromConfigTest(unittest.TestCase):
    """Smoke tests for `from_config` via tempdir + DAZEL_RC_FILE override."""

    def test_loads_simple_dazelrc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "ws")
            cache = os.path.join(tmpdir, "cache")
            os.makedirs(workspace, exist_ok=True)
            os.makedirs(cache, exist_ok=True)

            # `find_workspace_directory` walks up looking for MODULE.bazel.
            _touch(os.path.join(workspace, "MODULE.bazel"), "")

            rc_path = os.path.join(tmpdir, ".dazelrc")
            _touch(
                rc_path,
                "DAZEL_INSTANCE_NAME = 'unit-test'\n"
                "DAZEL_IMAGE_NAME = 'unit-image'\n"
                "DAZEL_NETWORK = 'host'\n"
                "DAZEL_VOLUMES = ['/x:/X']\n"
                f"DAZEL_BAZEL_USER_OUTPUT_ROOT = {cache!r}\n"
                "DAZEL_DELEGATED_VOLUME = False\n"
                "DAZEL_IMAGE_TAG = 'pinned'\n",
            )

            with mock.patch.dict(
                os.environ,
                {"DAZEL_RC_FILE": rc_path, "DAZEL_DIRECTORY": workspace},
                clear=False,
            ):
                # _find_workspace_directory is @functools.cache'd; bust it for this test.
                dazel._find_workspace_directory.cache_clear()
                di = dazel.DockerInstance.from_config()

            self.assertEqual(di.instance_name, "unit-test")
            self.assertEqual(di.image_name, "unit-image")
            self.assertEqual(di.network, "host")
            self.assertEqual(di.image_tag, "pinned")
            self.assertIn("/x:/X", di.volumes)


# --------------------------------------------------------------------------------------
# Subprocess wrappers (`_run_argv`, `_run_capture`, `_run_command`, `_run_silent_command`)
# --------------------------------------------------------------------------------------


class SubprocessWrapperTest(unittest.TestCase):
    """Pin the subprocess plumbing used by every higher-level docker call.

    These wrappers are thin but load-bearing - the kwargs they pass to
    ``subprocess.call`` / ``subprocess.run`` / ``subprocess.check_output``
    determine quiet-mode behavior, timeout handling, and stream capture for
    every other method on the class.
    """

    def test_run_argv_streams_to_stderr_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch("subprocess.call", return_value=7) as m:
                rc = di._run_argv(["docker", "ps"])
            self.assertEqual(rc, 7)
            args, kwargs = m.call_args
            self.assertEqual(args[0], ["docker", "ps"])
            self.assertIs(kwargs["stdout"], sys.stderr)

    def test_run_argv_ignore_output_routes_to_devnull(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch("subprocess.call", return_value=0) as m:
                di._run_argv(["docker", "stop", "x"], ignore_output=True)
            _, kwargs = m.call_args
            self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)

    def test_run_silent_command_default_routes_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch("subprocess.call", return_value=0) as m:
                di._run_silent_command("echo hi")
            args, kwargs = m.call_args
            self.assertEqual(args[0], "echo hi")
            self.assertTrue(kwargs.get("shell"))
            self.assertIs(kwargs["stdout"], sys.stderr)

    def test_run_silent_command_ignore_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch("subprocess.call", return_value=0) as m:
                di._run_silent_command("echo hi", ignore_output=True)
            _, kwargs = m.call_args
            self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)

    def test_run_command_returns_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch("subprocess.check_output", return_value="hi\n") as m:
                self.assertEqual(di._run_command("echo hi"), "hi\n")
            _, kwargs = m.call_args
            self.assertTrue(kwargs["shell"])
            self.assertTrue(kwargs["text"])
            self.assertEqual(kwargs["timeout"], dazel.DEFAULT_DOCKER_TIMEOUT_S)

    def test_run_capture_uses_capture_output_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            fake = subprocess.CompletedProcess(args=["docker", "ps"], returncode=0, stdout="OK\n", stderr="")
            with mock.patch("subprocess.run", return_value=fake) as m:
                result = di._run_capture(["docker", "ps"])
            self.assertEqual(result.stdout, "OK\n")
            args, kwargs = m.call_args
            self.assertEqual(args[0], ["docker", "ps"])
            self.assertTrue(kwargs["capture_output"])
            self.assertTrue(kwargs["text"])
            self.assertEqual(kwargs["timeout"], dazel.DEFAULT_DOCKER_TIMEOUT_S)


# --------------------------------------------------------------------------------------
# Cancel forwarding for `send_command`
# --------------------------------------------------------------------------------------


class RunWithCancelForwardingTest(unittest.TestCase):
    """`_run_with_cancel_forwarding` installs SIGINT/SIGTERM handlers, waits on the child,
    then restores the prior handlers. Ensure each branch does what it says."""

    def _fake_proc(self, return_code: int = 0) -> mock.Mock:
        proc = mock.Mock()
        proc.wait.return_value = return_code
        return proc

    def test_clean_exit_returns_child_rc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with (
                mock.patch("subprocess.Popen", return_value=self._fake_proc(0)),
                mock.patch("signal.signal", return_value=None),
            ):
                self.assertEqual(di._run_with_cancel_forwarding(["true"]), 0)

    def test_negative_rc_remapped_to_128_minus_rc(self) -> None:
        # Signal-killed children come back as rc < 0 (e.g. SIGINT -> -2);
        # the wrapper maps that to shell convention 128+N (impl: 128 - rc).
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with (
                mock.patch("subprocess.Popen", return_value=self._fake_proc(-2)),
                mock.patch("signal.signal", return_value=None),
            ):
                self.assertEqual(di._run_with_cancel_forwarding(["x"]), 130)

    def test_signal_handlers_installed_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with (
                mock.patch("subprocess.Popen", return_value=self._fake_proc(0)),
                mock.patch("signal.signal", return_value=None) as m_sig,
            ):
                di._run_with_cancel_forwarding(["x"])
            # On non-win32: SIGINT install + SIGTERM install + SIGINT restore + SIGTERM restore = 4
            # On win32: SIGINT install + SIGINT restore = 2
            expected = 4 if sys.platform != "win32" else 2
            self.assertEqual(m_sig.call_count, expected)

    def test_cancel_handler_forwards_bazel_shutdown(self) -> None:
        """The captured SIGINT handler must docker-exec ``bazel shutdown`` exactly once."""
        captured: dict[str, Any] = {}

        def fake_signal(sig: int, handler: Any) -> Any:
            if sig == signal.SIGINT and "int" not in captured:
                captured["int"] = handler
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            proc = mock.Mock()

            def trigger_cancel() -> int:
                # Simulate Ctrl-C while the child is "running": fire the captured handler twice to also exercise the
                # cancel_sent guard.
                captured["int"](signal.SIGINT, None)
                captured["int"](signal.SIGINT, None)
                return 0

            proc.wait.side_effect = trigger_cancel
            with (
                mock.patch("subprocess.Popen", return_value=proc),
                mock.patch("signal.signal", side_effect=fake_signal),
                mock.patch("subprocess.run") as m_run,
            ):
                rc = di._run_with_cancel_forwarding(["x"])
            self.assertEqual(rc, 0)
            self.assertEqual(m_run.call_count, 1, "second SIGINT must be a no-op due to cancel_sent guard")
            args = m_run.call_args.args[0]
            self.assertEqual(args, ["docker", "exec", di.instance_name, "bazel", "shutdown"])

    def test_cancel_handler_swallows_subprocess_errors(self) -> None:
        """If forwarding ``bazel shutdown`` itself raises, the handler must not propagate."""
        captured: dict[str, Any] = {}

        def fake_signal(sig: int, handler: Any) -> Any:
            if sig == signal.SIGINT and "int" not in captured:
                captured["int"] = handler
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            proc = mock.Mock()

            def trigger_cancel() -> int:
                captured["int"](signal.SIGINT, None)
                return 0

            proc.wait.side_effect = trigger_cancel
            with (
                mock.patch("subprocess.Popen", return_value=proc),
                mock.patch("signal.signal", side_effect=fake_signal),
                mock.patch("subprocess.run", side_effect=subprocess.SubprocessError("kaboom")),
            ):
                # Must not raise.
                self.assertEqual(di._run_with_cancel_forwarding(["x"]), 0)


# --------------------------------------------------------------------------------------
# `send_command` shell-string composition
# --------------------------------------------------------------------------------------


class SendCommandTest(unittest.TestCase):
    """`send_command` builds a single shell-string and runs it via cancel-forwarding.
    These tests pin the exact substrings every consumer depends on (output_user_root
    plumbing, env-var propagation, quiet-mode redirect, etc.).
    """

    def _capture_command(self, di: dazel.DockerInstance, args: list[str], **kw: Any) -> str:
        captured: dict[str, str] = {}

        def fake(cmd: Any, **kwargs: Any) -> int:
            captured["cmd"] = cmd
            return 0

        with mock.patch.object(type(di), "_run_with_cancel_forwarding", side_effect=fake):
            di.send_command(args, **kw)
        return captured["cmd"]

    def test_basic_composition_includes_docker_exec_and_args(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, user="alice", run_command=None)
            cmd = self._capture_command(di, ["build", "//..."])
            self.assertIn("docker exec -i", cmd)
            self.assertIn("--user=alice", cmd)
            self.assertIn(di.instance_name, cmd)
            self.assertIn('"build" "//..."', cmd)

    def test_quiet_mode_appends_redirect(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, user="alice")
            cmd = self._capture_command(di, ["test"], quiet=True)
            self.assertTrue(cmd.endswith(" > /dev/null 2>&1"))

    def test_env_vars_rendered_quoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, user="alice", raw_env_vars=["FOO=1", "BAR=baz qux"])
            cmd = self._capture_command(di, ["build"])
            self.assertIn('-e "FOO=1"', cmd)
            self.assertIn('-e "BAR=baz qux"', cmd)

    def test_privileged_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, user="alice", docker_run_privileged=True)
            cmd = self._capture_command(di, ["build"])
            self.assertIn("--privileged", cmd)

    def test_no_user_uses_temp_output_user_root(self) -> None:
        # When user is empty, the command embeds TEMP_BAZEL_OUTPUT_USER_ROOT and an output_base.
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, user="")
            cmd = self._capture_command(di, ["build"])
            self.assertIn(f"--output_user_root={dazel.TEMP_BAZEL_OUTPUT_USER_ROOT}", cmd)

    def test_user_uses_per_user_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, user="alice")
            cmd = self._capture_command(di, ["build"])
            self.assertIn(f"--output_user_root={di.bazel_user_output_root}", cmd)

    def test_bazelrc_flag_when_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, user="alice", bazel_rc_file=".bazelrc")
            cmd = self._capture_command(di, ["build"])
            self.assertIn("--bazelrc=.bazelrc", cmd)


# --------------------------------------------------------------------------------------
# Argv-assembly for `_run_container` and `_run_dep_container`
# --------------------------------------------------------------------------------------


def _last_run_argv(mock_run_argv: mock.Mock) -> list[str]:
    """Return the argv list from the LAST call to a mocked `_run_argv`.

    `_run_container` and `_run_dep_container` precede the real ``docker run``
    with best-effort ``stop`` + ``rm`` cleanup calls, so the actual argv we
    care about is always the last one.
    """
    return mock_run_argv.call_args_list[-1].args[0]


class RunContainerArgvTest(unittest.TestCase):
    def test_minimal_argv_with_repository(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_file = os.path.join(tmpdir, ".dazel_run")
            di = _make_instance(tmpdir, repository="myrepo", dazel_run_file=run_file)
            with mock.patch.object(type(di), "_run_argv", return_value=0) as m:
                rc = di._run_container()
            self.assertEqual(rc, 0)
            argv = _last_run_argv(m)
            self.assertEqual(argv[:4], ["docker", "run", "-id", "--name=dazel"])
            self.assertEqual(argv[-2], "myrepo/dazel:test-tag")
            self.assertEqual(argv[-1], "/bin/bash")
            # Side effect: dazel_run_file written with instance_name.
            self.assertTrue(os.path.exists(run_file))
            with open(run_file) as f:
                self.assertEqual(f.read().strip(), "dazel")

    def test_no_repository_omits_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(
                tmpdir,
                repository=None,
                dazel_run_file=os.path.join(tmpdir, ".dazel_run"),
            )
            with mock.patch.object(type(di), "_run_argv", return_value=0) as m:
                di._run_container()
            argv = _last_run_argv(m)
            # Image token should be just "name:tag" without a "repo/" prefix.
            self.assertIn("dazel:test-tag", argv)
            self.assertNotIn("/dazel:test-tag", " ".join(argv))

    def test_all_optional_flags_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(
                tmpdir,
                user="alice",
                docker_run_privileged=True,
                platform="linux/amd64",
                shm_size="4g",
                raw_volumes=["/x:/X"],
                raw_ports=["0.0.0.0:80:80"],
                raw_env_vars=["A=1", "B=2"],
                raw_gpus=["0", "1"],
                network="custom-net",
                dazel_run_file=os.path.join(tmpdir, ".dazel_run"),
            )
            with mock.patch.object(type(di), "_run_argv", return_value=0) as m:
                di._run_container()
            argv = _last_run_argv(m)
            self.assertIn("--privileged", argv)
            self.assertIn("--user=alice", argv)
            self.assertIn("--platform=linux/amd64", argv)
            self.assertIn("--shm-size=4g", argv)
            self.assertIn("--net=custom-net", argv)
            # `--gpus 0,1` joined with comma.
            gpus_idx = argv.index("--gpus")
            self.assertEqual(argv[gpus_idx + 1], "0,1")
            # Repeated -v / -p / -e flags.
            self.assertEqual(argv.count("-p"), 1)
            self.assertEqual(argv.count("-e"), 2)
            self.assertGreaterEqual(argv.count("-v"), 1)

    def test_nonzero_run_rc_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_file = os.path.join(tmpdir, ".dazel_run")
            di = _make_instance(tmpdir, dazel_run_file=run_file)
            # Stop/rm return 0 (ignore_output), the actual run returns 5.
            calls = [0, 0, 5]
            with mock.patch.object(type(di), "_run_argv", side_effect=lambda *a, **kw: calls.pop(0)):
                rc = di._run_container()
            self.assertEqual(rc, 5)
            # The dazel_run_file must NOT be written when run failed.
            self.assertFalse(os.path.exists(run_file))


class RunDepContainerArgvTest(unittest.TestCase):
    def test_argv_with_env_and_volumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, network="dazel")
            dep = dazel.RunDep.parse(
                "dazel",
                {
                    "image": "postgres:14",
                    "name": "db",
                    "env": {"POSTGRES_PASSWORD": "x"},
                    "volumes": ["data:/var/lib/postgresql/data"],
                },
            )
            with mock.patch.object(type(di), "_run_argv", return_value=0) as m:
                rc = di._run_dep_container(dep)
            self.assertEqual(rc, 0)
            argv = _last_run_argv(m)
            self.assertEqual(argv[:4], ["docker", "run", "-id", "--name=db"])
            self.assertIn("--net=dazel", argv)
            self.assertIn("-e", argv)
            self.assertIn("POSTGRES_PASSWORD=x", argv)
            self.assertIn("-v", argv)
            self.assertIn("data:/var/lib/postgresql/data", argv)
            self.assertEqual(argv[-1], "postgres:14")

    def test_no_network_omits_net_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, network="")
            dep = dazel.RunDep.parse("dazel", {"image": "redis:7", "name": "cache"})
            with mock.patch.object(type(di), "_run_argv", return_value=0) as m:
                di._run_dep_container(dep)
            argv = _last_run_argv(m)
            self.assertFalse(any(a.startswith("--net=") for a in argv))


# --------------------------------------------------------------------------------------
# Image / network / dep helpers (`_image_exists`, `_pull`, `_build`, `_network_exists`,
# `_start_network`, `_dep_is_running`, `_dump_run_dep_logs`, `_docker_*_exists`)
# --------------------------------------------------------------------------------------


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class ImageAndNetworkHelpersTest(unittest.TestCase):
    def test_image_exists_true_when_ls_has_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, repository="repo")
            # `docker image ls` prints a header + one row per match.
            stdout = "REPOSITORY  TAG  IMAGE ID  CREATED  SIZE\nrepo/dazel  latest  abc  ...  ...\n"
            with mock.patch.object(type(di), "_run_capture", return_value=_completed(stdout)):
                self.assertTrue(di._image_exists())

    def test_image_exists_false_when_only_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, repository="repo")
            stdout = "REPOSITORY  TAG  IMAGE ID  CREATED  SIZE\n"
            with mock.patch.object(type(di), "_run_capture", return_value=_completed(stdout)):
                self.assertFalse(di._image_exists())

    def test_image_exists_includes_repo_prefix_when_repository_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, repository="myrepo", image_name="build")
            with mock.patch.object(type(di), "_run_capture", return_value=_completed("")) as m:
                di._image_exists()
            self.assertEqual(m.call_args.args[0], ["docker", "image", "ls", "myrepo/build"])

    def test_image_exists_omits_repo_prefix_when_repository_empty(self) -> None:
        """Regression: empty repository must not produce a leading-slash reference like '/build'.

        In standalone / OSS-portable mode the dazel image is built locally with no remote
        registry. The previous formatting produced an invalid Docker reference and `image ls`
        silently returned no rows, forcing a re-build that then failed in `_get_build_command`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, repository="", image_name="build")
            with mock.patch.object(type(di), "_run_capture", return_value=_completed("")) as m:
                di._image_exists()
            argv = m.call_args.args[0]
            self.assertEqual(argv, ["docker", "image", "ls", "build"])
            self.assertNotIn("/build", argv, f"leading-slash leaked into argv: {argv!r}")

    def test_pull_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, repository="myrepo")
            with mock.patch.object(type(di), "_run_argv", return_value=0) as m:
                self.assertEqual(di._pull(), 0)
            argv = m.call_args.args[0]
            self.assertEqual(argv, ["docker", "pull", "myrepo/dazel"])

    def test_pull_without_repository_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, repository=None)
            with self.assertRaisesRegex(RuntimeError, "No repository"):
                di._pull()

    def test_build_invokes_run_argv_with_build_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(tmpdir, dockerfile=dockerfile)
            with mock.patch.object(type(di), "_run_argv", return_value=0) as m:
                self.assertEqual(di._build(), 0)
            argv = m.call_args.args[0]
            self.assertEqual(argv[:2], ["docker", "build"])
            self.assertEqual(argv[-1], di.directory)

    def test_network_exists_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, network="dazel")
            with mock.patch.object(type(di), "_run_capture", return_value=_completed("NETWORK ID NAME\n123 dazel\n")):
                self.assertTrue(di._network_exists())

    def test_network_exists_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, network="dazel")
            with mock.patch.object(type(di), "_run_capture", return_value=_completed("NETWORK ID NAME\n123 other\n")):
                self.assertFalse(di._network_exists())

    def test_start_network_argv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, network="dazel")
            with mock.patch.object(type(di), "_run_argv", return_value=0) as m:
                self.assertEqual(di._start_network(), 0)
            argv = m.call_args.args[0]
            self.assertEqual(argv, ["docker", "network", "create", "dazel"])

    def test_start_network_empty_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, network="")
            with mock.patch.object(type(di), "_run_argv") as m:
                self.assertEqual(di._start_network(), 0)
            m.assert_not_called()

    def test_docker_exists_delegates_to_command_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dazel._command_exists.cache_clear()
            with mock.patch("shutil.which", return_value="/usr/bin/docker"):
                self.assertTrue(di._docker_exists())
            dazel._command_exists.cache_clear()
            with mock.patch("shutil.which", return_value=None):
                self.assertFalse(di._docker_exists())
            dazel._command_exists.cache_clear()

    def test_docker_compose_exists_delegates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dazel._command_exists.cache_clear()
            with mock.patch("shutil.which", return_value="/usr/local/bin/docker-compose"):
                self.assertTrue(di._docker_compose_exists())
            dazel._command_exists.cache_clear()


class DepIsRunningTest(unittest.TestCase):
    def test_true_when_name_and_image_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = dazel.RunDep.parse("net", {"image": "img", "name": "dep1"})
            with mock.patch.object(type(di), "_run_capture", return_value=_completed("dep1\timg\nother\tother-img\n")):
                self.assertTrue(di._dep_is_running(dep))

    def test_false_when_only_other_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = dazel.RunDep.parse("net", {"image": "img", "name": "dep1"})
            with mock.patch.object(type(di), "_run_capture", return_value=_completed("dep2\timg\nother\timg\n")):
                self.assertFalse(di._dep_is_running(dep))

    def test_false_when_image_differs(self) -> None:
        """A container holding the right name but the wrong image must not satisfy the dep."""
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = dazel.RunDep.parse("net", {"image": "img", "name": "dep1"})
            with (
                mock.patch.object(type(di), "_run_capture", return_value=_completed("dep1\tother-img\n")),
                self.assertLogs(dazel.logger, level="WARNING"),
            ):
                self.assertFalse(di._dep_is_running(dep))

    def test_mismatch_warning_names_both_images(self) -> None:
        """The warning has to identify what is running and what was expected, or a stale sidecar stays invisible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = dazel.RunDep.parse("net", {"image": "database:1.13.0", "name": "cosmos-eval-local-db"})
            with (
                mock.patch.object(
                    type(di), "_run_capture", return_value=_completed("cosmos-eval-local-db\tdatabase:1.12.0\n")
                ),
                self.assertLogs(dazel.logger, level="WARNING") as logs,
            ):
                self.assertFalse(di._dep_is_running(dep))
            self.assertEqual(len(logs.output), 1)
            self.assertIn("database:1.12.0", logs.output[0])
            self.assertIn("database:1.13.0", logs.output[0])

    def test_false_when_no_container_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = dazel.RunDep.parse("net", {"image": "img", "name": "dep1"})
            with mock.patch.object(type(di), "_run_capture", return_value=_completed("")):
                self.assertFalse(di._dep_is_running(dep))


class DumpRunDepLogsTest(unittest.TestCase):
    def test_writes_header_body_and_footer_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            buf = io.StringIO()
            with (
                mock.patch.object(type(di), "_run_capture", return_value=_completed("body line\n", "stderr line\n")),
                mock.patch.object(sys, "stderr", buf),
            ):
                di._dump_run_dep_logs("mydep", n=10)
            out = buf.getvalue()
            self.assertIn("--- last 10 log lines from run dependency 'mydep' ---", out)
            self.assertIn("body line", out)
            self.assertIn("stderr line", out)
            self.assertIn("--- end of 'mydep' logs ---", out)

    def test_appends_newlines_when_streams_lack_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            buf = io.StringIO()
            with (
                mock.patch.object(type(di), "_run_capture", return_value=_completed("nope", "fail")),
                mock.patch.object(sys, "stderr", buf),
            ):
                di._dump_run_dep_logs("mydep")
            out = buf.getvalue()
            # There should be a newline between the body and the trailing "--- end" footer
            # even when neither stdout nor stderr ended with one.
            self.assertIn("nope\n", out)
            self.assertIn("fail\n", out)


# --------------------------------------------------------------------------------------
# `_wait_for_ready_probe` loop
# --------------------------------------------------------------------------------------


class WaitForReadyProbeTest(unittest.TestCase):
    """The probe loop has 4 distinct exit paths: empty cmd / probe success / container
    died / timeout. Pin all of them."""

    def _dep(self, **probe_kwargs: Any) -> dazel.RunDep:
        return dazel.RunDep.parse(
            "net",
            {
                "image": "img",
                "name": "dep",
                "ready_probe": {
                    "cmd": ["echo", "ok"],
                    "timeout_s": 5,
                    "interval_s": 0.001,
                    "progress_every_s": 1,
                    "description": "ready",
                    **probe_kwargs,
                },
            },
        )

    def test_empty_cmd_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = dazel.RunDep.parse("net", {"image": "img", "name": "dep"})
            self.assertEqual(di._wait_for_ready_probe(dep), 0)

    def test_succeeds_on_first_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = self._dep()
            captures = [
                _completed("running\n"),  # status check
                _completed("", returncode=0),  # probe rc=0
            ]
            with (
                mock.patch.object(type(di), "_run_capture", side_effect=captures),
                mock.patch("time.sleep") as m_sleep,
            ):
                self.assertEqual(di._wait_for_ready_probe(dep), 0)
            m_sleep.assert_not_called()

    def test_succeeds_after_initial_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = self._dep()
            captures = [
                _completed("running\n"),  # status (1)
                _completed("", returncode=1),  # probe fails
                _completed("running\n"),  # status (2)
                _completed("", returncode=0),  # probe succeeds
            ]
            with (
                mock.patch.object(type(di), "_run_capture", side_effect=captures),
                mock.patch("time.sleep"),
            ):
                self.assertEqual(di._wait_for_ready_probe(dep), 0)

    def test_returns_one_when_container_died(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = self._dep()
            captures = [
                _completed("exited\n"),  # status: not running
                # _dump_run_dep_logs will also call _run_capture for `docker logs`
                _completed("logs body\n"),
            ]
            with (
                mock.patch.object(type(di), "_run_capture", side_effect=captures),
                mock.patch("time.sleep"),
                mock.patch.object(sys, "stderr", io.StringIO()),
            ):
                self.assertEqual(di._wait_for_ready_probe(dep), 1)

    def test_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = self._dep(timeout_s=0)  # immediate timeout on first iter

            # Container is up but the probe never succeeds. The loop will:
            #   1. status -> running
            #   2. probe -> rc=1
            #   3. (timeout check) -> dump logs -> return 1
            def capture_factory() -> Any:
                yield _completed("running\n")
                yield _completed("", returncode=1)
                # _dump_run_dep_logs will also call _run_capture
                while True:
                    yield _completed("logs\n")

            with (
                mock.patch.object(type(di), "_run_capture", side_effect=capture_factory()),
                mock.patch("time.sleep"),
                mock.patch.object(sys, "stderr", io.StringIO()),
            ):
                self.assertEqual(di._wait_for_ready_probe(dep), 1)


# --------------------------------------------------------------------------------------
# `is_running` and `start` orchestration
# --------------------------------------------------------------------------------------


class IsRunningTest(unittest.TestCase):
    def test_returns_false_when_ps_does_not_list_instance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch.object(type(di), "_run_capture", return_value=_completed("CONTAINER ID NAMES\n")):
                self.assertFalse(di.is_running())

    def test_returns_false_when_image_tag_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want")
            captures = [
                _completed(f"...{di.instance_name}...\n"),  # ps shows running
                _completed("repo/dazel:other\n"),  # inspect Image -> wrong tag
            ]
            with mock.patch.object(type(di), "_run_capture", side_effect=captures):
                self.assertFalse(di.is_running())

    def test_returns_false_when_workspace_bind_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want")
            captures = [
                _completed(f"...{di.instance_name}...\n"),  # ps
                _completed("repo/dazel:want\n"),  # image
                _completed('["/elsewhere:/elsewhere"]\n'),  # binds
            ]
            with mock.patch.object(type(di), "_run_capture", side_effect=captures):
                self.assertFalse(di.is_running())

    def test_returns_false_when_extra_bind_missing(self) -> None:
        """Regression for stale long-lived containers that predate a `.dazelrc` mount-config change. The workspace bind
        still matches, but a newly-required bind (e.g. the bazel cache root) isn't present on the running container.
        Without this check dazel would silently `docker exec` into the stale container and bazel inside would fail with
        an obscure ENOENT when it tries to create its `--output_user_root` directory.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want", raw_volumes=["/extra-cache:/extra-cache"])
            real = os.path.realpath(di.directory)
            captures = [
                _completed(f"...{di.instance_name}...\n"),  # ps
                _completed("repo/dazel:want\n"),  # image
                # Workspace bind is present but the /extra-cache bind expected by the current config is missing.
                _completed(f'["{real}:{real}:rw"]\n'),
            ]
            with mock.patch.object(type(di), "_run_capture", side_effect=captures):
                self.assertFalse(di.is_running())

    def test_returns_false_when_only_prefix_collision_present(self) -> None:
        """The bind check matches anchored at the `host:container` boundary, not as an unanchored substring; otherwise
        an existing bind that just shares a path prefix (e.g. `/cache:/cache-other:rw`) would falsely satisfy the
        check for a different bind (`/cache:/cache`).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want", raw_volumes=["/cache:/cache"])
            real = os.path.realpath(di.directory)
            captures = [
                _completed(f"...{di.instance_name}...\n"),
                _completed("repo/dazel:want\n"),
                # Workspace bind matches; the only other entry shares `/cache:/cache` as a prefix but is a distinct
                # bind to `/cache-other`, so the `/cache:/cache` config volume must still be reported as missing.
                _completed(f'["{real}:{real}:rw", "/cache:/cache-other:rw"]\n'),
            ]
            with mock.patch.object(type(di), "_run_capture", side_effect=captures):
                self.assertFalse(di.is_running())

    def test_returns_false_when_inspect_returns_non_json(self) -> None:
        """`docker inspect` can return empty/non-JSON output when the container disappears between the `ps` and
        `inspect` calls. We fall back to an empty bind list, which classifies the container as stale so the caller
        recreates it cleanly instead of crashing in `json.loads`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want")
            captures = [
                _completed(f"...{di.instance_name}...\n"),
                _completed("repo/dazel:want\n"),
                _completed("not-json\n"),
            ]
            with mock.patch.object(type(di), "_run_capture", side_effect=captures):
                self.assertFalse(di.is_running())

    def test_returns_true_when_everything_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want", network="dazel")
            real = os.path.realpath(di.directory)
            captures = [
                _completed(f"...{di.instance_name}...\n"),
                _completed("repo/dazel:want\n"),
                _completed(f'["{real}:{real}:rw"]\n'),
                _completed("{dazel:0xC0FFEE}\n"),  # networks
            ]
            with mock.patch.object(type(di), "_run_capture", side_effect=captures):
                self.assertTrue(di.is_running())

    def test_returns_false_when_gpus_requested_but_container_has_none(self) -> None:
        """Regression for GPU jobs inheriting a container that a `DAZEL_GPUS="" dazel ...` command created. Devices can
        only be attached at `docker run` time, so reusing it makes every CUDA call inside fail with cudaErrorNoDevice.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want", network="dazel", raw_gpus="all")
            real = os.path.realpath(di.directory)
            captures = [
                _completed(f"...{di.instance_name}...\n"),
                _completed("repo/dazel:want\n"),
                _completed(f'["{real}:{real}:rw"]\n'),
                _completed("{dazel:0xC0FFEE}\n"),
                _completed("null\n"),  # DeviceRequests -> container was started without --gpus
            ]
            with mock.patch.object(type(di), "_run_capture", side_effect=captures) as capture:
                self.assertFalse(di.is_running())
            # Guard against the assertion above passing for the wrong reason: every other check has to have been
            # satisfied, leaving DeviceRequests as the one that rejected the container.
            self.assertIn("--format={{json .HostConfig.DeviceRequests}}", capture.call_args.args[0])

    def test_returns_false_when_device_requests_inspect_returns_non_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want", network="dazel", raw_gpus="all")
            real = os.path.realpath(di.directory)
            captures = [
                _completed(f"...{di.instance_name}...\n"),
                _completed("repo/dazel:want\n"),
                _completed(f'["{real}:{real}:rw"]\n'),
                _completed("{dazel:0xC0FFEE}\n"),
                _completed("not-json\n"),
            ]
            with (
                mock.patch.object(type(di), "_run_capture", side_effect=captures),
                self.assertLogs(dazel.logger, level="WARNING"),
            ):
                self.assertFalse(di.is_running())

    def test_returns_true_when_gpus_requested_and_container_holds_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want", network="dazel", raw_gpus="all")
            real = os.path.realpath(di.directory)
            captures = [
                _completed(f"...{di.instance_name}...\n"),
                _completed("repo/dazel:want\n"),
                _completed(f'["{real}:{real}:rw"]\n'),
                _completed("{dazel:0xC0FFEE}\n"),
                _completed('[{"Driver":"","Count":-1,"DeviceIDs":null,"Capabilities":[["gpu"]],"Options":{}}]\n'),
            ]
            with mock.patch.object(type(di), "_run_capture", side_effect=captures):
                self.assertTrue(di.is_running())

    def test_returns_true_when_container_holds_gpus_this_invocation_does_not_want(self) -> None:
        """A GPU-holding container serves a GPU-less command fine. Recreating it would make alternating GPU/non-GPU jobs
        on one host tear the container down on every invocation, so the staleness check is intentionally one-directional
        (and never inspects DeviceRequests at all when no GPU was requested).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want", network="dazel")
            real = os.path.realpath(di.directory)
            captures = [
                _completed(f"...{di.instance_name}...\n"),
                _completed("repo/dazel:want\n"),
                _completed(f'["{real}:{real}:rw"]\n'),
                _completed("{dazel:0xC0FFEE}\n"),
            ]
            with mock.patch.object(type(di), "_run_capture", side_effect=captures) as capture:
                self.assertTrue(di.is_running())
            self.assertNotIn("DeviceRequests", str(capture.call_args_list))

    def test_returns_true_when_all_extra_binds_present(self) -> None:
        """Counterpart to `test_returns_false_when_extra_bind_missing`: every bind the current config declares is
        present on the running container, so it should be considered live.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, image_tag="want", network="dazel", raw_volumes=["/extra-cache:/extra-cache"])
            real = os.path.realpath(di.directory)
            captures = [
                _completed(f"...{di.instance_name}...\n"),
                _completed("repo/dazel:want\n"),
                _completed(f'["{real}:{real}:rw", "/extra-cache:/extra-cache:rw"]\n'),
                _completed("{dazel:0xC0FFEE}\n"),
            ]
            with mock.patch.object(type(di), "_run_capture", side_effect=captures):
                self.assertTrue(di.is_running())


class StartTest(unittest.TestCase):
    """`start` is a thin orchestrator. We mock every leaf method and assert on the
    branching: build vs pull, compose vs network+run-deps, etc."""

    def test_returns_one_when_docker_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch.object(type(di), "_docker_exists", return_value=False):
                self.assertEqual(di.start(), 1)

    def test_build_path_when_dockerfile_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(tmpdir, dockerfile=dockerfile, network="host")
            with (
                mock.patch.object(type(di), "_docker_exists", return_value=True),
                mock.patch.object(type(di), "_build", return_value=0) as m_build,
                mock.patch.object(type(di), "_pull") as m_pull,
                mock.patch.object(type(di), "_start_run_deps", return_value=0),
                mock.patch.object(type(di), "_run_container", return_value=0),
            ):
                self.assertEqual(di.start(), 0)
            m_build.assert_called_once()
            m_pull.assert_not_called()

    def test_pull_path_when_dockerfile_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(
                tmpdir,
                dockerfile=os.path.join(tmpdir, "no_such_dockerfile"),
                network="host",
            )
            with (
                mock.patch.object(type(di), "_docker_exists", return_value=True),
                mock.patch.object(type(di), "_pull", return_value=0) as m_pull,
                mock.patch.object(type(di), "_image_exists", return_value=True),
                mock.patch.object(type(di), "_start_run_deps", return_value=0),
                mock.patch.object(type(di), "_run_container", return_value=0),
            ):
                self.assertEqual(di.start(), 0)
            m_pull.assert_called_once()

    def test_pull_failure_tolerated_when_image_locally_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(
                tmpdir,
                dockerfile=os.path.join(tmpdir, "missing"),
                network="host",
            )
            with (
                mock.patch.object(type(di), "_docker_exists", return_value=True),
                mock.patch.object(type(di), "_pull", return_value=2),  # pull failed...
                mock.patch.object(type(di), "_image_exists", return_value=True),  # but image exists
                mock.patch.object(type(di), "_start_run_deps", return_value=0),
                mock.patch.object(type(di), "_run_container", return_value=0) as m_run,
            ):
                self.assertEqual(di.start(), 0)
            m_run.assert_called_once()

    def test_creates_network_when_missing_and_not_predefined(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(tmpdir, dockerfile=dockerfile, network="custom")
            with (
                mock.patch.object(type(di), "_docker_exists", return_value=True),
                mock.patch.object(type(di), "_build", return_value=0),
                mock.patch.object(type(di), "_network_exists", return_value=False),
                mock.patch.object(type(di), "_start_network", return_value=0) as m_net,
                mock.patch.object(type(di), "_start_run_deps", return_value=0),
                mock.patch.object(type(di), "_run_container", return_value=0),
            ):
                self.assertEqual(di.start(), 0)
            m_net.assert_called_once()


# --------------------------------------------------------------------------------------
# `_start_compose_services` / `_start_run_deps`
# --------------------------------------------------------------------------------------


class StartComposeServicesTest(unittest.TestCase):
    def test_no_compose_file_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, docker_compose_file="")
            self.assertEqual(di._start_compose_services(), 0)

    def test_pulls_builds_and_brings_up(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(
                tmpdir,
                docker_compose_file="/path/to/docker-compose.yml",
                docker_compose_command="docker-compose",
                docker_compose_project_name="dazel",
                raw_docker_compose_services=["api", "web"],
            )
            with mock.patch.object(type(di), "_run_silent_command", return_value=0) as m:
                self.assertEqual(di._start_compose_services(), 0)
            cmd = m.call_args.args[0]
            self.assertIn("docker-compose -f /path/to/docker-compose.yml pull", cmd)
            self.assertIn("docker-compose -f /path/to/docker-compose.yml build", cmd)
            self.assertIn("docker-compose -f /path/to/docker-compose.yml up --force-recreate -d", cmd)
            self.assertIn("api web", cmd)
            self.assertIn("COMPOSE_PROJECT_NAME=dazel", cmd)


class StartRunDepsTest(unittest.TestCase):
    def test_starts_only_deps_that_arent_already_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(
                tmpdir,
                raw_run_deps=[
                    {"image": "imgA", "name": "depA"},
                    {"image": "imgB", "name": "depB"},
                ],
            )
            running_deps = {"depA"}
            with (
                mock.patch.object(type(di), "_dep_is_running", side_effect=lambda d: d.name in running_deps),
                mock.patch.object(type(di), "_run_dep_container", return_value=0) as m_start,
                mock.patch.object(type(di), "_wait_for_ready_probe", return_value=0),
            ):
                self.assertEqual(di._start_run_deps(), 0)
            # Only depB should have been started.
            self.assertEqual(m_start.call_count, 1)
            self.assertEqual(m_start.call_args.args[0].name, "depB")

    def test_propagates_start_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, raw_run_deps=[{"image": "img", "name": "dep"}])
            with (
                mock.patch.object(type(di), "_dep_is_running", return_value=False),
                mock.patch.object(type(di), "_run_dep_container", return_value=5) as m_start,
                mock.patch.object(type(di), "_wait_for_ready_probe") as m_probe,
            ):
                self.assertEqual(di._start_run_deps(), 5)
            m_start.assert_called_once()
            m_probe.assert_not_called()


# --------------------------------------------------------------------------------------
# `_add_volumes` / `_build_run_deps` extra branches
# --------------------------------------------------------------------------------------


class AddVolumesBranchesTest(unittest.TestCase):
    def test_delegated_volume_creates_per_user_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "ws")
            cache = os.path.join(tmpdir, "cache")
            os.makedirs(workspace)
            di = dazel.DockerInstance(
                directory=workspace,
                bazel_user_output_root=cache,
                delegated_volume=True,
                dockerfile=os.path.join(tmpdir, "Dockerfile"),
                image_tag="t",
                workspace_hex=True,
            )
            # Per-output dirs (external, action_cache, execroot, <workspace_basename>)
            # must exist after __post_init__ when delegated_volume is on.
            base = os.path.join(cache, di.workspace_hex_digest)
            for sub in dazel.DEFAULT_BAZEL_USER_OUTPUT_PATHS + [os.path.basename(workspace)]:
                self.assertTrue(os.path.isdir(os.path.join(base, sub)), f"missing {sub}")
            # Each per-output bind appears in volumes with :delegated.
            self.assertTrue(any(":delegated" in v for v in di.volumes))


class BuildRunDepsErrorTest(unittest.TestCase):
    def test_invalid_type_raises(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self.assertRaisesRegex(TypeError, "comma-separated string"),
        ):
            _make_instance(tmpdir, raw_run_deps=42)


# --------------------------------------------------------------------------------------
# Workspace + config file resolution
# --------------------------------------------------------------------------------------


class FindWorkspaceDirectoryTest(unittest.TestCase):
    def test_walks_up_to_module_bazel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            sub = os.path.join(ws, "a", "b", "c")
            os.makedirs(sub)
            _touch(os.path.join(ws, "MODULE.bazel"), "")
            dazel._find_workspace_directory.cache_clear()
            self.assertEqual(dazel._find_workspace_directory(sub), ws)

    def test_raises_when_no_module_bazel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = os.path.realpath(os.path.join(tmpdir, "x"))
            os.makedirs(sub)
            dazel._find_workspace_directory.cache_clear()
            with self.assertRaises(FileNotFoundError):
                dazel._find_workspace_directory(sub)


class ConfigFromFilesTest(unittest.TestCase):
    def test_returns_only_directory_when_no_dazelrc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            os.makedirs(ws)
            _touch(os.path.join(ws, "MODULE.bazel"), "")
            dazel._find_workspace_directory.cache_clear()
            with mock.patch.dict(
                os.environ,
                {"DAZEL_DIRECTORY": ws, "DAZEL_RC_FILE": os.path.join(tmpdir, "no-such-file")},
                clear=False,
            ):
                cfg = dazel.DockerInstance._config_from_files()
            self.assertEqual(cfg, {"DAZEL_DIRECTORY": ws})

    def test_user_dazelrc_overrides_dazelrc(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            os.makedirs(ws)
            _touch(os.path.join(ws, "MODULE.bazel"), "")
            _touch(
                os.path.join(ws, ".dazelrc"),
                "DAZEL_INSTANCE_NAME = 'base'\nDAZEL_VOLUMES = ['/x:/X']\n",
            )
            _touch(
                os.path.join(ws, "user.dazelrc"),
                "DAZEL_VOLUMES += ['/y:/Y']\nDAZEL_INSTANCE_NAME = 'override'\n",
            )
            dazel._find_workspace_directory.cache_clear()
            with mock.patch.dict(os.environ, {"DAZEL_DIRECTORY": ws}, clear=False):
                if "DAZEL_RC_FILE" in os.environ:
                    del os.environ["DAZEL_RC_FILE"]
                cfg = dazel.DockerInstance._config_from_files()
            self.assertEqual(cfg["DAZEL_INSTANCE_NAME"], "override")
            self.assertEqual(cfg["DAZEL_VOLUMES"], ["/x:/X", "/y:/Y"])


# --------------------------------------------------------------------------------------
# Module-level helpers (`_update_apt_lockfiles`, `_update_requirements_txt`,
# `_update_gazelle_manifest`, `_run_linter`, `main`)
# --------------------------------------------------------------------------------------


class UpdateAptLockfilesTest(unittest.TestCase):
    def test_no_module_bazel_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "ws"))
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            di = _make_instance(tmpdir)
            dazel._find_workspace_directory.cache_clear()
            _touch(os.path.join(ws, "MODULE.bazel"), "")
            os.remove(os.path.join(ws, "MODULE.bazel"))  # workspace has nothing
            with mock.patch.object(dazel.DockerInstance, "find_workspace_directory", return_value=ws):
                self.assertEqual(dazel._update_apt_lockfiles(di), 0)

    def test_regenerates_each_apt_install(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            os.makedirs(ws)
            _touch(
                os.path.join(ws, "MODULE.bazel"),
                'apt.install(\n    name = "egomotion_packages",\n)\napt.install(\n    name = "vlm_packages",\n)\n',
            )
            di = _make_instance(tmpdir)
            with (
                mock.patch.object(dazel.DockerInstance, "find_workspace_directory", return_value=ws),
                mock.patch.object(type(di), "send_command", return_value=0) as m,
            ):
                self.assertEqual(dazel._update_apt_lockfiles(di), 0)
            self.assertEqual(m.call_count, 2)
            self.assertEqual(m.call_args_list[0].args[0], ["run", "@egomotion_packages//:lock"])
            self.assertEqual(m.call_args_list[1].args[0], ["run", "@vlm_packages//:lock"])

    def test_propagates_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            os.makedirs(ws)
            _touch(os.path.join(ws, "MODULE.bazel"), 'apt.install(name = "pkg")\n')
            di = _make_instance(tmpdir)
            with (
                mock.patch.object(dazel.DockerInstance, "find_workspace_directory", return_value=ws),
                mock.patch.object(type(di), "send_command", return_value=3),
            ):
                self.assertEqual(dazel._update_apt_lockfiles(di), 3)


class UpdateRequirementsTxtTest(unittest.TestCase):
    def test_in_sync_returns_zero_without_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch.object(type(di), "send_command", side_effect=[0]) as m:
                self.assertEqual(dazel._update_requirements_txt(di), 0)
            self.assertEqual(m.call_count, 1)
            self.assertEqual(m.call_args.args[0], ["test", "//:requirements.test", "--test_output=errors"])

    def test_out_of_sync_runs_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch.object(type(di), "send_command", side_effect=[1, 0]) as m:
                self.assertEqual(dazel._update_requirements_txt(di), 0)
            self.assertEqual(m.call_count, 2)
            self.assertEqual(
                m.call_args_list[1].args[0],
                ["run", "//:requirements.update"],
            )

    def test_update_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch.object(type(di), "send_command", side_effect=[1, 7]):
                self.assertEqual(dazel._update_requirements_txt(di), 7)


class UpdateGazelleManifestTest(unittest.TestCase):
    def test_in_sync_returns_zero_without_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch.object(type(di), "send_command", side_effect=[0]) as m:
                self.assertEqual(dazel._update_gazelle_manifest(di), 0)
            self.assertEqual(m.call_count, 1)
            self.assertEqual(
                m.call_args.args[0],
                ["test", "//:gazelle_python_manifest.test", "--test_output=errors"],
            )

    def test_out_of_sync_runs_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch.object(type(di), "send_command", side_effect=[1, 0]) as m:
                self.assertEqual(dazel._update_gazelle_manifest(di), 0)
            self.assertEqual(m.call_count, 2)
            self.assertEqual(m.call_args_list[1].args[0], ["run", "//:gazelle_python_manifest.update"])

    def test_update_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            with mock.patch.object(type(di), "send_command", side_effect=[1, 7]):
                self.assertEqual(dazel._update_gazelle_manifest(di), 7)


class RunLinterTest(unittest.TestCase):
    def test_invokes_linter_script_from_di_attribute(self) -> None:
        # DAZEL_LINTER_SCRIPT comes off the DockerInstance (sourced from .dazelrc + user.dazelrc + env merged at
        # DockerInstance construction). _run_linter invokes that path verbatim, with targets and --apply-patches
        # (in fix mode) forwarded, and the workspace dir set as cwd.
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            os.makedirs(ws)
            wrapper = os.path.join(tmpdir, "custom_wrapper.sh")
            _touch(wrapper)
            di = _make_instance(tmpdir, linter_script=wrapper)
            with (
                mock.patch.object(dazel.DockerInstance, "find_workspace_directory", return_value=ws),
                mock.patch.object(type(di), "_run_with_cancel_forwarding", return_value=0) as m,
            ):
                self.assertEqual(dazel._run_linter(di, ["//foo:bar"], fix_mode=True), 0)
            argv = m.call_args.args[0]
            self.assertEqual(argv[0], wrapper)
            self.assertIn("--apply-patches", argv)
            self.assertIn("//foo:bar", argv)
            self.assertEqual(m.call_args.kwargs["cwd"], ws)

    def test_returns_error_when_linter_script_unset(self) -> None:
        # With no linter_script configured on the DockerInstance, dazel has no wrapper to invoke; rather than
        # guessing a default, _run_linter logs an actionable error and returns 1 so the operator knows to set
        # DAZEL_LINTER_SCRIPT in their workspace .dazelrc (or via env-var override).
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            os.makedirs(ws)
            di = _make_instance(tmpdir)  # linter_script defaults to ""
            with (
                mock.patch.object(dazel.DockerInstance, "find_workspace_directory", return_value=ws),
                mock.patch.object(type(di), "_run_with_cancel_forwarding") as m,
            ):
                self.assertEqual(dazel._run_linter(di, ["//foo:bar"]), 1)
            m.assert_not_called()

    def test_handles_subprocess_error(self) -> None:
        # When the linter wrapper invocation itself raises (e.g. the script path is unreadable), _run_linter
        # converts the subprocess/OS error into a non-zero exit code rather than propagating the exception.
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            os.makedirs(ws)
            wrapper = os.path.join(tmpdir, "custom_wrapper.sh")
            _touch(wrapper)
            di = _make_instance(tmpdir, linter_script=wrapper)
            with (
                mock.patch.object(dazel.DockerInstance, "find_workspace_directory", return_value=ws),
                mock.patch.object(type(di), "_run_with_cancel_forwarding", side_effect=OSError("boom")),
            ):
                self.assertEqual(dazel._run_linter(di, []), 1)


class MainDispatchTest(unittest.TestCase):
    """`main()` is essentially a switch between three commands: lint, fix, and 'pass-through'.
    Mock the underlying methods to verify each branch dispatches correctly without
    actually invoking docker."""

    def _stub_di(self) -> mock.MagicMock:
        di = mock.MagicMock(spec=dazel.DockerInstance)
        di.dazel_run_file = "/tmp/.never-exists"
        di.dockerfile = "/tmp/.never-exists"
        di.is_running.return_value = True
        di.start.return_value = 0
        di._start_run_deps.return_value = 0
        di.send_command.return_value = 0
        return di

    def test_passthrough_command_calls_send_command(self) -> None:
        di = self._stub_di()
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "build", "//..."]),
        ):
            self.assertEqual(dazel.main(), 0)
        di.send_command.assert_called_once_with(["build", "//..."])

    def test_lint_command_calls_run_linter(self) -> None:
        di = self._stub_di()
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "lint", "//foo"]),
            mock.patch.object(dazel, "_run_linter", return_value=0) as m,
        ):
            self.assertEqual(dazel.main(), 0)
        m.assert_called_once_with(di, ["//foo"])
        di.send_command.assert_not_called()

    def test_fix_command_runs_full_pipeline(self) -> None:
        di = self._stub_di()
        # The formatter target comes off di.formatter_target (sourced from .dazelrc). Set it to the consumer-style
        # label so this test exercises main()'s dispatch logic without depending on a real .dazelrc.
        di.formatter_target = "//tools/format:format_multirun"
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "fix"]),
            mock.patch.object(dazel, "_update_requirements_txt", return_value=0) as m_req,
            mock.patch.object(dazel, "_update_gazelle_manifest", return_value=0) as m_gaz,
            mock.patch.object(dazel, "_update_apt_lockfiles", return_value=0) as m_apt,
            mock.patch.object(dazel, "_run_linter", return_value=0) as m_lint,
        ):
            # Attach the pipeline-stage mocks to a shared parent so we can assert not just that each ran once
            # but that they ran in the required order: requirements -> gazelle manifest -> apt lockfiles.
            manager = mock.Mock()
            manager.attach_mock(m_req, "req")
            manager.attach_mock(m_gaz, "gaz")
            manager.attach_mock(m_apt, "apt")
            self.assertEqual(dazel.main(), 0)
        m_req.assert_called_once()
        m_gaz.assert_called_once()
        m_apt.assert_called_once()
        self.assertEqual(manager.mock_calls, [mock.call.req(di), mock.call.gaz(di), mock.call.apt(di)])
        # _run_linter is called twice in fix mode: once with fix_mode=True (apply
        # patches) and once afterwards as a verification pass.
        self.assertEqual(m_lint.call_count, 2)
        self.assertTrue(m_lint.call_args_list[0].kwargs.get("fix_mode"))
        # send_command gets called for the formatter step.
        di.send_command.assert_called_once_with(["run", "//tools/format:format_multirun"])

    def test_fix_command_aborts_when_formatter_target_unset(self) -> None:
        # If di.formatter_target is empty (no .dazelrc entry, no env override), the formatter step bails out with
        # an actionable error and rc=1 — and crucially, it does so AFTER the first lint pass has already run
        # successfully (today's behavior; documented so a future refactor doesn't accidentally re-order).
        di = self._stub_di()
        di.formatter_target = ""
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "fix"]),
            mock.patch.object(dazel, "_update_requirements_txt", return_value=0),
            mock.patch.object(dazel, "_update_gazelle_manifest", return_value=0),
            mock.patch.object(dazel, "_update_apt_lockfiles", return_value=0),
            mock.patch.object(dazel, "_run_linter", return_value=0) as m_lint,
        ):
            self.assertEqual(dazel.main(), 1)
        m_lint.assert_called_once()
        self.assertTrue(m_lint.call_args.kwargs.get("fix_mode"))
        di.send_command.assert_not_called()

    def test_fix_command_aborts_on_lint_failure(self) -> None:
        di = self._stub_di()
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "fix"]),
            mock.patch.object(dazel, "_update_requirements_txt", return_value=0),
            mock.patch.object(dazel, "_update_gazelle_manifest", return_value=0),
            mock.patch.object(dazel, "_update_apt_lockfiles", return_value=0),
            mock.patch.object(dazel, "_run_linter", return_value=4),
        ):
            self.assertEqual(dazel.main(), 4)

    def test_starts_container_when_not_running(self) -> None:
        # Exercises the cold-start branch in `main` (start is called and its rc gates main).
        di = self._stub_di()
        di.is_running.return_value = False
        di.start.return_value = 0
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "test", "//x"]),
        ):
            self.assertEqual(dazel.main(), 0)
        di.start.assert_called_once()

    def test_aborts_when_start_fails(self) -> None:
        di = self._stub_di()
        di.is_running.return_value = False
        di.start.return_value = 9
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "build"]),
        ):
            self.assertEqual(dazel.main(), 9)
        di.send_command.assert_not_called()

    def test_aborts_when_run_deps_fail(self) -> None:
        di = self._stub_di()
        di.is_running.return_value = True
        di._start_run_deps.return_value = 11
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "build"]),
        ):
            self.assertEqual(dazel.main(), 11)
        di.send_command.assert_not_called()

    def test_fix_command_aborts_when_requirements_update_fails(self) -> None:
        di = self._stub_di()
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "fix"]),
            mock.patch.object(dazel, "_update_requirements_txt", return_value=2),
        ):
            self.assertEqual(dazel.main(), 2)

    def test_fix_command_aborts_when_gazelle_manifest_update_fails(self) -> None:
        # The gazelle manifest step sits between requirements and apt lockfiles; a nonzero rc must short-circuit
        # main() with that same code and skip the later apt step entirely.
        di = self._stub_di()
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "fix"]),
            mock.patch.object(dazel, "_update_requirements_txt", return_value=0),
            mock.patch.object(dazel, "_update_gazelle_manifest", return_value=5),
            mock.patch.object(dazel, "_update_apt_lockfiles", return_value=0) as m_apt,
        ):
            self.assertEqual(dazel.main(), 5)
        m_apt.assert_not_called()

    def test_fix_command_aborts_when_apt_update_fails(self) -> None:
        di = self._stub_di()
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "fix"]),
            mock.patch.object(dazel, "_update_requirements_txt", return_value=0),
            mock.patch.object(dazel, "_update_gazelle_manifest", return_value=0),
            mock.patch.object(dazel, "_update_apt_lockfiles", return_value=3),
        ):
            self.assertEqual(dazel.main(), 3)

    def test_fix_command_aborts_when_formatter_fails(self) -> None:
        di = self._stub_di()
        di.send_command.return_value = 6  # formatter step
        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "fix"]),
            mock.patch.object(dazel, "_update_requirements_txt", return_value=0),
            mock.patch.object(dazel, "_update_gazelle_manifest", return_value=0),
            mock.patch.object(dazel, "_update_apt_lockfiles", return_value=0),
            mock.patch.object(dazel, "_run_linter", return_value=0),
        ):
            self.assertEqual(dazel.main(), 6)

    def test_fix_command_reports_remaining_violations(self) -> None:
        # Verify the verification-pass branch: fix-mode lint succeeds (rc=0) but the
        # follow-up check-mode lint reports new violations.
        di = self._stub_di()

        def fake_lint(_di: Any, _args: list[str], fix_mode: bool = False) -> int:
            return 0 if fix_mode else 1  # apply OK, verify FAIL

        with (
            mock.patch.object(dazel.DockerInstance, "from_config", return_value=di),
            mock.patch.object(sys, "argv", ["dazel", "fix"]),
            mock.patch.object(dazel, "_update_requirements_txt", return_value=0),
            mock.patch.object(dazel, "_update_gazelle_manifest", return_value=0),
            mock.patch.object(dazel, "_update_apt_lockfiles", return_value=0),
            mock.patch.object(dazel, "_run_linter", side_effect=fake_lint),
        ):
            self.assertEqual(dazel.main(), 1)


# --------------------------------------------------------------------------------------
# Additional coverage for `start`, `_start_run_deps`, `_run_linter`, `_add_volumes` edges
# --------------------------------------------------------------------------------------


class StartFailureBranchesTest(unittest.TestCase):
    def test_build_failure_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(tmpdir, dockerfile=dockerfile, network="host")
            with (
                mock.patch.object(type(di), "_docker_exists", return_value=True),
                mock.patch.object(type(di), "_build", return_value=2),
                mock.patch.object(type(di), "_run_container") as m_run,
            ):
                self.assertEqual(di.start(), 2)
            m_run.assert_not_called()

    def test_compose_path_when_compose_file_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(
                tmpdir,
                dockerfile=dockerfile,
                docker_compose_file="/path/docker-compose.yml",
                network="host",
            )
            with (
                mock.patch.object(type(di), "_docker_exists", return_value=True),
                mock.patch.object(type(di), "_build", return_value=0),
                mock.patch.object(type(di), "_docker_compose_exists", return_value=True),
                mock.patch.object(type(di), "_start_compose_services", return_value=0) as m_compose,
                mock.patch.object(type(di), "_run_container", return_value=0),
            ):
                self.assertEqual(di.start(), 0)
            m_compose.assert_called_once()

    def test_network_create_failure_short_circuits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(tmpdir, dockerfile=dockerfile, network="custom")
            with (
                mock.patch.object(type(di), "_docker_exists", return_value=True),
                mock.patch.object(type(di), "_build", return_value=0),
                mock.patch.object(type(di), "_network_exists", return_value=False),
                mock.patch.object(type(di), "_start_network", return_value=8),
                mock.patch.object(type(di), "_run_container") as m_run,
            ):
                self.assertEqual(di.start(), 8)
            m_run.assert_not_called()


class StartRunDepsProbeFailureTest(unittest.TestCase):
    def test_probe_failure_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir, raw_run_deps=[{"image": "img", "name": "dep"}])
            with (
                mock.patch.object(type(di), "_dep_is_running", return_value=True),
                mock.patch.object(type(di), "_run_dep_container") as m_start,
                mock.patch.object(type(di), "_wait_for_ready_probe", return_value=4),
            ):
                self.assertEqual(di._start_run_deps(), 4)
            # Already-running dep means start_dep_container is NOT called even though the probe ultimately failed; only
            # the probe rc propagates.
            m_start.assert_not_called()


class RunLinterFailureLoggingTest(unittest.TestCase):
    def test_nonzero_rc_logs_and_returns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            os.makedirs(ws)
            wrapper = os.path.join(tmpdir, "custom_wrapper.sh")
            _touch(wrapper)
            di = _make_instance(tmpdir, linter_script=wrapper)
            with (
                mock.patch.object(dazel.DockerInstance, "find_workspace_directory", return_value=ws),
                mock.patch.object(type(di), "_run_with_cancel_forwarding", return_value=7),
            ):
                self.assertEqual(dazel._run_linter(di, []), 7)


class UpdateAptLockfilesNoNameTest(unittest.TestCase):
    def test_apt_install_without_name_attribute_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = os.path.realpath(os.path.join(tmpdir, "ws"))
            os.makedirs(ws)
            _touch(
                os.path.join(ws, "MODULE.bazel"),
                # First apt.install() lacks `name = "..."` and should be silently skipped; second has a name and must
                # be regenerated.
                "apt.install(\n    pkg = 'foo',\n)\napt.install(\n    name = \"vlm_packages\",\n)\n",
            )
            di = _make_instance(tmpdir)
            with (
                mock.patch.object(dazel.DockerInstance, "find_workspace_directory", return_value=ws),
                mock.patch.object(type(di), "send_command", return_value=0) as m,
            ):
                self.assertEqual(dazel._update_apt_lockfiles(di), 0)
            self.assertEqual(m.call_count, 1)
            self.assertEqual(m.call_args.args[0], ["run", "@vlm_packages//:lock"])


class AddVolumesNoUserOutputRootTest(unittest.TestCase):
    def test_falls_back_to_real_bazelout_bind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "ws")
            os.makedirs(workspace)
            di = dazel.DockerInstance(
                directory=workspace,
                bazel_user_output_root="",
                delegated_volume=False,
                dockerfile=os.path.join(tmpdir, "Dockerfile"),
                image_tag="t",
            )
            # Without bazel_user_output_root, _add_volumes binds real_bazelout to itself
            # and stores it as bazel_output_base.
            self.assertTrue(di.bazel_output_base)
            self.assertTrue(any(di.bazel_output_base in v for v in di.volumes))


class StartRunDepsFailureBubblesUpTest(unittest.TestCase):
    def test_run_deps_failure_short_circuits_start(self) -> None:
        # Pin the `if rc: return rc` after `_start_run_deps` in `start()`.
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = _make_instance(tmpdir, dockerfile=dockerfile, network="host")
            with (
                mock.patch.object(type(di), "_docker_exists", return_value=True),
                mock.patch.object(type(di), "_build", return_value=0),
                mock.patch.object(type(di), "_start_run_deps", return_value=12),
                mock.patch.object(type(di), "_run_container") as m_run,
            ):
                self.assertEqual(di.start(), 12)
            m_run.assert_not_called()


class WaitForReadyProbeProgressLoggingTest(unittest.TestCase):
    def test_emits_still_waiting_log_after_progress_window(self) -> None:
        # `_wait_for_ready_probe` periodically logs a "still waiting" line every `progress_every_s` seconds. Mock
        # `time.monotonic` to advance past that threshold on the second iteration so the log branch is exercised.
        # threshold on the second iteration so the log branch is exercised.
        with tempfile.TemporaryDirectory() as tmpdir:
            di = _make_instance(tmpdir)
            dep = dazel.RunDep.parse(
                "net",
                {
                    "image": "img",
                    "name": "dep",
                    "ready_probe": {
                        "cmd": ["true"],
                        "timeout_s": 60,
                        "interval_s": 0,
                        "progress_every_s": 2,
                        "description": "ready",
                    },
                },
            )
            captures = [
                _completed("running\n"),
                _completed("", returncode=1),  # iter 1 probe fails
                _completed("running\n"),
                _completed("", returncode=1),  # iter 2 probe fails
                _completed("running\n"),
                _completed("", returncode=0),  # iter 3 probe succeeds
            ]
            # Time progression: start=0, iter1=0, iter2=5 (>=progress_every_s), iter3=6
            times = iter([0.0, 0.0, 0.0, 5.0, 5.0, 6.0, 6.0, 6.0])
            with (
                mock.patch.object(type(di), "_run_capture", side_effect=captures),
                mock.patch("time.sleep"),
                mock.patch("time.monotonic", side_effect=lambda: next(times)),
                mock.patch.object(dazel.logger, "info") as m_log,
            ):
                self.assertEqual(di._wait_for_ready_probe(dep), 0)
            still_waiting_calls = [c for c in m_log.call_args_list if "still waiting" in c.args[0]]
            self.assertGreaterEqual(len(still_waiting_calls), 1)


class AddVolumesBazelCacheInferenceTest(unittest.TestCase):
    def test_user_output_root_inferred_from_real_bazelout(self) -> None:
        # When the workspace lives under `<bazel_cache>/_bazel_<user>/<hex>/execroot/...`, `_add_volumes` reconstructs
        # the bazel cache root from the path. Build that exact directory structure in a tempdir and confirm the
        # inference fires.
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = os.path.join(tmpdir, "cache")
            user_root = os.path.join(cache_root, "_bazel_alice")
            workspace = os.path.join(user_root, "abc123", "execroot", "ws")
            os.makedirs(workspace)
            di = dazel.DockerInstance(
                directory=workspace,
                bazel_user_output_root="",  # force inference
                delegated_volume=False,
                dockerfile=os.path.join(tmpdir, "Dockerfile"),
                image_tag="t",
            )
            self.assertEqual(di.bazel_user_output_root, user_root)


class WorkspaceHexWithDockerfileTest(unittest.TestCase):
    def test_image_name_is_suffixed_when_dockerfile_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = os.path.join(tmpdir, "ws")
            cache = os.path.join(tmpdir, "cache")
            os.makedirs(workspace)
            os.makedirs(cache)
            dockerfile = os.path.join(tmpdir, "Dockerfile")
            _touch(dockerfile, "FROM scratch\n")
            di = dazel.DockerInstance(
                directory=workspace,
                bazel_user_output_root=cache,
                delegated_volume=False,
                dockerfile=dockerfile,
                image_tag="t",
                workspace_hex=True,
            )
            self.assertTrue(di.image_name.startswith("dazel_"))
            self.assertEqual(len(di.image_name), len("dazel_") + 32)


if __name__ == "__main__":
    unittest.main()
