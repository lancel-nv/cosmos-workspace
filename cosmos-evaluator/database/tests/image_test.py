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

"""Integration test for the generic //database:image PostgreSQL container image.

Boots the OCI image with only the standard POSTGRES_* contract (no BACKUP_FILE) and verifies the generic postgres
init / schema-apply / app-user / grant path that the startup script in //database:BUILD performs.

Regression guards:
  * Required env validation: the script aborts unless POSTGRES_USER/PASSWORD/DB are set.
  * Empty-database init: with no BACKUP_FILE the container still boots, creating the target database and the
    limited application role.
  * Schema application: the bundled orchestration schema (listed in SCHEMA_DIR/tables_manifest.txt) is applied on
    every boot, so an existing data directory keeps up with the schema files instead of staying frozen at the schema
    it was created with. Initialization and the dump restore remain first-boot-only.
  * GRANT scope + default privileges: the app user gets SELECT/INSERT/UPDATE/DELETE on tables created by the
    owner, but no cluster-level DDL (CREATE ROLE).
  * Pinned major: the server is the PostgreSQL major the image exports as PG_MAJOR.
  * SCRAM authentication: remote connections are configured for scram-sha-256, authenticate with the role's
    password, and are refused with a wrong one.
  * Major mismatch: a data directory created by another major is rejected before PostgreSQL is downloaded, and the log
    states the remedy instead of a catalog-version error.
  * Restart install skip: a restarted container finds the packages its first boot installed instead of downloading
    them again.
"""

import asyncio
from collections.abc import Callable
import json
import logging
import subprocess
import time
from typing import ClassVar
import unittest

import asyncpg
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from utils.container_images import load_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_USER = "appuser"
APP_PASSWORD = "test-password-not-a-secret"
POSTGRES_DB = "appdb"
# A boot that installs PostgreSQL downloads it, so its duration follows the network. A boot that skips the install,
# because the packages are already present or because it rejects its data directory first, finishes in seconds.
STARTUP_TIMEOUT_SECONDS = 600
BOOT_WITHOUT_INSTALL_TIMEOUT_SECONDS = 120
SMOKE_TABLE = "smoke"

# init-db.sh logs exactly one of these on every boot that reaches the install step.
INSTALLING_MARKER = "Installing PostgreSQL"
INSTALL_SKIPPED_MARKER = "is already installed; skipping the package install."

# run-postgres.sh logs this only after the temporary server that init-db.sh and tables_manifest.sh ran against has
# been stopped and the foreground postgres has been launched. Every boot therefore serves on port 5432 twice, and
# this marker is the only thing that distinguishes the server that stays up from the one that is about to go away.
FOREGROUND_READY_MARKER = "PostgreSQL started"


def _image_pg_major(image_name: str) -> int:
    """Return the PostgreSQL major the image declares in its PG_MAJOR environment variable."""
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .Config.Env}}", image_name],
        capture_output=True,
        text=True,
        check=True,
    )
    for entry in json.loads(result.stdout):
        name, _, value = entry.partition("=")
        if name == "PG_MAJOR":
            return int(value)
    raise ValueError(f"Image {image_name} does not set PG_MAJOR")


def _boot_container(image_name: str) -> DockerContainer:
    """Start the image under the standard POSTGRES_* contract and block until postgres is serving."""
    container = DockerContainer(image_name)
    container.with_env("POSTGRES_USER", APP_USER)
    container.with_env("POSTGRES_PASSWORD", APP_PASSWORD)
    container.with_env("POSTGRES_DB", POSTGRES_DB)
    # Published so the SCRAM test can authenticate the way a remote client does: in-container connections over
    # 127.0.0.1 match initdb's trust entry and never reach the password rule.
    container.with_exposed_ports(5432)

    # The run-postgres phase logs the marker only after init-db, the manifest schema apply + grants, and the
    # foreground postgres start have all succeeded.
    container.waiting_for(LogMessageWaitStrategy(FOREGROUND_READY_MARKER).with_startup_timeout(STARTUP_TIMEOUT_SECONDS))
    container.start()
    return container


class _ContainerCommands:
    """Shared ways to run SQL and read logs inside a booted container.

    A mixin rather than a base ``TestCase`` so it is never collected as a test in its own right; the concrete classes
    below supply ``container`` and differ only in how they boot and what they assert.
    """

    container: ClassVar[DockerContainer | None] = None

    def _exec(self, *cmd: str) -> tuple[int, str]:
        """Run a command inside the running container and return (exit_code, output)."""
        assert self.container is not None
        wrapped = self.container.get_wrapped_container()
        result = wrapped.exec_run(list(cmd), demux=False)
        output: bytes = result.output if isinstance(result.output, bytes) else b""
        return int(result.exit_code or 0), output.decode("utf-8", errors="replace")

    def _psql_postgres(self, sql: str) -> tuple[int, str]:
        """Execute SQL inside the container as the postgres superuser against POSTGRES_DB."""
        psql_cmd = f'psql -h 127.0.0.1 -p 5432 -U postgres -d {POSTGRES_DB} -tAc "{sql}"'
        return self._exec("su", "postgres", "-c", psql_cmd)

    def _psql_app_user(self, sql: str) -> tuple[int, str]:
        """Execute SQL inside the container as the application user against POSTGRES_DB."""
        env_assign = f"PGPASSWORD={APP_PASSWORD}"
        psql_cmd = f'{env_assign} psql -h 127.0.0.1 -p 5432 -U {APP_USER} -d {POSTGRES_DB} -tAc "{sql}"'
        return self._exec("sh", "-c", psql_cmd)

    def _logs(self) -> str:
        assert self.container is not None
        stdout, stderr = self.container.get_logs()
        return stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")

    def _wait_until(self, ready: Callable[[], bool], timeout_seconds: int, expectation: str) -> None:
        """Poll ``ready`` once a second for up to ``timeout_seconds``, failing with the tail of the container logs."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if ready():
                return
            time.sleep(1)
        raise AssertionError(f"{expectation} within {timeout_seconds}s:\n{self._logs()[-2000:]}")


class TestImageGeneric(_ContainerCommands, unittest.TestCase):
    """Boot the generic image with the standard POSTGRES_* contract and no BACKUP_FILE."""

    image_name: ClassVar[str] = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.image_name = load_image("database/tests/test_image_load.sh")
        logger.info("Booting container from %s", cls.image_name)
        cls.container = _boot_container(cls.image_name)
        logger.info("Container is ready")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.container is not None:
            try:
                cls.container.stop()
            finally:
                cls.container = None
            logger.info("Container stopped")

    def test_started_with_empty_database(self) -> None:
        """With no BACKUP_FILE the script logs the empty-database path and still boots."""
        logs = self._logs()
        self.assertIn("No BACKUP_FILE provided", logs, f"Missing empty-DB log line in:\n{logs[-2000:]}")
        self.assertIn(f"Database: {POSTGRES_DB}", logs, f"Did not find target DB log in:\n{logs[-2000:]}")

    def test_app_user_can_connect(self) -> None:
        """The application role was created and can authenticate against the target database."""
        exit_code, output = self._psql_app_user("SELECT 1;")
        self.assertEqual(exit_code, 0, f"App-user connection failed: {output}")
        self.assertEqual(output.strip(), "1", f"Unexpected SELECT output: {output!r}")

    def test_orchestration_schema_applied(self) -> None:
        """The bundled orchestration schema was applied on first init: core tables exist and are app-user readable."""
        exit_code, output = self._psql_app_user("SELECT to_regclass('public.clients') IS NOT NULL;")
        self.assertEqual(exit_code, 0, f"App-user schema query failed: {output}")
        self.assertEqual(output.strip(), "t", f"Expected orchestration table 'clients' to exist; got: {output!r}")

    def test_app_user_can_insert_and_select_owner_table(self) -> None:
        """Default privileges + GRANT block let the app user read/write owner-created tables."""
        exit_code, output = self._psql_postgres(f"CREATE TABLE IF NOT EXISTS {SMOKE_TABLE} (id int, name text);")
        self.assertEqual(exit_code, 0, f"Owner CREATE TABLE failed: {output}")

        exit_code, output = self._psql_app_user(
            f"INSERT INTO {SMOKE_TABLE} (id, name) VALUES (1, 'sentinel'); "
            f"SELECT count(*) FROM {SMOKE_TABLE}; "
            f"DELETE FROM {SMOKE_TABLE} WHERE id = 1;"
        )
        self.assertEqual(exit_code, 0, f"App-user INSERT/SELECT/DELETE failed: {output}")

    def test_app_user_cannot_create_role(self) -> None:
        """The app user must lack cluster-level CREATEROLE privilege (createuser with no flags)."""
        exit_code, output = self._psql_app_user("CREATE ROLE image_test_role;")
        self.assertNotEqual(exit_code, 0, f"CREATE ROLE unexpectedly succeeded; output: {output}")
        self.assertIn(
            "permission denied",
            output.lower(),
            f"Expected 'permission denied' in CREATE ROLE output, got: {output!r}",
        )

    def test_server_runs_the_pinned_major(self) -> None:
        """The running server is the major the image declares, not whatever the base image's archive carries."""
        exit_code, output = self._psql_postgres("SHOW server_version_num;")
        self.assertEqual(exit_code, 0, f"Could not read the server version: {output}")
        self.assertEqual(int(output.strip()) // 10000, _image_pg_major(self.image_name), output)

    def test_remote_connections_require_scram(self) -> None:
        """Both remote host rules use scram-sha-256, and the application role holds a SCRAM verifier to match."""
        exit_code, output = self._psql_postgres(
            "SELECT DISTINCT auth_method FROM pg_hba_file_rules WHERE address IN ('0.0.0.0', '::');"
        )
        self.assertEqual(exit_code, 0, f"Could not read pg_hba rules: {output}")
        self.assertEqual(output.split(), ["scram-sha-256"], f"Unexpected remote auth methods: {output!r}")

        exit_code, output = self._psql_postgres(
            f"SELECT rolpassword LIKE 'SCRAM-SHA-256\\$%' FROM pg_authid WHERE rolname = '{APP_USER}';"
        )
        self.assertEqual(exit_code, 0, f"Could not read the role's verifier: {output}")
        self.assertEqual(output.strip(), "t", f"The application role's password is not a SCRAM verifier: {output!r}")

    def test_remote_login_authenticates_with_the_password(self) -> None:
        """A client outside the container gets in with the role's password and is refused with any other."""
        assert self.container is not None
        host = self.container.get_container_host_ip()
        port = int(self.container.get_exposed_port(5432))

        async def select_one(password: str) -> int:
            conn = await asyncpg.connect(
                host=host, port=port, user=APP_USER, password=password, database=POSTGRES_DB, timeout=10
            )
            try:
                return int(await conn.fetchval("SELECT 1"))
            finally:
                await conn.close()

        self.assertEqual(asyncio.run(select_one(APP_PASSWORD)), 1)
        with self.assertRaises(asyncpg.InvalidPasswordError):
            asyncio.run(select_one("not-the-password"))


class TestDataDirectoryFromAnotherMajorIsRejected(_ContainerCommands, unittest.TestCase):
    """A data directory created by a different PostgreSQL major must stop the boot with the remedy in the log.

    The directory is seeded inside the container, before the entrypoint runs, so the test needs no host bind mount.
    Its own container because the boot under test is meant to fail.
    """

    image_name: ClassVar[str] = ""
    stale_major: ClassVar[int] = 0

    @classmethod
    def setUpClass(cls) -> None:
        cls.image_name = load_image("database/tests/test_image_load.sh")
        cls.stale_major = _image_pg_major(cls.image_name) - 1
        container = DockerContainer(cls.image_name)
        container.with_env("POSTGRES_USER", APP_USER)
        container.with_env("POSTGRES_PASSWORD", APP_PASSWORD)
        container.with_env("POSTGRES_DB", POSTGRES_DB)
        container.with_kwargs(
            entrypoint=[
                "bash",
                "-c",
                "mkdir -p /var/lib/postgresql/data && "
                f"echo {cls.stale_major} > /var/lib/postgresql/data/PG_VERSION && "
                "exec /usr/local/bin/startup.sh",
            ]
        )
        container.start()
        cls.container = container

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.container is not None:
            try:
                cls.container.stop()
            finally:
                cls.container = None

    def _exit_code(self) -> int | None:
        """The container's exit code once its entrypoint has returned, or None while it is still running."""
        assert self.container is not None
        wrapped = self.container.get_wrapped_container()
        wrapped.reload()
        if str(wrapped.status) != "exited":
            return None
        return int(wrapped.attrs["State"]["ExitCode"])

    def test_boot_fails_and_names_the_remedy(self) -> None:
        self._wait_until(
            lambda: self._exit_code() is not None, BOOT_WITHOUT_INSTALL_TIMEOUT_SECONDS, "The container did not exit"
        )
        self.assertNotEqual(self._exit_code(), 0, self._logs()[-2000:])

        logs = self._logs()
        self.assertIn(f"was initialized by PostgreSQL {self.stale_major}", logs, logs[-2000:])
        self.assertIn("FORCE_REINIT=true", logs, logs[-2000:])
        self.assertNotIn(FOREGROUND_READY_MARKER, logs, "The server started on a data directory it cannot read")
        self.assertNotIn(INSTALLING_MARKER, logs, "The boot downloaded PostgreSQL before rejecting the data directory")


class TestSchemaReappliedOnRestart(_ContainerCommands, unittest.TestCase):
    """The schema files must reach an *existing* data directory, not just a freshly initialized one.

    Its own container because it restarts the thing it is testing, which would disturb the assertions above.
    """

    image_name: ClassVar[str] = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.image_name = load_image("database/tests/test_image_load.sh")
        cls.container = _boot_container(cls.image_name)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.container is not None:
            try:
                cls.container.stop()
            finally:
                cls.container = None

    def _restart_and_wait(self) -> None:
        """Restart postgres in place and block until the foreground server serves, leaving the data directory intact.

        Waiting on connectivity alone answers the wrong question. init-db.sh starts a temporary server on the same
        port and run-postgres.sh stops it again, so the first connection to succeed after a restart is usually to a
        server with about a tenth of a second left to live, and whatever the caller does next lands in the gap before
        the foreground server is listening. Counting the marker is what tells the two apart. Connectivity is then
        polled on top of it because run-postgres.sh logs the marker when it launches postgres, not when postgres
        begins answering.
        """
        assert self.container is not None
        boots_before = self._logs().count(FOREGROUND_READY_MARKER)
        self.container.get_wrapped_container().restart()
        self._wait_until(
            lambda: self._logs().count(FOREGROUND_READY_MARKER) > boots_before,
            BOOT_WITHOUT_INSTALL_TIMEOUT_SECONDS,
            "The foreground postgres did not start after the restart",
        )
        self._wait_until(
            lambda: self._psql_postgres("SELECT 1;")[0] == 0,
            BOOT_WITHOUT_INSTALL_TIMEOUT_SECONDS,
            "The foreground postgres did not accept a connection after the restart",
        )

    def test_a_column_missing_from_an_existing_data_directory_is_restored_on_restart(self) -> None:
        """Guards the failure where a months-old volume stays frozen at the schema it was created with.

        Dropping a coordination column stands in for the real case: a data directory created before that column
        existed. Initialization and the dump restore are correctly skipped on an existing directory, so if the schema
        files are skipped along with them the column never arrives, every CREATE TABLE IF NOT EXISTS is a no-op, and
        the only symptom is application code failing on a database that booted perfectly.
        """
        exit_code, output = self._psql_postgres("ALTER TABLE checker_executions DROP COLUMN claimed_by;")
        self.assertEqual(exit_code, 0, f"Could not drop the column under test: {output}")
        exit_code, output = self._psql_postgres(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = 'checker_executions' AND column_name = 'claimed_by';"
        )
        self.assertEqual(output.strip(), "0", f"Column was still present after the drop: {output!r}")

        self._restart_and_wait()

        exit_code, output = self._psql_postgres(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = 'checker_executions' AND column_name = 'claimed_by';"
        )
        self.assertEqual(exit_code, 0, f"Column lookup failed after restart: {output}")
        self.assertEqual(output.strip(), "1", "Restart did not re-apply the schema files to the existing data dir")

    def test_the_restart_skips_initialization_but_still_applies_the_schema(self) -> None:
        """The two halves are deliberately separate: the restore stays first-init-only, the schema files do not.

        Restarts rather than reading whatever the preceding test left behind: the skip line is only written on a second
        boot, so asserting over the logs alone would make this pass or fail on method ordering.
        """
        self._restart_and_wait()

        logs = self._logs()
        self.assertIn("Existing data found, skipping initialization and restore.", logs, logs[-2000:])
        self.assertIn("Applying schema file:", logs.split("Existing data found")[-1], logs[-2000:])

    def test_the_restart_does_not_download_postgres_again(self) -> None:
        """A restarted container keeps its filesystem, so it reuses the packages its first boot installed."""
        self._restart_and_wait()

        # The text between the last two readiness markers spans the shutdown before the restart and the boot it
        # triggered, so it holds this boot's install decision and no other.
        restarted_boot = self._logs().split(FOREGROUND_READY_MARKER)[-2]
        self.assertIn(INSTALL_SKIPPED_MARKER, restarted_boot, restarted_boot[-2000:])
        self.assertNotIn(INSTALLING_MARKER, restarted_boot, restarted_boot[-2000:])


if __name__ == "__main__":
    unittest.main()
