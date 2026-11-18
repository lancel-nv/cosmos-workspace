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

load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load", "oci_push")
load("@rules_pkg//pkg:mappings.bzl", "pkg_files")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_library")
load(":version.bzl", "VERSION")

# PostgreSQL major version the image installs. Exported to the container as PG_MAJOR, which selects the server
# binaries and is checked against the data directory on every boot. Changing it requires a dump and restore of every
# deployed data directory, and a major bump of VERSION.
PG_MAJOR = "18"

py_library(
    name = "api_models",
    srcs = ["api_models.py"],
    visibility = ["//visibility:public"],
    deps = ["@pypi//pydantic"],
)

py_library(
    name = "statuses",
    srcs = ["statuses.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "api_client",
    srcs = ["api_client.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":api_models",
        "@pypi//httpx",
    ],
)

filegroup(
    name = "config_yaml",
    srcs = ["config.yaml"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "db_config",
    srcs = ["db_config.py"],
    data = [":config_yaml"],
    visibility = ["//visibility:public"],
    deps = [
        "//utils:bazel",
        "@pypi//pyyaml",
    ],
)

# Generic PostgreSQL container startup script
#
#   init-db.sh         - validate the POSTGRES_* env contract and the data directory's major, install PostgreSQL unless
#                        it is already installed, initialize the cluster, create the limited application role, and
#                        optionally restore a dump. Leaves a TEMPORARY server running and exports NEEDS_INIT for the
#                        later phases. Applies no schemas and no grants.
#   tables_manifest.sh - on every boot, apply the schema files listed in the manifest (in listed order) and then
#                        grant the application role. Runs after restore (schemas top up a restored dump) and applies
#                        every manifest entry before the grant block, so GRANT ... ON ALL TABLES covers them all.
#   run-postgres.sh    - stop the temporary server and run postgres in the foreground under a graceful-shutdown trap.
#   startup.sh         - orchestrator + image entrypoint: source init-db.sh, source tables_manifest.sh, exec
#                        run-postgres.sh. Sourcing shares NEEDS_INIT and the POSTGRES_* contract across the phases.
#
# The schema *.sql files ship in //database/schemas:schema_layer; the ordered manifest ships separately in
# :manifest_layer (a dedicated final layer) so the load set/order can be overridden by layering a different
# tables_manifest.txt on top (or by pointing SCHEMA_DIR elsewhere). This image ships a manifest containing only
# orchestration.sql.
#
# This image carries no deployment-specific logic so it can be reused by any consumer that needs a self-contained
# PostgreSQL container.
#
# Environment variables:
#   POSTGRES_USER     (required) application role to create with limited privileges
#   POSTGRES_PASSWORD (required) password for the application role
#   POSTGRES_DB       (required) target database name to create / restore into
#   BACKUP_FILE       (optional) path to a `pg_dump -Fc` archive to restore on first init
#   RESTORE_SCHEMA_ONLY (optional) "true" restores only the schema (no table data) from BACKUP_FILE; intended for
#                     tests that exercise the restore path without the bulk-data cost. Leave unset to restore data.
#   FORCE_REINIT      (optional) "true" clears the data directory and reinitializes
#   SCHEMA_DIR        (optional) directory holding the schema files + manifest, applied on every boot (default
#                     /etc/postgres-schemas)
#   PG_MAJOR          (set by the image) PostgreSQL major version installed at startup. A data directory created by
#                     any other major is rejected before the server starts.
genrule(
    name = "init_db_script",
    outs = ["init-db.sh"],
    cmd = """
cat > $@ << 'EOF'
#!/bin/bash
# Phase 1 of the generic PostgreSQL lifecycle. Sourced by startup.sh. Validates the POSTGRES_* contract and the data
# directory's major, installs packages, initializes the data directory, starts a temporary server, creates the limited
# application role, and optionally restores a dump. Leaves the temporary server RUNNING and exports NEEDS_INIT for the
# later phases. Does NOT apply schemas or grants (tables_manifest.sh) and does NOT start the foreground server
# (run-postgres.sh).
set -e

: "$${PG_MAJOR:?PG_MAJOR must be set; the image defines it}"
PG_BIN="/usr/lib/postgresql/$$PG_MAJOR/bin"

# Validate required configuration.
if [ -z "$$POSTGRES_USER" ]; then
    echo "ERROR: POSTGRES_USER environment variable is required"
    exit 1
fi
if [ -z "$$POSTGRES_PASSWORD" ]; then
    echo "ERROR: POSTGRES_PASSWORD environment variable is required"
    exit 1
fi
if [ -z "$$POSTGRES_DB" ]; then
    echo "ERROR: POSTGRES_DB environment variable is required"
    exit 1
fi

# POSTGRES_USER and POSTGRES_DB are interpolated unquoted into SQL (CREATE DATABASE / ALTER USER / GRANT) and the
# createuser CLI below, so reject anything that is not a plain SQL identifier (a letter or underscore followed by
# letters, digits or underscores) to prevent command/SQL injection via these values.
if ! [[ "$$POSTGRES_USER" =~ ^[A-Za-z_][A-Za-z0-9_]*$$ ]]; then
    echo "ERROR: POSTGRES_USER must be a valid SQL identifier (letter/underscore then letters, digits or underscores)"
    exit 1
fi
if ! [[ "$$POSTGRES_DB" =~ ^[A-Za-z_][A-Za-z0-9_]*$$ ]]; then
    echo "ERROR: POSTGRES_DB must be a valid SQL identifier (letter/underscore then letters, digits or underscores)"
    exit 1
fi

echo "Configuring PostgreSQL"
echo "  User: $$POSTGRES_USER"
echo "  Database: $$POSTGRES_DB"

# PostgreSQL data directory
PGDATA="/var/lib/postgresql/data"

# Force reinitialization if requested.
if [ "$${FORCE_REINIT:-false}" = "true" ]; then
    echo "FORCE_REINIT=true: Clearing existing data directory..."
    # Defensively guard against an empty/unset PGDATA before recursive deletion (shellcheck SC2115). Using
    # `find -delete` instead of `rm -rf "$$PGDATA"/*` avoids any glob expansion against the filesystem root if
    # PGDATA is unset.
    if [ -z "$$PGDATA" ] || [ ! -d "$$PGDATA" ]; then
        echo "ERROR: PGDATA is empty or not a directory; refusing to wipe"
        exit 1
    fi
    find "$$PGDATA" -mindepth 1 -delete
fi

# A data directory records the major version that created it in PG_VERSION, and a server of any other major refuses
# to start on it. Initialization is skipped for any non-empty directory, so without this check the mismatch would
# surface as a catalog error from the server rather than as the remedy.
if [ -f "$$PGDATA/PG_VERSION" ]; then
    DATA_MAJOR="$$(cat "$$PGDATA/PG_VERSION")"
    if [ "$$DATA_MAJOR" != "$$PG_MAJOR" ]; then
        echo "ERROR: $$PGDATA was initialized by PostgreSQL $$DATA_MAJOR, but this image runs PostgreSQL $$PG_MAJOR."
        echo "A major-version change needs a new data directory: dump the databases with the previous image, start"
        echo "this image on an empty data directory (FORCE_REINIT=true deletes the existing one), and restore the dumps."
        exit 1
    fi
fi

# Install PostgreSQL $$PG_MAJOR from the PostgreSQL Global Development Group (PGDG) apt repository. The Ubuntu archive
# carries one major per release, so pinning a specific major requires the upstream repository. Its signing key ships
# in the image (:pgdg_key_layer) and is referenced through signed-by, which needs neither gnupg nor a key download at
# boot. The createcluster.d drop-in stops the server package from creating a default cluster of its own, since this
# script initializes PGDATA itself.
#
# The install follows every check above, so a misconfigured container or a data directory from another major fails
# without downloading anything. It is skipped when both packages are already installed, which holds when a container
# restarts with its filesystem intact; a new container always installs.
PG_PACKAGES_INSTALLED=true
for pg_package in "postgresql-$$PG_MAJOR" "postgresql-client-$$PG_MAJOR"; do
    if [ "$$(dpkg-query -W -f='$${Status}' "$$pg_package" 2>/dev/null)" != "install ok installed" ]; then
        PG_PACKAGES_INSTALLED=false
    fi
done
if [ "$$PG_PACKAGES_INSTALLED" = true ]; then
    echo "PostgreSQL $$PG_MAJOR is already installed; skipping the package install."
else
    echo "Installing PostgreSQL $$PG_MAJOR from the PGDG apt repository..."
    apt-get update && apt-get install -y --no-install-recommends ca-certificates
    PGDG_KEYRING="/usr/share/keyrings/apt.postgresql.org.asc"
    PGDG_CODENAME="$$(. /etc/os-release && echo "$$VERSION_CODENAME")"
    echo "deb [signed-by=$$PGDG_KEYRING] https://apt.postgresql.org/pub/repos/apt $${PGDG_CODENAME}-pgdg main" \
        > /etc/apt/sources.list.d/pgdg.list
    mkdir -p /etc/postgresql-common/createcluster.d
    echo "create_main_cluster = false" > /etc/postgresql-common/createcluster.d/no-main-cluster.conf
    apt-get update && apt-get install -y --no-install-recommends \
        "postgresql-$$PG_MAJOR" \
        "postgresql-client-$$PG_MAJOR" \
        && rm -rf /var/lib/apt/lists/*
fi

# Check if this is a fresh initialization (no existing data)
NEEDS_INIT=false
if [ ! -d "$$PGDATA" ] || [ -z "$$(ls -A $$PGDATA 2>/dev/null)" ]; then
    NEEDS_INIT=true
    mkdir -p "$$PGDATA"
    chown postgres:postgres "$$PGDATA"
    chmod 700 "$$PGDATA"

    # Initialize the database as the postgres user with UTF-8 encoding
    su postgres -c "$$PG_BIN/initdb -D $$PGDATA --encoding=UTF8 --locale=C.UTF-8"

    # Configure PostgreSQL to accept connections from any host (0.0.0.0)
    echo "listen_addresses = '*'" >> "$$PGDATA/postgresql.conf"
    echo "port = 5432" >> "$$PGDATA/postgresql.conf"

    # Allow password authentication from any IP address. SCRAM rather than md5: MD5 password authentication is
    # deprecated, and password_encryption defaults to scram-sha-256, so every role created here has a SCRAM verifier.
    echo "host all all 0.0.0.0/0 scram-sha-256" >> "$$PGDATA/pg_hba.conf"
    echo "host all all ::/0 scram-sha-256" >> "$$PGDATA/pg_hba.conf"
fi

# Connection budget, applied on every boot rather than only at initdb time. The initdb default of 100 leaves no
# margin: every Database REST API replica holds a pool of its own, and running out is a hard "too many clients
# already" that also locks out the operator diagnosing it, whereas an exhausted pool merely queues. A backend process
# costs ~5-10 MB, so the headroom is cheap. Keep pool_max_size x api_replicas, plus room for an operator's own
# session, under this number.
#
# Written here, before the temporary server starts, so both it and the foreground server run with the setting -- and
# re-written on every boot for the same reason the schema files are re-applied: gating it on first init left a
# long-lived data directory pinned to whatever it was created with, which surfaces far from its cause as
# "too many clients already" on a container that booted perfectly well. The prior line is dropped rather than appended
# to, so a data directory that has booted a hundred times still has exactly one. Run as postgres so the rewritten
# config keeps the ownership the server needs to read it.
su postgres -c "sed -i '/^max_connections *=/d' $$PGDATA/postgresql.conf"
su postgres -c "echo 'max_connections = 300' >> $$PGDATA/postgresql.conf"

# Start PostgreSQL temporarily to set up users/databases. It is left running for the schema-application phase
# (tables_manifest.sh); run-postgres.sh stops it before starting the foreground server.
su postgres -c "$$PG_BIN/pg_ctl -D $$PGDATA -w start"

# Only create user/database and restore on first initialization
if [ "$$NEEDS_INIT" = true ]; then
    echo "First-time initialization: creating user and database..."

    # Create the application user FIRST. A restored dump references the application role as the owner of every
    # object; if the role doesn't exist when pg_restore runs, every "ALTER ... OWNER TO" statement fails and
    # (with set -e) aborts startup. Note: Password is escaped to prevent SQL injection (single quotes doubled).
    echo "Creating application user with limited privileges..."
    ESCAPED_PASSWORD="$${POSTGRES_PASSWORD//\\'/\\'\\'}"
    # Create the role only if it does not already exist (idempotent) rather than `createuser ... || true`, which hid
    # genuine failures. The name is passed as a psql variable and quoted via %I to prevent SQL injection;
    # ON_ERROR_STOP makes a real failure abort startup instead of being swallowed.
    su postgres -c "psql -v ON_ERROR_STOP=1 -v role='$$POSTGRES_USER'" << 'EOSQL'
SELECT format('CREATE USER %I', :'role')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'role')
\\gexec
EOSQL
    su postgres -c "psql -v ON_ERROR_STOP=1" << EOSQL
ALTER USER $$POSTGRES_USER WITH PASSWORD '$$ESCAPED_PASSWORD';
EOSQL

    # Create the target database (initially owned by postgres for restore) only if it does not already exist
    # (idempotent), mirroring the role creation above, rather than `... || true` which hid real connection/permission
    # failures. The name is passed as a psql variable and quoted via %I; ON_ERROR_STOP makes a genuine failure abort.
    su postgres -c "psql -v ON_ERROR_STOP=1 -v dbname='$$POSTGRES_DB'" << 'EOSQL'
SELECT format('CREATE DATABASE %I', :'dbname')
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'dbname')
\\gexec
EOSQL

    # Restore from the backup if one was provided; otherwise start with an empty database.
    if [ -n "$${BACKUP_FILE}" ]; then
        if [ ! -f "$${BACKUP_FILE}" ]; then
            echo "ERROR: BACKUP_FILE specified but not found: $${BACKUP_FILE}"
            exit 1
        fi
        echo "Restoring from backup: $${BACKUP_FILE}"
        # RESTORE_SCHEMA_ONLY restores DDL only (no row data). It lets callers exercise the full init/restore/grant
        # path against the real dump without paying the bulk-data COPY + index-build cost; default is a full restore.
        RESTORE_ARGS="--clean --if-exists"
        if [ "$${RESTORE_SCHEMA_ONLY:-false}" = "true" ]; then
            RESTORE_ARGS="$$RESTORE_ARGS --schema-only"
            echo "RESTORE_SCHEMA_ONLY=true: restoring schema only (no table data)."
        fi
        # Tolerate per-object restore warnings: pg_restore returns non-zero whenever any TOC entry fails (missing
        # extensions, leftover privileges, etc.) but the data and schema are still useful, and we don't want
        # set -e to abort startup. POSTGRES_DB / BACKUP_FILE are double-quoted and a -- terminator precedes the
        # filename so a path containing spaces (or one starting with '-') cannot word-split or be parsed as a
        # pg_restore option; RESTORE_ARGS is expanded unquoted on purpose as it holds only fixed, controlled flags.
        su postgres -c "pg_restore -d \\"$$POSTGRES_DB\\" $$RESTORE_ARGS -- \\"$${BACKUP_FILE}\\"" || true
    else
        echo "No BACKUP_FILE provided; starting with an empty database."
    fi
else
    echo "Existing data found, skipping initialization and restore."
fi

# Export NEEDS_INIT for the remainder of the lifecycle. Only user/database creation and the dump restore above are
# gated on it; tables_manifest.sh applies schemas + grants on every boot so an existing data directory keeps up with
# the schema files.
export NEEDS_INIT
EOF
    """,
    visibility = ["//visibility:public"],
)

genrule(
    name = "manifest_script",
    outs = ["tables_manifest.sh"],
    cmd = """
cat > $@ << 'EOF'
#!/bin/bash
# Phase 2 of the generic PostgreSQL lifecycle. Sourced by startup.sh after init-db.sh, so it inherits the running
# temporary server. Applies the schema files listed in the manifest (in listed order) and then grants the application
# role. Applying every manifest entry before the grant block means the one-shot GRANT ... ON ALL TABLES covers every
# table created here, regardless of which schema file defined it.
#
# This runs on every boot, not just first init, which is what makes a long-lived data directory self-updating: pull a
# revision that adds a column and restart, and the column is there. Gating it on first init instead left an existing
# volume frozen at whatever schema it was created with, and the failure surfaced far from its cause -- as an
# UndefinedColumnError from application code, on a database that restored and started up perfectly well.
#
# Re-applying is safe because every statement in every manifest file is idempotent: CREATE TABLE / CREATE INDEX /
# ADD COLUMN are all IF NOT EXISTS, and the files carry no seed data. That is the same property the ALTER TABLE
# top-up blocks already rely on to bring a restored dump up to date, so it is a constraint these files are written
# under rather than a new one imposed here. A manifest file that cannot be applied twice will now fail the boot
# loudly, which is the intended outcome.
set -e

# SCHEMA_DIR holds both the *.sql files and the ordered manifest that selects/orders them. The load set/order can be
# overridden by layering a different tables_manifest.txt on top (shipped as the final image layer) or by pointing
# SCHEMA_DIR at another directory.
SCHEMA_DIR="$${SCHEMA_DIR:-/etc/postgres-schemas}"
MANIFEST_FILE="$$SCHEMA_DIR/tables_manifest.txt"
if [ ! -f "$$MANIFEST_FILE" ]; then
    echo "ERROR: schema manifest not found: $$MANIFEST_FILE"
    exit 1
fi

# Apply each manifest entry in order. Entries are filenames relative to SCHEMA_DIR; blank lines and #-comments are
# ignored, and surrounding whitespace is trimmed. ON_ERROR_STOP=1 plus the missing-file guard make a typo'd or absent
# entry fail the boot loudly rather than silently skip a schema.
echo "Applying schema files from manifest: $$MANIFEST_FILE"
while IFS= read -r _line || [ -n "$$_line" ]; do
    _entry="$${_line%%#*}"
    _entry="$${_entry#"$${_entry%%[![:space:]]*}"}"
    _entry="$${_entry%"$${_entry##*[![:space:]]}"}"
    [ -z "$$_entry" ] && continue
    _schema="$$SCHEMA_DIR/$$_entry"
    if [ ! -f "$$_schema" ]; then
        echo "ERROR: manifest entry not found: $$_schema"
        exit 1
    fi
    echo "Applying schema file: $$_schema"
    su postgres -c "psql -v ON_ERROR_STOP=1 -d $$POSTGRES_DB -f $$_schema"
done < "$$MANIFEST_FILE"

# Grant only the permissions needed for the application. Runs after every manifest schema is applied, so the one-shot
# GRANT ... ON ALL TABLES covers them all; the ALTER DEFAULT PRIVILEGES lines cover objects created later at runtime.
# ON_ERROR_STOP (no `|| true`) makes a failed grant abort startup rather than silently leaving the app role
# under-privileged.
su postgres -c "psql -v ON_ERROR_STOP=1 -d $$POSTGRES_DB" << EOSQL
GRANT CONNECT ON DATABASE $$POSTGRES_DB TO $$POSTGRES_USER;
GRANT USAGE ON SCHEMA public TO $$POSTGRES_USER;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO $$POSTGRES_USER;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO $$POSTGRES_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO $$POSTGRES_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO $$POSTGRES_USER;
EOSQL
EOF
    """,
    visibility = ["//visibility:public"],
)

genrule(
    name = "run_postgres_script",
    outs = ["run-postgres.sh"],
    cmd = """
cat > $@ << 'EOF'
#!/bin/bash
# Phase 3 of the generic PostgreSQL lifecycle. Exec'd by startup.sh as the container's main process. Stops the
# temporary server left running by init-db.sh, then runs postgres in the foreground under a graceful-shutdown trap so
# SIGTERM/SIGINT (container stop) shut the database down cleanly.
set -e

: "$${PG_MAJOR:?PG_MAJOR must be set; the image defines it}"
PG_BIN="/usr/lib/postgresql/$$PG_MAJOR/bin"

# PostgreSQL data directory
PGDATA="/var/lib/postgresql/data"

# Stop the temporary server started by init-db.sh, then relaunch postgres as a direct child of this script so the
# trap + wait below can manage its lifecycle.
su postgres -c "$$PG_BIN/pg_ctl -D $$PGDATA -w stop"

# Graceful shutdown handler for container termination
cleanup() {
    echo "Received shutdown signal, stopping services gracefully..."
    echo "Stopping PostgreSQL..."
    su postgres -c "$$PG_BIN/pg_ctl -D $$PGDATA -m fast -w stop"
    echo "PostgreSQL stopped"
    echo "Graceful shutdown complete"
    exit 0
}

# Trap SIGTERM (container stop) and SIGINT (ctrl+c)
trap cleanup SIGTERM SIGINT

# Start PostgreSQL in background so trap can be processed
echo "Starting PostgreSQL..."
su postgres -c "$$PG_BIN/postgres -D $$PGDATA" &
POSTGRES_PID=$$!

echo "PostgreSQL started (PID $$POSTGRES_PID), waiting for shutdown signal..."

# Wait for PostgreSQL process (allows signal handling)
wait $$POSTGRES_PID
EOF
    """,
    visibility = ["//visibility:public"],
)

genrule(
    name = "startup_script",
    outs = ["startup.sh"],
    cmd = """
cat > $@ << 'EOF'
#!/bin/bash
# Generic PostgreSQL container entrypoint. Orchestrates the three lifecycle phases. init-db.sh and tables_manifest.sh
# are SOURCED (not executed) so NEEDS_INIT and the POSTGRES_* contract are shared across them; run-postgres.sh is
# exec'd so the foreground postgres + shutdown trap become the container's main process.
set -e

# shellcheck source=/dev/null
. /usr/local/bin/init-db.sh
# shellcheck source=/dev/null
. /usr/local/bin/tables_manifest.sh
exec /usr/local/bin/run-postgres.sh
EOF
    """,
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "startup_layer",
    srcs = [
        ":init_db_script",
        ":manifest_script",
        ":run_postgres_script",
        ":startup_script",
    ],
    mode = "0755",
    package_dir = "/usr/local/bin",
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

# Signing key of the PGDG apt repository init-db.sh installs PostgreSQL from, referenced through signed-by. It must be
# the key published at https://www.postgresql.org/media/keys/ACCC4CF8.asc, fingerprint
# B97B 0AFC AA1A 47F0 44F2 44A0 7FCC 7D46 ACCC 4CF8; apt rejects the repository if it is anything else.
pkg_files(
    name = "pgdg_key_files",
    srcs = ["apt.postgresql.org.asc"],
    prefix = "/usr/share/keyrings",
)

pkg_tar(
    name = "pgdg_key_layer",
    srcs = [":pgdg_key_files"],
    tags = ["manual"],
)

# Ordered manifest of schema files (relative to /etc/postgres-schemas) applied on every boot by tables_manifest.sh.
# Shipped as its own dedicated layer, applied LAST (see image_base tars), so the load set/order can be overridden by
# layering a different tables_manifest.txt on top. This manifest lists only orchestration.sql.
pkg_files(
    name = "manifest_files",
    srcs = ["tables_manifest.txt"],
    prefix = "/etc/postgres-schemas",
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "manifest_layer",
    srcs = [":manifest_files"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

# Base image without the git_sha label. Used directly by tests so the test cache key doesn't change on every
# commit (the git_sha label on :image would otherwise force a full rebuild + image reload before every test run).
oci_image(
    name = "image_base",
    base = "@ubuntu_22_04_image",
    entrypoint = ["/usr/local/bin/startup.sh"],
    env = {
        "DEBIAN_FRONTEND": "noninteractive",
        "PG_MAJOR": PG_MAJOR,
    },
    exposed_ports = ["5432"],
    tags = ["manual"],
    # manifest_layer is applied LAST so a layered tables_manifest.txt cleanly overrides this one.
    tars = [
        ":pgdg_key_layer",
        "//database/schemas:schema_layer",
        ":startup_layer",
        ":manifest_layer",
    ],
    visibility = ["//visibility:public"],
)

oci_image(
    name = "image",
    base = ":image_base",
    labels = "//utils:git_sha",
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

oci_load(
    name = "image_load",
    image = ":image",
    repo_tags = ["database-postgres:" + VERSION],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

oci_push(
    name = "image_push",
    image = ":image",
    remote_tags = [],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)
