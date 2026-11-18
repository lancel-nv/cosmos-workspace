-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- Database schema for Cosmos Evaulator Arbitrator Service
-- PostgreSQL schema for tracking API requests and checker executions
--
-- This schema is bundled with the generic PostgreSQL container image (//database:image) and applied automatically on
-- every boot. To run with a different schema set or load order, supply your own *.sql files and a
-- tables_manifest.txt through SCHEMA_DIR (see the database image README).
--
-- Every statement here must therefore be re-appliable: CREATE and ADD COLUMN carry IF NOT EXISTS, and nothing seeds
-- rows. That is what lets a long-lived data directory keep up with this file instead of staying frozen at the schema
-- it was created with.
--
-- Design Principles:
-- - Clients are tracked by IP address with optional name/description
-- - Each arbitrator request creates an execution record with full request/response capture
-- - Individual checker executions are linked to their parent arbitrator request
-- - JSON/JSONB fields store flexible request/response payloads and metadata

-- =============================================================================
-- Clients Table
-- =============================================================================
-- Tracks API clients by IP address. Each unique client IP gets a record.
-- Optional name and description fields allow human-friendly identification.

CREATE TABLE IF NOT EXISTS clients (
    client_id   INTEGER     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ip_address  inet        NOT NULL UNIQUE,
    name        TEXT,
    description TEXT
);

-- Unique constraint on ip_address prevents duplicate client records for the same IP
-- This is enforced at the database level to handle concurrent insert race conditions

-- =============================================================================
-- Arbitrator Executions Table
-- =============================================================================
-- Records each arbitration request submitted via POST /process/async.
-- Captures the full lifecycle: queued -> in_progress -> success/fail
--
-- Fields:
--   request_id    - Auto-generated unique identifier
--   client_id     - Foreign key to clients table
--   timestamp     - When the request was received
--   status        - Current state. One of:
--                     'queued'      - Admitted to the waiting room; no pod has claimed any of its jobs yet.
--                     'in_progress' - At least one pod has claimed a job from this request.
--                     'success'     - All jobs reached a terminal state and the response was assembled.
--                     'fail'        - Request terminated without producing a response (every job failed before any
--                                     was claimed, or an orchestration error occurred).
--   request_json  - Full request payload as received
--   response_json - Full response payload as returned
--   duration_ms   - Total processing time in milliseconds
--   metadata      - Additional structured data (JSONB for indexing)
--   version       - Service version string
--   commit_sha    - Git commit SHA of the service
--
-- Ownership fields. The aggregation of a request lives in one arbitrator replica's memory, so the row also records
-- which replica that is. When the claim goes stale another replica adopts the row and rebuilds the aggregation from
-- request_json plus this request's checker_executions rows.
--
--   claimed_by    - Identity of the arbitrator replica orchestrating this request, and the fencing token for its
--                   writes: after adoption the previous owner's updates match zero rows and it stops. NULL for rows
--                   predating adoption support, which are never adopted.
--   lease_expires - When the owning replica's claim goes stale. The owner pushes this forward while the request is
--                   live; a non-terminal row past this instant has lost its orchestrator.

CREATE TABLE IF NOT EXISTS arbitrator_executions (
    request_id    INTEGER     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    client_id     INTEGER     NOT NULL,
    timestamp     TIMESTAMPTZ NOT NULL,
    status        VARCHAR     NOT NULL,
    request_json  JSON        NOT NULL,
    response_json JSON        NOT NULL,
    duration_ms   INTEGER     NOT NULL,
    metadata      JSONB,
    version       VARCHAR     NOT NULL,
    commit_sha    VARCHAR(64) NOT NULL,
    claimed_by    VARCHAR,
    lease_expires TIMESTAMPTZ,
    CONSTRAINT fk_arbitrator_client
        FOREIGN KEY (client_id)
        REFERENCES clients(client_id)
);

-- See the note below the checker_executions ALTER block for why both forms exist.
ALTER TABLE arbitrator_executions
    ADD COLUMN IF NOT EXISTS claimed_by    VARCHAR,
    ADD COLUMN IF NOT EXISTS lease_expires TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_arbitrator_executions_client_id ON arbitrator_executions(client_id);
CREATE INDEX IF NOT EXISTS idx_arbitrator_executions_timestamp ON arbitrator_executions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_arbitrator_executions_status ON arbitrator_executions(status);

-- The sweeper's only query: non-terminal rows whose owner stopped renewing. Partial so it stays the size of the live
-- set rather than the whole history, which is what makes a sweep on every replica affordable.
CREATE INDEX IF NOT EXISTS idx_arbitrator_executions_adoptable
    ON arbitrator_executions(lease_expires)
    WHERE status IN ('queued', 'in_progress');

-- =============================================================================
-- Checker Executions Table
-- =============================================================================
-- Records each individual checker execution within an arbitrator request.
-- Multiple checker executions can be linked to a single arbitrator execution.
--
-- Fields:
--   request_id      - Foreign key to parent arbitrator_executions record
--   checker_name    - Base name of the checker service (e.g., 'av.objects_single_view',
--                     'metropolis.hallucination'). For per-camera fan-out checkers, the camera identity is stored
--                     separately in camera_name (NOT embedded in checker_name).
--   camera_name     - Full RDS camera stream id (e.g., 'camera_front_wide_120fov') for per-camera (fanned-out)
--                     checker executions. NULL for non-parameterized checkers, such as cross-camera aggregators.
--   start_time      - When the claiming pod began processing, written as it takes the job off the queue. Before that
--                     it holds the row-insertion moment as a placeholder. NOW() - start_time is how long a job that
--                     has not finished has been running, so elapsed time is never stored.
--   end_time        - When checker execution completed
--   duration_ms     - Checker processing time in milliseconds
--   status          - Execution state. One of:
--                       'pending'     - The row exists but the job has not been offered to anybody. This is what
--                                       admission inserts, and the payload it carries is a placeholder: a dependent
--                                       job's inputs are not knowable until its dependencies have produced results.
--                                       Not claimable, which is the whole point of it being distinct from 'queued': a
--                                       row born claimable would be offered to the fleet before the payload it is
--                                       supposed to run had been written.
--                       'queued'      - On the queue, and carrying the payload to run: the arbitrator has written the
--                                       real request onto the row, and any pod with room may claim it. Reached from
--                                       'pending' once dependencies are met, or from 'in_progress' when the reaper
--                                       takes a job back from an owner that stopped renewing.
--                       'in_progress' - A pod has claimed the job and is processing it. ``claimed_by`` names the
--                                       owner and ``start_time`` is set at this transition.
--                       'success'     - Terminal: response stored in ``response_json``.
--                       'fail'        - Terminal: checker error, or the job exhausted its attempts.
--                       'timeout'     - Terminal: the job outlived the arbitrator's patience for it.
--                       'skipped'     - Terminal: a dependency hit a non-success terminal state, so this job went
--                                       from 'pending' straight to terminal without ever being queued.
--   request_json    - Request payload the claiming pod reads off the row. Only meaningful from 'queued' onwards; a
--                     'pending' row holds whatever admission could assemble, which for a dependent job is nothing.
--   response_json   - Response payload received from the checker
--   error_message   - Error details if status is 'fail'
--   output_metadata - Additional structured output data (JSONB for indexing)
--   version         - Version of the checker pod that claimed the job, written by the claim. Empty until a pod
--                     claims it, so a job that never ran has none; a requeued job carries its latest claimer's.
--   commit_sha      - Git commit SHA of that same pod, recorded alongside version
--
-- Coordination fields. The row is not just a record of a job, it is the channel the job reports through: the checker
-- pod claims it, keeps its claim alive on it, and writes its own terminal status. The arbitrator reads the row instead
-- of holding a connection open to the pod doing the work.
--
--   claimed_by       - Identity of the pod that owns this job, and the fencing token for every write it makes. A pod
--                      whose lease expired can be superseded, so all of its subsequent writes carry
--                      "AND claimed_by = <me>" and a zero-row update means it has been replaced and must stop.
--                      NULL before anyone claims the job.
--   lease_expires    - When the owning pod's claim goes stale. The owner pushes this forward as it heartbeats; a
--                      non-terminal row past this instant means the pod died and the job can be taken over.
--   attempt          - Number of times the job has been claimed. Bounded so a job that reliably kills whichever pod
--                      picks it up fails loudly instead of cycling through the fleet forever.
--   cancel_requested - Set by the arbitrator to ask the owner to stop. A column rather than a status so the status
--                      vocabulary above stays closed, and because the owner is not addressable directly: it is
--                      whichever pod holds the claim, so cancellation has to be left somewhere it will look.

CREATE TABLE IF NOT EXISTS checker_executions (
    execution_id     INTEGER     GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id       INTEGER     NOT NULL,
    checker_name     VARCHAR     NOT NULL,
    camera_name      VARCHAR,
    start_time       TIMESTAMPTZ NOT NULL,
    end_time         TIMESTAMPTZ NOT NULL,
    duration_ms      INTEGER     NOT NULL,
    status           VARCHAR     NOT NULL,
    request_json     JSON        NOT NULL,
    response_json    JSON        NOT NULL,
    error_message    TEXT,
    output_metadata  JSONB,
    version          VARCHAR     NOT NULL,
    commit_sha       VARCHAR(64) NOT NULL,
    claimed_by       VARCHAR,
    lease_expires    TIMESTAMPTZ,
    attempt          INTEGER     NOT NULL DEFAULT 0,
    cancel_requested BOOLEAN     NOT NULL DEFAULT false,
    CONSTRAINT fk_checker_request
        FOREIGN KEY (request_id)
        REFERENCES arbitrator_executions(request_id)
);

-- Restate the coordination columns as ALTERs. There is no migration framework here, and whenever the table already
-- exists -- a boot that restored a dump, or any boot after the first -- the CREATE TABLE above is a no-op, so columns
-- added to it after that data directory was created would never appear. This block is what actually brings such a
-- database up to date. Both forms are required and must be kept in agreement; the CREATE TABLE keeps the table
-- readable in one place, this keeps restored and long-lived databases correct.
ALTER TABLE checker_executions
    ADD COLUMN IF NOT EXISTS claimed_by       VARCHAR,
    ADD COLUMN IF NOT EXISTS lease_expires    TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS attempt          INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS cancel_requested BOOLEAN NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS idx_checker_executions_request_id ON checker_executions(request_id);
CREATE INDEX IF NOT EXISTS idx_checker_executions_checker_name ON checker_executions(checker_name);
CREATE INDEX IF NOT EXISTS idx_checker_executions_camera_name ON checker_executions(camera_name);
CREATE INDEX IF NOT EXISTS idx_checker_executions_status ON checker_executions(status);

-- The claim query every checker pod runs while it has a free slot: the oldest unstarted job for one checker. Partial so
-- it holds only the work actually waiting rather than every job ever run, which is what keeps a poll from the whole
-- fleet cheap. Ordered by execution_id so the index supplies the FIFO directly and the claim needs no sort.
--
-- The predicate is also the enforcement point for the 'pending' / 'queued' split above: a job becomes visible to the
-- fleet exactly when the arbitrator writes its payload and moves it to 'queued'. Widening this to include 'pending'
-- would restore the defect the split exists to remove -- pods claiming rows whose payload has not been written.
CREATE INDEX IF NOT EXISTS idx_checker_executions_claimable
    ON checker_executions(checker_name, execution_id)
    WHERE status = 'queued';

-- The reaper's query: claims whose owner stopped renewing. Partial on the one status a job can be abandoned in.
CREATE INDEX IF NOT EXISTS idx_checker_executions_expired
    ON checker_executions(lease_expires)
    WHERE status = 'in_progress';
