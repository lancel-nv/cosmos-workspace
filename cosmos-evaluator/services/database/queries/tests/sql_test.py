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

"""Unit tests for the PartialUpdate statement builder.

The property worth pinning is that the ``$n`` in each assignment indexes the matching value in the returned
parameter list, for every subset of columns a caller might supply -- that is the invariant the builder exists to
hold, and the one a mocked connection cannot check.
"""

import re
import unittest

from services.database.queries.sql import PartialUpdate


def _assignment_positions(statement: str) -> dict[str, int]:
    """Map each assigned column to the 1-based parameter index it reads, from the rendered SET list."""
    set_list = statement.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
    return {
        match.group(1): int(match.group(2))
        for match in (re.fullmatch(r"(\w+) = \$(\d+)", part.strip()) for part in set_list.split(","))
        if match
    }


class TestParameterNumbering(unittest.TestCase):
    """Each rendered ``$n`` must index the value the caller passed for that column."""

    def test_every_assignment_indexes_its_own_value(self) -> None:
        update = PartialUpdate()
        update.set_all([("status", "failed"), ("result", {"score": 1}), ("error", "boom")])
        statement, params = update.statement("kpi_run_tasks", key="task_id", key_value=3)

        for column, position in _assignment_positions(statement).items():
            self.assertEqual(
                params[position - 1], {"status": "failed", "result": {"score": 1}, "error": "boom"}[column]
            )

    def test_skipped_columns_do_not_leave_gaps(self) -> None:
        # The middle column is omitted, so ``error`` must renumber down to $2 rather than stay at $3.
        update = PartialUpdate()
        update.set_all([("status", "failed"), ("result", None), ("error", "boom")])
        statement, params = update.statement("kpi_run_tasks", key="task_id", key_value=3)

        self.assertEqual(_assignment_positions(statement), {"status": 1, "error": 2})
        self.assertEqual(params, ["failed", "boom", 3])

    def test_the_key_is_bound_after_every_assignment(self) -> None:
        update = PartialUpdate()
        update.set("status", "queued")
        statement, params = update.statement("checker_executions", key="execution_id", key_value=42)

        self.assertEqual(statement, "UPDATE checker_executions SET status = $1 WHERE execution_id = $2")
        self.assertEqual(params, ["queued", 42])

    def test_assignments_are_emitted_in_call_order(self) -> None:
        # Call order is the contract: ``update_checker_execution`` sets status before request_json so that a
        # payload rewrite renders as ``request_json = $2``.
        update = PartialUpdate()
        update.set_all([("status", "queued"), ("request_json", {"video_url": "s3://clip.mp4"})])
        statement, _ = update.statement("checker_executions", key="execution_id", key_value=1)

        self.assertEqual(_assignment_positions(statement), {"status": 1, "request_json": 2})


class TestUnconditionalAssignments(unittest.TestCase):
    """Assignments with no bind parameter lead the SET list and do not count as caller-supplied columns."""

    def test_they_render_before_the_bound_columns(self) -> None:
        update = PartialUpdate("updated_at = CURRENT_TIMESTAMP")
        update.set("status", "succeeded")
        statement, params = update.statement("kpi_runs", key="run_id", key_value=7)

        self.assertEqual(statement, "UPDATE kpi_runs SET updated_at = CURRENT_TIMESTAMP, status = $1 WHERE run_id = $2")
        self.assertEqual(params, ["succeeded", 7])

    def test_they_do_not_make_an_empty_patch_look_populated(self) -> None:
        update = PartialUpdate("updated_at = CURRENT_TIMESTAMP")
        self.assertIs(update.has_columns(), False)
        update.set("status", None)
        self.assertIs(update.has_columns(), False)
        update.set("status", "queued")
        self.assertIs(update.has_columns(), True)


class TestReturning(unittest.TestCase):
    """The optional RETURNING clause is appended verbatim after the predicate."""

    def test_it_is_appended_when_requested(self) -> None:
        update = PartialUpdate()
        update.set("status", "succeeded")
        statement, _ = update.statement("kpi_runs", key="run_id", key_value=7, returning="run_id, status")

        self.assertTrue(statement.endswith("WHERE run_id = $2 RETURNING run_id, status"))

    def test_it_is_omitted_when_empty(self) -> None:
        update = PartialUpdate()
        update.set("status", "succeeded")
        statement, _ = update.statement("kpi_runs", key="run_id", key_value=7)

        self.assertNotIn("RETURNING", statement)


class TestSetAlways(unittest.TestCase):
    """``set_always`` writes the column even when the value is ``None``, unlike ``set``."""

    def test_none_is_written_as_a_bound_null(self) -> None:
        # What a terminal write needs: a retried job reporting success must clear the previous attempt's error, not
        # inherit it because the caller passed nothing this time.
        update = PartialUpdate()
        update.set("status", "success")
        update.set_always("error_message", None)
        statement, params = update.statement("checker_executions", key="execution_id", key_value=1)

        self.assertEqual(_assignment_positions(statement), {"status": 1, "error_message": 2})
        self.assertEqual(params, ["success", None, 1])

    def test_it_counts_as_a_caller_supplied_column(self) -> None:
        update = PartialUpdate()
        self.assertIs(update.has_columns(), False)
        update.set_always("error_message", None)
        self.assertIs(update.has_columns(), True)


class TestParts(unittest.TestCase):
    """``parts`` renders the SET list for a caller supplying its own predicate."""

    def test_it_renders_the_assignments_without_a_statement(self) -> None:
        update = PartialUpdate("end_time = NOW()")
        update.set_all([("status", "success"), ("duration_ms", 12)])

        self.assertEqual(update.parts(), ("end_time = NOW(), status = $1, duration_ms = $2", ["success", 12]))

    def test_the_returned_params_are_a_copy(self) -> None:
        # The caller appends its own predicate parameters to what it gets back, so handing over the live list would
        # have it corrupt the builder's numbering for any later call.
        update = PartialUpdate()
        update.set("status", "queued")
        _, params = update.parts()
        params.append("scribble")

        self.assertEqual(update.parts(), ("status = $1", ["queued"]))


class TestFalsyValues(unittest.TestCase):
    """Only ``None`` means "leave this column alone"; a falsy value is still a value to write."""

    def test_zero_and_empty_string_are_written(self) -> None:
        update = PartialUpdate()
        update.set_all([("duration_ms", 0), ("camera_variant", ""), ("output_metadata", {})])
        statement, params = update.statement("checker_executions", key="execution_id", key_value=1)

        self.assertEqual(
            _assignment_positions(statement), {"duration_ms": 1, "camera_variant": 2, "output_metadata": 3}
        )
        self.assertEqual(params, [0, "", {}, 1])


if __name__ == "__main__":
    unittest.main()
