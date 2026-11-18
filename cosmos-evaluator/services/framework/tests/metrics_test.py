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

"""Tests for the checker Prometheus exposition."""

import unittest

from services.framework.job_slots import JobSlots
from services.framework.metrics import PROMETHEUS_CONTENT_TYPE, render_checker_metrics


def _series(body: str) -> dict[str, str]:
    """Parse an exposition body into ``{series: value}``, dropping HELP/TYPE comments."""
    return {
        line.rsplit(" ", 1)[0]: line.rsplit(" ", 1)[1]
        for line in body.splitlines()
        if line and not line.startswith("#")
    }


class TestCheckerMetrics(unittest.TestCase):
    """The exposition a scraper reads off a single checker replica."""

    def test_content_type_declares_the_text_format(self) -> None:
        """Scrapers pick text vs protobuf off the version token rather than sniffing the body."""
        self.assertIn("version=0.0.4", PROMETHEUS_CONTENT_TYPE)

    def test_occupancy_is_reported_per_checker(self) -> None:
        slots = JobSlots(3)
        slots.take(2)

        series = _series(render_checker_metrics("av.egomotion", slots, jobs_in_flight=2))

        self.assertEqual(series['checker_slots_active{checker="av.egomotion"}'], "2")
        self.assertEqual(series['checker_slots_limit{checker="av.egomotion"}'], "3")
        self.assertEqual(series['checker_jobs_in_flight{checker="av.egomotion"}'], "2")

    def test_headroom_is_visible_as_the_gap_between_active_and_limit(self) -> None:
        """What a fleet-wide alert reads: how much of the configured capacity is actually free to claim work."""
        slots = JobSlots(3)
        slots.take()

        series = _series(render_checker_metrics("av.objects_single_view", slots, jobs_in_flight=1))

        self.assertEqual(series['checker_slots_active{checker="av.objects_single_view"}'], "1")
        self.assertEqual(series['checker_slots_limit{checker="av.objects_single_view"}'], "3")

    def test_slots_held_by_something_other_than_a_job_are_distinguishable(self) -> None:
        """A direct ``/process`` call holds a slot without owning a job row, and a leak looks the same from outside."""
        slots = JobSlots(2)
        slots.take()

        series = _series(render_checker_metrics("av.objects_multi_view", slots, jobs_in_flight=0))

        self.assertEqual(series['checker_slots_active{checker="av.objects_multi_view"}'], "1")
        self.assertEqual(series['checker_jobs_in_flight{checker="av.objects_multi_view"}'], "0")

    def test_body_is_newline_terminated_and_typed(self) -> None:
        """Exposition parsers reject a body without a trailing newline, and untyped series scrape as untyped."""
        body = render_checker_metrics("av.objects_multi_view", JobSlots(1), jobs_in_flight=0)

        self.assertTrue(body.endswith("\n"))
        self.assertIn("# TYPE checker_slots_active gauge", body)
        self.assertIn("# TYPE checker_jobs_in_flight gauge", body)


if __name__ == "__main__":
    unittest.main()
