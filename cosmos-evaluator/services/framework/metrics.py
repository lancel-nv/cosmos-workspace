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

"""Prometheus exposition for checker services.

Each replica reports only its own occupancy, and summing across replicas is the scraper's job -- the same per-replica
contract the arbitrator's ``/metrics`` follows.

The counterpart fields on ``/health`` stay where they are: ``/health`` answers "should this replica be sent work",
while these series answer "how loaded is the deployment, and is it losing capacity", which needs a time axis.
"""

from services.framework.job_slots import JobSlots

# The version token lets scrapers detect the textual format (rather than the protobuf format) without parsing the
# body. Matches the arbitrator's exposition so one scrape config covers both.
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def render_checker_metrics(checker: str, slots: JobSlots, jobs_in_flight: int, gpu_healthy: bool = True) -> str:
    """Render one checker replica's occupancy as Prometheus v0.0.4 text exposition.

    Args:
        checker: Qualified checker name used as the ``checker`` label (e.g. ``av.objects_single_view``), so a single
            scrape job covers every checker deployment. Qualified rather than bare because the arbitrator labels the
            same checker this way: correlating demand against capacity is the reason both sides are exported, and
            unequal label values would make that a relabeling exercise
        slots: The replica's slot pool; both its occupancy and its configured ceiling are exported, because the gap
            between them is what says how much headroom the fleet actually has
        jobs_in_flight: Detached jobs this replica currently owns a lease on. Normally tracks the active slot count;
            a persistent gap between the two means slots are being held by something other than a claimed job.
        gpu_healthy: Whether this replica can still use its GPU. Exported as a gauge because it is the one replica
            state that is invisible in the occupancy numbers: a pod whose GPU died reads as idle, exactly like a pod
            waiting on an empty queue. Defaults to True for checkers that never touch a GPU.

    Returns:
        The exposition body, newline-terminated.
    """
    label = f'{{checker="{checker}"}}'
    lines = [
        "# HELP checker_slots_active Processing slots currently in use on this replica.",
        "# TYPE checker_slots_active gauge",
        f"checker_slots_active{label} {slots.active}",
        "# HELP checker_slots_limit Configured processing slot ceiling for this replica.",
        "# TYPE checker_slots_limit gauge",
        f"checker_slots_limit{label} {slots.limit}",
        (
            "# HELP checker_jobs_in_flight Claimed jobs this replica is running detached and holding a lease on. "
            "Work in this state survives whatever asked for it, so this is the only place it is visible."
        ),
        "# TYPE checker_jobs_in_flight gauge",
        f"checker_jobs_in_flight{label} {jobs_in_flight}",
        (
            "# HELP checker_gpu_healthy 1 while this replica can use its GPU, 0 once it could not. "
            "A replica that reports 0 is on its way out and is no longer claiming work."
        ),
        "# TYPE checker_gpu_healthy gauge",
        f"checker_gpu_healthy{label} {int(gpu_healthy)}",
    ]
    return "\n".join(lines) + "\n"
