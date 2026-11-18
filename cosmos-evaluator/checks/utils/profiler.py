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

from collections import defaultdict
import logging
import time


class Profiler:
    """Simple profiler to track timing of different operations."""

    def __init__(self) -> None:
        self.timings: defaultdict[str, list[float]] = defaultdict(list)
        self.current_timer: float | None = None
        self.current_operation: str | None = None

    def start(self, operation: str) -> None:
        """Start timing an operation."""
        if self.current_timer is not None:
            self.end()
        self.current_operation = operation
        self.current_timer = time.time()

    def end(self) -> None:
        """End timing the current operation."""
        if self.current_timer is not None and self.current_operation is not None:
            duration = time.time() - self.current_timer
            self.timings[self.current_operation].append(duration)
            self.current_timer = None
            self.current_operation = None

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get timing summary statistics."""
        summary = {}
        for operation, times in self.timings.items():
            if times:
                summary[operation] = {
                    "total_time": sum(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "count": len(times),
                }
        return summary

    def print_summary(self, logger: logging.Logger) -> None:
        """Print timing summary to logger."""
        summary = self.get_summary()
        if not summary:
            return

        LINE_WIDTH = 60
        logger.info("=" * LINE_WIDTH)
        logger.info("PERFORMANCE PROFILING SUMMARY")
        logger.info("=" * LINE_WIDTH)

        # Sort by total time and get top 5
        sorted_ops = sorted(summary.items(), key=lambda x: x[1]["total_time"], reverse=True)
        top_5_ops = sorted_ops[:5]

        for operation, stats in top_5_ops:
            percentage = (stats["total_time"] / sum(s["total_time"] for _, s in sorted_ops)) * 100
            logger.info("%s:", operation)
            logger.info("  Total: %.3fs (%.1f%%)", stats["total_time"], percentage)
            logger.info("  Average: %.3fs", stats["avg_time"])
            logger.info("  Count: %s", stats["count"])
            logger.info("  Range: %.3fs - %.3fs", stats["min_time"], stats["max_time"])

        total_time = sum(stats["total_time"] for stats in summary.values())
        logger.info("\nTotal processing time: %.3fs", total_time)
        logger.info("=" * LINE_WIDTH)
