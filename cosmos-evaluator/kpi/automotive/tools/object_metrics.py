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
"""Add object-level metric verdicts to object-check result JSON files.

The tool accepts result files or directories and recursively searches directories for
``*.objects.results.json``. It evaluates each input with the required importance policy, then either updates
every input in place or writes a single input to a separate output path. Unless ``--quiet`` is set, it also
prints a per-object-type summary of the computed verdicts.

Example:

    dazel run //kpi/automotive/tools:object_metrics -- \\
        --importance ego_relevant --in-place /path/to/results/
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

from kpi.automotive.augment import RESULT_GLOB, augment_file
from kpi.automotive.policy import default_policy
from kpi.automotive.result_view import ImportancePolicy

logger = logging.getLogger(__name__)


def _collect_inputs(paths: list[Path]) -> list[Path]:
    """Expand each argument into result files, erroring on a path that matches nothing."""
    collected: list[Path] = []
    for path in paths:
        if path.is_dir():
            found = sorted(path.rglob(RESULT_GLOB))
            if not found:
                raise ValueError(f"no {RESULT_GLOB} under {path}")
            collected.extend(found)
        elif path.is_file():
            collected.append(path)
        else:
            raise ValueError(f"no such file or directory: {path}")
    return collected


def _log_summary(path: Path, metrics: dict[str, Any]) -> None:
    """Print the per-type table, which is what makes the tool useful interactively."""
    logger.info("%s (importance=%s)", path.name, metrics["policy"]["importance"])
    header = (
        f"  {'object_type':<16}{'obs':>6}{'unsc':>7}{'present':>9}{'missing':>9}{'misal':>8}"
        f"{'wrong':>8}{'confl':>8}{'extra':>7}"
    )
    logger.info("%s", header)
    for name, row in metrics["by_type"].items():
        # "n/a" rather than 0 for a signal no rule measured for this type - the same distinction the JSON draws
        # with null, kept in the interactive view so the two cannot be read as agreeing counts of zero.
        signals = ("misaligned", "wrong_state", "conflicting")
        counts = {key: "n/a" if row[key] is None else str(row[key]) for key in signals}
        logger.info(
            "  %-16s%6d%7d%9d%9d%8s%8s%8s%7d",
            name,
            row["observed"],
            row["unscored"],
            row["present"],
            row["missing"],
            counts["misaligned"],
            counts["wrong_state"],
            counts["conflicting"],
            row["hallucinated"],
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help=f"result JSON files, or directories of {RESULT_GLOB}")
    parser.add_argument(
        "--importance",
        required=True,
        choices=[str(policy) for policy in ImportancePolicy],
        help="which scored cells count; there is no default because the choice moves every rate",
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="write here (only valid with a single input file)")
    destination.add_argument("--in-place", action="store_true", help="rewrite each input file")
    parser.add_argument("--quiet", action="store_true", help="suppress the per-type summary table")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = _collect_inputs(args.paths)
    if args.output is not None and len(inputs) != 1:
        raise ValueError(f"--output takes a single input file, got {len(inputs)}; use --in-place instead")

    policy = default_policy(ImportancePolicy(args.importance))
    for path in inputs:
        metrics = augment_file(path, args.output or path, policy)
        if not args.quiet:
            _log_summary(path, metrics)
    logger.info("augmented %d file(s)", len(inputs))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        sys.exit(main())
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception:
        logger.exception("unexpected failure in object_metrics")
        sys.exit(1)
