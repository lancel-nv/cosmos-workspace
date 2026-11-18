import argparse
import json
from pathlib import Path

from src.push_t_data.io import validate_episode
from src.push_t_data.outcomes import OutcomeQuota


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=int)
    parser.add_argument("--min-episodes", type=int, default=0)
    parser.add_argument("--min-successes", type=int, default=0)
    parser.add_argument("--min-failures", type=int, default=0)
    args = parser.parse_args()
    paths = sorted(args.root.glob("episode_*"))
    if not paths or (args.expected is not None and len(paths) != args.expected):
        raise ValueError(f"Found {len(paths)} episodes; expected {args.expected}")
    results = [validate_episode(path) for path in paths]
    summary = {
        "episodes": len(results),
        "success": sum(row["success"] for row in results),
        "failure": sum(not row["success"] for row in results),
        "task_failure": sum(row["termination_reason"] == "task_failure" for row in results),
        "time_limit": sum(row["termination_reason"] == "time_limit" for row in results),
        "actions": sum(row["actions"] for row in results),
        "episodes_with_object_motion_over_1mm": sum(row["object_displacement_max_m"] > 0.001 for row in results),
        "partial_episodes": len(list(args.root.glob(".*.partial"))),
        "details": results,
    }
    quota = OutcomeQuota(args.min_episodes, args.min_successes, args.min_failures)
    summary["outcome_quota"] = vars(quota)
    summary["outcome_quota_met"] = quota.met(summary["episodes"], summary["success"])
    (args.root / "quality_report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, indent=2))
    quota.require(summary["episodes"], summary["success"])


if __name__ == "__main__":
    main()
