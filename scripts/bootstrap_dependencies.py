"""Fetch pinned optional dependencies and omitted upstream runtime resources."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True)


def checkout(url, commit, target):
    run(["git", "init", str(target)])
    run(["git", "remote", "add", "origin", url], target)
    run(["git", "fetch", "--depth", "1", "origin", commit], target)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        choices=("RoboDojo", "cosmos-curator", "cosmos-evaluator"),
        required=True,
    )
    args = parser.parse_args()
    lock = json.loads((ROOT / "upstream-lock.json").read_text())
    for relative, source in lock["dependencies"].items():
        if not relative.startswith(args.component + "/"):
            continue
        target = ROOT / relative
        if target.exists():
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=target, text=True
            ).strip()
            if actual != source["commit"]:
                raise RuntimeError(
                    f"Existing dependency differs: {relative}; inspect it, do not overwrite"
                )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            checkout(source["url"], source["commit"], target)
        if source.get("patch"):
            patch = ROOT / source["patch"]
            applied = subprocess.run(
                ["git", "apply", "--reverse", "--check", str(patch)],
                cwd=target,
                capture_output=True,
            )
            if applied.returncode:
                run(["git", "apply", "--check", str(patch)], target)
                run(["git", "apply", str(patch)], target)
    if args.component == "cosmos-evaluator":
        source = lock["repositories"][args.component]
        paths = [
            path
            for path in source["omitted_optional_files"]
            if path.startswith(("checks/", "third_party/"))
        ]
        with tempfile.TemporaryDirectory() as temporary:
            scratch = Path(temporary) / "upstream"
            checkout(source["url"], source["commit"], scratch)
            run(["git", "lfs", "pull", "--include", ",".join(paths)], scratch)
            for relative in paths:
                target = ROOT / args.component / relative
                if target.exists():
                    raise FileExistsError(f"Refusing to overwrite {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(scratch / relative, target)


if __name__ == "__main__":
    main()
