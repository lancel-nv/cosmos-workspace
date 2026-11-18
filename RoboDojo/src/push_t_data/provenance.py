"""Record pinned external checkouts in the consolidated workspace."""

import hashlib
import json
from pathlib import Path
import subprocess


def dependency_provenance():
    workspace = Path(__file__).resolve().parents[3]
    lock = json.loads((workspace / "upstream-lock.json").read_text())
    versions, changes = [], hashlib.sha256()
    for name, expected in sorted(lock["dependencies"].items()):
        if not name.startswith("RoboDojo/"):
            continue
        directory = workspace / name
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        if actual != expected["commit"]:
            raise ValueError(f"Dependency commit differs from upstream-lock.json: {name}")
        versions.append(f"{actual} {name}")
        changes.update(name.encode())
        changes.update(subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=directory))
    return "\n".join(versions), changes.hexdigest()
