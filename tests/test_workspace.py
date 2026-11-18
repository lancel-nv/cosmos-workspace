import importlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
raw_release = importlib.import_module("raw_release")
sha256, verify = raw_release.sha256, raw_release.verify
commands = importlib.import_module("process_raw").commands


class RawReleaseTests(unittest.TestCase):
    def test_gitignore_preserves_runtime_data_modules(self):
        result = subprocess.run(
            [
                "git",
                "check-ignore",
                "--no-index",
                "cosmos-framework/cosmos_framework/data/generator/action/robodojo_eef.py",
                "cosmos-curator/cosmos_curator/core/sensors/data/camera_data.py",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_checksum_rejects_changed_and_extra_raw_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "raw/episode_000001"
            folder.mkdir(parents=True)
            payload = folder / "trajectory.npz"
            payload.write_bytes(b"original")
            manifest = {
                "episodes": 1,
                "files_sha256": {"raw/episode_000001/trajectory.npz": sha256(payload)},
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify(root)["episodes"], 1)
            payload.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verify(root)
            payload.write_bytes(b"original")
            (folder / "extra.txt").write_text("unexpected")
            with self.assertRaises(ValueError):
                verify(root)

    def test_collection_config_remains_valid(self):
        sys.path.insert(0, str(ROOT / "RoboDojo"))
        from src.push_t_data.workflow import load_config

        config = load_config(ROOT / "RoboDojo/configs/push_t_collection.json")
        self.assertEqual(config["min_successes"], 200)
        self.assertEqual(config["start"], 840)

    def test_processing_plan_preserves_virtualenv_interpreter_and_separates_outputs(
        self,
    ):
        from argparse import Namespace

        args = Namespace(
            work=Path("/tmp/work"),
            raw=Path("/tmp/raw"),
            lerobot_python=Path("/tmp/venv/bin/python"),
            curator_python=Path("/tmp/curator/bin/python"),
            framework_python=Path("/tmp/framework/bin/python"),
            tokenizer=Path("/tmp/tokenizer"),
            robot_assets=Path("/tmp/assets"),
        )
        plan = commands(args)
        self.assertEqual(plan[0][2][0], str(args.lerobot_python))
        self.assertEqual(
            [stage for stage, _, _ in plan],
            [
                "export",
                "curate",
                "augment",
                "staging",
                "bundle",
                "audit-short",
                "windows",
                "audit-final",
            ],
        )
        self.assertIn("/tmp/work-head-var593", plan[-1][2])
        self.assertNotIn("train", [stage for stage, _, _ in plan])

    def test_no_secrets_or_large_payloads_in_workspace_entrypoints(self):
        for directory in (ROOT / "scripts", ROOT / ".agents/skills/push-t-workflow"):
            for path in directory.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    self.assertLess(path.stat().st_size, 1024 * 1024)
                    self.assertNotIn("/home/lancel", path.read_text())


if __name__ == "__main__":
    unittest.main()
