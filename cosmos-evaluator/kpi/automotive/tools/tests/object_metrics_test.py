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

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from kpi.automotive.augment import METRICS_KEY, build_git_sha
from kpi.automotive.existence import is_conflicting, is_misaligned, is_missing, is_wrong_state
from kpi.automotive.policy import default_policy
from kpi.automotive.result_view import ImportancePolicy, read_result
from kpi.automotive.tests.fixtures import load_dynamic_result, load_traffic_light_result
from kpi.automotive.tools.object_metrics import main


class ObjectMetricsTestBase(unittest.TestCase):
    """Stages the fixture as a realistically-named result file in a scratch directory."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.path = self.directory / "clip123.dynamic.objects.results.json"
        self.original = load_dynamic_result()
        self.path.write_text(json.dumps(self.original, indent=2), encoding="utf-8")

    def run_tool(self, *args: str) -> int:
        return main([*args, "--quiet"])

    def augmented(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))


class TestBuildGitSha(unittest.TestCase):
    def test_returns_a_sha_or_none_but_never_the_sentinel(self) -> None:
        """An unstamped build must not leak the status script's placeholder into the emitted block."""
        sha = build_git_sha()
        self.assertNotEqual(sha, "unknown")
        if sha is not None:
            self.assertRegex(sha, r"^[0-9a-f]{40}$")


class TestAugmentation(ObjectMetricsTestBase):
    def test_adds_the_block_and_leaves_every_other_key_untouched(self) -> None:
        self.assertEqual(self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place"), 0)
        augmented = self.augmented()
        self.assertIn(METRICS_KEY, augmented)
        del augmented[METRICS_KEY]
        self.assertEqual(augmented, self.original, "the tool must be purely additive")

    def test_per_track_verdicts_match_the_library(self) -> None:
        """The tool is a serializer, not a second implementation of the rules."""
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        policy = default_policy(ImportancePolicy.EGO_RELEVANT)
        view = read_result(self.original, importance=ImportancePolicy.EGO_RELEVANT)
        entries = {entry["track_id"]: entry for entry in self.augmented()[METRICS_KEY]["tracks"]}
        self.assertEqual(set(entries), set(view.tracks))
        for track_id, track in view.tracks.items():
            self.assertEqual(entries[track_id]["missing"], is_missing(track, policy), track_id)
            self.assertEqual(entries[track_id]["misaligned"], is_misaligned(track, policy), track_id)

    def test_the_block_records_the_build_it_came_from(self) -> None:
        """The rules live in code, so the commit is the only complete statement of what produced a verdict."""
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        block = self.augmented()[METRICS_KEY]
        self.assertIn("git_sha", block, "the key is always present, null when the build was not stamped")
        self.assertNotIn("schema_version", block)
        sha = block["git_sha"]
        if sha is not None:
            self.assertRegex(sha, r"^[0-9a-f]{40}$")
        self.assertNotEqual(sha, "unknown", "the unstamped sentinel must be normalized to null, not emitted")

    def test_the_block_records_no_processor_mode(self) -> None:
        """Object types are read from the file, so there is nothing about the processor to declare."""
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        self.assertNotIn("mode", self.augmented()[METRICS_KEY])

    def test_the_block_records_the_policy_that_produced_it(self) -> None:
        self.run_tool(str(self.path), "--importance", "all_scored", "--in-place")
        policy = self.augmented()[METRICS_KEY]["policy"]
        self.assertEqual(policy["importance"], "all_scored")
        self.assertEqual(policy["name"], "default")
        self.assertEqual(policy["existence"]["Car"], {"rule": "drawn_empty", "atol": 1e-8})

    def test_the_importance_choice_changes_the_numbers(self) -> None:
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        ego = self.augmented()[METRICS_KEY]["by_type"]["Car"]
        self.path.write_text(json.dumps(self.original, indent=2), encoding="utf-8")
        self.run_tool(str(self.path), "--importance", "all_scored", "--in-place")
        every = self.augmented()[METRICS_KEY]["by_type"]["Car"]
        self.assertEqual((ego["observed"], ego["unscored"]), (2, 1))
        self.assertEqual((every["observed"], every["unscored"]), (3, 0))

    def test_extra_objects_are_recorded_with_their_group(self) -> None:
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        extras = self.augmented()[METRICS_KEY]["extra_objects"]
        self.assertEqual([extra["id"] for extra in extras], [1, 3])
        self.assertEqual(extras[0]["group"], "four_wheeler")
        self.assertEqual(extras[1]["group"], "vru")

    def test_untyped_track_ids_are_reported(self) -> None:
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        self.assertEqual(self.augmented()[METRICS_KEY]["untyped_track_ids"], [7])

    def test_emits_no_nan_token(self) -> None:
        """JSON has no NaN literal; an unscored track carries a null mean instead."""
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        raw = self.path.read_text(encoding="utf-8")
        self.assertNotIn("NaN", raw)
        json.loads(raw, parse_constant=lambda token: self.fail(f"non-standard JSON constant {token}"))
        unscored = [entry for entry in self.augmented()[METRICS_KEY]["tracks"] if entry["unscored"]]
        self.assertTrue(unscored)
        for entry in unscored:
            self.assertIsNone(entry["mean_score"])
            self.assertIsNone(entry["missing"])

    def test_running_twice_is_the_same_as_running_once(self) -> None:
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        once = self.path.read_text(encoding="utf-8")
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        self.assertEqual(self.path.read_text(encoding="utf-8"), once)

    def test_rerunning_with_another_policy_replaces_rather_than_merges(self) -> None:
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        self.run_tool(str(self.path), "--importance", "all_scored", "--in-place")
        self.assertEqual(self.augmented()[METRICS_KEY]["policy"]["importance"], "all_scored")

    def test_output_writes_elsewhere_and_leaves_the_input_alone(self) -> None:
        destination = self.directory / "augmented.json"
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--output", str(destination))
        self.assertIn(METRICS_KEY, json.loads(destination.read_text(encoding="utf-8")))
        self.assertNotIn(METRICS_KEY, json.loads(self.path.read_text(encoding="utf-8")))

    def test_a_directory_argument_augments_every_result_file_under_it(self) -> None:
        second = self.directory / "nested" / "clip456.dynamic.objects.results.json"
        second.parent.mkdir()
        second.write_text(json.dumps(self.original, indent=2), encoding="utf-8")
        self.run_tool(str(self.directory), "--importance", "ego_relevant", "--in-place")
        self.assertIn(METRICS_KEY, json.loads(second.read_text(encoding="utf-8")))
        self.assertIn(METRICS_KEY, self.augmented())


class TestUnevaluatedTypes(ObjectMetricsTestBase):
    def test_a_hallucination_class_with_no_rules_is_named_rather_than_silently_dropped(self) -> None:
        """Otherwise a file whose hallucinations were all dropped looks like one that had none."""
        result = load_dynamic_result()
        result["hallucination_detections"]["WaitLine"] = [{"frame_idx": 0, "bbox_xywh": [1, 2, 3, 4]}]
        result["hallucination_tracks"].append({"id": 9, "class": "WaitLine", "detections": [0], "relevancy": 0})
        self.path.write_text(json.dumps(result), encoding="utf-8")
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        block = self.augmented()[METRICS_KEY]
        self.assertEqual(block["unevaluated_hallucination_classes"], ["WaitLine"])
        self.assertNotIn(9, [extra["id"] for extra in block["extra_objects"]])

    def test_a_governed_class_its_rule_rejects_is_not_reported_as_unevaluated(self) -> None:
        """Hallucination 2 is a Vehicle with relevancy 0: scored and rejected, not ungoverned."""
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        block = self.augmented()[METRICS_KEY]
        self.assertEqual(block["unevaluated_hallucination_classes"], [])
        self.assertNotIn(2, [extra["id"] for extra in block["extra_objects"]])

    def test_a_hallucination_with_no_class_surfaces_as_an_empty_label(self) -> None:
        result = load_dynamic_result()
        result["hallucination_tracks"].append({"id": 9, "detections": [], "relevancy": 0})
        self.path.write_text(json.dumps(result), encoding="utf-8")
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        self.assertEqual(self.augmented()[METRICS_KEY]["unevaluated_hallucination_classes"], [""])

    def test_a_type_with_no_rules_is_named_rather_than_silently_absent(self) -> None:
        result = load_dynamic_result()
        result["tracks"].append({"track_id": 8, "object_type": "WaitLine"})
        self.path.write_text(json.dumps(result), encoding="utf-8")
        self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        block = self.augmented()[METRICS_KEY]
        self.assertEqual(block["unevaluated_object_types"], ["WaitLine"])
        self.assertNotIn("WaitLine", block["by_type"])


class TestFailureModes(ObjectMetricsTestBase):
    def test_a_filename_carrying_no_mode_is_processed_anyway(self) -> None:
        """Nothing is inferred from the name, so an unconventional one is not a failure."""
        renamed = self.directory / "results.json"
        self.path.rename(renamed)
        self.assertEqual(main([str(renamed), "--importance", "ego_relevant", "--in-place", "--quiet"]), 0)
        self.assertIn(METRICS_KEY, json.loads(renamed.read_text(encoding="utf-8")))

    def test_output_with_several_inputs_raises(self) -> None:
        second = self.directory / "clip456.dynamic.objects.results.json"
        second.write_text(json.dumps(self.original), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_tool(str(self.directory), "--importance", "ego_relevant", "--output", str(second))

    def test_missing_path_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.run_tool(str(self.directory / "absent.json"), "--importance", "ego_relevant", "--in-place")

    def test_malformed_json_raises_and_leaves_the_file_untouched(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{not json")

    def test_a_payload_that_is_not_an_object_raises(self) -> None:
        self.path.write_text("[1, 2, 3]", encoding="utf-8")
        with self.assertRaises(TypeError):
            self.run_tool(str(self.path), "--importance", "ego_relevant", "--in-place")

    def test_importance_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            main([str(self.path), "--in-place"])

    def test_a_destination_is_required(self) -> None:
        with self.assertRaises(SystemExit):
            main([str(self.path), "--importance", "ego_relevant"])


class TestTrafficLightAugmentation(unittest.TestCase):
    """The static path: the color verdicts must reach the file per object, not only as a per-type count."""

    def setUp(self) -> None:
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory)
        self.path = self.directory / "clip123.static.objects.results.json"
        self.original = load_traffic_light_result()
        self.path.write_text(json.dumps(self.original, indent=2), encoding="utf-8")
        main([str(self.path), "--importance", "all_scored", "--in-place", "--quiet"])
        self.block = json.loads(self.path.read_text(encoding="utf-8"))[METRICS_KEY]

    def test_per_track_state_verdicts_match_the_library(self) -> None:
        policy = default_policy(ImportancePolicy.ALL_SCORED)
        view = read_result(self.original, importance=ImportancePolicy.ALL_SCORED)
        entries = {entry["track_id"]: entry for entry in self.block["tracks"]}
        for track_id, track in view.tracks.items():
            self.assertEqual(entries[track_id]["wrong_state"], is_wrong_state(track, policy), track_id)
            self.assertEqual(entries[track_id]["conflicting"], is_conflicting(track, policy), track_id)
        # Without this the loop above would pass by comparing None to None for every track.
        self.assertIn(True, [entry["wrong_state"] for entry in self.block["tracks"]])
        self.assertIn(True, [entry["conflicting"] for entry in self.block["tracks"]])

    def test_the_block_carries_the_state_rules_it_applied(self) -> None:
        self.assertEqual(self.block["policy"]["wrong_state"]["TrafficLight"]["rule"], "wrong_color_episode")
        self.assertEqual(self.block["policy"]["conflicting"]["TrafficLight"]["rule"], "conflicting_colors")

    def test_by_type_publishes_a_rate_for_every_signal_it_counted(self) -> None:
        row = self.block["by_type"]["TrafficLight"]
        self.assertEqual((row["observed"], row["color_observed"]), (4, 3))
        self.assertEqual((row["wrong_state"], row["conflicting"]), (1, 1))
        # Color divides by the lights the state check measured, geometry by the ones the overlap check scored.
        self.assertAlmostEqual(row["wrong_state_rate"], 1 / 3)
        self.assertAlmostEqual(row["conflicting_rate"], 1 / 3)
        self.assertAlmostEqual(row["missing_rate"], 0.25)
        self.assertIsNone(row["misaligned_rate"], "a signal no rule measured stays null, never 0.0")


if __name__ == "__main__":
    unittest.main()
