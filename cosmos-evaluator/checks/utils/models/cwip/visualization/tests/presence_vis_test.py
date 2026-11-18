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

"""Unit tests for the static-check presence masks and their visualization (CPU only)."""

import json
import os
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest import mock

import numpy as np
import torch

from checks.utils.models.cwip.modules.presence import PresenceSource, TypeMaskGate
from checks.utils.models.cwip.visualization import presence_vis


def _similarity_logits(labels: list[list[int]]) -> torch.Tensor:
    """``(1, 5, h, w)`` similarity logits whose argmax is ``labels`` (PatchSimilarityType values)."""
    one_hot = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=5).permute(2, 0, 1).float()
    return (one_hot * 10.0 - 5.0).unsqueeze(0)


def _type_logits(top_probs: list[list[float]], num_classes: int = 3) -> torch.Tensor:
    """``(1, num_classes, h, w)`` type logits: class 1 wins every patch with probability ``top_probs``."""
    top = torch.tensor(top_probs)
    probs = torch.full((num_classes, *top.shape), 0.01)
    probs[1] = top
    return torch.log(probs / (1 - probs)).unsqueeze(0)


class TestPresenceMasks(unittest.TestCase):
    """``compute_presence_masks``, ``compare_masks``, ``write_presence_outputs``, and the labels."""

    MATCH, EXTRA, MISSING = 1, 3, 4

    def setUp(self) -> None:
        # Columns: MATCH / EXTRA / MISSING similarity labels; type-head top probability 0.9 / 0.9 / 0.3.
        self.sim = _similarity_logits([[self.MATCH, self.EXTRA, self.MISSING]])
        self.typ = _type_logits([[0.9, 0.9, 0.3]])

    def _masks(self, **kwargs: Any) -> tuple[list, list]:
        sim = kwargs.pop("sim", self.sim)
        kwargs.setdefault("presence_source", PresenceSource.SIMILARITY)
        corr, hall = presence_vis.compute_presence_masks(sim, self.typ, presence_vis.PresenceOptions(**kwargs), None)
        self.assertEqual(corr.dtype, np.uint8)
        return corr.tolist(), hall.tolist()

    def test_similarity_source(self) -> None:
        self.assertEqual(self._masks(), ([[[1, 0, 0]]], [[[0, 1, 0]]]))

    def test_wrong_label_counts_as_match_like_the_checker(self) -> None:
        corr, _ = self._masks(sim=_similarity_logits([[2, 3, 4]]))  # WRONG / EXTRA / MISSING
        self.assertEqual(corr, [[[1, 0, 0]]])

    def test_type_source(self) -> None:
        self.assertEqual(self._masks(presence_source=PresenceSource.TYPE), ([[[1, 1, 0]]], [[[1, 1, 0]]]))

    def test_union_and_intersection(self) -> None:
        self.assertEqual(self._masks(presence_source=PresenceSource.UNION), ([[[1, 1, 0]]], [[[1, 1, 0]]]))
        self.assertEqual(self._masks(presence_source=PresenceSource.INTERSECTION), ([[[1, 0, 0]]], [[[0, 1, 0]]]))

    def test_threshold_and_per_class_override(self) -> None:
        self.assertEqual(
            self._masks(presence_source=PresenceSource.TYPE, type_confidence_threshold=0.95), ([[[0, 0, 0]]],) * 2
        )
        # Identity mapping (no class mapping): ObjectType EDGE == model class 1, the winning class here.
        self.assertEqual(
            self._masks(presence_source=PresenceSource.TYPE, per_class_thresholds={"EDGE": 0.2}), ([[[1, 1, 1]]],) * 2
        )

    def test_argmax_gate(self) -> None:
        self.assertEqual(
            self._masks(presence_source=PresenceSource.TYPE, type_mask_gate=TypeMaskGate.ARGMAX), ([[[1, 1, 1]]],) * 2
        )

    def test_camera_only_uses_type_mask(self) -> None:
        for source in ("similarity", "union", "intersection"):
            self.assertEqual(self._masks(sim=None, presence_source=source), ([[[1, 1, 0]]],) * 2, source)

    def test_background_patches_cleared(self) -> None:
        # Background (class 0) wins the middle patch with probability 0.9: present per the type head, but the checker
        # never scores a background patch, so both masks clear it, whatever the source.
        self.typ[0, 0, 0, 1] = 5.0
        self.typ[0, 1, 0, 1] = -5.0
        self.assertEqual(self._masks(presence_source=PresenceSource.TYPE), ([[[1, 0, 0]]], [[[1, 0, 0]]]))
        self.assertEqual(self._masks(), ([[[1, 0, 0]]], [[[0, 0, 0]]]))
        mapping = mock.MagicMock(__name__="FakeMapping")
        with mock.patch.object(presence_vis, "get_model_class_index_for_object_type", return_value=1):
            corr, _ = presence_vis.compute_presence_masks(
                self.sim, self.typ, presence_vis.PresenceOptions(presence_source=PresenceSource.TYPE), mapping
            )
        self.assertEqual(corr.tolist(), [[[0, 1, 0]]])  # the mapping puts background at model class 1

    def test_similarity_resized_to_type_grid(self) -> None:
        typ = _type_logits([[0.9] * 6, [0.9] * 6])
        options = presence_vis.PresenceOptions(presence_source=PresenceSource.SIMILARITY)
        corr, _ = presence_vis.compute_presence_masks(self.sim, typ, options, None)
        self.assertEqual(corr.shape, (1, 2, 6))
        self.assertEqual(corr[0, 0].tolist(), [1, 1, 0, 0, 0, 0])

    def test_type_source_without_type_head_raises(self) -> None:
        options = presence_vis.PresenceOptions(presence_source=PresenceSource.UNION)
        with self.assertRaises(ValueError):
            presence_vis.compute_presence_masks(self.sim, None, options, None)
        # The similarity source does not need it.
        options = presence_vis.PresenceOptions(presence_source=PresenceSource.SIMILARITY)
        corr, _ = presence_vis.compute_presence_masks(self.sim, None, options, None)
        self.assertEqual(corr.tolist(), [[[1, 0, 0]]])

    def test_resolve_class_thresholds(self) -> None:
        self.assertEqual(presence_vis.resolve_class_thresholds({"TRAFFIC_LIGHT_RED": 0.7}, None), {8: 0.7})
        with self.assertRaises(ValueError):
            presence_vis.resolve_class_thresholds({"NOT_A_TYPE": 0.7}, None)
        mapping = mock.MagicMock(__name__="FakeMapping")
        with (
            mock.patch.object(presence_vis, "get_model_class_index_for_object_type", return_value=None),
            self.assertRaises(ValueError),
        ):
            presence_vis.resolve_class_thresholds({"TRAFFIC_LIGHT_RED": 0.7}, mapping)
        with mock.patch.object(presence_vis, "get_model_class_index_for_object_type", return_value=4):
            self.assertEqual(presence_vis.resolve_class_thresholds({"TRAFFIC_LIGHT_RED": 0.7}, mapping), {4: 0.7})

    def _write(self, corr: np.ndarray, hall: np.ndarray, compared: tuple | None, **kwargs: Any) -> dict:
        """Run ``write_presence_outputs`` on a black 120x240 camera; return the npz, summary, and video-writer args."""
        options = kwargs.pop("options", presence_vis.PresenceOptions(presence_source=PresenceSource.UNION))
        with (
            tempfile.TemporaryDirectory() as out,
            mock.patch.object(presence_vis.image_video_utils, "write_video_with_ffmpeg") as mock_write,
        ):
            presence_vis.write_presence_outputs(
                Path(out),
                torch.zeros(len(corr), 3, 120, 240),
                corr,
                hall,
                options,
                compared=compared,
                fps=10.0,
                **kwargs,
            )
            with np.load(os.path.join(out, "presence_masks_iter_000000.npz")) as npz:
                arrays = {k: npz[k] for k in npz.files}
            summary = json.loads(Path(out, "presence_summary_iter_000000.json").read_text())
        frames, path, out_w, out_h, fps = mock_write.call_args.args
        self.assertTrue(path.endswith("presence_mask_vis_iter_000000.mp4"))
        self.assertEqual(fps, 10.0)
        return {"arrays": arrays, "summary": summary, "frames": frames, "size": (out_w, out_h)}

    def test_write_presence_outputs_two_stream(self) -> None:
        # Mask column 0 covers x < 120 of each panel, column 1 x >= 120.
        corr = np.array([[[1, 0]], [[0, 0]]], dtype=np.uint8)
        hall = np.array([[[0, 0]], [[0, 1]]], dtype=np.uint8)
        similarity = np.array([[[1, 0]], [[1, 1]]], dtype=np.uint8)
        type_mask = np.array([[[0, 1]], [[0, 1]]], dtype=np.uint8)
        compared = (
            presence_vis.ComparedMask("similarity", "similarity", similarity),
            presence_vis.ComparedMask("type", "type_confidence>=0.5", type_mask),
        )
        options = presence_vis.PresenceOptions(presence_source=PresenceSource.UNION, per_class_thresholds={"EDGE": 0.2})
        out = self._write(corr, hall, compared, options=options, camera_only=False)
        arrays, summary = out["arrays"], out["summary"]
        np.testing.assert_array_equal(arrays["presence"], [[[1, 0]], [[0, 1]]])
        np.testing.assert_array_equal(arrays["correspondence"], corr)
        np.testing.assert_array_equal(arrays["hallucination"], hall)
        np.testing.assert_array_equal(arrays["similarity_presence"], similarity)
        np.testing.assert_array_equal(arrays["type_presence"], type_mask)
        self.assertEqual(
            {k: summary[k] for k in ("mode", "presence_source", "presence", "compared", "per_class_thresholds")},
            {
                "mode": "two_stream",
                "presence_source": "union",
                "presence": "presence_source=union, type_confidence>=0.5 (+1 per-class)",
                "compared": {"similarity": "similarity", "type": "type_confidence>=0.5"},
                "per_class_thresholds": {"EDGE": 0.2},
            },
        )
        self.assertEqual(
            summary["per_frame_fraction_present"],
            {
                "presence": [0.5, 0.5],
                "similarity_presence": [0.5, 1.0],
                "type_presence": [0.5, 0.5],
                "similarity_only": [0.5, 0.5],
                "type_only": [0.5, 0.0],
                "both": [0.0, 0.5],
            },
        )
        self.assertAlmostEqual(summary["mean_fraction_present"]["type_only"], 0.25)

        # 2x2 grid of 120x240 panels. The camera is black: a tinted pixel is half the color, an untinted one black.
        self.assertEqual(out["size"], (480, 240))
        f0, f1 = out["frames"]
        self.assertEqual(f0.shape, (240, 480, 3))
        # Top-left: presence tinted onto the camera; bottom-left: presence alone.
        self.assertEqual(f0[110, 30].tolist(), [0, 100, 127])
        self.assertEqual(f0[110, 200].tolist(), [0, 0, 0])
        self.assertEqual(f0[230, 30].tolist(), [0, 200, 255])
        self.assertEqual(f0[230, 200].tolist(), [0, 0, 0])
        # Top-right: similarity-only (green) / type-only (magenta) tinted; bottom-right: alone.
        self.assertEqual(f0[110, 240 + 30].tolist(), [0, 127, 0])
        self.assertEqual(f0[110, 240 + 200].tolist(), [127, 0, 127])
        self.assertEqual(f0[230, 240 + 30].tolist(), [0, 255, 0])
        self.assertEqual(f0[230, 240 + 200].tolist(), [255, 0, 255])
        # Frame 1, column 1: present in both compared masks (yellow).
        self.assertEqual(f1[230, 240 + 200].tolist(), [255, 220, 0])
        self.assertEqual(f1[110, 240 + 200].tolist(), [127, 110, 0])
        self.assertEqual(f1[230, 240 + 30].tolist(), [0, 255, 0])
        # Each panel has a legend (white text over the otherwise black frame) in its top-left corner.
        for top, left in ((0, 0), (0, 240), (120, 0), (120, 240)):
            self.assertGreater(int(f0[top : top + 40, left : left + 120].max()), 200, (top, left))

    def test_write_presence_outputs_camera_only(self) -> None:
        mask = np.array([[[1, 0]]], dtype=np.uint8)
        compared = (
            presence_vis.ComparedMask("default_type", "type_confidence>=0.5 (default)", np.ones((1, 1, 2), np.uint8)),
            presence_vis.ComparedMask("type", "type_confidence>=0.6", mask),
        )
        options = presence_vis.PresenceOptions(type_confidence_threshold=0.6)
        out = self._write(mask, mask, compared, options=options, camera_only=True)
        summary = out["summary"]
        self.assertEqual(
            {k: summary[k] for k in ("mode", "presence_source", "presence", "compared")},
            {
                "mode": "camera_only",
                "presence_source": "type",
                "presence": "type_confidence>=0.6",
                "compared": {"default_type": "type_confidence>=0.5 (default)", "type": "type_confidence>=0.6"},
            },
        )
        self.assertEqual(summary["per_frame_fraction_present"]["default_type_only"], [0.5])
        self.assertEqual(summary["per_frame_fraction_present"]["type_only"], [0.0])
        self.assertEqual(summary["per_frame_fraction_present"]["both"], [0.5])
        # The stricter threshold drops column 1: present only at the default threshold (green).
        frame = out["frames"][0]
        self.assertEqual(out["size"], (480, 240))
        self.assertEqual(frame[230, 240 + 200].tolist(), [0, 255, 0])
        self.assertEqual(frame[230, 240 + 30].tolist(), [255, 220, 0])  # column 0 present under both

    def test_compare_masks_without_type_head(self) -> None:
        # A similarity-only model (no type head) has no type-head mask to contrast.
        options = presence_vis.PresenceOptions(presence_source=PresenceSource.SIMILARITY)
        self.assertIsNone(presence_vis.compare_masks(self.sim, None, options, None))

    def test_write_presence_outputs_without_comparison(self) -> None:
        corr = np.array([[[1, 0]]], dtype=np.uint8)
        hall = np.array([[[0, 1]]], dtype=np.uint8)
        options = presence_vis.PresenceOptions(presence_source=PresenceSource.SIMILARITY)
        out = self._write(corr, hall, None, options=options, camera_only=False)
        self.assertEqual(set(out["arrays"]), {"presence", "correspondence", "hallucination"})
        self.assertIsNone(out["summary"]["compared"])
        self.assertEqual(out["summary"]["per_frame_fraction_present"], {"presence": [1.0]})
        # Left column only: presence on the camera above the presence mask alone.
        self.assertEqual(out["size"], (240, 240))
        frame = out["frames"][0]
        self.assertEqual(frame[110, 30].tolist(), [0, 100, 127])
        self.assertEqual(frame[230, 200].tolist(), [0, 200, 255])

    def test_compare_masks(self) -> None:
        # Two-stream: the similarity mask vs the type-head mask under the selected type settings.
        options = presence_vis.PresenceOptions(
            presence_source=PresenceSource.INTERSECTION, type_confidence_threshold=0.95
        )
        compared = presence_vis.compare_masks(self.sim, self.typ, options, None)
        assert compared is not None
        first, second = compared
        self.assertEqual((first.key, first.label), ("similarity", "similarity"))
        self.assertEqual((second.key, second.label), ("type", "type_confidence>=0.95"))
        self.assertEqual(first.mask.tolist(), [[[1, 1, 0]]])
        self.assertEqual(second.mask.tolist(), [[[0, 0, 0]]])
        # Camera-only: the type-head mask at the default settings vs the selected ones.
        compared = presence_vis.compare_masks(None, self.typ, options, None)
        assert compared is not None
        first, second = compared
        self.assertEqual((first.key, first.label), ("default_type", "type_confidence>=0.5 (default)"))
        self.assertEqual(first.mask.tolist(), [[[1, 1, 0]]])
        self.assertEqual(second.mask.tolist(), [[[0, 0, 0]]])

    def test_presence_label(self) -> None:
        label = presence_vis.presence_label
        options = presence_vis.PresenceOptions
        # The default source is the union of the similarity verdict and the type-head mask.
        self.assertEqual(options().presence_source, "union")
        self.assertEqual(label(options(), camera_only=False), "presence_source=union, type_confidence>=0.5")
        # Two-stream names the type-head settings only when the source uses the type-head mask.
        self.assertEqual(
            label(options(presence_source=PresenceSource.SIMILARITY, type_confidence_threshold=0.7), camera_only=False),
            "presence_source=similarity",
        )
        self.assertEqual(
            label(options(presence_source=PresenceSource.UNION, type_confidence_threshold=0.7), camera_only=False),
            "presence_source=union, type_confidence>=0.7",
        )
        self.assertEqual(
            label(
                options(presence_source=PresenceSource.INTERSECTION, type_mask_gate=TypeMaskGate.ARGMAX),
                camera_only=False,
            ),
            "presence_source=intersection, type_argmax",
        )
        # Camera-only always uses the type-head mask: only its settings are named, whatever presence_source says.
        self.assertEqual(label(options(presence_source=PresenceSource.UNION), camera_only=True), "type_confidence>=0.5")
        self.assertEqual(
            label(options(per_class_thresholds={"EDGE": 0.2, "TRAFFIC_LIGHT_RED": 0.7}), camera_only=True),
            "type_confidence>=0.5 (+2 per-class)",
        )

    def test_overlay_panel_width_caps_resolution(self) -> None:
        mask = np.ones((1, 1, 1), dtype=np.uint8)
        compared = (presence_vis.ComparedMask("a", "a", mask), presence_vis.ComparedMask("b", "b", mask))
        frames = presence_vis._presence_overlay_frames(torch.zeros(1, 3, 1080, 1920), mask, compared, 960, "label")
        self.assertEqual(frames[0].shape, (1080, 1920, 3))


if __name__ == "__main__":
    unittest.main()
