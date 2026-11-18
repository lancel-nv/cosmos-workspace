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

"""Unit tests for api module."""

from typing import Any
import unittest
from unittest.mock import MagicMock, patch

from checks.objects.single_view.api import run_object_processors as run
from checks.objects.single_view.object_processor_base import ObjectProcessorBase


class TestRunObjectProcessors(unittest.TestCase):
    """Test case for the run_object_processors() function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.base_args: dict[str, Any] = {
            "input_data": "/path/to/data",
            "clip_id": "test_clip",
            "camera_name": "front_camera",
            "video_path": "/path/to/video.mp4",
            "output_dir": "/path/to/output",
            "model_device": "cpu",
            "verbose": "INFO",
            "trial_frames": 5,
        }
        self.mock_dynamic_result = {"processed_frames": 10, "mean_score": 0.85}
        self.mock_static_result = {"processed_frames": 10, "mean_score": 0.90}

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_with_only_dynamic_objects(self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock) -> None:
        """Test that only dynamic processor is run when only dynamic objects are configured."""
        # Config with only dynamic objects (vehicle, pedestrian)
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "vehicle": {"method": "ratio"},
                    "pedestrian": {"method": "cluster"},
                }
            }
        }

        # Setup mocks
        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_instance = MagicMock()
        mock_static_instance.process_clip.return_value = self.mock_static_result
        mock_static_cls.return_value = mock_static_instance

        # Call run with world_video_path (but should not matter since no static objects)
        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        # Verify dynamic processor was created and called
        mock_dynamic_cls.assert_called_once()
        mock_dynamic_instance.process_clip.assert_called_once()

        # Verify static processor was NOT created or called
        mock_static_cls.assert_not_called()
        mock_static_instance.process_clip.assert_not_called()

        # Verify results
        self.assertIn("dynamic", result)
        self.assertNotIn("static", result)
        self.assertEqual(result["dynamic"], self.mock_dynamic_result)

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_with_only_static_objects(self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock) -> None:
        """Test that only static processor is run when only static objects are configured."""
        # Config with only static objects (traffic_sign, crosswalk)
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "traffic_sign": {"method": "ratio"},
                    "crosswalk": {"method": "ratio"},
                }
            }
        }

        # Setup mocks
        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_instance = MagicMock()
        mock_static_instance.process_clip.return_value = self.mock_static_result
        mock_static_cls.return_value = mock_static_instance

        # Call run with world_video_path
        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        # Verify dynamic processor was NOT created or called
        mock_dynamic_cls.assert_not_called()
        mock_dynamic_instance.process_clip.assert_not_called()

        # Verify static processor was created and called
        mock_static_cls.assert_called_once()
        mock_static_instance.process_clip.assert_called_once()

        # Verify results
        self.assertNotIn("dynamic", result)
        self.assertIn("static", result)
        self.assertEqual(result["static"], self.mock_static_result)

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_with_both_object_types(self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock) -> None:
        """Test that both processors are run when both types of objects are configured."""
        # Config with both dynamic and static objects
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "vehicle": {"method": "ratio"},
                    "pedestrian": {"method": "cluster"},
                    "traffic_sign": {"method": "ratio"},
                    "crosswalk": {"method": "ratio"},
                }
            }
        }

        # Setup mocks
        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_instance = MagicMock()
        mock_static_instance.process_clip.return_value = self.mock_static_result
        mock_static_cls.return_value = mock_static_instance

        # Call run with world_video_path
        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        # Verify both processors were created and called
        mock_dynamic_cls.assert_called_once()
        mock_dynamic_instance.process_clip.assert_called_once()
        mock_static_cls.assert_called_once()
        mock_static_instance.process_clip.assert_called_once()

        # Verify results contain both
        self.assertIn("dynamic", result)
        self.assertIn("static", result)
        self.assertEqual(result["dynamic"], self.mock_dynamic_result)
        self.assertEqual(result["static"], self.mock_static_result)

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_static_objects_camera_only_without_world_video_path(
        self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock
    ) -> None:
        """An absent world_video_path runs the static processor camera-only rather than skipping it."""
        # Config with only static objects
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "traffic_sign": {"method": "ratio"},
                    "crosswalk": {"method": "ratio"},
                }
            }
        }

        # Setup mocks
        mock_dynamic_instance = MagicMock()
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_instance = MagicMock()
        mock_static_instance.process_clip.return_value = self.mock_static_result
        mock_static_cls.return_value = mock_static_instance

        # Call run WITHOUT world_video_path
        result = run(config=config, world_video_path=None, **self.base_args)

        # The static processor ran camera-only; the dynamic one did not (config has only static objects).
        mock_dynamic_cls.assert_not_called()
        mock_static_cls.assert_called_once()
        _, process_kwargs = mock_static_instance.process_clip.call_args
        self.assertIs(process_kwargs["camera_only"], True)
        self.assertIsNone(process_kwargs["world_video_path"])
        self.assertNotIn("dynamic", result)
        self.assertEqual(result["static"], self.mock_static_result)

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_camera_only_runs_static_without_world_video(
        self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock
    ) -> None:
        """Omitting the world video runs the static processor camera-only and forwards camera_only=True."""
        config = {"av.objects_single_view": {"overlap_check": {"traffic_light": {"method": "ratio"}}}}

        mock_static_instance = MagicMock()
        mock_static_instance.process_clip.return_value = self.mock_static_result
        mock_static_cls.return_value = mock_static_instance

        result = run(config=config, world_video_path=None, **self.base_args)

        mock_static_cls.assert_called_once()
        mock_static_instance.process_clip.assert_called_once()
        _, call_kwargs = mock_static_instance.process_clip.call_args
        self.assertIsNone(call_kwargs["world_video_path"])
        self.assertIs(call_kwargs["camera_only"], True)
        self.assertEqual(result["static"], self.mock_static_result)

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_with_empty_overlap_check(self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock) -> None:
        """Test that no processors are run when overlap_check is empty."""
        # Config with empty overlap_check
        config: dict[str, Any] = {"av.objects_single_view": {"overlap_check": {}}}

        # Setup mocks
        mock_dynamic_instance = MagicMock()
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_instance = MagicMock()
        mock_static_cls.return_value = mock_static_instance

        # Call run
        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        # Verify neither processor was created or called
        mock_dynamic_cls.assert_not_called()
        mock_static_cls.assert_not_called()

        # Verify empty results
        self.assertEqual(result, {})

    @patch.object(
        ObjectProcessorBase,
        "get_default_config",
        return_value={"av.objects_single_view": {"overlap_check": {"vehicle": {"method": "ratio"}}}},
    )
    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_with_empty_config_uses_defaults(
        self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock, _mock_defaults: MagicMock
    ) -> None:
        """An empty {} config resolves to the image defaults, so the processors the defaults configure still run."""
        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_instance = MagicMock()
        mock_static_cls.return_value = mock_static_instance

        result = run(config={}, world_video_path="/path/to/world.mp4", **self.base_args)

        # The default config configures a dynamic object, so the dynamic processor runs rather than silently
        # processing nothing.
        mock_dynamic_cls.assert_called_once()
        mock_static_cls.assert_not_called()
        self.assertEqual(result, {"dynamic": self.mock_dynamic_result})

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_with_non_empty_config_missing_section_raises(
        self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock
    ) -> None:
        """A non-empty config replaces the defaults, so omitting 'av.objects_single_view' is a hard error.

        Previously this silently selected no processors at all and reported a successful, empty run.
        """
        with self.assertRaises(ValueError) as ctx:
            run(config={"unrelated_key": 1}, world_video_path="/path/to/world.mp4", **self.base_args)

        self.assertIn("av.objects_single_view", str(ctx.exception))
        mock_dynamic_cls.assert_not_called()
        mock_static_cls.assert_not_called()

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_with_section_missing_overlap_check_raises(
        self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock
    ) -> None:
        """'overlap_check' gates which processors run, so a section omitting it must fail rather than no-op."""
        with self.assertRaises(ValueError) as ctx:
            run(
                config={"av.objects_single_view": {"importance_filter": {}}},
                world_video_path="/path/to/world.mp4",
                **self.base_args,
            )

        self.assertIn("overlap_check", str(ctx.exception))
        mock_dynamic_cls.assert_not_called()
        mock_static_cls.assert_not_called()

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_with_all_dynamic_object_types(self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock) -> None:
        """Test with all dynamic object types: bicycle, motorcycle, pedestrian, vehicle."""
        # Config with all dynamic objects
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "bicycle": {"method": "cluster"},
                    "motorcycle": {"method": "cluster"},
                    "pedestrian": {"method": "cluster"},
                    "vehicle": {"method": "ratio"},
                }
            }
        }

        # Setup mocks
        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_instance = MagicMock()
        mock_static_cls.return_value = mock_static_instance

        # Call run
        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        # Verify only dynamic processor was called
        mock_dynamic_cls.assert_called_once()
        mock_dynamic_instance.process_clip.assert_called_once()
        mock_static_cls.assert_not_called()

        self.assertIn("dynamic", result)
        self.assertNotIn("static", result)

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_with_all_static_object_types(self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock) -> None:
        """Test with all static object types: crosswalk, lane_line, road_boundary, traffic_light, traffic_sign,
        wait_line."""
        # Config with all static objects
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "crosswalk": {"method": "ratio"},
                    "lane_line": {"method": "ratio"},
                    "road_boundary": {"method": "ratio"},
                    "traffic_light": {"method": "ratio"},
                    "traffic_sign": {"method": "ratio"},
                    "wait_line": {"method": "ratio"},
                }
            }
        }

        # Setup mocks
        mock_dynamic_instance = MagicMock()
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_instance = MagicMock()
        mock_static_instance.process_clip.return_value = self.mock_static_result
        mock_static_cls.return_value = mock_static_instance

        # Call run
        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        # Verify only static processor was called
        mock_dynamic_cls.assert_not_called()
        mock_static_cls.assert_called_once()
        mock_static_instance.process_clip.assert_called_once()

        self.assertNotIn("dynamic", result)
        self.assertIn("static", result)

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_with_unknown_objects_only(self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock) -> None:
        """Test that no processors are run when only unknown object types are configured."""
        # Config with unknown object types
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "unknown_object": {"method": "ratio"},
                    "another_unknown": {"method": "cluster"},
                }
            }
        }

        # Setup mocks
        mock_dynamic_instance = MagicMock()
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_instance = MagicMock()
        mock_static_cls.return_value = mock_static_instance

        # Call run
        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        # Verify neither processor was created or called
        mock_dynamic_cls.assert_not_called()
        mock_static_cls.assert_not_called()

        # Verify empty results
        self.assertEqual(result, {})

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_static_skipped_on_runtime_error(self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock) -> None:
        """Test that static processing is skipped when ObjectProcessorStatic raises RuntimeError."""
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "vehicle": {"method": "ratio"},
                    "traffic_sign": {"method": "ratio"},
                }
            }
        }

        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_cls.side_effect = RuntimeError("CWIP model path not found")

        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        mock_static_cls.assert_called_once()
        self.assertIn("dynamic", result)
        self.assertNotIn("static", result)
        self.assertEqual(result["dynamic"], self.mock_dynamic_result)

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_static_only_skipped_on_runtime_error(
        self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock
    ) -> None:
        """Test that result is empty when only static objects are configured and init raises RuntimeError."""
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "traffic_sign": {"method": "ratio"},
                }
            }
        }

        mock_static_cls.side_effect = RuntimeError("CWIP model path not found")

        result = run(config=config, world_video_path="/path/to/world.mp4", **self.base_args)

        mock_static_cls.assert_called_once()
        mock_dynamic_cls.assert_not_called()
        self.assertNotIn("static", result)
        self.assertEqual(result, {})

    @patch("checks.objects.single_view.api.ObjectProcessorStatic")
    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_run_processor_arguments(self, mock_dynamic_cls: MagicMock, mock_static_cls: MagicMock) -> None:
        """Test that processors are initialized and called with correct arguments."""
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "vehicle": {"method": "ratio"},
                    "traffic_sign": {"method": "ratio"},
                }
            }
        }

        # Setup mocks
        mock_dynamic_instance = MagicMock()
        mock_dynamic_instance.process_clip.return_value = self.mock_dynamic_result
        mock_dynamic_cls.return_value = mock_dynamic_instance

        mock_static_instance = MagicMock()
        mock_static_instance.process_clip.return_value = self.mock_static_result
        mock_static_cls.return_value = mock_static_instance

        world_video_path = "/path/to/world.mp4"

        # Call run
        run(config=config, world_video_path=world_video_path, **self.base_args)

        # Verify dynamic processor initialization. ``seg_session=None`` is the local-CLI shape: no caller supplied a
        # pre-built session, so the processor builds and owns one.
        mock_dynamic_cls.assert_called_once_with(
            config=config,
            model_device=self.base_args["model_device"],
            verbose=self.base_args["verbose"],
            clip_id=self.base_args["clip_id"],
            output_dir=self.base_args["output_dir"],
            seg_session=None,
        )

        # Verify dynamic processor process_clip call
        mock_dynamic_instance.process_clip.assert_called_once_with(
            input_data=self.base_args["input_data"],
            clip_id=self.base_args["clip_id"],
            camera_name=self.base_args["camera_name"],
            video_path=self.base_args["video_path"],
            output_dir=self.base_args["output_dir"],
            trial_frames=self.base_args["trial_frames"],
            mask_cache_dir=None,
            cancel_event=None,
        )

        # Verify static processor initialization
        mock_static_cls.assert_called_once_with(
            config=config,
            model_device=self.base_args["model_device"],
            verbose=self.base_args["verbose"],
            clip_id=self.base_args["clip_id"],
            output_dir=self.base_args["output_dir"],
            seg_model=None,
        )

        # Verify static processor process_clip call
        mock_static_instance.process_clip.assert_called_once_with(
            input_data=self.base_args["input_data"],
            clip_id=self.base_args["clip_id"],
            camera_name=self.base_args["camera_name"],
            camera_video_path=self.base_args["video_path"],
            world_video_path=world_video_path,
            output_dir=self.base_args["output_dir"],
            trial_frames=self.base_args["trial_frames"],
            mask_cache_dir=None,
            cancel_event=None,
            camera_only=False,
        )

    @patch("checks.objects.single_view.api.ObjectProcessorDynamic")
    def test_supplied_seg_session_reaches_dynamic_processor(self, mock_dynamic_cls: MagicMock) -> None:
        """A caller-supplied SegFormer session must be handed to the processor rather than quietly ignored.

        This is the whole point of the argument: a service that pre-builds one session and then finds each job
        building its own anyway would keep racing on ``cublasCreate`` while looking like it had been fixed. Asserting
        identity rather than presence, since a copy would be a separate CUDA context.
        """
        mock_dynamic_cls.return_value.process_clip.return_value = self.mock_dynamic_result
        session = object()

        run(
            config={"av.objects_single_view": {"overlap_check": {"vehicle": {"method": "ratio"}}}},
            dynamic_seg_session=session,
            **self.base_args,
        )

        self.assertIs(mock_dynamic_cls.call_args.kwargs["seg_session"], session)


if __name__ == "__main__":
    unittest.main()
