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

"""Unit tests for the attribute verification VLM verifier.

One test per code branch for 100% coverage without redundant tests.
"""

from io import BytesIO
import logging
import subprocess
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import openai

from checks.attribute_verification.vlm_verifier import VLMVerifier, _extract_mcq_letter, _resolve_api_key


def _make_mock_chat_completion(content: str | None) -> MagicMock:
    """Build a mock chat completion with the given message content."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = content
    return completion


class TestVLMVerifier(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """Set up for VLM verifier tests."""
        self.logger = logging.getLogger(__name__)
        with patch("checks.attribute_verification.vlm_verifier.AsyncOpenAI"):
            self.verifier = VLMVerifier(
                system_prompt="You are a helpful assistant.",
                retry=1,
                temperature=0.0,
                top_p=1.0,
                frequency_penalty=0.0,
                max_tokens=512,
                stream=False,
                endpoint="http://test.invalid",
                model="test-model",
                logger=self.logger,
            )

    def test_from_config_returns_instance(self) -> None:
        """from_config builds a VLMVerifier with config params."""
        with patch("checks.attribute_verification.vlm_verifier.AsyncOpenAI"):
            instance = VLMVerifier.from_config(
                config_params={"retry": 2, "temperature": 0.1},
                system_prompt="System",
                endpoint="http://ep",
                model="model",
                logger=self.logger,
            )
        self.assertIsInstance(instance, VLMVerifier)
        self.assertEqual(instance.max_retries, 2)
        self.assertEqual(instance.temperature, 0.1)

    def test_from_config_sets_frame_count(self) -> None:
        with patch("checks.attribute_verification.vlm_verifier.AsyncOpenAI"):
            instance = VLMVerifier.from_config(
                config_params={},
                system_prompt="System",
                endpoint="http://ep",
                model="model",
                logger=self.logger,
                frames=6,
            )
        self.assertEqual(instance.frames, 6)

    def test_reasoning_answer_parser_uses_final_letter(self) -> None:
        self.assertEqual(_extract_mcq_letter("A may fit, but the final answer is C"), "C")
        self.assertEqual(_extract_mcq_letter("<think>A then B then D</think>\nC"), "C")
        self.assertEqual(_extract_mcq_letter("<think>A then B"), "UNKNOWN")

    def test_api_key_resolution_prefers_configured_environment_variable(self) -> None:
        with patch.dict(
            "checks.attribute_verification.vlm_verifier.os.environ",
            {"CUSTOM_KEY": "custom", "VLM_API_KEY": "role", "BUILD_NVIDIA_API_KEY": "legacy"},
            clear=True,
        ):
            self.assertEqual(_resolve_api_key("CUSTOM_KEY"), "custom")

    def test_extract_first_frame_succeeds_when_ffmpeg_writes_frame(self) -> None:
        """_extract_first_frame returns True when ffmpeg exits 0 and writes a non-empty output file."""
        with (
            patch("checks.attribute_verification.vlm_verifier.subprocess.run") as mock_run,
            patch("checks.attribute_verification.vlm_verifier.os.path.exists", return_value=True),
            patch("checks.attribute_verification.vlm_verifier.os.path.getsize", return_value=1024),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            result = self.verifier._extract_first_frame("/fake/video.mp4", "/fake/out.jpg")
        self.assertTrue(result)
        mock_run.assert_called_once()

        # Pin the ffmpeg invocation so regressions in the argv are caught.
        args = mock_run.call_args[0][0]
        # The packaged ffmpeg binary is invoked by name (resolved on PATH via //services:ffmpeg_layer).
        self.assertEqual(args[0], "ffmpeg")
        # Exactly the first video frame is extracted.
        self.assertEqual(args[args.index("-frames:v") + 1], "1")
        # Errors-only logging keeps stderr meaningful for the failure path.
        self.assertEqual(args[args.index("-loglevel") + 1], "error")
        # The caller-provided output path is the final positional target.
        self.assertIn("/fake/out.jpg", args)
        # A timeout guards against a hung ffmpeg process.
        self.assertEqual(mock_run.call_args.kwargs["timeout"], 30)

    def test_format_question_returns_formatted_string(self) -> None:
        """_format_question returns question plus sorted options and instruction."""
        result = self.verifier._format_question(
            "What color?",
            {"A": "red", "B": "blue", "C": "green"},
        )
        self.assertIn("What color?", result)
        self.assertIn("A) red", result)
        self.assertIn("B) blue", result)
        self.assertIn("C) green", result)
        self.assertIn("Answer with only a single letter (A, B, C, or D).", result)

    async def test_verify_question_returns_correct_and_answer_when_mock_returns_letter(self) -> None:
        """verify_question returns (True, correct_answer) when msc and AsyncOpenAI mock return success."""
        video_path = "storage/video.mp4"
        question = "What color?"
        options = {"A": "red", "B": "blue"}
        correct_answer = "A"

        def fake_extract(video_path_arg: str, output_path: str) -> bool:
            with open(output_path, "wb") as f:
                f.write(b"\xff\xd8\xff")  # minimal JPEG-like bytes for base64
            return True

        self.verifier.client = AsyncMock()
        self.verifier.client.chat.completions.create = AsyncMock(return_value=_make_mock_chat_completion("A"))

        with patch("checks.attribute_verification.vlm_verifier.msc") as mock_msc:
            mock_msc.is_file.return_value = True
            mock_msc.open.return_value.__enter__.return_value = BytesIO(b"fake video bytes")
            mock_msc.open.return_value.__exit__.return_value = None
            with patch.object(self.verifier, "_extract_first_frame", side_effect=fake_extract):
                is_correct, vlm_answer, reasoning = await self.verifier.verify_question(
                    video_path, question, options, correct_answer
                )

        self.assertTrue(is_correct)
        self.assertEqual(vlm_answer, "A")
        self.assertIsNone(reasoning)
        self.verifier.client.chat.completions.create.assert_called_once()

    # --- One test per additional branch for 100% coverage ---

    def test_extract_first_frame_returns_false_when_ffmpeg_nonzero(self) -> None:
        """_extract_first_frame: ffmpeg exits non-zero → return False."""
        with patch("checks.attribute_verification.vlm_verifier.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="decode error")
            result = self.verifier._extract_first_frame("/fake/video.mp4", "/fake/out.jpg")
        self.assertFalse(result)

    def test_extract_first_frame_returns_false_when_output_missing(self) -> None:
        """_extract_first_frame: ffmpeg exits 0 but produced no output file → return False."""
        with (
            patch("checks.attribute_verification.vlm_verifier.subprocess.run") as mock_run,
            patch("checks.attribute_verification.vlm_verifier.os.path.exists", return_value=False),
        ):
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            result = self.verifier._extract_first_frame("/fake/video.mp4", "/fake/out.jpg")
        self.assertFalse(result)

    def test_extract_first_frame_returns_false_on_timeout(self) -> None:
        """_extract_first_frame: ffmpeg exceeds the timeout → return False."""
        with patch("checks.attribute_verification.vlm_verifier.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)
            result = self.verifier._extract_first_frame("/fake/video.mp4", "/fake/out.jpg")
        self.assertFalse(result)

    def test_extract_first_frame_returns_false_on_exception(self) -> None:
        """_extract_first_frame: ffmpeg binary missing (OSError) → return False."""
        with patch("checks.attribute_verification.vlm_verifier.subprocess.run") as mock_run:
            mock_run.side_effect = OSError("ffmpeg not found")
            result = self.verifier._extract_first_frame("/fake/video.mp4", "/fake/out.jpg")
        self.assertFalse(result)

    async def test_verify_question_raises_when_video_not_found(self) -> None:
        """verify_question: msc.is_file(video_path) False → Exception."""
        with patch("checks.attribute_verification.vlm_verifier.msc") as mock_msc:
            mock_msc.is_file.return_value = False
            with self.assertRaises(Exception) as ctx:
                await self.verifier.verify_question("missing.mp4", "Q?", {"A": "a"}, "A")
            self.assertIn("Media file not found", str(ctx.exception))

    async def test_verify_question_raises_when_extract_first_frame_fails(self) -> None:
        """verify_question: _extract_first_frame returns False → Exception."""
        with patch("checks.attribute_verification.vlm_verifier.msc") as mock_msc:
            mock_msc.is_file.return_value = True
            mock_msc.open.return_value.__enter__.return_value = BytesIO(b"x")
            mock_msc.open.return_value.__exit__.return_value = None
            with patch.object(self.verifier, "_extract_first_frame", return_value=False):
                with self.assertRaises(Exception) as ctx:
                    await self.verifier.verify_question("v.mp4", "Q?", {"A": "a"}, "A")
                self.assertIn("Failed to extract first frame", str(ctx.exception))

    async def test_verify_question_returns_unknown_when_content_none(self) -> None:
        """A response without content is an uncommitted UNKNOWN verdict."""

        def fake_extract(video_path_arg: str, output_path: str) -> bool:
            with open(output_path, "wb") as f:
                f.write(b"\xff\xd8")
            return True

        self.verifier.client = AsyncMock()
        self.verifier.client.chat.completions.create = AsyncMock(return_value=_make_mock_chat_completion(None))

        with patch("checks.attribute_verification.vlm_verifier.msc") as mock_msc:
            mock_msc.is_file.return_value = True
            mock_msc.open.return_value.__enter__.return_value = BytesIO(b"x")
            mock_msc.open.return_value.__exit__.return_value = None
            with patch.object(self.verifier, "_extract_first_frame", side_effect=fake_extract):
                is_correct, answer, reasoning = await self.verifier.verify_question("v.mp4", "Q?", {"A": "a"}, "A")
        self.assertFalse(is_correct)
        self.assertEqual(answer, "UNKNOWN")
        self.assertIsNone(reasoning)

    async def test_verify_question_accepts_image_and_returns_reasoning(self) -> None:
        self.verifier.client = AsyncMock()
        self.verifier.client.chat.completions.create = AsyncMock(
            return_value=_make_mock_chat_completion("The views show matching clothing.\nB")
        )
        with patch("checks.attribute_verification.vlm_verifier.msc") as mock_msc:
            mock_msc.is_file.return_value = True
            mock_msc.open.return_value.__enter__.return_value = BytesIO(b"png bytes")
            mock_msc.open.return_value.__exit__.return_value = None
            is_correct, answer, reasoning = await self.verifier.verify_question(
                "storage/person.png",
                "Are the views consistent?",
                {"A": "no", "B": "yes"},
                "B",
                request_reasoning=True,
            )
        self.assertTrue(is_correct)
        self.assertEqual(answer, "B")
        self.assertEqual(reasoning, "The views show matching clothing.")
        request = self.verifier.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["max_tokens"], 512)
        self.assertEqual(request["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        image_url = request["messages"][1]["content"][1]["image_url"]["url"]
        self.assertTrue(image_url.startswith("data:image/png;base64,"))

    async def test_verify_question_sends_all_sampled_video_frames(self) -> None:
        """Configured video frames are sent together in chronological order."""
        self.verifier.frames = 3
        self.verifier.client = AsyncMock()
        self.verifier.client.chat.completions.create = AsyncMock(return_value=_make_mock_chat_completion("A"))

        def fake_extract_frames(video_path: str, frame_count: int, output_dir: str) -> list[str]:
            self.assertEqual(frame_count, 3)
            paths = []
            for index in range(frame_count):
                path = f"{output_dir}/frame_{index:02d}.jpg"
                with open(path, "wb") as image_file:
                    image_file.write(f"frame-{index}".encode())
                paths.append(path)
            return paths

        with patch("checks.attribute_verification.vlm_verifier.msc") as mock_msc:
            mock_msc.is_file.return_value = True
            mock_msc.open.return_value.__enter__.return_value = BytesIO(b"video")
            mock_msc.open.return_value.__exit__.return_value = None
            with patch.object(self.verifier, "_extract_frames", side_effect=fake_extract_frames):
                is_correct, answer, reasoning = await self.verifier.verify_question(
                    "storage/video.mp4", "What changed?", {"A": "weather", "B": "nothing"}, "A"
                )

        self.assertTrue(is_correct)
        self.assertEqual(answer, "A")
        self.assertIsNone(reasoning)
        request = self.verifier.client.chat.completions.create.call_args.kwargs
        content = request["messages"][1]["content"]
        self.assertEqual(
            content[0]["text"],
            "You are shown 3 frames sampled in chronological order from a single generated video; "
            "consider all of them when answering.\n\n"
            "What changed?\nA) weather\nB) nothing\n\n"
            "Answer with only a single letter (A, B, C, or D).",
        )
        self.assertEqual([item["type"] for item in content], ["text", "image_url", "image_url", "image_url"])

    async def test_verify_question_returns_unknown_when_content_not_parseable(self) -> None:
        """verify_question: no A-D regex match, strip/upper not in A-D → vlm_answer UNKNOWN."""

        def fake_extract(video_path_arg: str, output_path: str) -> bool:
            with open(output_path, "wb") as f:
                f.write(b"\xff\xd8")
            return True

        self.verifier.client = AsyncMock()
        self.verifier.client.chat.completions.create = AsyncMock(
            return_value=_make_mock_chat_completion("unclear response")
        )

        with patch("checks.attribute_verification.vlm_verifier.msc") as mock_msc:
            mock_msc.is_file.return_value = True
            mock_msc.open.return_value.__enter__.return_value = BytesIO(b"x")
            mock_msc.open.return_value.__exit__.return_value = None
            with patch.object(self.verifier, "_extract_first_frame", side_effect=fake_extract):
                is_correct, vlm_answer, _ = await self.verifier.verify_question(
                    "v.mp4", "Q?", {"A": "a", "B": "b"}, "A"
                )
        self.assertEqual(vlm_answer, "UNKNOWN")
        self.assertFalse(is_correct)

    async def test_verify_question_returns_false_when_answer_wrong(self) -> None:
        """verify_question: vlm_answer != correct_answer → is_correct False."""

        def fake_extract(video_path_arg: str, output_path: str) -> bool:
            with open(output_path, "wb") as f:
                f.write(b"\xff\xd8")
            return True

        self.verifier.client = AsyncMock()
        self.verifier.client.chat.completions.create = AsyncMock(return_value=_make_mock_chat_completion("B"))

        with patch("checks.attribute_verification.vlm_verifier.msc") as mock_msc:
            mock_msc.is_file.return_value = True
            mock_msc.open.return_value.__enter__.return_value = BytesIO(b"x")
            mock_msc.open.return_value.__exit__.return_value = None
            with patch.object(self.verifier, "_extract_first_frame", side_effect=fake_extract):
                is_correct, vlm_answer, _ = await self.verifier.verify_question(
                    "v.mp4", "Q?", {"A": "a", "B": "b"}, "A"
                )
        self.assertFalse(is_correct)
        self.assertEqual(vlm_answer, "B")

    def test_init_uses_placeholder_key_when_env_unset(self) -> None:
        """__init__: BUILD_NVIDIA_API_KEY not set → AsyncOpenAI called with api_key 'not-used'."""
        with patch("checks.attribute_verification.vlm_verifier.AsyncOpenAI") as mock_cls:
            with patch("checks.attribute_verification.vlm_verifier.os.environ.get", return_value=None):
                VLMVerifier(
                    system_prompt="",
                    retry=0,
                    temperature=0.0,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    max_tokens=512,
                    stream=False,
                    endpoint="http://test",
                    model="m",
                    logger=self.logger,
                )
            mock_cls.assert_called_once()
            self.assertEqual(mock_cls.call_args[1]["api_key"], "not-used")

    async def test_verify_question_retries_on_retryable_error(self) -> None:
        """verify_question: retryable error then success (retry > 0) → retry and return."""

        def fake_extract(video_path_arg: str, output_path: str) -> bool:
            with open(output_path, "wb") as f:
                f.write(b"\xff\xd8")
            return True

        self.verifier.max_retries = 1
        self.verifier.client = AsyncMock()
        self.verifier.client.chat.completions.create = AsyncMock(
            side_effect=[
                openai.APITimeoutError(request=MagicMock()),
                _make_mock_chat_completion("A"),
            ]
        )

        with patch("checks.attribute_verification.vlm_verifier.msc") as mock_msc:
            mock_msc.is_file.return_value = True
            mock_msc.open.return_value.__enter__.return_value = BytesIO(b"x")
            mock_msc.open.return_value.__exit__.return_value = None
            with patch.object(self.verifier, "_extract_first_frame", side_effect=fake_extract):
                is_correct, vlm_answer, _ = await self.verifier.verify_question("v.mp4", "Q?", {"A": "a"}, "A")
        self.assertTrue(is_correct)
        self.assertEqual(vlm_answer, "A")
        self.assertEqual(self.verifier.client.chat.completions.create.call_count, 2)

    async def test_verify_question_raises_runtime_error_after_exhausted_retries(self) -> None:
        """verify_question: retryable error and retry == 0 → RuntimeError."""

        def fake_extract(video_path_arg: str, output_path: str) -> bool:
            with open(output_path, "wb") as f:
                f.write(b"\xff\xd8")
            return True

        self.verifier.max_retries = 0
        self.verifier.client = AsyncMock()
        self.verifier.client.chat.completions.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=MagicMock())
        )

        with patch("checks.attribute_verification.vlm_verifier.msc") as mock_msc:
            mock_msc.is_file.return_value = True
            mock_msc.open.return_value.__enter__.return_value = BytesIO(b"x")
            mock_msc.open.return_value.__exit__.return_value = None
            with patch.object(self.verifier, "_extract_first_frame", side_effect=fake_extract):
                with self.assertRaises(RuntimeError) as ctx:
                    await self.verifier.verify_question("v.mp4", "Q?", {"A": "a"}, "A")
                self.assertIn("Max retries reached", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
