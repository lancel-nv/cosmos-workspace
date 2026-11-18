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

"""Unit tests for the attribute verification processor."""

import asyncio
import threading
from typing import cast
import unittest
from unittest.mock import MagicMock

import openai

from checks.attribute_verification.processor import AttributeVerificationProcessor, AttributeVerificationResult
from checks.attribute_verification.question_generator import GeneratedQuestion, LLMQuestionGenerator
from checks.attribute_verification.vlm_verifier import VLMVerifier
from utils.cancellation import ProcessingCancelled

# Ceiling on every await in the cancellation tests. Generous next to the sub-millisecond work they actually do, so it
# only ever fires when cancellation failed to propagate -- which must fail the test rather than hang the suite.
_CANCEL_TEST_TIMEOUT_SECONDS = 10.0


def _make_openai_status_error_response(status_code: int = 404) -> MagicMock:
    """Build a minimal fake httpx.Response for openai.APIStatusError subclasses."""
    response = MagicMock()
    response.request = MagicMock()
    response.status_code = status_code
    response.headers = {"x-request-id": "test-request-id"}
    return response


class TestAttributeVerificationProcessor(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        """Set up the test processor."""
        self.processor = AttributeVerificationProcessor({})
        self.processor.question_generator = cast(LLMQuestionGenerator, _DummyQuestionGenerator())
        self.processor.vlm_verifier = cast(VLMVerifier, _DummyVLMVerifier())

    def test_deep_merge(self) -> None:
        """Test the deep merge method."""
        base = {
            "a": 1,
            "b": 2,
            "c": {
                "d": 3,
                "e": 4,
            },
        }
        update = {
            "a": 10,
            "b": 20,
            "c": {
                "d": 30,
                "e": 40,
                "f": 50,
                "g": {
                    "h": 60,
                    "i": 70,
                },
            },
        }
        expected = {
            "a": 10,
            "b": 20,
            "c": {
                "d": 30,
                "e": 40,
                "f": 50,
                "g": {
                    "h": 60,
                    "i": 70,
                },
            },
        }
        result = AttributeVerificationProcessor._deep_merge(base, update)
        self.assertDictEqual(result, expected)

    def test_get_default_config(self) -> None:
        """Test the get default config method."""
        config = AttributeVerificationProcessor.get_default_config()
        self.assertIsInstance(config, dict)
        self.assertIn("metropolis.attribute_verification", config)
        self.assertIsInstance(config["metropolis.attribute_verification"], dict)
        self.assertIn("question_generation", config["metropolis.attribute_verification"])
        self.assertIn("vlm_verification", config["metropolis.attribute_verification"])
        self.assertIn("parameters", config["metropolis.attribute_verification"]["question_generation"])
        self.assertIn("parameters", config["metropolis.attribute_verification"]["vlm_verification"])

    async def test_process(self) -> None:
        """Test the process method."""
        clip_id = "test_clip"
        augmented_video_path = "test_video.mp4"

        result = await self.processor.process(clip_id, augmented_video_path)
        self.assertIsInstance(result, AttributeVerificationResult)
        self.assertEqual(result.clip_id, clip_id)
        self.assertEqual(result.passed, True)
        self.assertEqual(result.summary.total_checks, 2)
        self.assertEqual(result.summary.passed_checks, 2)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(len(result.checks), 2)
        self.assertEqual(result.checks[0].variable, "color")
        self.assertEqual(result.checks[0].value, "blue")
        self.assertEqual(result.checks[0].question, "What is the color of the car?")
        self.assertEqual(result.checks[0].options, {"A": "blue", "B": "red", "C": "green", "D": "yellow"})
        self.assertEqual(result.checks[0].expected_answer, "A")
        self.assertEqual(result.checks[0].vlm_answer, "A")
        self.assertEqual(result.checks[0].passed, True)
        self.assertEqual(result.checks[0].error, None)
        self.assertEqual(result.checks[1].variable, "type")
        self.assertEqual(result.checks[1].value, "sedan")
        self.assertEqual(result.checks[1].question, "What type of car is it?")
        self.assertEqual(result.checks[1].options, {"A": "truck", "B": "SUV", "C": "sedan", "D": "van"})
        self.assertEqual(result.checks[1].expected_answer, "C")
        self.assertEqual(result.checks[1].vlm_answer, "C")
        self.assertEqual(result.checks[1].error, None)
        self.assertGreaterEqual(result.duration_seconds, 0.0)

    async def test_failed_verification_logs_multiple_choice_options(self) -> None:
        """A failed answer log includes the text associated with every choice."""
        self.processor.vlm_verifier = cast(VLMVerifier, _IncorrectVLMVerifier())

        with self.assertLogs(self.processor.logger, level="WARNING") as captured:
            await self.processor.process("test_clip", "test_video.mp4")

        self.assertIn(
            "Verification failed for 'color': expected A, got B; options: A='blue', B='red', C='green', D='yellow'",
            captured.output[0],
        )

    async def test_unknown_vlm_verdict_is_recorded_as_errored_check(self) -> None:
        """An unparseable VLM verdict is not an explicit verification failure."""
        self.processor.vlm_verifier = cast(VLMVerifier, _UnknownVLMVerifier())

        result = await self.processor.process("test_clip", "test_video.mp4")

        self.assertFalse(result.passed)
        self.assertEqual(result.summary.passed_checks, 0)
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(result.summary.errored_checks, result.summary.total_checks)
        self.assertTrue(all(check.vlm_answer == "UNKNOWN" and check.error is None for check in result.checks))

    async def test_fixed_question_can_run_without_selected_variables(self) -> None:
        self.processor.config = {
            "selected_variables": {},
            "variable_options": {},
            "extra_questions": [
                {
                    "variable": "consistency",
                    "value": "consistent",
                    "question": "Are the views consistent?",
                    "options": {"A": "no", "B": "yes"},
                    "correct_answer": "B",
                    "request_reasoning": True,
                }
            ],
        }
        result = await self.processor.process("clip", "person.png")
        self.assertTrue(result.passed)
        self.assertEqual(result.summary.total_checks, 1)
        self.assertTrue(result.checks[0].request_reasoning)
        self.assertEqual(result.checks[0].vlm_reasoning, "reason")

    async def test_excluding_every_variable_returns_structured_no_questions_failure(self) -> None:
        self.processor.config = {
            "selected_variables": {"weather": "sunny"},
            "variable_options": {"weather": ["sunny", "rainy"]},
            "exclude_variables": ["weather"],
        }
        result = await self.processor.process("clip", "video.mp4")
        self.assertFalse(result.passed)
        self.assertEqual(result.summary.reason, "no_questions")
        self.assertEqual(result.checks, [])

    async def test_invalid_fixed_questions_are_skipped(self) -> None:
        self.processor.config = {
            "selected_variables": {},
            "variable_options": {},
            "extra_questions": [
                {"question": "Missing options"},
                {
                    "question": "Too many options",
                    "options": {"A": "one", "B": "two", "C": "three", "D": "four", "E": "five"},
                    "correct_answer": "E",
                },
                {
                    "question": "Ambiguous option keys",
                    "options": {"A": "one", "a": "two"},
                    "correct_answer": "A",
                },
                "not-an-object",
            ],
        }
        result = await self.processor.process("clip", "video.mp4")
        self.assertFalse(result.passed)
        self.assertEqual(result.summary.reason, "no_questions")

    def test_fixed_question_normalizes_lowercase_option_keys(self) -> None:
        self.processor.config = {
            "extra_questions": [
                {
                    "question": "Is the light on?",
                    "options": {"a": "no", "b": "yes"},
                    "correct_answer": "b",
                }
            ]
        }

        questions = self.processor._fixed_questions()

        self.assertEqual(questions[0].options, {"A": "no", "B": "yes"})
        self.assertEqual(questions[0].correct_answer, "B")

    def test_fixed_question_remaps_non_mcq_keys_and_correct_answer(self) -> None:
        self.processor.config = {
            "extra_questions": [
                {
                    "question": "Are the views consistent?",
                    "options": {"E": "no", "F": "yes"},
                    "correct_answer": "F",
                }
            ]
        }

        questions = self.processor._fixed_questions()

        self.assertEqual(questions[0].options, {"A": "no", "B": "yes"})
        self.assertEqual(questions[0].correct_answer, "B")

    def _make_processor_with_dummy_question_generator(self) -> AttributeVerificationProcessor:
        """Return a processor with dummy VLM verifier; caller sets question_generator."""
        processor = AttributeVerificationProcessor({})
        processor.vlm_verifier = cast(VLMVerifier, _DummyVLMVerifier())
        return processor

    def _make_processor_with_dummy_vlm_verifier(self) -> AttributeVerificationProcessor:
        """Return a processor with dummy question generator; caller sets vlm_verifier."""
        processor = AttributeVerificationProcessor({})
        processor.question_generator = cast(LLMQuestionGenerator, _DummyQuestionGenerator())
        return processor

    # --- generate_questions error tests ---

    async def test_process_generate_questions_runtime_error(self) -> None:
        """RuntimeError from generate_questions is propagated as-is."""
        processor = self._make_processor_with_dummy_question_generator()
        processor.question_generator = cast(LLMQuestionGenerator, _RaisingQuestionGenerator(RuntimeError("LLM failed")))

        with self.assertRaises(RuntimeError) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("LLM failed", str(ctx.exception))

    async def test_process_generate_questions_not_found_error(self) -> None:
        """openai.NotFoundError from generate_questions is re-raised as Exception with LLM endpoint message."""
        processor = self._make_processor_with_dummy_question_generator()
        exc = openai.NotFoundError(
            "not found",
            response=_make_openai_status_error_response(404),
            body=None,
        )
        processor.question_generator = cast(LLMQuestionGenerator, _RaisingQuestionGenerator(exc))

        with self.assertRaises(Exception) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("LLM endpoint was not found", str(ctx.exception))

    async def test_process_generate_questions_authentication_error(self) -> None:
        """openai.AuthenticationError from generate_questions is re-raised as Exception with API key message."""
        processor = self._make_processor_with_dummy_question_generator()
        exc = openai.AuthenticationError(
            "unauthorized",
            response=_make_openai_status_error_response(401),
            body=None,
        )
        processor.question_generator = cast(LLMQuestionGenerator, _RaisingQuestionGenerator(exc))

        with self.assertRaises(Exception) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("API key is invalid", str(ctx.exception))

    async def test_process_generate_questions_permission_denied_error(self) -> None:
        """openai.PermissionDeniedError from generate_questions is re-raised as Exception with API key message."""
        processor = self._make_processor_with_dummy_question_generator()
        exc = openai.PermissionDeniedError(
            "forbidden",
            response=_make_openai_status_error_response(403),
            body=None,
        )
        processor.question_generator = cast(LLMQuestionGenerator, _RaisingQuestionGenerator(exc))

        with self.assertRaises(Exception) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("does not have permission", str(ctx.exception))

    async def test_process_generate_questions_unexpected_error(self) -> None:
        """Generic Exception from generate_questions is propagated as-is."""
        processor = self._make_processor_with_dummy_question_generator()
        processor.question_generator = cast(LLMQuestionGenerator, _RaisingQuestionGenerator(ValueError("unexpected")))

        with self.assertRaises(ValueError) as ctx:
            await processor.process("test_clip", "test_video.mp4")

        self.assertIn("unexpected", str(ctx.exception))

    # --- verify_question error tests ---

    def _assert_recorded_failure(self, result: AttributeVerificationResult, expected_in_error: str) -> None:
        """Common assertions for the per-question failure pattern.

        After the asyncio.gather fix, each ``_verify_one`` failure is recorded on its check
        (``passed=False``, ``error`` populated) instead of cancelling the whole batch.
        ``_DummyQuestionGenerator`` yields multiple questions; when ``_RaisingVLMVerifier``
        raises on every call, every check fails with the same error.
        """
        self.assertFalse(result.passed)
        self.assertGreaterEqual(len(result.checks), 1)
        for check in result.checks:
            self.assertFalse(check.passed)
            self.assertEqual(check.vlm_answer, "ERROR")
            self.assertIsNotNone(check.error)
            self.assertIn(expected_in_error, str(check.error))

    async def test_process_verify_question_runtime_error_recorded_as_failed_check(self) -> None:
        """RuntimeError from verify_question is captured on the check, not raised."""
        processor = self._make_processor_with_dummy_vlm_verifier()
        processor.vlm_verifier = cast(VLMVerifier, _RaisingVLMVerifier(RuntimeError("Max retries reached")))

        result = await processor.process("test_clip", "test_video.mp4")
        self._assert_recorded_failure(result, "Max retries reached")
        self.assertEqual(result.summary.failed_checks, 0)
        self.assertEqual(result.summary.errored_checks, result.summary.total_checks)

    async def test_process_verify_question_not_found_error(self) -> None:
        """openai.NotFoundError from verify_question is captured on the check, not raised."""
        processor = self._make_processor_with_dummy_vlm_verifier()
        exc = openai.NotFoundError(
            "not found",
            response=_make_openai_status_error_response(404),
            body=None,
        )
        processor.vlm_verifier = cast(VLMVerifier, _RaisingVLMVerifier(exc))

        result = await processor.process("test_clip", "test_video.mp4")
        self._assert_recorded_failure(result, "not found")

    async def test_process_verify_question_authentication_error(self) -> None:
        """openai.AuthenticationError from verify_question is captured on the check, not raised."""
        processor = self._make_processor_with_dummy_vlm_verifier()
        exc = openai.AuthenticationError(
            "unauthorized",
            response=_make_openai_status_error_response(401),
            body=None,
        )
        processor.vlm_verifier = cast(VLMVerifier, _RaisingVLMVerifier(exc))

        result = await processor.process("test_clip", "test_video.mp4")
        self._assert_recorded_failure(result, "unauthorized")

    async def test_process_verify_question_permission_denied_error(self) -> None:
        """openai.PermissionDeniedError from verify_question is captured on the check, not raised."""
        processor = self._make_processor_with_dummy_vlm_verifier()
        exc = openai.PermissionDeniedError(
            "forbidden",
            response=_make_openai_status_error_response(403),
            body=None,
        )
        processor.vlm_verifier = cast(VLMVerifier, _RaisingVLMVerifier(exc))

        result = await processor.process("test_clip", "test_video.mp4")
        self._assert_recorded_failure(result, "forbidden")

    async def test_process_verify_question_unexpected_error(self) -> None:
        """Generic Exception from verify_question is captured on the check, not raised."""
        processor = self._make_processor_with_dummy_vlm_verifier()
        processor.vlm_verifier = cast(VLMVerifier, _RaisingVLMVerifier(Exception("VLM crashed")))

        result = await processor.process("test_clip", "test_video.mp4")
        self._assert_recorded_failure(result, "VLM crashed")

    async def test_process_verify_question_one_failure_does_not_cancel_others(self) -> None:
        """A single failing question must not cancel sibling _verify_one tasks (asyncio.gather).

        ``_DummyQuestionGenerator`` returns two questions; the first verify call crashes and the
        second succeeds. We assert the batch completes and the second check is still scored.
        """
        processor = self._make_processor_with_dummy_vlm_verifier()
        processor.vlm_verifier = cast(
            VLMVerifier,
            _PartiallyFailingVLMVerifier(exceptions_in_order=[Exception("first verify crashed"), None]),
        )

        result = await processor.process("test_clip", "test_video.mp4")
        self.assertEqual(len(result.checks), 2)
        self.assertFalse(result.passed)
        first, second = result.checks
        self.assertFalse(first.passed)
        self.assertIsNotNone(first.error)
        self.assertIn("first verify crashed", str(first.error))
        self.assertTrue(second.passed)
        self.assertIsNone(second.error)

    # --- cancellation tests ---

    async def test_an_already_cancelled_job_never_reaches_the_llm(self) -> None:
        """Cancellation that arrives before the work starts costs nothing at all."""
        generator = _DummyQuestionGenerator()
        self.processor.question_generator = cast(LLMQuestionGenerator, generator)
        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaises(ProcessingCancelled):
            await self.processor.process("test_clip", "test_video.mp4", cancel_event)

        self.assertEqual(generator.calls, 0)

    async def test_cancelling_mid_batch_abandons_the_outstanding_vlm_calls(self) -> None:
        """Every question is in flight at once, so cancellation has to tear the batch down rather than wait it out."""
        verifier = _HangingVLMVerifier()
        processor = self._make_processor_with_dummy_vlm_verifier()
        processor.vlm_verifier = cast(VLMVerifier, verifier)
        cancel_event = threading.Event()

        async def cancel_once_underway() -> None:
            # Bounded so a verifier that never reports itself started fails this test instead of hanging it: the
            # hanging verifier's own sleep is an hour long, and nothing else here would ever end the wait.
            await asyncio.wait_for(verifier.started.wait(), _CANCEL_TEST_TIMEOUT_SECONDS)
            cancel_event.set()

        canceller = asyncio.create_task(cancel_once_underway())
        try:
            with self.assertRaises(ProcessingCancelled):
                await asyncio.wait_for(
                    processor.process("test_clip", "test_video.mp4", cancel_event), _CANCEL_TEST_TIMEOUT_SECONDS
                )
            await asyncio.wait_for(canceller, _CANCEL_TEST_TIMEOUT_SECONDS)
        finally:
            canceller.cancel()

        # Both questions had begun and neither was left running once the batch was abandoned.
        self.assertEqual(verifier.in_flight, 0)
        self.assertEqual(verifier.cancelled, 2)

    async def test_cancelling_while_the_questions_are_being_generated_abandons_the_llm_call(self) -> None:
        """The generation call is as long as a verification one, so waiting it out would cost the whole round trip."""
        generator = _HangingQuestionGenerator()
        processor = self._make_processor_with_dummy_vlm_verifier()
        processor.question_generator = cast(LLMQuestionGenerator, generator)
        cancel_event = threading.Event()

        async def cancel_once_underway() -> None:
            await asyncio.wait_for(generator.started.wait(), _CANCEL_TEST_TIMEOUT_SECONDS)
            cancel_event.set()

        canceller = asyncio.create_task(cancel_once_underway())
        try:
            with self.assertRaises(ProcessingCancelled):
                await asyncio.wait_for(
                    processor.process("test_clip", "test_video.mp4", cancel_event), _CANCEL_TEST_TIMEOUT_SECONDS
                )
            await asyncio.wait_for(canceller, _CANCEL_TEST_TIMEOUT_SECONDS)
        finally:
            canceller.cancel()

        self.assertEqual(generator.cancelled, 1)


class _DummyQuestionGenerator:
    """Returns a fixed list of questions (used by tests that only need verify_question to run)."""

    def __init__(self) -> None:
        self.calls = 0

    async def generate_questions(self, selected_variables: dict, variable_options: dict) -> list[GeneratedQuestion]:
        self.calls += 1
        return [
            GeneratedQuestion(
                variable="color",
                value="blue",
                question="What is the color of the car?",
                options={"A": "blue", "B": "red", "C": "green", "D": "yellow"},
                correct_answer="A",
            ),
            GeneratedQuestion(
                variable="type",
                value="sedan",
                question="What type of car is it?",
                options={"A": "truck", "B": "SUV", "C": "sedan", "D": "van"},
                correct_answer="C",
            ),
        ]


class _DummyVLMVerifier:
    """Returns (True, correct_answer) for every verify_question call."""

    async def verify_question(
        self,
        augmented_video_path: str,
        question: str,
        options: dict,
        correct_answer: str,
        request_reasoning: bool = False,
    ) -> tuple[bool, str, str | None]:
        return True, correct_answer, "reason" if request_reasoning else None


class _IncorrectVLMVerifier:
    """Returns the wrong answer for every verify_question call."""

    async def verify_question(
        self,
        augmented_video_path: str,
        question: str,
        options: dict,
        correct_answer: str,
        request_reasoning: bool = False,
    ) -> tuple[bool, str, str | None]:
        return False, "B", None


class _UnknownVLMVerifier:
    """Returns an unparseable verdict for every question."""

    async def verify_question(
        self,
        augmented_video_path: str,
        question: str,
        options: dict,
        correct_answer: str,
        request_reasoning: bool = False,
    ) -> tuple[bool, str, str | None]:
        return False, "UNKNOWN", None


class _HangingVLMVerifier:
    """Never answers, and records how each of its calls ended.

    Stands in for a VLM round trip that outlives the job: ``started`` fires once every question is genuinely in
    flight, which is the only moment at which cancelling the batch proves anything.
    """

    def __init__(self, expected_calls: int = 2) -> None:
        self.started = asyncio.Event()
        self.in_flight = 0
        self.cancelled = 0
        self._expected_calls = expected_calls

    async def verify_question(
        self,
        augmented_video_path: str,
        question: str,
        options: dict,
        correct_answer: str,
        request_reasoning: bool = False,
    ) -> tuple[bool, str, str | None]:
        self.in_flight += 1
        if self.in_flight == self._expected_calls:
            self.started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.in_flight -= 1
        raise AssertionError("unreachable: the verifier is meant to be cancelled")


class _HangingQuestionGenerator:
    """Never answers, and records whether its call was cancelled.

    Stands in for an LLM round trip that outlives the job: ``started`` fires once the call is genuinely in flight,
    which is the only moment at which cancelling it proves anything.
    """

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = 0

    async def generate_questions(self, selected_variables: dict, variable_options: dict) -> list[GeneratedQuestion]:
        self.started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        raise AssertionError("unreachable: the generator is meant to be cancelled")


class _RaisingQuestionGenerator:
    """Question generator that raises a given exception from generate_questions."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def generate_questions(self, selected_variables: dict, variable_options: dict) -> list[GeneratedQuestion]:
        raise self._exc


class _RaisingVLMVerifier:
    """VLM verifier that raises a given exception from verify_question."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def verify_question(
        self,
        augmented_video_path: str,
        question: str,
        options: dict,
        correct_answer: str,
        request_reasoning: bool = False,
    ) -> tuple[bool, str, str | None]:
        raise self._exc


class _PartiallyFailingVLMVerifier:
    """VLM verifier that raises ``exceptions_in_order[i]`` (or succeeds when ``None``) per call."""

    def __init__(self, exceptions_in_order: list[BaseException | None]) -> None:
        self._exceptions = exceptions_in_order
        self._idx = 0

    async def verify_question(
        self,
        augmented_video_path: str,
        question: str,
        options: dict,
        correct_answer: str,
        request_reasoning: bool = False,
    ) -> tuple[bool, str, str | None]:
        idx = self._idx
        self._idx += 1
        exc = self._exceptions[idx] if idx < len(self._exceptions) else None
        if exc is not None:
            raise exc
        return True, correct_answer, "reason" if request_reasoning else None


if __name__ == "__main__":
    unittest.main()
