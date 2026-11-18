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

"""
Module for the attribute verification checker.

This module handles the verification of attributes in augmented image or video media using a VLM.
"""

import asyncio
import logging
from pathlib import Path
import threading
import time
from typing import Any

from fastapi import HTTPException
import openai
from pydantic import BaseModel, Field

from checks.attribute_verification.question_generator import GeneratedQuestion, LLMQuestionGenerator
from checks.attribute_verification.vlm_verifier import VLMVerifier
from checks.utils.config_manager import ConfigManager
from utils import bazel
from utils.cancellation import ProcessingCancelled, check_cancelled

logger = logging.getLogger(__name__)

# How often the VLM fan-out looks at its cancellation event. Short relative to a VLM round trip, so a cancelled job
# stops within a fraction of one question rather than waiting out the whole batch.
_CANCEL_POLL_SECONDS = 0.5


async def _wait_until_set(cancel_event: threading.Event, poll_seconds: float = _CANCEL_POLL_SECONDS) -> None:
    """Resolve once ``cancel_event`` is set.

    Polled rather than awaited because the cancellation contract is a :class:`threading.Event`, shared with the
    checkers whose work runs in an executor thread and so cannot be handed an asyncio primitive.
    """
    while not cancel_event.is_set():
        await asyncio.sleep(poll_seconds)


async def _cancellation_won(work: asyncio.Future[Any], cancel_event: threading.Event) -> bool:
    """Wait for ``work`` or for ``cancel_event``, whichever lands first, and report whether cancellation got there.

    Every phase of this check is a single long ``await`` on somebody else's endpoint, with no loop in which to poll a
    flag, so cancellation has to be raced against the work rather than checked between steps. The caller decides how to
    tear ``work`` down, because abandoning one request and abandoning a whole fan-out are not the same operation.
    """
    watcher = asyncio.create_task(_wait_until_set(cancel_event))
    # Annotated because the two have no common type more specific than ``object``, which ``wait`` rejects.
    racers: set[asyncio.Future[Any]] = {work, watcher}
    try:
        await asyncio.wait(racers, return_when=asyncio.FIRST_COMPLETED)
    finally:
        watcher.cancel()
    return not work.done()


class AttributeVerificationCheck(BaseModel):
    variable: str = Field(..., description="Name of the variable being verified")
    value: str = Field(..., description="The expected value of the variable")
    question: str = Field(..., description="The question for the VLM to verify")
    options: dict[str, str] = Field(..., description="The multiple choice answers for the question")
    expected_answer: str = Field(..., description="The correct multiple choice answer to the question")
    request_reasoning: bool = Field(False, description="Whether the VLM was asked to explain its answer")
    vlm_answer: str = Field(..., description="The multiple choice answer from the VLM to the question")
    vlm_reasoning: str | None = Field(None, description="The optional explanation returned by the VLM")
    passed: bool = Field(..., description="Whether the VLM answer matches the expected answer")
    error: str | None = Field(None, description="The error message if the check failed to execute")


class AttributeVerificationSummary(BaseModel):
    total_checks: int = Field(..., description="The total number of checks performed")
    passed_checks: int = Field(..., description="The number of checks that passed")
    failed_checks: int = Field(..., description="The number of checks that failed")
    errored_checks: int = Field(0, description="The number of checks for which the VLM returned no verdict")
    reason: str | None = Field(None, description="Why no checks could be performed")


class AttributeVerificationResult(BaseModel):
    clip_id: str
    passed: bool
    summary: AttributeVerificationSummary
    checks: list[AttributeVerificationCheck]
    duration_seconds: float = Field(0.0, description="Elapsed checker runtime in seconds")

    @property
    def output_dir(self) -> str | None:
        """Always ``None``: this check reports per-question verdicts and writes no artifacts to upload.

        A property rather than a field so the answer stays out of the serialized result, which is recorded on the job
        row and read by callers who have no interest in the transport's cleanup contract.
        """
        return None


class AttributeVerificationProcessor:
    """
    Processor for the attribute verification checker.
    """

    def __init__(self, params: dict, config_dir: str | None = None, verbose: str = "INFO"):
        """
        Initialize the processor.
        """
        self.logger = self._setup_logging(verbose)
        self.config = AttributeVerificationProcessor.get_default_config(config_dir=config_dir).get(
            "metropolis.attribute_verification", {}
        )
        if params is not None:
            self.config = self._deep_merge(self.config, params)
            # These mappings are complete data sets, not partial model configuration. An explicitly empty mapping
            # must therefore clear the bundled examples rather than recursively retaining them.
            for replace_key in ("selected_variables", "variable_options"):
                if replace_key in params:
                    self.config[replace_key] = params[replace_key]
        self.question_generation_config = self.config.get("question_generation", {})
        self.vlm_verification_config = self.config.get("vlm_verification", {})
        self.question_generator = LLMQuestionGenerator.from_config(
            self.question_generation_config.get("parameters", {}),
            self.question_generation_config.get("system_prompt", ""),
            self.question_generation_config.get("llm", {}).get("endpoint", ""),
            self.question_generation_config.get("llm", {}).get("model", ""),
            self.logger,
            api_key_env=self.question_generation_config.get("llm", {}).get("api_key_env"),
            generate_options=bool(self.question_generation_config.get("generate_options", False)),
        )
        self.vlm_verifier = VLMVerifier.from_config(
            self.vlm_verification_config.get("parameters", {}),
            self.vlm_verification_config.get("system_prompt", ""),
            self.vlm_verification_config.get("vlm", {}).get("endpoint", ""),
            self.vlm_verification_config.get("vlm", {}).get("model", ""),
            self.logger,
            api_key_env=self.vlm_verification_config.get("vlm", {}).get("api_key_env"),
            frames=int(self.vlm_verification_config.get("frames", 1) or 1),
        )

    def _setup_logging(self, verbose: str) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)

        # Set logging level based on verbose mode
        log_level = getattr(logging, verbose.upper())
        logger.setLevel(log_level)

        # Ensure propagation is enabled
        logger.propagate = True

        # Clear any existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        logger.info("Logging level set to: %s", verbose)
        return logger

    @staticmethod
    def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        """
        Deep merge two dictionaries, recursively merging nested dictionaries.

        Args:
            base: Base dictionary to merge into
            update: Dictionary with updates to apply

        Returns:
            New dictionary with merged values
        """
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = AttributeVerificationProcessor._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def get_default_config(config_dir: str | None = None) -> dict[str, Any]:
        """
        Get the default configuration for attribute verification processing.

        Args:
            config_dir: Path to the configuration directory. If None, the directory holding the bundled
                       cosmos_evaluator.yaml is resolved from the cosmos_eval module's runfiles.

        Returns:
            Default configuration for attribute verification processing loaded from cosmos_evaluator.yaml
        """
        try:
            if config_dir is None:
                config_path = bazel.get_runfiles_path(
                    "checks/cosmos_evaluator.yaml", external_repo="cosmos_eval", source_repo=""
                )
                if not config_path:
                    raise FileNotFoundError("Configuration file not found: checks/cosmos_evaluator.yaml")
                config_dir = str(Path(config_path).parent)
            config_manager = ConfigManager(config_dir)
            config = config_manager.load_config("cosmos_evaluator")
            return {"metropolis.attribute_verification": config["metropolis.attribute_verification"]}
        except Exception:
            logger.exception("Error loading default configuration")
            raise

    async def _verify_one(
        self, question_data: GeneratedQuestion, augmented_video_path: str
    ) -> AttributeVerificationCheck:
        """Run one VLM verification, recording endpoint/media failures as errored checks."""
        variable = question_data.variable
        value = question_data.value
        question = question_data.question
        options = question_data.options
        correct_answer = question_data.correct_answer
        request_reasoning = question_data.request_reasoning

        self.logger.info("Verifying variable '%s' with value '%s'", variable, value)
        self.logger.debug("Question: %s", question)

        try:
            is_correct, vlm_answer, vlm_reasoning = await self.vlm_verifier.verify_question(
                augmented_video_path,
                question,
                options,
                correct_answer,
                request_reasoning=request_reasoning,
            )
            if is_correct:
                self.logger.info(
                    "✓ Verification passed for '%s': expected %s, got %s", variable, correct_answer, vlm_answer
                )
            else:
                formatted_options = ", ".join(f"{letter}={text!r}" for letter, text in options.items())
                self.logger.warning(
                    "✗ Verification failed for '%s': expected %s, got %s; options: %s",
                    variable,
                    correct_answer,
                    vlm_answer,
                    formatted_options,
                )
            return AttributeVerificationCheck(
                variable=variable,
                value=value,
                question=question,
                options=options,
                expected_answer=correct_answer,
                request_reasoning=request_reasoning,
                vlm_answer=vlm_answer,
                vlm_reasoning=vlm_reasoning,
                passed=is_correct,
            )
        except Exception as e:
            # Record the per-question failure on the check and let the gather complete the rest.
            # Raising would cancel sibling _verify_one tasks via asyncio.gather, dropping results
            # for every other variable in the batch. Covers the openai.* error classes
            # (APIConnectionError / NotFoundError / AuthenticationError / PermissionDeniedError)
            # plus any other transient VLM failure.
            self.logger.exception("VLM verification failed for variable '%s'", variable)
            return AttributeVerificationCheck(
                variable=variable,
                value=value,
                question=question,
                options=options,
                expected_answer=correct_answer,
                request_reasoning=request_reasoning,
                vlm_answer="ERROR",
                passed=False,
                error=str(e) or e.__class__.__name__,
            )

    async def _gather_checks(
        self, tasks: list[asyncio.Task[AttributeVerificationCheck]], cancel_event: threading.Event | None
    ) -> list[AttributeVerificationCheck]:
        """Await the whole verification batch, abandoning it if cancellation arrives first.

        Unlike the frame-by-frame checkers there is no loop to poll: once the fan-out starts, every question is already
        in flight. So cancellation is watched alongside the batch and tears it down, which matters more here than in a
        frame loop because each outstanding task is a VLM call that keeps costing after the job has been abandoned.
        """
        batch = asyncio.gather(*tasks)
        if cancel_event is None:
            return list(await batch)

        if await _cancellation_won(batch, cancel_event):
            batch.cancel()
            # Await the children rather than the gather: awaiting a cancelled gather would raise CancelledError
            # here and be mistaken for this coroutine itself being cancelled.
            await asyncio.gather(*tasks, return_exceptions=True)
            self.logger.info("Attribute verification cancelled with %d question(s) outstanding", len(tasks))
            raise ProcessingCancelled()
        return list(await batch)

    async def _generate_questions(
        self,
        selected_variables: dict[str, Any],
        variable_options: dict[str, Any],
        cancel_event: threading.Event | None,
    ) -> list[GeneratedQuestion]:
        """Ask the LLM for the verification questions, abandoning the call if cancellation arrives first.

        Watched the same way the VLM fan-out is, and for the same reason: this is one round trip to a model endpoint,
        so a job cancelled here would otherwise sit out the whole call before noticing -- and pay for it.
        """
        task = asyncio.create_task(self.question_generator.generate_questions(selected_variables, variable_options))
        if cancel_event is None:
            return await task

        if await _cancellation_won(task, cancel_event):
            task.cancel()
            # Suppressed rather than awaited directly: the CancelledError this raises is the task's, not this
            # coroutine's, and letting it out would look like the caller cancelling us.
            await asyncio.gather(task, return_exceptions=True)
            self.logger.info("Attribute verification cancelled while its questions were still being generated")
            raise ProcessingCancelled()
        return await task

    def _fixed_questions(self) -> list[GeneratedQuestion]:
        """Validate configured fixed questions, skipping malformed entries with a warning."""
        questions: list[GeneratedQuestion] = []
        for raw_question in self.config.get("extra_questions", []) or []:
            if not isinstance(raw_question, dict):
                self.logger.warning("Skipping invalid extra question because it is not an object: %r", raw_question)
                continue
            question = raw_question.get("question")
            options = raw_question.get("options")
            answer_text = str(raw_question.get("correct_answer", "")).strip().upper()
            correct_answer = answer_text[:1]
            if not isinstance(question, str) or not question:
                self.logger.warning("Skipping extra question with an invalid question: %r", raw_question)
                continue
            if not isinstance(options, dict) or not 2 <= len(options) <= 4:
                self.logger.warning("Skipping extra question with invalid options: %r", raw_question)
                continue
            normalized_options = {str(key).strip().upper(): str(value) for key, value in options.items()}
            if len(normalized_options) != len(options):
                self.logger.warning("Skipping extra question with invalid options: %r", raw_question)
                continue
            if not correct_answer or correct_answer not in normalized_options:
                self.logger.warning("Skipping extra question whose correct answer is not an option: %r", raw_question)
                continue
            if not normalized_options.keys() <= {"A", "B", "C", "D"}:
                original_keys = list(normalized_options)
                correct_answer = "ABCD"[original_keys.index(correct_answer)]
                normalized_options = dict(zip("ABCD", normalized_options.values(), strict=False))
            questions.append(
                GeneratedQuestion(
                    variable=str(raw_question.get("variable") or "extra"),
                    value=str(raw_question.get("value", "N/A")),
                    question=question,
                    options=normalized_options,
                    correct_answer=correct_answer,
                    request_reasoning=bool(raw_question.get("request_reasoning", False)),
                )
            )
        return questions

    async def process(
        self, clip_id: str, augmented_video_path: str, cancel_event: threading.Event | None = None
    ) -> AttributeVerificationResult:
        """
        Verify that the media contains the selected variable values.

        Args:
            augmented_video_path: Path to the generated image or video (legacy parameter name)
            cancel_event: Optional cooperative-cancellation event checked between stages and before each VLM call.
                ``None`` disables cancellation (local CLI).

        Returns:
            AttributeVerificationResult with fields:
                - clip_id (str): The clip identifier passed in.
                - passed (bool): True iff every entry in ``checks`` passed.
                - summary (AttributeVerificationSummary): Aggregate counters — ``total_checks``,
                  ``passed_checks``, ``failed_checks``, ``errored_checks``, and optional ``reason``.
                - checks (list[AttributeVerificationCheck]): Per-question verification record
                  (``variable``, ``value``, ``question``, ``options``, ``expected_answer``,
                  ``request_reasoning``, ``vlm_answer``, ``vlm_reasoning``, ``passed``, ``error``).
        """

        start_time = time.perf_counter()
        excluded_variables = set(self.config.get("exclude_variables", []) or [])
        selected_variables = {
            key: value
            for key, value in self.config.get("selected_variables", {}).items()
            if key not in excluded_variables
        }
        variable_options = {
            key: value
            for key, value in self.config.get("variable_options", {}).items()
            if key not in excluded_variables
        }

        self.logger.info("Starting attribute verification for video: %s", augmented_video_path)
        self.logger.info("Selected variables: %s", selected_variables)

        summary: AttributeVerificationSummary = AttributeVerificationSummary(
            total_checks=0,
            passed_checks=0,
            failed_checks=0,
            errored_checks=0,
        )

        # Step 1: Generate verification questions using LLM
        check_cancelled(cancel_event)
        questions: list[GeneratedQuestion] = []
        try:
            if selected_variables:
                self.logger.info("Generating verification questions...")
                questions = await self._generate_questions(selected_variables, variable_options, cancel_event)
            else:
                self.logger.info("No selected variables remain; skipping LLM question generation")
        except openai.APIConnectionError as e:
            message = "The provided LLM endpoint is not reachable. Please check the LLM endpoint."
            self.logger.exception(message)
            raise HTTPException(status_code=400, detail=message) from e
        except openai.NotFoundError as e:
            message = "The provided LLM endpoint was not found. Please check the LLM endpoint and model."
            self.logger.exception(message)
            raise HTTPException(status_code=400, detail=message) from e
        except openai.AuthenticationError as e:
            message = (
                "The API key is invalid or the provided LLM endpoint is not authorized. "
                "Please check the LLM endpoint and API key."
            )
            self.logger.exception(message)
            raise HTTPException(status_code=401, detail=message) from e
        except openai.PermissionDeniedError as e:
            message = (
                "The API key does not have permission to access the provided LLM endpoint. "
                "Please check the LLM endpoint and API key."
            )
            self.logger.exception(message)
            raise HTTPException(status_code=403, detail=message) from e
        questions.extend(self._fixed_questions())
        summary.total_checks = len(questions)

        self.logger.info("Generated %s verification questions", len(questions))
        if not questions:
            summary.reason = "no_questions"
            self.logger.warning("Attribute verification produced no valid questions; marking the result as failed")
            return AttributeVerificationResult(
                clip_id=clip_id,
                passed=False,
                summary=summary,
                checks=[],
                duration_seconds=time.perf_counter() - start_time,
            )

        # Step 2: Verify all questions using VLM in parallel (each call is independent I/O)
        check_cancelled(cancel_event)
        tasks = [asyncio.create_task(self._verify_one(question, augmented_video_path)) for question in questions]
        checks = await self._gather_checks(tasks, cancel_event)

        all_passed = True
        for check in checks:
            if check.error is not None or check.vlm_answer == "UNKNOWN":
                summary.errored_checks += 1
                all_passed = False
            elif check.passed:
                summary.passed_checks += 1
            else:
                summary.failed_checks += 1
                all_passed = False

        # Log summary
        self.logger.info("=" * 80)
        self.logger.info("Verification Summary:")
        self.logger.info("  Total checks: %s", summary.total_checks)
        self.logger.info("  Passed: %s", summary.passed_checks)
        self.logger.info("  Failed: %s", summary.failed_checks)
        self.logger.info("  Errored: %s", summary.errored_checks)
        self.logger.info("  Overall result: %s", "PASSED" if all_passed else "FAILED")
        self.logger.info("=" * 80)

        return AttributeVerificationResult(
            clip_id=clip_id,
            passed=all_passed,
            summary=summary,
            checks=checks,
            duration_seconds=time.perf_counter() - start_time,
        )
