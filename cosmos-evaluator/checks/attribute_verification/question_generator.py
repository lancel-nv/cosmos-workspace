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
import logging
import os
import re
import time
from typing import Any, ClassVar

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormatJSONSchema
from pydantic import BaseModel, Field

from checks.attribute_verification.common import validate_and_cast_config_params


def _format_variable_label(variable_name: str) -> str:
    """Convert a config variable name into a readable label for prompting."""
    return variable_name.replace("_", " ").strip()


def _canonicalize(text: str) -> str:
    """Normalize an option value for comparison."""
    return str(text).strip().lower()


_CONFUSING_COLOR_DISTRACTORS = {
    "beige": {"brown", "white", "yellow", "orange"},
    "black": {"brown", "grey", "blue"},
    "blue": {"black", "green", "purple"},
    "brown": {"beige", "green", "orange", "yellow", "camouflage"},
    "camouflage": {"brown", "green", "grey"},
    "green": {"blue", "brown", "camouflage", "yellow"},
    "grey": {"black", "white", "blue"},
    "orange": {"beige", "brown", "red", "yellow"},
    "pink": {"beige", "purple", "red"},
    "purple": {"blue", "pink", "red"},
    "red": {"brown", "orange", "pink", "purple"},
    "white": {"beige", "grey", "yellow"},
    "yellow": {"beige", "brown", "green", "orange", "white"},
}

_CONFUSING_VALUE_PAIRS = {
    "bottom_type": {"dress": {"skirt"}, "skirt": {"dress"}},
}


def _is_confusing_option(variable_name: str, selected_value: str, selected_allowed: str, candidate: str) -> bool:
    """Return whether a distractor is too close to the selected answer."""
    candidate_norm = _canonicalize(candidate)
    selected_norm = _canonicalize(selected_allowed)
    original_norm = _canonicalize(selected_value)

    if candidate_norm == selected_norm:
        return False
    if candidate_norm and original_norm and (candidate_norm in original_norm or original_norm in candidate_norm):
        return True
    if variable_name.endswith(("_color", " color")):
        return candidate_norm in _CONFUSING_COLOR_DISTRACTORS.get(selected_norm, set())
    return candidate_norm in _CONFUSING_VALUE_PAIRS.get(variable_name, {}).get(selected_norm, set())


def _resolve_api_key(configured_env: str | None, role_env: str) -> str:
    """Resolve endpoint credentials while preserving the legacy autograder variable."""
    for env_name in (configured_env, role_env, "BUILD_NVIDIA_API_KEY"):
        if env_name and (api_key := os.environ.get(env_name)):
            return api_key
    return "not-used"


class GeneratedQuestion(BaseModel):
    """One multiple-choice verification question, as it travels from the LLM that wrote it to the VLM that answers it.

    The payload the generator produces and the processor consumes, so neither side has to agree on the keys of a bare
    dict. Mirrors :attr:`LLMQuestionGenerator._QUESTION_SCHEMA`, and is built at the one point where model output stops
    being arbitrary JSON: the end of :meth:`LLMQuestionGenerator.generate_question`. A response that does not fit is a
    retryable failure there rather than a ``KeyError`` in the verifier.
    """

    variable: str = Field(..., description="Name of the variable being verified")
    value: str = Field(..., description="The selected value to verify")
    question: str = Field(..., description="The multiple choice question text")
    options: dict[str, str] = Field(..., description="Answer options keyed by letter (A-D)")
    correct_answer: str = Field(..., description="Letter of the option matching ``value``")
    request_reasoning: bool = Field(False, description="Whether the VLM should return a brief explanation")


class LLMQuestionGenerator:
    """Generate multiple choice verification questions using LLM."""

    # Define expected parameter types for validation
    _PARAM_TYPES: ClassVar[dict[str, type]] = {
        "system_prompt": str,
        "retry": int,
        "temperature": float,
        "top_p": float,
        "frequency_penalty": float,
        "presence_penalty": float,
        "max_tokens": int,
        "stream": bool,
        "endpoint": str,
        "model": str,
    }

    # JSON schema for structured output (guided JSON)
    _QUESTION_SCHEMA: ClassVar[ResponseFormatJSONSchema] = {
        "type": "json_schema",
        "json_schema": {
            "name": "verification_question",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string", "description": "Name of the variable being verified"},
                    "value": {"type": "string", "description": "The selected value to verify"},
                    "question": {"type": "string", "description": "The multiple choice question text"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "A": {"type": "string"},
                            "B": {"type": "string"},
                            "C": {"type": "string"},
                            "D": {"type": "string"},
                        },
                        "required": ["A", "B"],
                        "additionalProperties": False,
                    },
                    "correct_answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                },
                "required": ["variable", "value", "question", "options", "correct_answer"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(
        self,
        system_prompt: str,
        retry: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        max_tokens: int,
        stream: bool,
        endpoint: str,
        model: str,
        logger: logging.Logger,
        api_key_env: str | None = None,
        generate_options: bool = False,
    ):
        # Validate and cast parameters to ensure correct types
        params = {
            "system_prompt": system_prompt,
            "retry": retry,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": max_tokens,
            "stream": stream,
            "endpoint": endpoint,
            "model": model,
        }

        self.logger = logger
        self.logger.info("Initializing QuestionGenerator with params: %s", params)
        validated_params = validate_and_cast_config_params(params, self._PARAM_TYPES, logger)

        self.system_prompt = validated_params["system_prompt"]
        self.max_retries = validated_params["retry"]
        self.temperature = validated_params["temperature"]
        self.top_p = validated_params["top_p"]
        self.frequency_penalty = validated_params["frequency_penalty"]
        self.presence_penalty = validated_params["presence_penalty"]
        self.max_tokens = validated_params["max_tokens"]
        self.stream = validated_params["stream"]
        self.endpoint = validated_params["endpoint"]
        self.model = validated_params["model"]
        self.generate_options = bool(generate_options)

        api_key = _resolve_api_key(api_key_env, "LLM_API_KEY")
        if api_key == "not-used":
            logger.warning("No LLM API key is set, using 'not-used' as placeholder")

        self.client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
            timeout=7200,
        )

        self.logger = logger

    @classmethod
    def from_config(
        cls,
        config_params: dict,
        system_prompt: str,
        endpoint: str,
        model: str,
        logger: logging.Logger,
        api_key_env: str | None = None,
        generate_options: bool = False,
    ) -> "LLMQuestionGenerator":
        """
        Create LLMQuestionGenerator instance from configuration dictionary with type validation.

        Args:
            config_params: Dictionary containing LLM configuration parameters
            system_prompt: System prompt text
            endpoint: LLM endpoint URL
            model: LLM model name
            logger: Logger instance

        Returns:
            LLMQuestionGenerator: Configured instance with validated parameters
        """
        # Extract and validate parameters
        params = {
            "system_prompt": system_prompt,
            "retry": config_params.get("retry", 1),
            "temperature": config_params.get("temperature", 0.0),
            "top_p": config_params.get("top_p", 0.95),
            "frequency_penalty": config_params.get("frequency_penalty", 0.0),
            "presence_penalty": config_params.get("presence_penalty", 0.0),
            "max_tokens": config_params.get("max_tokens", 2048),
            "stream": config_params.get("stream", True),
            "endpoint": endpoint,
            "model": model,
        }

        validated_params = validate_and_cast_config_params(params, cls._PARAM_TYPES, logger)

        return cls(
            system_prompt=validated_params["system_prompt"],
            retry=validated_params["retry"],
            temperature=validated_params["temperature"],
            top_p=validated_params["top_p"],
            frequency_penalty=validated_params["frequency_penalty"],
            presence_penalty=validated_params["presence_penalty"],
            max_tokens=validated_params["max_tokens"],
            stream=validated_params["stream"],
            endpoint=validated_params["endpoint"],
            model=validated_params["model"],
            logger=logger,
            api_key_env=api_key_env,
            generate_options=generate_options,
        )

    def _sanitize_model_output(self, text: str) -> str:
        """Remove artifacts from model output."""
        # Remove Markdown code fences
        sanitized = re.sub(r"```[a-zA-Z]*", "", text)
        sanitized = re.sub(r"```", "", sanitized)

        # Remove block comments
        sanitized = re.sub(r"/\*.*?\*/", "", sanitized, flags=re.DOTALL)

        return sanitized

    def _parse_llm_response(self, result: str) -> dict[str, Any]:
        """
        Parse the LLM response to extract a single question JSON.

        Args:
            result (str): The raw LLM response

        Returns:
            Dict[str, Any]: Question dictionary

        Raises:
            ValueError: If the response cannot be parsed as JSON
        """
        # Step 1: cut away any leading chain-of-thought like </think> blocks
        think_close_idx = result.rfind("</think>")
        base_text = result[think_close_idx + len("</think>") :].strip() if think_close_idx != -1 else result.strip()
        self.logger.debug("Post-think text length=%s (think_found=%s)", len(base_text), think_close_idx != -1)

        # Step 2: prefer content inside the first fenced code block, if present.
        # The capture group is deliberately not bracketed by `\s*` quantifiers: pairing `\s*` with the adjacent lazy
        # `[\s\S]*?` creates overlapping whitespace matches that backtrack polynomially (ReDoS) on a long whitespace run
        # with no closing fence. `.strip()` below trims the result, so the surrounding `\s*` were redundant anyway.
        fence_match = re.search(
            r"```(?:json|javascript|js|python)?([\s\S]*?)```",
            base_text,
            flags=re.IGNORECASE,
        )
        candidate_text = fence_match.group(1).strip() if fence_match else base_text
        self.logger.debug(
            "Fenced block %s; candidate length=%s", "found" if fence_match else "not found", len(candidate_text)
        )

        # Step 3: sanitize comments and stray fences
        sanitized = self._sanitize_model_output(candidate_text)
        self.logger.debug("Sanitized candidate length=%s", len(sanitized))

        # Step 4: attempt to decode JSON
        try:
            decoder = json.JSONDecoder()
            # Scan for likely JSON starts
            for match in re.finditer(r"[\[{]", sanitized):
                start_idx = match.start()
                try:
                    obj, end_idx = decoder.raw_decode(sanitized, idx=start_idx)
                    self.logger.debug("Decoded JSON segment start=%s end=%s", start_idx, end_idx)
                    return self._normalize_question_output(obj)
                except json.JSONDecodeError:
                    continue
            raise json.JSONDecodeError("No decodable JSON value found", sanitized, 0)

        except json.JSONDecodeError as e:
            self.logger.error("JSON decode failed: msg=%s, pos=%s", getattr(e, "msg", ""), getattr(e, "pos", None))
            error_msg = "Could not parse JSON from model output. Enable debug logs to inspect sanitized text."
            raise ValueError(error_msg) from e

    def _normalize_question_output(self, obj: Any) -> dict[str, Any]:
        """
        Normalize the LLM output to expected format for a single question.

        Expected format:
        {
            "variable": "weather_condition",
            "value": "cloudy",
            "question": "What are the weather conditions?",
            "options": {"A": "clear", "B": "cloudy", "C": "rainy", "D": "snowy"},
            "correct_answer": "B"
        }
        """
        required_fields = ["variable", "value", "question", "options", "correct_answer"]

        # If it's already a dict with required fields, validate and return
        if isinstance(obj, dict):
            # Check if it has all required fields directly
            if all(field in obj for field in required_fields):
                return obj

            # Check if it's a wrapper dict
            for key in ("question", "item", "data", "result", "output"):
                inner = obj.get(key)
                if isinstance(inner, dict) and all(field in inner for field in required_fields):
                    return inner

        # If it's a list with one item, extract that item
        if (
            isinstance(obj, list)
            and len(obj) == 1
            and isinstance(obj[0], dict)
            and all(field in obj[0] for field in required_fields)
        ):
            return obj[0]

        raise ValueError(f"Output must be a dict with required fields: {required_fields}. Got: {type(obj).__name__}")

    def _create_user_prompt(self, variable_name: str, selected_value: str, all_options: list[str]) -> str:
        """
        Create the user prompt for generating a single question.

        Args:
            variable_name: Name of the variable (e.g., "weather_condition")
            selected_value: Selected value for this variable (e.g., "cloudy")
            all_options: All possible values for this variable

        Returns:
            str: The formatted user prompt
        """
        variable_label = _format_variable_label(variable_name)
        if not all_options:
            return f"""Generate a verification question for the following attribute:

Variable: {variable_name}
Readable label: {variable_label}
Correct value: {selected_value}

Create ONE multiple choice question that verifies whether '{selected_value}' is visible in the video frames.

Requirements:
1. The question MUST explicitly name the attribute (use the readable label). No vague wording like "What is shown?".
2. Provide 3-4 total options. Exactly ONE option must be the correct value '{selected_value}'.
3. Invent the OTHER options yourself: plausible but clearly WRONG, visually distinct alternatives for this attribute. Do NOT use near-synonyms, paraphrases, parent/child labels, or overlapping categories of the correct value.
4. Keep options short and mutually exclusive.
5. Format the correct answer as a single letter (A, B, C, or D).

Output MUST be a single JSON object with the following structure:
{{
    "variable": "{variable_name}",
    "value": "{selected_value}",
    "question": "Question text?",
    "options": {{"A": "option1", "B": "option2", "C": "option3", "D": "option4"}},
    "correct_answer": "B"
}}

Do NOT include any explanatory text, markdown fences, or comments. Output ONLY the JSON object."""  # noqa: E501

        return f"""Generate a verification question for the following variable:

Variable: {variable_name}
Readable label: {variable_label}
Selected value: {selected_value}
All possible values: {all_options}

Create ONE multiple choice question that verifies whether the selected value is present in the image.

Requirements:
1. The question must have 2-4 answer options (A, B, C, D)
2. The selected value '{selected_value}' MUST be one of the options
3. The question should be simple and direct
4. Options should include other possible values from the list above
5. Format the correct answer as a single letter (A, B, C, or D)
6. CRITICAL: The question text MUST explicitly name the variable or attribute being asked about. Use the readable label above when helpful. Do NOT use vague wording like "What is shown?" or "Which option is correct?"
7. Keep the wording domain-agnostic. Do not assume the variable refers to clothing, color, weather, or any other specific use case unless that meaning is already explicit in the variable name or options.
8. Distractor options must be visually and semantically distinct from the selected value. Do NOT include near-synonyms, parent/child labels, or overlapping color families as distractors. For example, if the selected value is "olive brown", do not offer both "olive" and "brown"; choose one correct option and use clearly different colors as distractors. For garment types, avoid ambiguous pairs such as "dress" and "skirt" in the same question when one is the correct answer.

Output MUST be a single JSON object with the following structure:
{{
    "variable": "{variable_name}",
    "value": "{selected_value}",
    "question": "Question text?",
    "options": {{"A": "option1", "B": "option2", "C": "option3", "D": "option4"}},
    "correct_answer": "B"
}}

Do NOT include any explanatory text, markdown fences, or comments. Output ONLY the JSON object."""  # noqa: E501

    @staticmethod
    def _resolve_selected_to_allowed(selected_value: str, all_options: list[str]) -> str:
        """Resolve a fine-grained selected value to its configured base option."""
        if not all_options:
            return selected_value
        selected_norm = _canonicalize(selected_value)
        allowed = {_canonicalize(option): str(option) for option in all_options}
        if selected_norm in allowed:
            return allowed[selected_norm]
        for normalized, raw in allowed.items():
            if normalized and (normalized in selected_norm or selected_norm in normalized):
                return raw
        return selected_value

    def _repair_question_options(
        self,
        question: dict[str, Any],
        variable_name: str,
        selected_value: str,
        all_options: list[str],
    ) -> dict[str, Any]:
        """Constrain, deduplicate, and relabel a model-generated option set."""
        repaired = dict(question)
        raw_options = repaired.get("options")
        options = raw_options if isinstance(raw_options, dict) else {}
        allowed_values = [str(value) for value in all_options]
        allowed_norm = {_canonicalize(value) for value in allowed_values}
        selected_allowed = self._resolve_selected_to_allowed(selected_value, allowed_values)
        selected_norm = _canonicalize(selected_allowed)

        kept_values: list[str] = []
        for _, value in sorted(options.items()):
            candidate = str(value).strip()
            kept_norm = {_canonicalize(item) for item in kept_values}
            if not candidate or _canonicalize(candidate) in kept_norm:
                continue
            if allowed_norm and _canonicalize(candidate) not in allowed_norm:
                continue
            if _is_confusing_option(variable_name, selected_value, selected_allowed, candidate):
                continue
            kept_values.append(candidate)

        if selected_norm not in {_canonicalize(item) for item in kept_values}:
            kept_values.insert(0, selected_allowed)

        for candidate in allowed_values:
            if _canonicalize(candidate) in {_canonicalize(item) for item in kept_values}:
                continue
            if _is_confusing_option(variable_name, selected_value, selected_allowed, candidate):
                continue
            kept_values.append(candidate)
            if len(kept_values) >= 4:
                break

        if len(kept_values) < 2:
            kept_values.append("other")

        final_options = dict(zip(("A", "B", "C", "D"), kept_values[:4], strict=False))
        correct_answer = next(
            (letter for letter, value in final_options.items() if _canonicalize(value) == selected_norm),
            "A",
        )
        repaired["value"] = selected_allowed
        repaired["options"] = final_options
        repaired["correct_answer"] = correct_answer
        return repaired

    async def generate_question(
        self, variable_name: str, selected_value: str, all_options: list[str]
    ) -> GeneratedQuestion:
        """
        Generate a verification question for a single variable.

        Args:
            variable_name: Name of the variable (e.g., "weather_condition")
            selected_value: Selected value for this variable (e.g., "cloudy")
            all_options: All possible values for this variable

        Returns:
            GeneratedQuestion: The validated question payload

        Raises:
            RuntimeError: If question generation fails
        """
        self.logger.debug("Generating question for variable '%s' with value '%s'", variable_name, selected_value)
        retries_remaining = self.max_retries
        option_pool = [] if self.generate_options else all_options
        user_prompt = self._create_user_prompt(variable_name, selected_value, option_pool)

        while retries_remaining >= 0:
            try:
                self.logger.debug("Making API call to generate question with structured output")
                # Try structured output first (guided JSON), fall back to regular parsing
                try:
                    messages: list[ChatCompletionMessageParam] = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    self.logger.info("LLM endpoint call starting (model=%s, variable='%s')", self.model, variable_name)
                    _llm_t0 = time.perf_counter()
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        frequency_penalty=self.frequency_penalty,
                        max_tokens=self.max_tokens,
                        presence_penalty=self.presence_penalty,
                        response_format=self._QUESTION_SCHEMA,
                        stream=False,  # Structured output doesn't support streaming
                    )
                    self.logger.info(
                        "LLM endpoint call completed in %.0fms (variable='%s')",
                        (time.perf_counter() - _llm_t0) * 1000,
                        variable_name,
                    )
                    result = completion.choices[0].message.content
                    if result is None:
                        # Raise a retryable error so the surrounding except falls back to manual parsing
                        # instead of failing the whole call immediately.
                        raise ValueError("No content returned from LLM")
                    question = json.loads(result)
                    self.logger.debug("Successfully used structured output (guided JSON)")
                except (openai.APIError, openai.BadRequestError, json.JSONDecodeError, ValueError) as e:
                    # Fall back to manual parsing if structured output not supported
                    self.logger.debug("Structured output not available (%s), falling back to manual parsing", e)
                    self.logger.info(
                        "LLM endpoint call starting (model=%s, variable='%s', fallback=True)", self.model, variable_name
                    )
                    _llm_t0 = time.perf_counter()
                    completion = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                        stream=self.stream,
                    )
                    self.logger.info(
                        "LLM endpoint call completed in %.0fms (variable='%s', fallback=True)",
                        (time.perf_counter() - _llm_t0) * 1000,
                        variable_name,
                    )

                    result = ""
                    if self.stream:
                        # Streaming: iterate over chunks and accumulate content
                        async for chunk in completion:
                            if chunk.choices[0].delta.content is not None:
                                result += chunk.choices[0].delta.content
                    else:
                        # Non-streaming: completion is a single response object
                        result = completion.choices[0].message.content
                        if result is None:
                            raise ValueError("No content returned from LLM response")

                    # Parse the result
                    question = self._parse_llm_response(result)

                question = self._repair_question_options(question, variable_name, selected_value, option_pool)
                self.logger.info("Successfully generated question for '%s'", variable_name)
                self.logger.debug("Generated question: %s", question)
                # Validated inside the retry block on purpose: a response missing a field or carrying the wrong type
                # is worth another attempt, the same as one that would not parse at all.
                return GeneratedQuestion.model_validate(question)
            except Exception as e:
                if isinstance(
                    e,
                    (
                        openai.NotFoundError,
                        openai.AuthenticationError,
                        openai.PermissionDeniedError,
                    ),
                ) or (isinstance(e, openai.APIConnectionError) and not isinstance(e, openai.APITimeoutError)):
                    raise
                self.logger.exception("Failed to generate question for '%s' with LLM", variable_name)

                if retries_remaining > 0:
                    retries_remaining -= 1
                    self.logger.info(
                        "Retrying failed call to LLM. Retries remaining: %s/%s", retries_remaining, self.max_retries
                    )
                    # Continue the while loop to retry
                else:
                    self.logger.exception(
                        "Max retries reached. LLM failed to generate question for variable_name='%s' "
                        "after %s total attempts",
                        variable_name,
                        self.max_retries + 1,
                    )
                    raise RuntimeError(
                        f"Max retries reached. LLM failed to generate question for variable_name="
                        f"'{variable_name}' after {self.max_retries + 1} total attempts"
                    ) from e
        # Unreachable: every path through the loop above either returns a question or raises.
        raise RuntimeError(f"LLM question generation for variable_name='{variable_name}' ended without a result")

    async def generate_questions(
        self, variables: dict[str, str], variable_options: dict[str, list[str]]
    ) -> list[GeneratedQuestion]:
        """
        Generate verification questions for all given variables (one at a time).

        Args:
            variables: Dictionary of variable names to selected values
            variable_options: Dictionary of variable names to all possible values

        Returns:
            list[GeneratedQuestion]: One validated question payload per variable

        Raises:
            RuntimeError: If question generation fails for any variable
        """
        self.logger.info("Generating questions for %s variables", len(variables))
        questions = []
        failures: list[str] = []

        for variable_name, selected_value in variables.items():
            all_options = variable_options.get(variable_name, [selected_value])

            try:
                question = await self.generate_question(variable_name, selected_value, all_options)
                questions.append(question)
            except (
                openai.APIConnectionError,
                openai.AuthenticationError,
                openai.NotFoundError,
                openai.PermissionDeniedError,
            ):
                # Preserve the service's established endpoint-specific HTTP error mapping.
                raise
            except Exception as e:  # noqa: BLE001 - collect every failed variable before rejecting the set
                self.logger.error("Failed to generate question for '%s': %s", variable_name, e)
                failures.append(f"{variable_name}: {e}")

        if failures:
            raise RuntimeError(
                f"Question generation failed for {len(failures)} of {len(variables)} variable(s), "
                f"so the sample cannot be fully verified: {'; '.join(failures)}"
            )

        if not questions:
            raise RuntimeError("Failed to generate any questions")

        self.logger.info("Successfully generated %s questions", len(questions))
        return questions
