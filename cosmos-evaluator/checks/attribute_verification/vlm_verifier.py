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

import asyncio
import base64
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import ClassVar

import multistorageclient as msc
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionContentPartParam, ChatCompletionMessageParam

from checks.attribute_verification.common import validate_and_cast_config_params

# Hard cap on the ffmpeg first-frame extraction so a stuck or corrupt input cannot hang the verifier (and, by extension,
# the inflight request) indefinitely.
_FFMPEG_TIMEOUT_SECONDS = 30
_IMAGE_MIME_TYPES = {
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".webp": "image/webp",
}
_IMAGE_EXTENSIONS = set(_IMAGE_MIME_TYPES)


def _strip_reasoning(content: str) -> str:
    """Remove complete or truncated ``<think>`` blocks from a model response."""
    if not content:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", " ", content, flags=re.DOTALL | re.IGNORECASE)
    lowered = cleaned.lower()
    if "</think>" in lowered:
        cleaned = cleaned[lowered.rfind("</think>") + len("</think>") :]
    elif "<think>" in lowered:
        cleaned = cleaned[: lowered.rfind("<think>")]
    return cleaned.strip()


def _extract_mcq_letter(content: str) -> str:
    """Return the final standalone A-D verdict outside model reasoning."""
    matches = re.findall(r"\b([A-D])\b", _strip_reasoning(content).upper())
    return matches[-1] if matches else "UNKNOWN"


def _resolve_api_key(configured_env: str | None) -> str:
    for env_name in (configured_env, "VLM_API_KEY", "BUILD_NVIDIA_API_KEY"):
        if env_name and (api_key := os.environ.get(env_name)):
            return api_key
    return "not-used"


class VLMVerifier:
    """Verify attributes using a VLM on images or sampled video frames."""

    # Define expected parameter types for validation
    _PARAM_TYPES: ClassVar[dict[str, type]] = {
        "retry": int,
        "temperature": float,
        "top_p": float,
        "frequency_penalty": float,
        "max_tokens": int,
        "stream": bool,
        "system_prompt": str,
        "endpoint": str,
        "model": str,
    }

    def __init__(
        self,
        system_prompt: str,
        retry: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        max_tokens: int,
        stream: bool,
        endpoint: str,
        model: str,
        logger: logging.Logger,
        api_key_env: str | None = None,
        frames: int = 1,
    ):
        # Validate and cast parameters to ensure correct types
        params = {
            "system_prompt": system_prompt,
            "retry": retry,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "stream": stream,
            "endpoint": endpoint,
            "model": model,
        }

        self.logger = logger
        self.logger.info("Initializing VLMVerifier with params: %s", params)
        validated_params = validate_and_cast_config_params(params, self._PARAM_TYPES, logger)

        self.system_prompt = validated_params["system_prompt"]
        self.max_retries = validated_params["retry"]
        self.temperature = validated_params["temperature"]
        self.top_p = validated_params["top_p"]
        self.frequency_penalty = validated_params["frequency_penalty"]
        self.max_tokens = validated_params["max_tokens"]
        self.stream = validated_params["stream"]

        # Ensure endpoint URL is properly formatted
        self.endpoint = validated_params["endpoint"].rstrip("/")
        self.model = validated_params["model"]
        self.frames = max(1, int(frames))

        api_key = _resolve_api_key(api_key_env)
        if api_key == "not-used":
            self.logger.warning("No VLM API key is set, using 'not-used' as placeholder")
        self.client = AsyncOpenAI(base_url=self.endpoint, api_key=api_key, timeout=7200)

    @classmethod
    def from_config(
        cls,
        config_params: dict,
        system_prompt: str,
        endpoint: str,
        model: str,
        logger: logging.Logger,
        api_key_env: str | None = None,
        frames: int = 1,
    ) -> "VLMVerifier":
        """
        Create VLMVerifier instance from configuration dictionary with type validation.

        Args:
            config_params: Dictionary containing VLM configuration parameters
            system_prompt: System prompt text
            endpoint: VLM endpoint URL
            model: VLM model name
            logger: Logger instance

        Returns:
            VLMVerifier: Configured instance with validated parameters
        """
        # Extract and validate parameters
        params = {
            "system_prompt": system_prompt,
            "retry": config_params.get("retry", 0),
            "temperature": config_params.get("temperature", 0.0),
            "top_p": config_params.get("top_p", 1.0),
            "frequency_penalty": config_params.get("frequency_penalty", 0.0),
            "max_tokens": config_params.get("max_tokens", 512),
            "stream": config_params.get("stream", False),
            "endpoint": endpoint,
            "model": model,
        }
        logger.info("Config params: %s", config_params)
        logger.info("Creating VLMVerifier from config with params: %s", params)

        validated_params = validate_and_cast_config_params(params, cls._PARAM_TYPES, logger)

        return cls(
            system_prompt=validated_params["system_prompt"],
            retry=validated_params["retry"],
            temperature=validated_params["temperature"],
            top_p=validated_params["top_p"],
            frequency_penalty=validated_params["frequency_penalty"],
            max_tokens=validated_params["max_tokens"],
            stream=validated_params["stream"],
            endpoint=validated_params["endpoint"],
            model=validated_params["model"],
            logger=logger,
            api_key_env=api_key_env,
            frames=frames,
        )

    def _extract_first_frame(self, video_path: str, output_image_path: str) -> bool:
        """
        Extract the first frame from a video file using the repo's packaged FFmpeg binary.

        We shell out to ``ffmpeg`` (the LGPL 9.0.1 NVENC/SVT-AV1 build packaged in third_party/ffmpeg and shipped on
        PATH via //services:ffmpeg_layer) instead of opencv's bundled decoder, so this checker shares a single,
        controlled FFmpeg build with the other AV checkers.

        Args:
            video_path: Path to the video file
            output_image_path: Path to save the extracted frame

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    output_image_path,
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
            )

            if result.returncode != 0:
                self.logger.error("ffmpeg failed to extract first frame from %s: %s", video_path, result.stderr.strip())
                return False

            if not os.path.exists(output_image_path) or os.path.getsize(output_image_path) == 0:
                self.logger.error("ffmpeg produced no output frame for: %s", video_path)
                return False

            self.logger.debug("Extracted first frame to: %s", output_image_path)
            return True

        except subprocess.TimeoutExpired:
            self.logger.error(
                "ffmpeg timed out after %ss extracting first frame from %s", _FFMPEG_TIMEOUT_SECONDS, video_path
            )
            return False
        except (OSError, subprocess.SubprocessError):
            self.logger.exception("Error extracting first frame")
            return False

    def _format_question(self, question: str, options: dict) -> str:
        """
        Format the question with options for the VLM.

        Args:
            question: The question text
            options: Dictionary of option letters to values

        Returns:
            str: Formatted question string
        """
        options_text = "\n".join([f"{k}) {v}" for k, v in sorted(options.items())])
        return f"{question}\n{options_text}\n\nAnswer with only a single letter (A, B, C, or D)."

    def _probe_frame_count(self, video_path: str) -> int:
        """Return the decoded frame count reported by the packaged FFprobe, or zero."""
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    video_path,
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return max(0, int(result.stdout.strip()))
        except (OSError, ValueError, subprocess.SubprocessError):
            self.logger.exception("Unable to determine frame count for %s", video_path)
        return 0

    def _extract_frames(self, video_path: str, frame_count: int, output_dir: str) -> list[str]:
        """Extract evenly spaced frames with the repository's controlled FFmpeg build."""
        if frame_count <= 1:
            output_image_path = os.path.join(output_dir, "frame_01.jpg")
            return [output_image_path] if self._extract_first_frame(video_path, output_image_path) else []

        total = self._probe_frame_count(video_path)
        output_pattern = os.path.join(output_dir, "frame_%02d.jpg")
        command = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path]
        if total > 0:
            wanted = sorted({round(index * (total - 1) / (frame_count - 1)) for index in range(frame_count)})
            expression = "+".join(f"eq(n\\,{index})" for index in wanted)
            command.extend(["-vf", f"select={expression}", "-fps_mode", "vfr"])
        else:
            self.logger.warning(
                "Unknown frame count for %s; extracting the first %s frames instead of evenly spaced frames",
                video_path,
                frame_count,
            )
            command.extend(["-frames:v", str(frame_count)])
        command.extend(["-q:v", "2", output_pattern])

        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            self.logger.exception("Error extracting sampled frames from %s", video_path)
            return []
        if result.returncode != 0:
            self.logger.error("ffmpeg failed to extract sampled frames from %s: %s", video_path, result.stderr.strip())
            return []
        return [str(path) for path in sorted(Path(output_dir).glob("frame_*.jpg")) if path.stat().st_size > 0]

    def _format_question_with_reasoning(self, question: str, options: dict[str, str]) -> str:
        options_text = "\n".join(f"{key}) {value}" for key, value in sorted(options.items()))
        return (
            f"{question}\n{options_text}\n\n"
            "First provide a brief explanation (1-2 sentences). Then give only your answer as a single letter "
            "(A, B, C, or D) on the last line."
        )

    async def verify_question(
        self,
        media_path: str,
        question: str,
        options: dict[str, str],
        correct_answer: str,
        request_reasoning: bool = False,
    ) -> tuple[bool, str, str | None]:
        """
        Verify a single question by asking the VLM about an image or sampled video frames.

        Args:
            media_path: Path to the image or video file
            question: The question to ask
            options: Dictionary of answer options
            correct_answer: The expected correct answer letter
            request_reasoning: Whether to ask the VLM for a brief explanation

        Returns:
            Tuple of (is_correct, VLM answer letter, optional VLM reasoning)

        Raises:
            Exception: If verification fails
        """

        retries_remaining = self.max_retries
        formatted_question = (
            self._format_question_with_reasoning(question, options)
            if request_reasoning
            else self._format_question(question, options)
        )

        def _prepare_media() -> list[tuple[str, str]]:
            """Download media and return MIME-typed, base64-encoded image evidence."""
            if not msc.is_file(media_path):
                raise FileNotFoundError(f"Media file not found: {media_path}")

            with tempfile.TemporaryDirectory() as temp_dir:
                extension = os.path.splitext(media_path)[1].lower()
                temp_media_path = os.path.join(temp_dir, f"media{extension or '.mp4'}")
                with msc.open(media_path, "rb") as src, open(temp_media_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                self.logger.debug("Downloaded media to %s", temp_media_path)

                if extension in _IMAGE_EXTENSIONS:
                    image_paths = [temp_media_path]
                elif self.frames == 1:
                    first_frame = os.path.join(temp_dir, "first_frame.jpg")
                    if not self._extract_first_frame(temp_media_path, first_frame):
                        raise RuntimeError("Failed to extract first frame from video")
                    image_paths = [first_frame]
                else:
                    image_paths = self._extract_frames(temp_media_path, self.frames, temp_dir)
                    if not image_paths:
                        first_frame = os.path.join(temp_dir, "first_frame.jpg")
                        if not self._extract_first_frame(temp_media_path, first_frame):
                            raise RuntimeError("Failed to extract any frame from video")
                        image_paths = [first_frame]

                encoded = []
                for image_path in image_paths:
                    with open(image_path, "rb") as image_file:
                        encoded.append(
                            (
                                _IMAGE_MIME_TYPES[os.path.splitext(image_path)[1].lower()],
                                base64.b64encode(image_file.read()).decode(),
                            )
                        )
                return encoded

        while retries_remaining >= 0:
            try:
                # Blocking file I/O runs in a thread; the event loop stays free for other requests.
                image_data = await asyncio.to_thread(_prepare_media)

                prompt = formatted_question
                if len(image_data) > 1:
                    prompt = (
                        f"You are shown {len(image_data)} frames sampled in "
                        f"chronological order from a single generated video; "
                        f"consider all of them when answering.\n\n{formatted_question}"
                    )
                image_items: list[ChatCompletionContentPartParam] = [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}}
                    for mime_type, image_b64 in image_data
                ]
                user_content: list[ChatCompletionContentPartParam] = [
                    {"type": "text", "text": prompt},
                    *image_items,
                ]

                conversation: list[ChatCompletionMessageParam] = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ]

                self.logger.info("VLM endpoint call starting (model=%s)", self.model)
                _vlm_t0 = time.perf_counter()
                chat_response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=conversation,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    max_tokens=max(self.max_tokens, 256) if request_reasoning else self.max_tokens,
                    stream=self.stream,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                self.logger.info("VLM endpoint call completed in %.0fms", (time.perf_counter() - _vlm_t0) * 1000)
                assistant_message = chat_response.choices[0].message
                self.logger.debug("VLM response: %s", assistant_message.content)

                content = (assistant_message.content or "").strip()
                vlm_answer = _extract_mcq_letter(content)
                reasoning: str | None = None
                if request_reasoning and content:
                    reasoning_text = _strip_reasoning(content) or content
                    matches = list(re.finditer(r"\b([A-D])\b", reasoning_text.upper()))
                    if matches:
                        reasoning = reasoning_text[: matches[-1].start()].strip() or None
                    else:
                        reasoning = reasoning_text
                elif vlm_answer == "UNKNOWN":
                    self.logger.warning("Could not parse clear answer from VLM response: %s", content)

                is_correct = vlm_answer == correct_answer.upper()
                self.logger.debug(
                    "Question verification: expected=%s, got=%s, correct=%s", correct_answer, vlm_answer, is_correct
                )

                return is_correct, vlm_answer, reasoning

            except Exception as e:
                self.logger.error("Failed to verify question with VLM", exc_info=False)
                if retries_remaining > 0:
                    retries_remaining -= 1
                    self.logger.info(
                        "Retrying failed call to VLM. Retries remaining: %s/%s", retries_remaining, self.max_retries
                    )
                    # Continue the while loop to retry
                else:
                    self.logger.exception(
                        "Max retries reached. VLM failed to verify question after %s total attempts",
                        self.max_retries + 1,
                        exc_info=False,
                    )
                    raise RuntimeError(
                        f"Max retries reached. VLM failed to verify question after "
                        f"{self.max_retries + 1} total attempts: {e}"
                    ) from e
        return False, "UNKNOWN", None
