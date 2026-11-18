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

"""Long-lived physical-plausibility service with in-process dynamic batching."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import logging
import threading
from typing import Any, Protocol

from checks.physical_plausibility.processor import (
    PhysicalPlausibilityProcessor,
    PhysicalPlausibilityResult,
)
from checks.utils.multistorage import is_remote_path, validate_uri
from services.framework.service_base import ServiceBase
from services.physical_plausibility.models import PhysicalPlausibilityRequest

logger = logging.getLogger(__name__)


@dataclass
class _Pending:
    request: PhysicalPlausibilityRequest
    future: asyncio.Future[PhysicalPlausibilityResult]


class _BatchProcessor(Protocol):
    def process_batch(
        self,
        items: Sequence[tuple[str, str, float | None]],
    ) -> list[PhysicalPlausibilityResult]: ...


class PhysicalPlausibilityService(ServiceBase[PhysicalPlausibilityRequest, PhysicalPlausibilityResult]):
    def __init__(
        self,
        processor: _BatchProcessor | None = None,
        *,
        max_batch_size: int = 4,
        max_batch_wait_ms: float = 10.0,
        checkpoint_uri: str | None = None,
        config_dir: str | None = None,
    ) -> None:
        super().__init__()
        if max_batch_size < 1 or max_batch_wait_ms < 0:
            raise ValueError("max_batch_size must be positive and max_batch_wait_ms must be non-negative")
        self.processor = processor or PhysicalPlausibilityProcessor(
            checkpoint_uri=checkpoint_uri,
            config_dir=config_dir,
        )
        self.config_dir = config_dir
        self.max_batch_size = max_batch_size
        self.max_batch_wait_seconds = max_batch_wait_ms / 1000
        self._queue: asyncio.Queue[_Pending] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._active_batch: list[_Pending] | None = None

    async def validate_input(self, request: PhysicalPlausibilityRequest) -> bool:
        if "://" in request.video_path and not is_remote_path(request.video_path):
            raise ValueError('field "video_path" has an unsupported URI scheme')
        # Remote existence is checked when the processor downloads/opens the URI. Avoid a second remote operation here.
        if not is_remote_path(request.video_path) and not await asyncio.to_thread(
            validate_uri, request.video_path, True
        ):
            raise ValueError('field "video_path" does not exist or cannot be accessed')
        return True

    async def get_default_config(self) -> dict[str, Any]:
        return (await asyncio.to_thread(PhysicalPlausibilityProcessor.get_default_config, self.config_dir))[
            "metropolis.physical_plausibility"
        ]

    async def process(
        self, request: PhysicalPlausibilityRequest, cancel_event: threading.Event | None = None
    ) -> PhysicalPlausibilityResult:
        # The current model encoder has no cooperative cancellation hook. Accept the shared checker-factory contract
        # so this service can participate in durable dispatch; cancellation is observed between queued jobs.
        del cancel_event
        loop = asyncio.get_running_loop()
        future: asyncio.Future[PhysicalPlausibilityResult] = loop.create_future()
        await self._queue.put(_Pending(request, future))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._batch_loop(), name="physical-plausibility-batcher")
        return await future

    async def close(self) -> None:
        if self._worker is None:
            return
        self._worker.cancel()
        try:
            await self._worker
        except asyncio.CancelledError:
            pass
        self._worker = None

    async def _batch_loop(self) -> None:
        try:
            while True:
                first = await self._queue.get()
                batch = [first]
                deadline = asyncio.get_running_loop().time() + self.max_batch_wait_seconds
                while len(batch) < self.max_batch_size:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self._queue.get(), remaining))
                    except TimeoutError:
                        break
                self._active_batch = [item for item in batch if not item.future.cancelled()]
                await self._run_batch(self._active_batch)
                self._active_batch = None
        except asyncio.CancelledError:
            if self._active_batch is not None:
                for pending in self._active_batch:
                    if not pending.future.done():
                        pending.future.cancel()
            while not self._queue.empty():
                pending = self._queue.get_nowait()
                pending.future.cancel()
            raise

    async def _run_batch(self, batch: list[_Pending]) -> None:
        if not batch:
            return
        logger.info("Dispatching physical-plausibility batch batch_size=%d", len(batch))
        items = [
            (
                item.request.clip_id,
                item.request.video_path,
                item.request.config.get("bad_threshold") if item.request.config else None,
            )
            for item in batch
        ]
        try:
            results = await asyncio.to_thread(self.processor.process_batch, items)
        except Exception as exc:
            if len(batch) == 1:
                if not batch[0].future.done():
                    batch[0].future.set_exception(exc)
                return
            logger.warning("Batch failed; retrying requests individually", exc_info=True)
            for pending, item in zip(batch, items):
                try:
                    individual_results = await asyncio.to_thread(self.processor.process_batch, [item])
                    if not pending.future.done():
                        pending.future.set_result(individual_results[0])
                except Exception as item_exc:  # noqa: BLE001 - isolate one failed request from its batch
                    if not pending.future.done():
                        pending.future.set_exception(item_exc)
            return
        if len(results) != len(batch):
            error = RuntimeError(f"processor returned {len(results)} results for {len(batch)} requests")
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(error)
            return
        for pending, result in zip(batch, results):
            if not pending.future.done():
                pending.future.set_result(result)
