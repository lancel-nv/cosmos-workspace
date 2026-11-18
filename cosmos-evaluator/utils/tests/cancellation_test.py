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
import sys
import threading
import time

import pytest

from utils.cancellation import ProcessingCancelled, check_cancelled, shielded_to_thread


def test_check_cancelled_none_is_noop() -> None:
    """A ``None`` event (local CLI path) must never raise."""
    check_cancelled(None)


def test_check_cancelled_unset_is_noop() -> None:
    """An unset event must not raise."""
    check_cancelled(threading.Event())


def test_check_cancelled_set_raises() -> None:
    """A set event raises ``ProcessingCancelled`` so executor-thread loops unwind promptly."""
    event = threading.Event()
    event.set()
    with pytest.raises(ProcessingCancelled):
        check_cancelled(event)


@pytest.mark.asyncio
async def test_shielded_to_thread_returns_value() -> None:
    """The uncancelled path behaves exactly like ``asyncio.to_thread``."""
    assert await shielded_to_thread(lambda a, b: a + b, 2, 3) == 5


@pytest.mark.asyncio
async def test_shielded_to_thread_forwards_kwargs() -> None:
    """Keyword arguments reach the callable, matching ``asyncio.to_thread``'s signature."""

    def build(base: str, *, suffix: str) -> str:
        return base + suffix

    assert await shielded_to_thread(build, "rds", suffix=".zip") == "rds.zip"


@pytest.mark.asyncio
async def test_shielded_to_thread_propagates_exceptions() -> None:
    """An exception raised in the worker thread surfaces to the caller unchanged."""
    with pytest.raises(ValueError, match="bad zip"):
        await shielded_to_thread(_raise_value_error)


def _raise_value_error() -> None:
    raise ValueError("bad zip")


@pytest.mark.asyncio
async def test_shielded_to_thread_waits_for_thread_before_cancelling() -> None:
    """Cancelling the awaiting task must not hand control back while the worker thread is still running.

    This is the whole point of the helper: a plain ``await asyncio.to_thread(...)`` raises immediately on cancel and
    lets the caller's ``finally`` delete the directory the thread is still writing into. Assert on thread liveness at
    the moment ``CancelledError`` is observed, not merely that the thread eventually finished.
    """
    running = threading.Event()
    finished = threading.Event()

    def slow_body() -> str:
        running.set()
        time.sleep(0.2)
        finished.set()
        return "done"

    async def caller() -> str:
        return await shielded_to_thread(slow_body)

    task = asyncio.create_task(caller())
    await asyncio.to_thread(running.wait, 5.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set(), "CancelledError escaped while the worker thread was still running"


@pytest.mark.asyncio
async def test_shielded_to_thread_suppresses_worker_error_during_cancellation() -> None:
    """A thread that fails while the caller is unwinding still yields ``CancelledError``, not the thread's exception."""
    running = threading.Event()

    def failing_body() -> None:
        running.set()
        time.sleep(0.1)
        raise RuntimeError("worker blew up during teardown")

    task = asyncio.create_task(shielded_to_thread(failing_body))
    await asyncio.to_thread(running.wait, 5.0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
