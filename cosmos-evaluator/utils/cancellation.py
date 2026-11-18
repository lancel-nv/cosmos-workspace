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

"""Cooperative-cancellation primitives shared by checker processing pipelines and their service transports.

A transport creates a per-request :class:`threading.Event` (``cancel_event``) and hands it down to the CPU/GPU work
running in an executor thread. When the request is abandoned — a checker's claim loop sees ``cancel_requested`` come
back on a lease renewal, or a streaming client disconnects — the transport sets the event; executor-thread work polls it
at loop boundaries via :func:`check_cancelled` and raises :class:`ProcessingCancelled` so the thread unwinds promptly
instead of running on with its temp dir already deleted (the "zombie thread" failure mode).

Both sides of that boundary live here. :func:`check_cancelled` is the synchronous checkpoint called from inside worker
threads; :func:`shielded_to_thread` is its async counterpart, used by transports to offload blocking work that has no
checkpoint to poll and therefore must be waited out rather than abandoned.
"""

import asyncio
from collections.abc import Callable
import contextlib
import threading


class ProcessingCancelled(Exception):
    """Raised inside a checker's processing pipeline when its request has been cancelled.

    Cooperative-cancellation signal: a transport creates a per-request :class:`threading.Event` (``cancel_event``) and
    hands it down to the CPU/GPU work running in an executor thread. When the request is abandoned the transport sets
    the event; the executor-thread work is expected to poll it at loop boundaries via :func:`check_cancelled` and raise
    this exception so the thread unwinds promptly instead of running on with its temp dir already deleted (the "zombie
    thread" failure mode).

    Carries a default message so that handlers which format it as ``%s`` -- the shape of a generic
    ``except Exception as e`` in a service's processing path -- cannot emit a blank line with nothing to identify it.
    """

    def __init__(self, message: str = "Processing was cancelled") -> None:
        super().__init__(message)


def check_cancelled(cancel_event: "threading.Event | None") -> None:
    """Raise :class:`ProcessingCancelled` if ``cancel_event`` is set.

    Convenience checkpoint for the hot per-frame loops inside executor-thread work. ``None`` is accepted (and is a
    no-op) so callers that have no cancellation wired up — e.g. the local CLI — can share the same code path.

    Args:
        cancel_event: The per-request cancellation event, or ``None`` when cancellation is not wired up.

    Raises:
        ProcessingCancelled: If ``cancel_event`` is not ``None`` and has been set.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise ProcessingCancelled()


async def shielded_to_thread[T](fn: Callable[..., T], /, *args: object, **kwargs: object) -> T:
    """Run ``fn`` in a worker thread, waiting for it to finish even if the awaiting task is cancelled.

    A plain ``await asyncio.to_thread(fn, ...)`` is unsafe whenever the caller cleans up state the thread is still
    touching. Cancelling the await raises immediately, but the thread itself cannot be cancelled and keeps running, so
    control returns to a ``finally`` block that deletes the request's temp dir out from under live work.

    Use this for blocking calls that have no cooperative checkpoint to poll -- zip extraction and similar bounded I/O,
    where the only correct response to cancellation is to wait the thread out. Work that runs long enough that waiting
    is unacceptable should instead take a ``cancel_event`` and poll :func:`check_cancelled`.

    ``CancelledError`` is still raised once the thread has finished, so callers cancel with the usual semantics; only
    the timing changes. An exception from ``fn`` raised during that post-cancellation wait is suppressed, since the
    caller is already unwinding and the cancellation is the more relevant outcome.

    Args:
        fn: The blocking callable to run in a worker thread.
        *args: Positional arguments forwarded to ``fn``.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        Whatever ``fn`` returns.
    """
    fut = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        # Shielded again: a bare ``await fut`` here would propagate a second cancellation into ``fut`` itself, marking
        # it cancelled while its thread runs on and leaving the caller with a future that lies about being finished.
        with contextlib.suppress(Exception):
            await asyncio.shield(fut)
        raise
