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

"""URL security utilities for SSRF prevention.

Provides helpers that reject URLs targeting private, reserved, loopback, or link-local IP addresses. Used both at
request-validation time (early feedback) and immediately before an HTTP fetch (TOCTOU / DNS-rebinding mitigation).
"""

from __future__ import annotations

from collections.abc import Collection
from concurrent.futures import ThreadPoolExecutor
import ipaddress
import socket
from urllib.parse import urlparse

# Module-level singleton: per-call DNS timeout is enforced via ``future.result(timeout=...)`` so we never have
# to call ``socket.setdefaulttimeout``, which is process-wide and would lower the timeout for every other
# socket operation in the interpreter during the lookup window. Two workers is enough to keep a new request
# moving even when a previous hostname is still hanging inside the kernel DNS resolver; an orphaned worker
# will exit once that lookup finally completes. A per-call ``ThreadPoolExecutor()`` context manager would
# defeat the purpose by calling ``shutdown(wait=True)`` on exit and re-introducing the blocking wait.
_RESOLVER_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ssrf-dns-resolver")


class InsecureUrlError(ValueError):
    """A URL targets a private/reserved address or uses an insecure scheme."""


def is_private_or_reserved_ip(ip_str: str) -> bool:
    """Return True if *ip_str* is a private, reserved, loopback, or link-local IP."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return addr.is_private or addr.is_reserved or addr.is_loopback or addr.is_link_local


def hostname_resolves_to_private_ip(hostname: str) -> bool:
    """Return True if *hostname* resolves to any private/reserved IP address.

    Resolution runs in a dedicated worker thread with a 2-second timeout enforced per call (no process-wide
    ``socket.setdefaulttimeout`` mutation). Returns False when the name cannot be resolved or the lookup times
    out. Note that ``socket.timeout`` and ``concurrent.futures.TimeoutError`` are both aliases of the builtin
    ``TimeoutError`` on Python 3.11+, so a single ``TimeoutError`` arm covers both the per-call deadline and
    the underlying socket-level timeout.
    """
    future = _RESOLVER_EXECUTOR.submit(socket.getaddrinfo, hostname, None, proto=socket.IPPROTO_TCP)
    try:
        results = future.result(timeout=2.0)
    except TimeoutError:
        # Cancel the future so a still-queued lookup doesn't sit in the executor work queue indefinitely
        # and consume a slot the next caller will need. ``Future.cancel()`` returns False (without raising)
        # if the worker has already started ``getaddrinfo`` — ``ThreadPoolExecutor`` cannot interrupt a
        # running blocking syscall, so the orphaned worker drains naturally once the kernel resolver returns.
        future.cancel()
        return False
    except socket.gaierror:
        return False
    return any(is_private_or_reserved_ip(str(sockaddr[0])) for _, _, _, _, sockaddr in results)


def check_url_security(url: str, trusted_hosts: Collection[str] = ()) -> None:
    """Raise :class:`InsecureUrlError` if *url* targets a private/reserved address.

    Designed to be called immediately before an HTTP fetch so that DNS-rebinding attacks that pass an earlier Pydantic
    validation are still caught.

    Only ``https://`` URLs are checked; ``s3://``, ``gs://``, and local paths are left to their respective providers.

    ``trusted_hosts`` is an opt-in allowlist of exact hostnames (e.g. an internal S3-compatible cluster such as
    ``pbss.s8k.io``) whose resolution to a private/reserved IP is permitted by design. An allowlisted ``https`` host
    skips the private-IP checks; TLS certificate verification on the fetch is the backstop here, since a DNS rebind to
    a different internal host would still have to present a valid certificate for the allowlisted name. ``http://`` is
    rejected even for allowlisted hosts. Empty (the default) preserves the original behavior for every caller.
    """
    parsed = urlparse(url)

    if parsed.scheme == "http":
        raise InsecureUrlError(f"Unencrypted HTTP URLs are not allowed: {url}")

    if parsed.scheme == "https":
        hostname = parsed.hostname
        if not hostname:
            raise InsecureUrlError("HTTPS URL must include a hostname")

        if hostname in trusted_hosts:
            return

        if is_private_or_reserved_ip(hostname):
            raise InsecureUrlError(f"URL targets a private/reserved IP address: {hostname}")

        if hostname_resolves_to_private_ip(hostname):
            raise InsecureUrlError(f"URL hostname resolves to a private/reserved IP address: {hostname}")
