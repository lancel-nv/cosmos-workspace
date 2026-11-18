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

"""Tests for SSRF/URL-security helpers in :mod:`services.framework.url.security`.

Backfills coverage for the existing module (no test file existed pre-relocation). DNS-resolution branches are exercised
by patching ``socket.getaddrinfo`` so the tests stay hermetic.
"""

import socket as _socket
import sys
from unittest.mock import patch

import pytest

from services.framework.url.security import (
    InsecureUrlError,
    check_url_security,
    hostname_resolves_to_private_ip,
    is_private_or_reserved_ip,
)


class TestIsPrivateOrReservedIp:
    @pytest.mark.parametrize(
        "ip",
        [
            "127.0.0.1",  # loopback
            "10.0.0.5",  # private RFC1918
            "172.16.0.1",  # private RFC1918
            "192.168.1.1",  # private RFC1918
            "169.254.42.42",  # link-local
            "::1",  # IPv6 loopback
            "fe80::1",  # IPv6 link-local
            "fc00::1",  # IPv6 unique local
            "0.0.0.0",  # reserved
        ],
    )
    def test_rejects_unsafe_addresses(self, ip: str) -> None:
        assert is_private_or_reserved_ip(ip) is True

    @pytest.mark.parametrize(
        "ip",
        [
            "8.8.8.8",
            "1.1.1.1",
            "2001:4860:4860::8888",  # public IPv6
        ],
    )
    def test_allows_public_addresses(self, ip: str) -> None:
        assert is_private_or_reserved_ip(ip) is False

    def test_unparseable_input_returns_false(self) -> None:
        # Hostnames (not IPs) and garbage strings should not raise.
        assert is_private_or_reserved_ip("example.com") is False
        assert is_private_or_reserved_ip("not-an-ip") is False
        assert is_private_or_reserved_ip("") is False


class TestHostnameResolvesToPrivateIp:
    def test_returns_true_when_any_result_is_private(self) -> None:
        # ``getaddrinfo`` returns a list of 5-tuples; sockaddr is index 4.
        # See https://docs.python.org/3/library/socket.html#socket.getaddrinfo
        with patch(
            "services.framework.url.security.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("127.0.0.1", 0))],
        ):
            assert hostname_resolves_to_private_ip("evil.example.com") is True

    def test_returns_false_when_all_results_are_public(self) -> None:
        with patch(
            "services.framework.url.security.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("8.8.8.8", 0))],
        ):
            assert hostname_resolves_to_private_ip("dns.google") is False

    def test_returns_false_on_gaierror(self) -> None:
        with patch(
            "services.framework.url.security.socket.getaddrinfo",
            side_effect=_socket.gaierror,
        ):
            assert hostname_resolves_to_private_ip("nonexistent.invalid") is False

    def test_returns_false_on_timeout(self) -> None:
        with patch(
            "services.framework.url.security.socket.getaddrinfo",
            side_effect=_socket.timeout,
        ):
            assert hostname_resolves_to_private_ip("slow.example.com") is False


class TestCheckUrlSecurity:
    def test_rejects_http_scheme(self) -> None:
        with pytest.raises(InsecureUrlError, match="HTTP URLs are not allowed"):
            check_url_security("http://example.com/foo")

    def test_rejects_https_with_literal_private_ip(self) -> None:
        with pytest.raises(InsecureUrlError, match="private/reserved IP address"):
            check_url_security("https://10.0.0.5/foo")

    def test_rejects_https_with_loopback_literal(self) -> None:
        with pytest.raises(InsecureUrlError, match="private/reserved IP address"):
            check_url_security("https://127.0.0.1/foo")

    def test_rejects_https_resolving_to_private_ip(self) -> None:
        with (
            patch(
                "services.framework.url.security.socket.getaddrinfo",
                return_value=[(None, None, None, None, ("192.168.1.5", 0))],
            ),
            pytest.raises(InsecureUrlError, match="resolves to a private/reserved IP"),
        ):
            check_url_security("https://intranet.example.com/foo")

    def test_rejects_https_without_hostname(self) -> None:
        with pytest.raises(InsecureUrlError, match="must include a hostname"):
            check_url_security("https:///path")

    def test_allows_public_https(self) -> None:
        with patch(
            "services.framework.url.security.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("8.8.8.8", 0))],
        ):
            check_url_security("https://example.com/foo")

    def test_ignores_non_http_schemes(self) -> None:
        # s3://, gs://, file:// etc. are left to their respective providers.
        check_url_security("s3://bucket/key")
        check_url_security("gs://bucket/key")
        check_url_security("file:///tmp/local")

    def test_trusted_host_allows_literal_private_ip(self) -> None:
        # An allowlisted host is exempt even when it is itself a literal private IP.
        check_url_security("https://10.0.0.5/foo", trusted_hosts={"10.0.0.5"})

    def test_trusted_host_allows_private_resolving_host(self) -> None:
        # An internal cluster (e.g. PBSS) resolves to a private IP by design; allowlisting it permits the fetch.
        with patch(
            "services.framework.url.security.socket.getaddrinfo",
            return_value=[(None, None, None, None, ("10.50.181.116", 0))],
        ):
            check_url_security("https://pbss.s8k.io/bucket/key", trusted_hosts={"pbss.s8k.io"})

    def test_non_allowlisted_host_still_rejected(self) -> None:
        # The allowlist is exact: a different private-resolving host stays blocked even when another host is trusted.
        with (
            patch(
                "services.framework.url.security.socket.getaddrinfo",
                return_value=[(None, None, None, None, ("192.168.1.5", 0))],
            ),
            pytest.raises(InsecureUrlError, match="resolves to a private/reserved IP"),
        ):
            check_url_security("https://intranet.example.com/foo", trusted_hosts={"pbss.s8k.io"})

    def test_trusted_host_does_not_bypass_http(self) -> None:
        # http:// is rejected even for an allowlisted host — the exemption is only for the private-IP check.
        with pytest.raises(InsecureUrlError, match="HTTP URLs are not allowed"):
            check_url_security("http://pbss.s8k.io/foo", trusted_hosts={"pbss.s8k.io"})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
