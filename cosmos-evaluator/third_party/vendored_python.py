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

"""Audit the Python metadata carried by the vendored repos under third_party/.

A vendored Python library is only attributable if it reaches the runfiles tree — and therefore an image layer — with a
readable ``.dist-info``. That is what ``importlib.metadata`` resolves and what an SBOM scanner reads out of a shipped
image, so a repo whose metadata is missing, misnamed, or stale is invisible to attribution tooling even though its code
is running in production.

Three things have to line up for that to hold, and each is checked here:

* the metadata exists where the declaration says it does, which for a source archive means the ``dist_info_*``
  attributes on its ``http_archive.archive(...)`` tag and for a local wheel means the wheel's own ``.dist-info``;
* it agrees with the declaration on name and version, so a wheel rebuilt without its local version label or an archive
  bumped to a new commit without touching ``dist_info_version`` fails loudly rather than silently misreporting;
* it ships license text, and every ``py_library`` in the repo's overlay carries the metadata as data so it is not
  dropped on the way into runfiles.

The declarations come from the ``//:vendored_python_facts`` genrule, which extracts them from MODULE.bazel in the
package that owns it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
import re

# The attributes the facts genrule extracts from each archive/wheel tag. Anything else in a tag is irrelevant here.
_FACT_KEYS = frozenset({"name", "src", "build_file", "dist_info_name", "dist_info_root", "dist_info_version"})

# PEP 503 name normalization, for comparing a declared name against the one in METADATA.
_NAME_SEPARATORS_RE = re.compile(r"[-_.]+")

# A license file as shipped by either metadata generation: PEP 639 records them under licenses/, while older wheels
# (opencv-python-headless among them) drop them at the .dist-info root.
_LICENSE_FILE_RE = re.compile(r"^(licenses/.+|(LICEN[CS]E|COPYING|NOTICE)[^/]*)$", re.IGNORECASE)


@dataclass(frozen=True)
class VendoredRepo:
    """One ``http_archive.archive(...)`` or ``local_wheel.wheel(...)`` declaration from a module's MODULE.bazel."""

    kind: str
    name: str
    src: str | None
    build_file: str | None
    dist_info_name: str | None
    dist_info_root: str | None
    dist_info_version: str | None


@dataclass(frozen=True)
class DistInfo:
    """The subset of a ``.dist-info`` an attribution audit cares about, as parsed from METADATA plus a file listing."""

    name: str
    version: str
    license_expression: str | None
    files: frozenset[str]


def normalized_name(name: str) -> str:
    """Return the PEP 503 normalized form of a distribution name."""
    return _NAME_SEPARATORS_RE.sub("-", name).lower()


def escaped_name(name: str) -> str:
    """Return the escaped form the extension writes into a synthesized ``.dist-info`` directory name.

    This has to track ``_escaped_dist_name`` in ``http_archive.bzl`` exactly — lowercased, with each run of ``-``, ``_``
    or ``.`` collapsed to a single underscore. Any divergence sends the audit looking for a directory the extension
    never wrote, which surfaces as a missing-metadata failure against a repo whose metadata is fine.
    """
    return _NAME_SEPARATORS_RE.sub("_", name).lower()


def parse_facts(facts_text: str) -> list[VendoredRepo]:
    """Return the vendored repo declarations emitted by the ``vendored_python_facts`` genrule.

    The genrule emits one ``<kind>\\t<attr>\\t<value>`` line per attribute of interest, in file order. ``name`` is
    buildifier-guaranteed to be a tag's first attribute, so it opens a new record.
    """
    repos: list[VendoredRepo] = []
    attributes: dict[str, str] = {}
    kind = ""

    def flush() -> None:
        if attributes:
            repos.append(
                VendoredRepo(
                    kind=kind,
                    name=attributes["name"],
                    src=attributes.get("src"),
                    build_file=attributes.get("build_file"),
                    dist_info_name=attributes.get("dist_info_name"),
                    dist_info_root=attributes.get("dist_info_root"),
                    dist_info_version=attributes.get("dist_info_version"),
                )
            )

    for line_number, line in enumerate(facts_text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            msg = f"line {line_number} of the vendored_python_facts output is not <kind>\\t<attr>\\t<value>: {line!r}"
            raise ValueError(msg)
        record_kind, attribute, value = fields
        if attribute not in _FACT_KEYS:
            msg = f"line {line_number} of the vendored_python_facts output has an unexpected attribute {attribute!r}"
            raise ValueError(msg)
        if attribute == "name":
            flush()
            attributes = {}
            kind = record_kind
        attributes[attribute] = value

    flush()
    return repos


def expected_distribution(repo: VendoredRepo) -> tuple[str, str] | None:
    """Return the ``(name, version)`` the repo's metadata must report, or None if it declares no metadata.

    An archive states both explicitly via ``dist_info_*``. A wheel states them in its filename, which is the whole
    point of checking it: the filename is what a reader sees, so the metadata inside has to agree with it.
    """
    if repo.kind == "wheel":
        if not repo.src:
            return None
        filename = repo.src.rpartition(":")[2]
        parts = filename.removesuffix(".whl").split("-")
        if len(parts) < 2:
            return None
        return parts[0], parts[1]
    if repo.dist_info_name and repo.dist_info_version:
        return repo.dist_info_name, repo.dist_info_version
    return None


def dist_info_dir(repo: VendoredRepo) -> str | None:
    """Return the repo-relative path of the repo's ``.dist-info`` directory, or None if it declares no metadata."""
    distribution = expected_distribution(repo)
    if distribution is None:
        return None
    name, version = distribution
    # A wheel's directory stem was escaped by whatever built the wheel, and the filename we read it from is that same
    # escaped spelling, so reuse it verbatim rather than re-escaping. An archive's directory is written by
    # http_archive.bzl from the declared name, so escape it the way that extension does.
    stem = name if repo.kind == "wheel" else escaped_name(name)
    directory = f"{stem}-{version}.dist-info"
    return f"{repo.dist_info_root}/{directory}" if repo.dist_info_root else directory


def read_dist_info(dist_info_path: Path) -> DistInfo:
    """Read a ``.dist-info`` directory the way attribution tooling does: METADATA headers plus the files beside it."""
    metadata = Parser().parsestr(dist_info_path.joinpath("METADATA").read_text(encoding="utf-8"), headersonly=True)
    files = frozenset(
        entry.relative_to(dist_info_path).as_posix() for entry in dist_info_path.rglob("*") if entry.is_file()
    )
    return DistInfo(
        name=metadata.get("Name", ""),
        version=metadata.get("Version", ""),
        license_expression=metadata.get("License-Expression") or metadata.get("License"),
        files=files,
    )


def overlay_path(repo: VendoredRepo) -> str | None:
    """Return the workspace-relative path of the repo's BUILD overlay, or None if it declares no ``build_file``."""
    if not repo.build_file:
        return None
    package, _, target = repo.build_file.removeprefix("//").partition(":")
    return f"{package}/{target}"


def audit_metadata(
    repos: list[VendoredRepo],
    dist_infos: Mapping[str, DistInfo | None],
    exemptions: Mapping[str, str],
) -> list[str]:
    """Check each repo's shipped metadata against its declaration, returning one message per violation.

    ``dist_infos`` maps a repo name to the metadata found in runfiles at ``dist_info_dir(repo)``, or None when nothing
    is there. ``exemptions`` maps a repo name to the reason it ships no metadata, so an unattributable repo is a
    deliberate, documented entry rather than an oversight.
    """
    violations: list[str] = []

    for repo in sorted(repos, key=lambda candidate: candidate.name):
        distribution = expected_distribution(repo)

        if distribution is None:
            if repo.name not in exemptions:
                violations.append(
                    f"{repo.name}: declares no Python metadata, so nothing attributes it in a runfiles tree or an "
                    f"SBOM. Add dist_info_name/dist_info_version (plus dist_info_license and dist_info_license_files) "
                    f"to its http_archive.archive() tag, or record it in this test's exemptions with a reason."
                )
            continue

        if repo.name in exemptions:
            violations.append(
                f"{repo.name}: exempted from the metadata requirement, but it now declares metadata. Drop the "
                f"exemption so the declaration is audited."
            )
            continue

        expected_name, expected_version = distribution
        location = dist_info_dir(repo)
        dist_info = dist_infos.get(repo.name)

        if dist_info is None:
            violations.append(
                f"{repo.name}: no readable METADATA at @{repo.name}//{location}. Either the declared name/version and "
                f"the shipped metadata have diverged, or the auditing target is missing @{repo.name}//:dist_info in "
                f"its data attribute."
            )
            continue

        if normalized_name(dist_info.name) != normalized_name(expected_name):
            violations.append(
                f"{repo.name}: METADATA reports the name {dist_info.name!r} but the declaration says "
                f"{expected_name!r}. Attribution keys off the metadata, so the two have to agree."
            )

        if dist_info.version != expected_version:
            violations.append(
                f"{repo.name}: METADATA reports version {dist_info.version!r} but the declaration says "
                f"{expected_version!r}. A stale version misreports which upstream revision is shipping; bump the "
                f"declaration, or rebuild the artifact so its own metadata carries the right version."
            )

        if not dist_info.license_expression:
            violations.append(
                f"{repo.name}: METADATA states no license. Set dist_info_license on the archive tag so the shipped "
                f"metadata names the license the code is under."
            )

        if not any(_LICENSE_FILE_RE.match(name) for name in dist_info.files):
            violations.append(
                f"{repo.name}: the .dist-info carries no license text. Point dist_info_license_files at the upstream "
                f"LICENSE committed beside the overlay so the text travels with the code."
            )

    for name, reason in sorted(exemptions.items()):
        if not reason.strip():
            violations.append(f"{name}: exempted from the metadata requirement without a reason. Document why.")
        if name not in {repo.name for repo in repos}:
            violations.append(
                f"{name}: exempted from the metadata requirement but no longer declared in MODULE.bazel. Remove the "
                f"stale exemption."
            )

    return violations


def audit_overlays(repos: list[VendoredRepo], overlays: Mapping[str, str | None]) -> list[str]:
    """Check that every ``py_library`` in each overlay carries the repo's metadata, returning one message per violation.

    A ``.dist-info`` synthesized into a repo still reaches runfiles only if a target claims it, so an overlay that
    grows a new ``py_library`` without ``data = [":dist_info"]`` silently drops attribution for whatever depends on
    that target alone. ``overlays`` maps a repo name to its overlay's text, or None when it is absent from runfiles.
    """
    violations: list[str] = []

    for repo in sorted(repos, key=lambda candidate: candidate.name):
        if expected_distribution(repo) is None:
            continue

        path = overlay_path(repo)
        if path is None:
            violations.append(f"{repo.name}: declares metadata but no build_file, so no target can carry it.")
            continue

        text = overlays.get(repo.name)
        if text is None:
            violations.append(
                f"{repo.name}: {path} is not in runfiles. Add it to the auditing target's data attribute so its "
                f"py_library rules can be checked."
            )
            continue

        for rule in _top_level_rules(text, "py_library"):
            name_match = re.search(r'name = "([^"]+)"', rule)
            target = name_match.group(1) if name_match else "<unnamed>"
            if '":dist_info"' not in rule:
                violations.append(
                    f"{repo.name}: py_library {target!r} in {path} does not carry the repo's metadata. Add "
                    f'":dist_info" to its data attribute, or the .dist-info never reaches the runfiles tree of a '
                    f"target that depends only on it."
                )

    return violations


def _top_level_rules(build_text: str, rule_kind: str) -> list[str]:
    """Return the text of each top-level ``rule_kind(...)`` invocation in a buildifier-formatted BUILD file.

    Buildifier puts a top-level rule's opening call at column 0 and its closing paren alone on a line, so the rules can
    be sliced out without a Starlark parser.
    """
    rules: list[str] = []
    current: list[str] = []

    for line in build_text.splitlines():
        if current:
            current.append(line)
            if line == ")":
                rules.append("\n".join(current))
                current = []
        elif line.startswith(f"{rule_kind}("):
            current = [line]

    return rules
