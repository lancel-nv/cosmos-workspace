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

"""Merge overlay dpkg stanzas into the base OCI image's /var/lib/dpkg/status and record each overlay package's files.

Reads the base image's OCI layout to extract /var/lib/dpkg/status, then merges overlay package stanzas from a
rules_distroless dpkg_status tar. Packages present in both are replaced with the overlay version (upgraded).

rules_distroless installs package contents without running dpkg, so nothing else writes the
/var/lib/dpkg/info/<pkg>.list and <pkg>.md5sums files that record which paths a package installed. Security scanners
rely on those files to attribute an installed file such as /usr/bin/openssl to its package; a file no package claims is
reported as a separate binary at the upstream version embedded in it, so distribution-backported fixes go unrecognized.
Both files are therefore generated for every overlay package from its data layer. The caller supplies the data layers
together with the rules_distroless lockfile, and each layer is matched to its package through the lockfile key that
names the repository directory holding it (``<repo>_<key>``).

The merged result is written as a tar containing /var/lib/dpkg/status and the generated info files.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import logging
from pathlib import Path
import posixpath
import sys
import tarfile
from typing import IO

logger = logging.getLogger(__name__)

_DPKG_STATUS_PATH = "./var/lib/dpkg/status"
_DPKG_STATUS_PATH_NO_PREFIX = "var/lib/dpkg/status"
_DPKG_INFO_DIR = "./var/lib/dpkg/info"

# Upper bound on the decompressed size of a single dpkg status file we are willing to read into memory. Real
# /var/lib/dpkg/status files are a few MiB even for large images; this generous cap guards against decompression
# ("zip bomb") attacks where a small archive expands to gigabytes.
_MAX_DPKG_STATUS_BYTES = 256 * 1024 * 1024  # 256 MiB

# Resource-consumption thresholds applied while expanding an archive, guarding against decompression-bomb attacks
# (small archives that expand to exhaust memory, disk, or inodes):
#   * total uncompressed bytes read across all members of a single archive,
#   * total number of members (entries) processed in a single archive, and
#   * the ratio of bytes actually read for one member to the size that member declares.
_MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024  # 1 GiB total uncompressed per archive
_MAX_ARCHIVE_ENTRIES = 100_000
_MAX_ENTRY_COMPRESSION_RATIO = 100
_READ_CHUNK_BYTES = 1024 * 1024  # 1 MiB read granularity


@dataclass(frozen=True)
class LockedPackage:
    """A package entry of a rules_distroless lockfile."""

    name: str
    version: str
    key: str


@dataclass(frozen=True)
class PackageLayer:
    """An overlay package together with the data layer that installs its files."""

    package: LockedPackage
    layer: Path


@dataclass(frozen=True)
class PackageFiles:
    """Contents of a package's /var/lib/dpkg/info ``.list`` and ``.md5sums`` files."""

    list_text: str
    md5sums_text: str


def _iter_member_guarded(
    extracted: IO[bytes],
    declared_size: int,
    source: str,
    max_bytes: int = _MAX_DPKG_STATUS_BYTES,
) -> Iterator[bytes]:
    """Yield a tar member's contents in chunks while defending against decompression bombs.

    Yields at most ``max_bytes`` and aborts early via ``ValueError`` if either the running byte count exceeds
    ``max_bytes`` or the ratio of bytes read to the member's declared size exceeds
    ``_MAX_ENTRY_COMPRESSION_RATIO``. Reading incrementally (rather than slurping the whole member) means an
    attacker-controlled archive that under-reports a member's size cannot force us to materialize gigabytes in
    memory before the cap trips.
    """
    size_entry = 0
    while True:
        # A member whose real (uncompressed) size dwarfs its declared size is the hallmark of a decompression
        # bomb; bound the ratio of bytes actually read to the declared size before reading the next chunk.
        if declared_size > 0 and size_entry / declared_size > _MAX_ENTRY_COMPRESSION_RATIO:
            msg = f"{source}: decompression ratio exceeds {_MAX_ENTRY_COMPRESSION_RATIO} (possible decompression bomb)"
            raise ValueError(msg)
        chunk = extracted.read(_READ_CHUNK_BYTES)
        if not chunk:
            break
        size_entry += len(chunk)
        if size_entry > max_bytes:
            msg = f"{source} exceeds the maximum allowed size of {max_bytes} bytes (possible decompression bomb)"
            raise ValueError(msg)
        yield chunk


def _read_member_guarded(
    extracted: IO[bytes],
    declared_size: int,
    source: str,
    max_bytes: int = _MAX_DPKG_STATUS_BYTES,
) -> bytes:
    """Read a whole tar member under the limits enforced by ``_iter_member_guarded``."""
    return b"".join(_iter_member_guarded(extracted, declared_size, source, max_bytes))


def _parse_stanzas(text: str) -> dict[str, str]:
    """Parse dpkg status text into {package_name: stanza_text} dict."""
    stanzas: dict[str, str] = {}
    for stanza in text.split("\n\n"):
        stanza = stanza.strip()
        if not stanza:
            continue
        for line in stanza.splitlines():
            if line.startswith("Package:"):
                pkg_name = line.split(":", 1)[1].strip()
                stanzas[pkg_name] = stanza
                break
    return stanzas


def _get_field(stanza: str, field: str) -> str | None:
    """Extract a field value from a dpkg stanza."""
    for line in stanza.splitlines():
        if line.startswith(f"{field}:"):
            return line.split(":", 1)[1].strip()
    return None


def _validate_safe_member_name(name: str, expected_prefix: str) -> str:
    """Validate that ``name`` is a safe tar member path confined to ``expected_prefix``.

    The merged tar this module writes is later expanded into a container image. Some member names are
    derived from package metadata in the untrusted overlay archive, so an attacker-controlled package
    name could embed ``..`` segments or an absolute path and escape the intended directory on
    extraction (path traversal / Zip-Slip). Reject anything that does not normalize to a path strictly
    under ``expected_prefix``.
    """
    normalized = posixpath.normpath(name)
    prefix = posixpath.normpath(expected_prefix)
    # normpath collapses any ``..``/``.`` segments and strips a leading ``./``; if the result is not strictly
    # nested under the prefix (or is absolute), the original name was attempting to escape.
    if name.startswith("/") or not normalized.startswith(f"{prefix}/"):
        msg = f"refusing to write unsafe tar member name {name!r} (must stay under {expected_prefix!r})"
        raise ValueError(msg)
    return name


def _dpkg_info_name(stanza: str, pkg_name: str) -> str:
    """Return the dpkg info filename stem (e.g. 'libssl3:amd64' or 'gnupg2')."""
    if _get_field(stanza, "Multi-Arch") == "same":
        arch = _get_field(stanza, "Architecture") or "amd64"
        return f"{pkg_name}:{arch}"
    return pkg_name


def _extract_base_dpkg_status(oci_layout_dir: Path) -> str:
    """Extract /var/lib/dpkg/status from an OCI image layout.

    Walks layers bottom-to-top so the last layer that contains the file wins
    (matching OCI overlay semantics).
    """
    index = json.loads((oci_layout_dir / "index.json").read_text())
    manifest_digest = index["manifests"][0]["digest"]
    manifest_blob = oci_layout_dir / "blobs" / manifest_digest.replace(":", "/")
    manifest = json.loads(manifest_blob.read_text())

    status_text = ""
    for layer_desc in manifest["layers"]:
        layer_digest = layer_desc["digest"]
        layer_path = oci_layout_dir / "blobs" / layer_digest.replace(":", "/")
        media_type = layer_desc.get("mediaType", "")

        try:
            opener = gzip.open if ("gzip" in media_type or layer_path.suffix == ".gz") else open
            with opener(layer_path, "rb") as raw, tarfile.open(fileobj=raw, mode="r|*") as tf:
                # Defense-in-depth: enforce the safe ``data`` filter so any extraction rejects absolute paths,
                # ``..`` traversal, and other escaping members. We only read a single known member in-memory below,
                # but setting the filter documents intent and guards against future extract/extractall use.
                tf.extraction_filter = tarfile.data_filter
                # Control resource consumption while expanding the archive: cap the number of entries and the total
                # uncompressed bytes read so a maliciously crafted layer cannot exhaust memory or inodes.
                total_size_archive = 0
                for idx, member in enumerate(tf, start=1):
                    if idx > _MAX_ARCHIVE_ENTRIES:
                        msg = (
                            f"layer {layer_path}: archive contains more than {_MAX_ARCHIVE_ENTRIES} entries "
                            "(possible decompression bomb)"
                        )
                        raise ValueError(msg)
                    if member.name in (_DPKG_STATUS_PATH_NO_PREFIX, _DPKG_STATUS_PATH):
                        if member.size > _MAX_DPKG_STATUS_BYTES:
                            msg = (
                                f"layer {layer_path}: {member.name!r} declares {member.size} bytes, exceeding the "
                                f"maximum allowed size of {_MAX_DPKG_STATUS_BYTES} bytes (possible decompression bomb)"
                            )
                            raise ValueError(msg)
                        extracted = tf.extractfile(member)
                        if extracted:
                            member_bytes = _read_member_guarded(
                                extracted, member.size, f"layer {layer_path}: {member.name!r}"
                            )
                            total_size_archive += len(member_bytes)
                            if total_size_archive > _MAX_ARCHIVE_BYTES:
                                msg = (
                                    f"layer {layer_path}: total uncompressed size exceeds {_MAX_ARCHIVE_BYTES} bytes "
                                    "(possible decompression bomb)"
                                )
                                raise ValueError(msg)
                            status_text = member_bytes.decode("utf-8")
                        break
        except (tarfile.TarError, gzip.BadGzipFile, OSError):
            logger.warning("Skipping layer %s: unable to read as tar archive", layer_path, exc_info=True)
            continue
        except ValueError:
            # A ValueError here is a deliberate guard signal (member exceeds _MAX_DPKG_STATUS_BYTES, a capped-read
            # overflow, or malformed UTF-8), not a "this blob isn't a tar archive" condition. Surface it instead of
            # logging a misleading warning and silently skipping the layer, which would defeat the size cap.
            logger.exception("Aborting dpkg status extraction for layer %s", layer_path)
            raise

    return status_text


def _extract_overlay_stanzas(dpkg_status_tar: Path) -> str:
    """Extract /var/lib/dpkg/status text from a rules_distroless dpkg_status tar."""
    with tarfile.open(dpkg_status_tar, "r:*") as tf:
        # Defense-in-depth: enforce the safe ``data`` filter so any extraction rejects absolute paths,
        # ``..`` traversal, and other escaping members. We only read a single known member in-memory below,
        # but setting the filter documents intent and guards against future extract/extractall use.
        tf.extraction_filter = tarfile.data_filter
        try:
            member = tf.getmember(_DPKG_STATUS_PATH)
        except KeyError:
            msg = f"_extract_overlay_stanzas: member {_DPKG_STATUS_PATH!r} not found in {dpkg_status_tar}"
            raise ValueError(msg) from None
        if member.size > _MAX_DPKG_STATUS_BYTES:
            msg = (
                f"_extract_overlay_stanzas: {_DPKG_STATUS_PATH!r} in {dpkg_status_tar} declares {member.size} bytes, "
                f"exceeding the maximum allowed size of {_MAX_DPKG_STATUS_BYTES} bytes (possible decompression bomb)"
            )
            raise ValueError(msg)
        f = tf.extractfile(member)
        if f is None:
            msg = (
                f"_extract_overlay_stanzas: cannot read {_DPKG_STATUS_PATH!r} from {dpkg_status_tar} "
                "(not a regular file)"
            )
            raise ValueError(msg)
        source = f"_extract_overlay_stanzas: {_DPKG_STATUS_PATH!r} in {dpkg_status_tar}"
        return _read_member_guarded(f, member.size, source).decode("utf-8")


def _load_lock(lock: Path) -> list[LockedPackage]:
    """Read the package entries of a rules_distroless lockfile."""
    entries = json.loads(lock.read_text(encoding="utf-8"))["packages"]
    return [LockedPackage(name=entry["name"], version=entry["version"], key=entry["key"]) for entry in entries]


def _match_data_layers(repo: str, locked: list[LockedPackage], data_layers: list[Path]) -> dict[str, PackageLayer]:
    """Map each package name to its data layer.

    rules_distroless builds the data layer of the lockfile entry with key ``<key>`` in the repository ``<repo>_<key>``,
    whose directory name additionally carries the module-extension prefix (e.g. ``rules_distroless++apt+``).

    Raises:
        ValueError: A layer's directory matches no lockfile key or more than one, or two layers belong to the same
            package. Either means the layers no longer follow that naming.
    """
    matched: dict[str, PackageLayer] = {}
    for layer in data_layers:
        directory = layer.parent.name
        candidates = [package for package in locked if directory.endswith(f"{repo}_{package.key}")]
        if len(candidates) != 1:
            msg = (
                f"data layer {layer} matches {len(candidates)} entries of the {repo!r} lockfile instead of exactly one; "
                f"its directory must be named <prefix>{repo}_<key> for a lockfile key"
            )
            raise ValueError(msg)
        package = candidates[0]
        if package.name in matched:
            msg = f"package {package.name!r} has more than one data layer: {matched[package.name].layer} and {layer}"
            raise ValueError(msg)
        matched[package.name] = PackageLayer(package=package, layer=layer)
    return matched


def _check_overlay_layers(overlay_stanzas: dict[str, str], layers: dict[str, PackageLayer]) -> None:
    """Require one data layer per overlay package, at the version its status stanza records, and no other layers.

    Raises:
        ValueError: An overlay package has no data layer, a data layer belongs to a package outside the overlay, or a
            stanza's ``Version`` differs from the lockfile's.
    """
    missing = sorted(set(overlay_stanzas) - set(layers))
    if missing:
        msg = f"no data layer for overlay packages: {', '.join(missing)}"
        raise ValueError(msg)
    unexpected = sorted(set(layers) - set(overlay_stanzas))
    if unexpected:
        msg = f"data layers for packages absent from the overlay status: {', '.join(unexpected)}"
        raise ValueError(msg)
    mismatched = [
        f"{name} (status {_get_field(stanza, 'Version')}, lockfile {layers[name].package.version})"
        for name, stanza in sorted(overlay_stanzas.items())
        if _get_field(stanza, "Version") != layers[name].package.version
    ]
    if mismatched:
        msg = f"overlay status and lockfile disagree on package versions: {', '.join(mismatched)}"
        raise ValueError(msg)


def _installed_path(member_name: str, layer: Path) -> str:
    """Return the path a data-layer member installs, relative to ``/``, or an empty string for the root directory.

    Raises:
        ValueError: The member name is absolute, escapes the root through ``..``, or contains a newline, which the
            line-oriented ``.list`` and ``.md5sums`` formats cannot represent.
    """
    normalized = posixpath.normpath(member_name)
    if member_name.startswith("/") or normalized == ".." or normalized.startswith("../") or "\n" in member_name:
        msg = f"data layer {layer}: refusing unsafe member name {member_name!r}"
        raise ValueError(msg)
    return "" if normalized == "." else normalized


def _read_package_files(layer: Path) -> PackageFiles:
    """Build a package's ``.list`` and ``.md5sums`` contents from its gzip-compressed data layer.

    ``.list`` holds ``/.`` followed by every installed path (directories, regular files, hard links and symbolic links)
    in archive order. ``.md5sums`` holds one ``<md5>  <path relative to />`` line per regular file or hard link; it is
    the ownership record syft reads, while Trivy reads ``.list``. Paths are taken from the layer as installed, so they
    already reflect the merged-/usr rewriting applied when the layer was built.

    Raises:
        ValueError: The layer breaches a decompression-bomb limit, holds an unsafe member name, or has a member whose
            contents cannot be read.
        OSError: The layer is not gzip-compressed.
    """
    paths = ["/."]
    seen = set(paths)
    md5sums: list[str] = []
    total_size_archive = 0
    with gzip.open(layer, "rb") as raw, tarfile.TarFile(fileobj=raw) as tf:
        # Defense-in-depth, as for the other archives read here: no member is extracted to disk.
        tf.extraction_filter = tarfile.data_filter
        for idx, member in enumerate(tf, start=1):
            if idx > _MAX_ARCHIVE_ENTRIES:
                msg = (
                    f"data layer {layer}: archive contains more than {_MAX_ARCHIVE_ENTRIES} entries "
                    "(possible decompression bomb)"
                )
                raise ValueError(msg)
            relative = _installed_path(member.name, layer)
            if not relative:
                continue
            path = f"/{relative}"
            if path not in seen:
                seen.add(path)
                paths.append(path)
            if not (member.isfile() or member.islnk()):
                continue
            try:
                extracted = tf.extractfile(member)
            except KeyError:
                extracted = None
            if extracted is None:
                msg = f"data layer {layer}: cannot read the contents of {member.name!r}"
                raise ValueError(msg)
            digest = hashlib.md5(usedforsecurity=False)
            source = f"data layer {layer}: {member.name!r}"
            for chunk in _iter_member_guarded(extracted, member.size, source, max_bytes=_MAX_ARCHIVE_BYTES):
                total_size_archive += len(chunk)
                if total_size_archive > _MAX_ARCHIVE_BYTES:
                    msg = (
                        f"data layer {layer}: total uncompressed size exceeds {_MAX_ARCHIVE_BYTES} bytes "
                        "(possible decompression bomb)"
                    )
                    raise ValueError(msg)
                digest.update(chunk)
            md5sums.append(f"{digest.hexdigest()}  {relative}\n")
    return PackageFiles(list_text="\n".join(paths) + "\n", md5sums_text="".join(md5sums))


def _add_text_member(tout: tarfile.TarFile, name: str, text: str) -> None:
    """Add a root-owned, mode 0644 regular file holding ``text`` to ``tout``."""
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    tout.addfile(info, io.BytesIO(data))


def merge(
    oci_layout_dir: Path, overlay_tar: Path, output_tar: Path, repo: str, lock: Path, data_layers: list[Path]
) -> None:
    """Write ``output_tar`` holding the merged dpkg status and the ``.list``/``.md5sums`` of every overlay package.

    ``data_layers`` must contain exactly the data layers of the overlay packages, as built by the rules_distroless
    ``apt.install`` repository ``repo`` from ``lock``. Info files of packages that only the base image contains are left
    to the base image.

    Raises:
        ValueError: An input archive breaches a decompression-bomb limit or holds an unsafe member name, or the data
            layers do not correspond one-to-one, at matching versions, to the overlay packages.
    """
    base_stanzas = _parse_stanzas(_extract_base_dpkg_status(oci_layout_dir))
    overlay_stanzas = _parse_stanzas(_extract_overlay_stanzas(overlay_tar))
    logger.info("Base packages: %d, overlay packages: %d", len(base_stanzas), len(overlay_stanzas))

    layers = _match_data_layers(repo, _load_lock(lock), data_layers)
    _check_overlay_layers(overlay_stanzas, layers)

    merged = {**base_stanzas, **overlay_stanzas}
    logger.info("Merged package count: %d", len(merged))
    merged_text = "\n\n".join(merged[pkg] for pkg in sorted(merged)) + "\n"

    with tarfile.TarFile(output_tar, mode="w") as tout:
        _add_text_member(tout, _DPKG_STATUS_PATH, merged_text)
        for pkg_name in sorted(overlay_stanzas):
            files = _read_package_files(layers[pkg_name].layer)
            info_stem = f"{_DPKG_INFO_DIR}/{_dpkg_info_name(overlay_stanzas[pkg_name], pkg_name)}"
            _add_text_member(tout, _validate_safe_member_name(f"{info_stem}.list", _DPKG_INFO_DIR), files.list_text)
            if files.md5sums_text:
                md5sums_name = _validate_safe_member_name(f"{info_stem}.md5sums", _DPKG_INFO_DIR)
                _add_text_member(tout, md5sums_name, files.md5sums_text)
    logger.info("Wrote %s with file lists for %d overlay packages", output_tar, len(overlay_stanzas))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge overlay dpkg stanzas into a base image's dpkg status and record each overlay package's files"
    )
    parser.add_argument("oci_layout_dir", type=Path, help="OCI layout directory of the base image")
    parser.add_argument("overlay_tar", type=Path, help="rules_distroless dpkg_status tar of the overlay packages")
    parser.add_argument("output_tar", type=Path, help="Output tar for the merged status and the generated info files")
    parser.add_argument("data_layers", type=Path, nargs="*", help="Data layer of every overlay package")
    parser.add_argument("--repo", required=True, help="Name of the rules_distroless apt.install repository")
    parser.add_argument("--lock", type=Path, required=True, help="Lockfile of the apt.install repository")
    return parser.parse_args()


def main() -> None:
    """Merge base and overlay dpkg status files and record the files of every overlay package."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    args = _parse_args()

    logger.info("OCI layout dir: %s", args.oci_layout_dir)
    logger.info("Overlay tar:    %s", args.overlay_tar)
    logger.info("Output tar:     %s", args.output_tar)
    logger.info("Data layers:    %d from %s (lockfile %s)", len(args.data_layers), args.repo, args.lock)

    if not args.oci_layout_dir.is_dir():
        logger.error("OCI layout directory not found: %s", args.oci_layout_dir)
        sys.exit(1)
    if not (args.oci_layout_dir / "index.json").is_file():
        logger.error("Not a valid OCI layout (missing index.json): %s", args.oci_layout_dir)
        sys.exit(1)
    if not args.overlay_tar.is_file():
        logger.error("Overlay dpkg_status tar not found: %s", args.overlay_tar)
        sys.exit(1)
    if not args.lock.is_file():
        logger.error("Lockfile not found: %s", args.lock)
        sys.exit(1)

    try:
        merge(args.oci_layout_dir, args.overlay_tar, args.output_tar, args.repo, args.lock, args.data_layers)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except (tarfile.TarError, OSError):
        logger.exception("Failed to merge the dpkg status into %s", args.output_tar)
        sys.exit(1)


if __name__ == "__main__":
    main()
