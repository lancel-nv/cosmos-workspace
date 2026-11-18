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

"""Derive the wheel's LICENSE-3RD-PARTY.txt, attributing third-party code the way OpenCV itself does.

opencv-python ships one LICENSE-3RD-PARTY.txt with every wheel it publishes, so the file attributes libraries that only
some of those wheels contain (FFmpeg, Qt, Intel IPP, the macOS codecs, ...), omits third-party code that OpenCV's own
build attributes, and leaves out OpenCV's copyright notice. The derived file contains, in order:

- opencv-python's opening text, which carries OpenCV's Apache License 2.0, verbatim.
- OpenCV's copyright notice, read from the COPYRIGHT file of the pinned source tree. As in OpenCV's own distribution,
  its closing statement that third party copyrights are property of their respective owners covers the third-party code
  embedded in OpenCV's module sources that carries no separately installed license.
- opencv-python's sections for the code this build contains, verbatim.
- Every license file that OpenCV's build installs for this configuration (share/licenses/opencv4), except those for
  libraries a kept section already attributes.

Two checks turn an opencv-python upgrade into a loud failure instead of a silent change of attribution. Every section
heading in opencv-python's file must appear in KEPT_SECTIONS or DROPPED_SECTIONS, and every library named in the
"3rdparty dependencies" line of the build configuration embedded in the cv2 extension must be attributed by a kept
section or an installed license.

Usage (the env var keeps bytecode out of the directory the script sits in, which is a bind-mounted checkout):
    PYTHONDONTWRITEBYTECODE=1 python3 curate_third_party_licenses.py <opencv-python LICENSE-3RD-PARTY.txt> \
        <cv2 extension> <OpenCV COPYRIGHT file> <OpenCV installed licenses dir> <output>
"""

import dataclasses
from pathlib import Path
import re
import sys

SEPARATOR = "-" * 78

# A block opening with this prefix continues the preceding section: opencv-python lists the per-file licenses of a
# library's own dependencies under that library's heading.
_CONTINUATION_PREFIX = "Files: "

_SECTION_START_RE = re.compile(rf"^(?={re.escape(SEPARATOR)}$)", re.MULTILINE)
_BUILD_DEPENDENCIES_RE = re.compile(rb"3rdparty dependencies:[ \t]*([^\n]*)")

COPYRIGHT_HEADING = "OpenCV copyright notice, from the COPYRIGHT file of the OpenCV source tree."

# opencv-python's headings, verbatim, for the code this build contains. Each maps to the names under which the build
# refers to the library it attributes: its entry in the "3rdparty dependencies" build-configuration line and the prefix
# of the license files OpenCV installs for it. KAZE and AKAZE are compiled from OpenCV's own source tree and appear in
# neither.
KEPT_SECTIONS: dict[str, tuple[str, ...]] = {
    "libpng is redistributed within all opencv-python Linux and macOS packages.": ("libpng",),
    "libwebp and libwebpmux are redistributed within all opencv-python packages.": ("libwebp",),
    "Libjpeg-turbo is redistributed within all opencv-python packages as build option.": ("libjpeg-turbo",),
    "Flatbuffers library is redistributed within all opencv-python packages.": ("flatbuffers",),
    "Protobuf library is redistributed within all opencv-python packages.": ("libprotobuf", "protobuf"),
    "OpenJPEG library is redistributed within all opencv-python packages.": ("libopenjp2",),
    "TIFF library is redistributed within all opencv-python packages.": ("libtiff",),
    "OpenEXR library is redistributed within all opencv-python packages.": ("IlmImf", "openexr"),
    "KAZE Features library is redistributed within all opencv-python packages.": (),
    "AKAZE Features library is redistributed within all opencv-python packages.": (),
}

_KEPT_NAMES = frozenset(name for names in KEPT_SECTIONS.values() for name in names)

# opencv-python's headings, verbatim, for code this build does not contain.
DROPPED_SECTIONS = frozenset(
    {
        # The FFmpeg video backend and the libraries it pulls in; the build sets WITH_FFMPEG=OFF.
        "libvpx is redistributed within all opencv-python Linux packages.",
        "FFmpeg is redistributed within all opencv-python packages.",
        "bzip2 is redistributed within all opencv-python Linux packages.",
        "libcrypto and libssl are redistributed within all opencv-python Linux and macOS packages.",
        # The GUI stack of the non-headless packages and the FreeType text renderer, neither of which is linked.
        "Qt 5 is redistributed within non-headless opencv-python Linux and macOS packages.",
        "libfreetype is redistributed within opencv-python Linux and macOS packages.",
        "Libxcb utility libraries are redistributed within opencv-python non-headless Linux packages.",
        "Libxcb-image is redistributed within opencv-python non-headless Linux packages.",
        "Libxcb-util is redistributed within opencv-python non-headless Linux packages.",
        "Libxcb-render-util is redistributed within opencv-python non-headless Linux packages.",
        "Libxcb-icccm is redistributed within opencv-python non-headless Linux packages.",
        "libXau is redistributed within opencv-python non-headless Linux packages.",
        # Libraries of the macOS packages only. The OpenJPEG section above covers the libopenjp2 this build links.
        "libfontconfig is redistributed within all opencv-python macOS packages.",
        "libdav1d is redistributed within opencv-python macOS packages.",
        "libffi is redistributed within opencv-python macOS packages.",
        "libogg is redistributed within opencv-python macOS packages.",
        "libopenjp2 is redistributed within opencv-python macOS packages.",
        "libopus is redistributed within opencv-python macOS packages.",
        "librav1e is redistributed within opencv-python macOS packages.",
        "libsnappy is redistributed within opencv-python macOS packages.",
        "libspeex is redistributed within opencv-python macOS packages.",
        "libsrt is redistributed within opencv-python macOS packages.",
        "libtheoradec and libtheoraenc are redistributed within opencv-python macOS packages.",
        "libvorbis and libvorbisenc are redistributed within opencv-python macOS packages.",
        "Orbbec SDK distributed with arm64 MacOS packages.",
        # The cv2 extension loads zlib from the host system rather than bundling it.
        "libz is redistributed within all opencv-python Linux packages.",
        # Optional components the build leaves disabled: Vulkan, libspng (libpng is used instead), the quirc QR decoder
        # (OpenCV's own decoder is used), AVIF and the AV1 codec it depends on, and Intel IPP (WITH_IPP=OFF).
        "Vulkan headers are redistributed within all opencv-python packages.",
        "Libspng is redistributed within all opencv-python packages as build option.",
        "QUIRC library is redistributed within all opencv-python packages.",
        "libavif library and it's dependnecies are redistributed within all opencv-python packages.",
        "aom library and it's dependnecies are redistributed within all opencv-python packages.",
        "Intel(R) IPP ICV library statically linked within x86 and x86_64 opencv-python packages.",
    }
)


@dataclasses.dataclass(frozen=True)
class Section:
    """One section of a LICENSE-3RD-PARTY.txt: its heading and its verbatim text, opening separator included."""

    heading: str
    text: str


@dataclasses.dataclass(frozen=True)
class AddedSection:
    """A section that opencv-python's file lacks, with the name of the file its text was read from."""

    heading: str
    body: str
    source: str

    def render(self) -> str:
        """Return the section in opencv-python's layout: separator, heading, blank line, text."""
        body = self.body.strip("\n")
        return f"{SEPARATOR}\n{self.heading}\n\n{body}\n\n"

    def attributes(self, library: str) -> bool:
        """Return whether this section is a license OpenCV installs for ``library``."""
        return self.source.startswith(f"{library}-")


def split_sections(text: str) -> tuple[str, list[Section]]:
    """Split a LICENSE-3RD-PARTY.txt into the text before its first separator and its sections.

    A section runs from a separator line to the next one, and its heading is its first non-blank line. A block headed by
    a ``Files:`` line is folded into the preceding section.

    Raises:
        ValueError: if a block has no heading, or the first block is a continuation.
    """
    preamble, *blocks = _SECTION_START_RE.split(text)
    sections: list[Section] = []
    for block in blocks:
        heading = next((line.strip() for line in block.splitlines()[1:] if line.strip()), "")
        if not heading:
            raise ValueError("a section of the third-party license file has no heading")
        if heading.startswith(_CONTINUATION_PREFIX):
            if not sections:
                raise ValueError(f"{heading!r} continues no section")
            sections[-1] = Section(sections[-1].heading, sections[-1].text + block)
        else:
            sections.append(Section(heading, block))
    return preamble, sections


def build_dependencies(extension: bytes) -> list[str]:
    """Return the libraries named in the "3rdparty dependencies" line of the build configuration in ``extension``.

    Raises:
        ValueError: if the extension carries no such line.
    """
    match = _BUILD_DEPENDENCIES_RE.search(extension)
    if match is None:
        raise ValueError("the cv2 extension carries no '3rdparty dependencies' build-configuration line")
    return match.group(1).decode("ascii").split()


def copyright_section(copyright_file: Path) -> AddedSection:
    """Return OpenCV's copyright notice as a section."""
    return AddedSection(COPYRIGHT_HEADING, copyright_file.read_text(encoding="utf-8"), copyright_file.name)


def installed_license_sections(licenses_dir: Path) -> list[AddedSection]:
    """Return a section per license file OpenCV installed into ``licenses_dir`` that no kept section covers.

    OpenCV installs each library's license files as ``<library>-<file name>``, and installs them only for third-party
    code that the configured build compiles in.

    Raises:
        ValueError: if ``licenses_dir`` holds no license files.
    """
    files = sorted((path for path in licenses_dir.iterdir() if path.is_file()), key=lambda path: path.name.lower())
    if not files:
        raise ValueError(f"OpenCV installed no third-party license files into {licenses_dir}")
    return [
        AddedSection(
            f"{path.name}, which OpenCV installs for third-party code compiled into this build.",
            path.read_text(encoding="utf-8"),
            path.name,
        )
        for path in files
        if not any(path.name.startswith(f"{name}-") for name in _KEPT_NAMES)
    ]


def curate(
    upstream: str, dependencies: list[str], opencv_copyright: AddedSection, installed: list[AddedSection]
) -> str:
    """Return the third-party license file for this build.

    Raises:
        ValueError: if opencv-python's file has a heading that is neither kept nor dropped, lacks a kept section, or if
            a build dependency is attributed by neither a kept section nor an installed license.
    """
    preamble, sections = split_sections(upstream)
    headings = [section.heading for section in sections]
    unclassified = [heading for heading in headings if heading not in KEPT_SECTIONS and heading not in DROPPED_SECTIONS]
    if unclassified:
        raise ValueError(
            "opencv-python's LICENSE-3RD-PARTY.txt has sections that are neither kept nor dropped; add each to "
            f"KEPT_SECTIONS or DROPPED_SECTIONS: {unclassified}"
        )
    missing = [heading for heading in KEPT_SECTIONS if heading not in headings]
    if missing:
        raise ValueError(f"opencv-python's LICENSE-3RD-PARTY.txt lacks kept sections: {missing}")
    unattributed = [
        dependency
        for dependency in dependencies
        if dependency not in _KEPT_NAMES and not any(section.attributes(dependency) for section in installed)
    ]
    if unattributed:
        raise ValueError(
            "the cv2 build configuration names libraries that neither a kept section nor an installed license "
            f"attributes; map each in KEPT_SECTIONS: {unattributed}"
        )
    kept = "".join(section.text for section in sections if section.heading in KEPT_SECTIONS)
    appended = "".join(section.render() for section in installed)
    return (preamble + opencv_copyright.render() + kept + appended).rstrip("\n") + "\n"


def main() -> None:
    """Write the third-party license file for the build named on the command line."""
    if len(sys.argv) != 6:
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} <opencv-python LICENSE-3RD-PARTY.txt> <cv2 extension> "
            "<OpenCV COPYRIGHT file> <OpenCV installed licenses dir> <output>"
        )

    upstream, extension, copyright_file, licenses_dir, output = (Path(argument) for argument in sys.argv[1:6])
    dependencies = build_dependencies(extension.read_bytes())
    installed = installed_license_sections(licenses_dir)
    curated = curate(upstream.read_text(encoding="utf-8"), dependencies, copyright_section(copyright_file), installed)
    output.write_text(curated, encoding="utf-8")
    print(
        f"OK: {output} carries OpenCV's copyright notice, keeps {len(KEPT_SECTIONS)} of opencv-python's sections, and "
        f"adds {len(installed)} licenses OpenCV installs ({', '.join(section.source for section in installed)})."
    )


if __name__ == "__main__":
    main()
