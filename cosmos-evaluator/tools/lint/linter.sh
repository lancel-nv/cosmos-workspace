#!/usr/bin/env bash
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
#
# Linting workflow for Bazel targets.
#
# This script runs on the host and calls dazel commands to enter the container.
# See https://docs.aspect.build/cli/commands/aspect_lint
#
# By default, shows violations without applying patches. Use --apply-patches to fix violations.
#
# Usage:
#   linter.sh [--fail-on-violation] [--apply-patches] [--output-base=PATH] [TARGETS...]
#
# Compatible with both bash and zsh.
#

# Enable zsh compatibility options if running in zsh
if [[ -n "${ZSH_VERSION:-}" ]]; then
    setopt SH_WORD_SPLIT  # Enable word splitting like bash
    setopt KSH_ARRAYS     # Use 0-based array indexing like bash
fi

set -euo pipefail

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Separator line used as a header/section divider in output
SEPARATOR='========================================================================'

# Default options
APPLY_PATCHES=false
ASPECT_PREFIX="//"
BUILDIFIER=true
GAZELLE=true
FAIL_ON_VIOLATION=false
MYPY=true
OUTPUT_BASE=""
TARGETS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --apply-patches)
            APPLY_PATCHES=true
            shift
            ;;
        --aspect-prefix)
            # Reject missing value or another flag as the value (would silently consume the next flag otherwise, since
            # `shift 2` cannot tell a value apart from a flag-shaped string).
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "ERROR: --aspect-prefix requires a non-flag value (got '${2:-<missing>}')" >&2
                exit 1
            fi
            ASPECT_PREFIX="$2"
            # Must be a Bazel label prefix (i.e. end with '//'), because it is composed directly into labels like
            # "${ASPECT_PREFIX}tools/lint:linters.bzl%ruff_aspect" below. The empty-string case is also caught here
            # since "" does not end with '//'.
            if [[ "$ASPECT_PREFIX" != *// ]]; then
                echo "ERROR: --aspect-prefix requires a Bazel label prefix ending with '//' (got '$ASPECT_PREFIX')" >&2
                exit 1
            fi
            shift 2
            ;;
        --aspect-prefix=*)
            ASPECT_PREFIX="${1#*=}"
            if [[ "$ASPECT_PREFIX" != *// ]]; then
                echo "ERROR: --aspect-prefix requires a Bazel label prefix ending with '//' (got '$ASPECT_PREFIX')" >&2
                exit 1
            fi
            shift
            ;;
        --fail-on-violation)
            FAIL_ON_VIOLATION=true
            shift
            ;;
        --no-buildifier)
            BUILDIFIER=false
            shift
            ;;
        --no-gazelle)
            GAZELLE=false
            shift
            ;;
        --no-mypy)
            MYPY=false
            shift
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --output-base=*)
            OUTPUT_BASE="${1#*=}"
            shift
            ;;
        --help|-h)
            cat << EOF
Usage: linter.sh [OPTIONS] [TARGETS...]

Run linters on Bazel targets

Options:
  --apply-patches           Auto-fix violations by applying patches (default: false)
  --aspect-prefix=PREFIX    Bazel label prefix used to load the ruff/mypy aspects (default: //).
  --fail-on-violation       Fail the build when a linter warning is present (default: false)
  --no-buildifier           Disable buildifier linting (default: false)
  --no-gazelle              Disable Gazelle BUILD dependency checks (default: false)
  --no-mypy                 Disable mypy type checking (default: false)
  --output-base=PATH        Set the output_base directory for dazel/bazel (default: none)

Arguments:
  TARGETS                   Bazel target patterns to lint (e.g., //... or //checks/...) (default: //...)

Examples:
  linter.sh                              # Check for violations (default)
  linter.sh //checks/...                 # Check specific targets
  linter.sh --apply-patches              # Fix violations in everything
  linter.sh --apply-patches //checks/... # Fix violations in specific targets
  linter.sh --fail-on-violation          # Check and fail if violations exist
  linter.sh --no-mypy                    # Run only ruff and buildifier (skip mypy)
  linter.sh --no-buildifier              # Run only ruff and mypy (skip buildifier)
  linter.sh --no-gazelle                 # Run without Gazelle BUILD checks
EOF
            exit 0
            ;;
        -*)
            echo "Error: Unknown option: $1" >&2
            echo "Use --help for usage information" >&2
            exit 1
            ;;
        *)
            TARGETS+=("$1")
            shift
            ;;
    esac
done

# Default to //... if no targets specified
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=("//...")
fi

build_dazel_cmd_array() {
    # Build dazel command array with optional output_base
    # Sets DAZEL_CMD_RESULT array (portable across bash/zsh)
    DAZEL_CMD_RESULT=()  # Clear any previous content

    # Use dazel command if available (e.g., from sourced env_setup.sh in bash)
    # Otherwise fall back to calling dazel.py directly (needed for zsh or unsourced shells)
    if command -v dazel &>/dev/null; then
        DAZEL_CMD_RESULT+=("dazel")
    else
        # Get the path to dazel.py relative to this script
        local script_dir
        script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
        DAZEL_CMD_RESULT+=("$script_dir/../dazel/dazel.py")
    fi

    if [[ -n "$OUTPUT_BASE" ]]; then
        DAZEL_CMD_RESULT+=("--output_base=$OUTPUT_BASE")
    fi

    return 0
}

find_workspace_root() {
    # Find workspace root using dazel
    local workspace
    build_dazel_cmd_array
    local -a dazel_cmd=("${DAZEL_CMD_RESULT[@]}")
    if workspace="$("${dazel_cmd[@]}" info workspace 2>/dev/null)" && [[ -n "$workspace" ]]; then
        echo "$workspace"
        return
    fi

    # Try git if dazel info fails
    if git rev-parse --show-toplevel &>/dev/null; then
        echo "$(git rev-parse --show-toplevel)"
        return
    fi

    echo "Error: Could not find workspace root" >&2
    exit 1
}

WORKSPACE_ROOT="$(find_workspace_root)"

echo "Workspace: $WORKSPACE_ROOT"
echo ""


# Resolve the Bazel label prefix used to reach cosmos_eval's shared lint tooling (buildifier, aspects, etc.). The same
# script serves both the module that owns these targets and any consumer that vendors it, so the prefix depends on the
# workspace it runs in: targets are self-referenced with a bare "//" from inside the owning module, but must be reached
# as an external repo ("@cosmos_eval//") from a consumer. The two are told apart by the workspace-root MODULE.bazel's
# module name (a name of "cosmos_eval" means we're running standalone; anything else means we're a consumer).
detect_cosmos_eval_label_prefix() {
    local module_file="$WORKSPACE_ROOT/MODULE.bazel"
    if [[ -f "$module_file" ]]; then
        local module_name
        module_name=$(awk '
            /^module[[:space:]]*\(/ { in_module = 1 }
            in_module && /name[[:space:]]*=/ {
                if (match($0, /"[^"]*"/)) {
                    print substr($0, RSTART + 1, RLENGTH - 2)
                    exit
                }
            }
        ' "$module_file")
        if [[ "$module_name" == "cosmos_eval" ]]; then
            echo "//"
            return
        fi
    fi
    echo "@cosmos_eval//"
}

COSMOS_EVAL_LABEL_PREFIX="$(detect_cosmos_eval_label_prefix)"

# Map Bazel target patterns to filesystem directories for Gazelle. Returns via GAZELLE_DIR_ARGS array; empty means full
# recursive update from the workspace root (//...).
build_gazelle_dir_args() {
    GAZELLE_DIR_ARGS=()
    for target in "${TARGETS[@]}"; do
        if [[ "$target" == "//..." ]]; then
            GAZELLE_DIR_ARGS=()
            return 0
        fi
        local pkg="${target#//}"
        pkg="${pkg%%:*}"
        pkg="${pkg%/...}"
        if [[ -n "$pkg" ]]; then
            GAZELLE_DIR_ARGS+=("$pkg")
        fi
    done
}

build_gazelle_dir_args

# Create temporary file for build events in workspace root
BUILD_EVENTS_FILE=$(mktemp --tmpdir="$WORKSPACE_ROOT" --suffix=.json)

# Build bazel command
BAZEL_ARGS=(
    "build"
    "--build_event_json_file=$BUILD_EVENTS_FILE"
    "--keep_going"
    "--remote_download_toplevel"
)

# Aspect labels are prefixed by $ASPECT_PREFIX (default //), so they load tools/lint:linters.bzl from the current
# workspace. Both a standalone checkout and a consumer that defines its own //tools/lint:linters.bzl get the right
# aspects for free. A consumer that instead wants to load the aspects from the module that provides them (e.g. to lint
# that module's vendored targets with its own pyproject.toml) can point at it via --aspect-prefix=<module>//.
ASPECTS=("${ASPECT_PREFIX}tools/lint:linters.bzl%ruff_aspect")
OUTPUT_GROUPS=("rules_lint_human")
if [[ "$MYPY" == "true" ]]; then
    ASPECTS+=("${ASPECT_PREFIX}tools/lint:linters.bzl%mypy_aspect")
    OUTPUT_GROUPS+=("+mypy")
fi

# Configure patch generation based on mode
if [[ "$APPLY_PATCHES" == "true" ]]; then
    # Generate and apply patches
    BAZEL_ARGS+=("--@aspect_rules_lint//lint:fix")
    OUTPUT_GROUPS+=("rules_lint_patch")
fi

# Join arrays with commas
ASPECTS_STR=$(IFS=, ; echo "${ASPECTS[*]}")
BAZEL_ARGS+=("--aspects=${ASPECTS_STR}")

OUTPUT_GROUPS_STR=$(IFS=, ; echo "${OUTPUT_GROUPS[*]}")
BAZEL_ARGS+=("--output_groups=${OUTPUT_GROUPS_STR}")

# Add targets
BAZEL_ARGS+=("${TARGETS[@]}")

# Build dazel command array
build_dazel_cmd_array
DAZEL_CMD=("${DAZEL_CMD_RESULT[@]}")

# Run bazel build
echo "Running: ${DAZEL_CMD[*]} ${BAZEL_ARGS[*]}"
echo ""

# Create temporary file for bazel output in workspace root
BAZEL_OUTPUT_FILE=$(mktemp --tmpdir="$WORKSPACE_ROOT" --suffix=.txt)
trap 'rm -f "$BUILD_EVENTS_FILE" "$BAZEL_OUTPUT_FILE"' EXIT

# Run bazel and capture output and exit code
set +e
"${DAZEL_CMD[@]}" "${BAZEL_ARGS[@]}" 2>&1 | tee "$BAZEL_OUTPUT_FILE"
BAZEL_EXIT_CODE=$?
set -e

BUILDIFIER_EXIT_CODE=0
GAZELLE_EXIT_CODE=0
if [[ "$GAZELLE" == "true" && "$APPLY_PATCHES" != "true" ]]; then
    set +e
    GAZELLE_CMD=(run //:gazelle -- update -mode diff)
    if [[ ${#GAZELLE_DIR_ARGS[@]} -gt 0 ]]; then
        GAZELLE_CMD+=("${GAZELLE_DIR_ARGS[@]}")
    fi
    "${DAZEL_CMD[@]}" "${GAZELLE_CMD[@]}" 2>&1 | tee -a "$BAZEL_OUTPUT_FILE"
    GAZELLE_EXIT_CODE=$?
    if [[ $GAZELLE_EXIT_CODE -eq 0 ]] && grep -q 'gazelle: ERROR' "$BAZEL_OUTPUT_FILE"; then
        GAZELLE_EXIT_CODE=1
    fi
    set -e
fi
if [[ "$BUILDIFIER" == "true" ]]; then
    set +e
    "${DAZEL_CMD[@]}" run "${COSMOS_EVAL_LABEL_PREFIX}tools/lint:buildifier.check" 2>&1 | tee -a "$BAZEL_OUTPUT_FILE"
    BUILDIFIER_EXIT_CODE=$?
    set -e
    # buildifier exits non-zero (e.g. 123) on ANY formatting/lint violation. Capture it as the authoritative pass/fail
    # signal (pipefail is set, and tee succeeds, so $? is buildifier.check's exit). The regex-based parsing below is
    # only for *displaying* and *counting* violations; new buildifier warning shapes (such as the
    # "load: Loaded symbol X is unused" lint warning, which is emitted as "<path>:<line>:" with no leading "./")
    # can slip past that regex. The exit code never does, so it is folded into TOTAL_VIOLATIONS below as a backstop.
fi

# Show linting results
echo ""
echo "$SEPARATOR"
echo "RUFF RESULTS"
echo "$SEPARATOR"
echo ""

TOTAL_VIOLATIONS=0
# Process ruff violations
if ! command -v jq &> /dev/null; then
    # Check if jq is available
    echo "Warning: jq not found, cannot extract report files" >&2
    echo "Install jq to see detailed reports: apt-get install jq" >&2
    exit 1
else
    # Parse build events to find .out files
    REPORT_FILES=$(jq -r '
        select(.namedSetOfFiles.files != null) |
        .namedSetOfFiles.files[] |
        select(.name | endswith(".out")) |
        select(.name | contains("AspectRulesLint")) |
        if .pathPrefix then
            (.pathPrefix | join("/")) + "/" + .name
        else
            .name
        end
    ' "$BUILD_EVENTS_FILE" 2>/dev/null | sort -u || true)

    # Process .out files
    FILES_WITH_RUFF_VIOLATIONS=0
    RUFF_VIOLATIONS=0

    while IFS= read -r report_file; do
        [[ -z "$report_file" ]] && continue
        full_path="$WORKSPACE_ROOT/$report_file"
        if [[ -f "$full_path" && -s "$full_path" ]]; then
            content=$(cat "$full_path")

            # Skip if "All checks passed!"
            if echo "$content" | grep -q "All checks passed!"; then
                continue
            fi

            # Extract source file from report path
            source_file=$(echo "$report_file" | sed -E 's/\.AspectRulesLint.*\.out$/.py/' | sed -E 's|.*/bin/||')

            # Count violations by counting source-location pointer lines, which ruff's "full" output format emits
            # exactly once per violation (e.g. "   --> path/to/file.py:12:34"). This is invariant across header styles
            # ("error[CODE]: ..." vs "CODE ..."), so it works for both the CLI ruff output and the aspect-rules-lint
            # .out report (which does not include a "Found N errors" summary). Strip ANSI color sequences first so we
            # can match the bare "-->" with POSIX-friendly `grep -E` instead of PCRE (`-P`), which isn't available on
            # BSD/macOS grep.
            clean_content=$(printf '%s' "$content" | sed $'s/\033\\[[0-9;]*m//g')
            violation_count=$(printf '%s\n' "$clean_content" | grep -cE '^[[:space:]]*-->[[:space:]]' || true)
            if [[ -z "$violation_count" || "$violation_count" -eq 0 ]]; then
                violation_count=1
            fi

            FILES_WITH_RUFF_VIOLATIONS=$((FILES_WITH_RUFF_VIOLATIONS + 1))
            RUFF_VIOLATIONS=$((RUFF_VIOLATIONS + violation_count))

            echo "$source_file: $violation_count violation(s)"

            # Always show violation content
            echo "$content"
            echo ""
        fi
    done <<< "$REPORT_FILES"

    if [[ $FILES_WITH_RUFF_VIOLATIONS -eq 0 ]]; then
        echo "✓ No ruff violations found!"
    else
        echo ""
        echo "Total: $FILES_WITH_RUFF_VIOLATIONS file(s) with $RUFF_VIOLATIONS violation(s)"
        echo ""
        # Add to total violations count
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + RUFF_VIOLATIONS))
    fi
fi

# Process mypy violations
FILES_WITH_MYPY_VIOLATIONS=0
MYPY_VIOLATIONS=0

if [[ "$MYPY" == "true" ]]; then
    echo ""
    echo "$SEPARATOR"
    echo "MYPY RESULTS"
    echo "$SEPARATOR"
    echo ""

    # Extract mypy errors from bazel output
    # Pattern: "filepath.py:line: error: message [error-code]"
    MYPY_ERRORS=$(grep -E '^[a-zA-Z0-9_/.-]+\.py:[0-9]+: error:' "$BAZEL_OUTPUT_FILE" || true)

    if [[ -z "$MYPY_ERRORS" ]]; then
        echo "✓ No mypy violations found!"
    else
        # Count unique files with errors
        FILES_WITH_MYPY_VIOLATIONS=$(echo "$MYPY_ERRORS" | cut -d: -f1 | sort -u | wc -l)

        # Count total errors
        MYPY_VIOLATIONS=$(echo "$MYPY_ERRORS" | wc -l)

        # Display errors grouped by file
        while IFS= read -r file; do
            [[ -z "$file" ]] && continue

            file_errors=$(echo "$MYPY_ERRORS" | grep "^$file:")
            error_count=$(echo "$file_errors" | wc -l)

            echo "$file: $error_count error(s)" >&2
            echo "$file_errors" >&2
            echo ""
        done < <(echo "$MYPY_ERRORS" | cut -d: -f1 | sort -u)

        echo "Total: $FILES_WITH_MYPY_VIOLATIONS file(s) with $MYPY_VIOLATIONS error(s)" >&2
        echo ""

        # Add to total violations count
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + MYPY_VIOLATIONS))
    fi
fi

# Process buildifier violations
BUILDIFIER_VIOLATION_COUNT=0
FILES_WITH_BUILDIFIER_VIOLATIONS=0
GAZELLE_VIOLATION_COUNT=0
FILES_WITH_GAZELLE_VIOLATIONS=0
if [[ "$BUILDIFIER" == "true" ]]; then
    echo ""
    echo "$SEPARATOR"
    echo "BUILDIFIER RESULTS"
    echo "$SEPARATOR"
    echo ""

    # Extract buildifier errors from bazel output. buildifier (-lint=warn -mode=check) emits two line shapes:
    #   Format 1: <path>:<line>: <category>: <description>   (lint warnings, e.g. "load: Loaded symbol X is unused")
    #   Format 2: <path> # reformat                          (formatting differences)
    # The leading "./" is NOT guaranteed so we make it optional.
    LINE_WARNING_RE='^(\./)?[^: ]*(BUILD[^:/ ]*|\.(bzl|bazel|sky)):[0-9]+: '
    REFORMAT_RE='^[a-zA-Z0-9_/.-]+(BUILD[^ ]*|\.(bzl|bazel|sky)) # reformat'
    BUILDIFIER_VIOLATIONS=$(grep -E "$LINE_WARNING_RE|$REFORMAT_RE" "$BAZEL_OUTPUT_FILE" || true)

    if [[ -n "$BUILDIFIER_VIOLATIONS" ]]; then
        # Count both format types (line-number warnings and reformat violations)
        BUILDIFIER_VIOLATION_COUNT=$(echo "$BUILDIFIER_VIOLATIONS" | wc -l)
        # For files with violations, count unique files from both formats
        DETAILED_VIOLATIONS=$(echo "$BUILDIFIER_VIOLATIONS" | grep -E "$LINE_WARNING_RE" || true)
        REFORMAT_VIOLATIONS=$(echo "$BUILDIFIER_VIOLATIONS" | grep -E "$REFORMAT_RE" || true)

        FILES_COUNT=0
        if [[ -n "$DETAILED_VIOLATIONS" ]]; then
            FILES_COUNT=$(echo "$DETAILED_VIOLATIONS" | cut -d: -f1 | sort -u | wc -l)
        fi
        if [[ -n "$REFORMAT_VIOLATIONS" ]]; then
            REFORMAT_FILES=$(echo "$REFORMAT_VIOLATIONS" | sed 's/ # reformat$//' | wc -l)
            FILES_COUNT=$((FILES_COUNT + REFORMAT_FILES))
        fi
        FILES_WITH_BUILDIFIER_VIOLATIONS=$FILES_COUNT
    fi

    if [[ -n "$BUILDIFIER_VIOLATIONS" ]]; then
        echo "Total: $FILES_WITH_BUILDIFIER_VIOLATIONS file(s) with $BUILDIFIER_VIOLATION_COUNT violation(s)"
        echo "$BUILDIFIER_VIOLATIONS"

        # Add to total violations count
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + BUILDIFIER_VIOLATION_COUNT))
    elif [[ $BUILDIFIER_EXIT_CODE -ne 0 ]]; then
        # buildifier reported a failure that the parsing above didn't recognize as a known line shape. Never let a
        # real buildifier failure pass silently: surface it and count it so TOTAL_VIOLATIONS (and the exit logic)
        # reflect it. The raw buildifier output is in the bazel log streamed above.
        echo "✗ buildifier reported violations (exit code $BUILDIFIER_EXIT_CODE) not matched by the summary parser."
        echo "  See the buildifier output above; run 'dazel fix' to apply available fixes."
        FILES_WITH_BUILDIFIER_VIOLATIONS=1
        BUILDIFIER_VIOLATION_COUNT=1
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + 1))
    else
        echo "✓ No buildifier violations found!"
    fi
fi

# Gazelle violation processing and the overall SUMMARY are emitted *after* the apply-patches block below. In fix mode
# `gazelle update` must run after ruff/mypy patches are applied (so it computes BUILD deps from the post-fix import
# set); deferring the summary lets that run's result be counted alongside ruff/mypy/buildifier instead of being
# reported after a summary that already claimed success.

# Apply patches if requested
if [[ "$APPLY_PATCHES" == "true" ]]; then
    echo ""
    echo "$SEPARATOR"
    echo "APPLYING AUTO-FIX PATCHES"
    echo "$SEPARATOR"
    echo ""

    # Extract patch files from build events using jq
    if ! command -v jq &> /dev/null; then
        echo "Warning: jq not found, cannot extract patch files" >&2
        echo "Install jq to enable auto-fix: apt-get install jq" >&2
        exit 1
    fi

    # Parse build events to find .patch files
    ALL_PATCH_FILES=$(jq -r '
        select(.namedSetOfFiles.files != null) |
        .namedSetOfFiles.files[] |
        select(.name | endswith(".patch")) |
        if .pathPrefix then
            (.pathPrefix | join("/")) + "/" + .name
        else
            .name
        end
    ' "$BUILD_EVENTS_FILE" 2>/dev/null | sort -u || true)

    # Filter out empty patch files
    NON_EMPTY_PATCHES=()
    while IFS= read -r patch_file; do
        [[ -z "$patch_file" ]] && continue
        full_path="$WORKSPACE_ROOT/$patch_file"
        if [[ -f "$full_path" && -s "$full_path" ]]; then
            NON_EMPTY_PATCHES+=("$patch_file")
        fi
    done <<< "$ALL_PATCH_FILES"

    if [[ ${#NON_EMPTY_PATCHES[@]} -eq 0 ]]; then
        echo "No patches to apply."
    else
        PATCH_COUNT=${#NON_EMPTY_PATCHES[@]}

        echo "Found $PATCH_COUNT non-empty patch(es):"
        printf '  - %s\n' "${NON_EMPTY_PATCHES[@]}"
        echo ""

        # Apply the patches
        echo "Applying $PATCH_COUNT patch(es)..."
        ERRORS=0
        APPLIED=0
        for patch_file in "${NON_EMPTY_PATCHES[@]}"; do
            full_path="$WORKSPACE_ROOT/$patch_file"
            if patch -p1 -d "$WORKSPACE_ROOT" < "$full_path" 2>&1; then
                APPLIED=$((APPLIED + 1))
                echo "✓ Applied $(basename "$patch_file")"
            else
                echo "✗ Failed to apply $(basename "$patch_file")" >&2
                ERRORS=$((ERRORS + 1))
            fi
        done

        if [[ $ERRORS -gt 0 ]]; then
            echo "" >&2
            echo "✗ $ERRORS patch(es) failed to apply." >&2
            exit 1
        else
            echo "✓ Successfully applied all $APPLIED patch(es)!"
        fi
    fi

    if [[ "$GAZELLE" == "true" ]]; then
        set +e
        GAZELLE_CMD=(run //:gazelle -- update)
        if [[ ${#GAZELLE_DIR_ARGS[@]} -gt 0 ]]; then
            GAZELLE_CMD+=("${GAZELLE_DIR_ARGS[@]}")
        fi
        "${DAZEL_CMD[@]}" "${GAZELLE_CMD[@]}" 2>&1 | tee -a "$BAZEL_OUTPUT_FILE"
        GAZELLE_EXIT_CODE=$?
        # `gazelle update` can exit 0 while still printing `gazelle: ERROR:` lines for targets it could not resolve
        # (ambiguous/unresolved deps); treat those as failures too so they are counted below rather than silently
        # applied-and-passed. The GAZELLE RESULTS + SUMMARY blocks after this loop report and count the outcome.
        if [[ $GAZELLE_EXIT_CODE -eq 0 ]] && grep -q 'gazelle: ERROR' "$BAZEL_OUTPUT_FILE"; then
            GAZELLE_EXIT_CODE=1
        fi
        set -e
        if [[ $GAZELLE_EXIT_CODE -eq 0 ]]; then
            echo "✓ Successfully ran Gazelle update"
        fi
    fi

    if [[ "$BUILDIFIER" == "true" ]]; then
        set +e
        "${DAZEL_CMD[@]}" run "${COSMOS_EVAL_LABEL_PREFIX}tools/lint:buildifier"
        BUILDIFIER_FIX_EXIT=$?
        set -e
        if [[ $BUILDIFIER_FIX_EXIT -ne 0 ]]; then
            echo "✗ Failed to run buildifier" >&2
            BAZEL_EXIT_CODE=$BUILDIFIER_FIX_EXIT
            exit 1
        else
            echo "✓ Successfully ran buildifier"
        fi
    fi
fi

# Process Gazelle violations (both check and fix mode). GAZELLE_EXIT_CODE is set either by the `update -mode diff`
# run in check mode (stale BUILD metadata detection) or by the `gazelle update` in the apply block above in fix mode
# (a non-zero exit there is an unfixable error — ambiguous/unresolved deps — that `update` could not auto-apply).
# Counting it into TOTAL_VIOLATIONS keeps the SUMMARY (and the two-pass wrapper that parses it) honest.
if [[ "$GAZELLE" == "true" ]]; then
    echo ""
    echo "$SEPARATOR"
    echo "GAZELLE RESULTS"
    echo "$SEPARATOR"
    echo ""

    if [[ $GAZELLE_EXIT_CODE -ne 0 ]]; then
        # Gazelle reports each unresolvable/ambiguous import as a block keyed by '"<file>", line <N>:'. A single bad
        # import fans out into one line per ancestor-module prefix (checks, checks.utils, ...) plus a paired
        # "multiple targets" line, so dedupe by (file, line) to count import statements — not the symptom lines that
        # a naive `grep -c 'invalid dependency'` (or the old hard-coded 1) would over-/under-count.
        GAZELLE_PROBLEM_KEYS=$(grep -oE '^"[^"]+", line [0-9]+:' "$BAZEL_OUTPUT_FILE" 2>/dev/null | sort -u || true)
        GAZELLE_PROBLEM_COUNT=0
        GAZELLE_PROBLEM_FILES=0
        if [[ -n "$GAZELLE_PROBLEM_KEYS" ]]; then
            GAZELLE_PROBLEM_COUNT=$(printf '%s\n' "$GAZELLE_PROBLEM_KEYS" | grep -c .)
            GAZELLE_PROBLEM_FILES=$(printf '%s\n' "$GAZELLE_PROBLEM_KEYS" | sed -E 's/", line [0-9]+:$//; s/^"//' | sort -u | grep -c .)
        fi
        # Stale-but-auto-fixable BUILD metadata surfaces as a diff (check mode) rather than a `gazelle: ERROR` line.
        GAZELLE_DIFF_FILES=$(grep -E '^diff |^--- ' "$BAZEL_OUTPUT_FILE" 2>/dev/null | grep -E 'BUILD(\.bazel)?' | sed -E 's/^diff .* b\/(.*)$/\\1/; s/^--- .*\/(.*BUILD(\.bazel)?).*$/\\1/' | sort -u || true)
        GAZELLE_DIFF_COUNT=0
        if [[ -n "$GAZELLE_DIFF_FILES" ]]; then
            GAZELLE_DIFF_COUNT=$(printf '%s\n' "$GAZELLE_DIFF_FILES" | grep -c .)
        fi

        GAZELLE_VIOLATION_COUNT=$((GAZELLE_PROBLEM_COUNT + GAZELLE_DIFF_COUNT))
        FILES_WITH_GAZELLE_VIOLATIONS=$((GAZELLE_PROBLEM_FILES + GAZELLE_DIFF_COUNT))
        if [[ $GAZELLE_VIOLATION_COUNT -eq 0 ]]; then
            # Non-zero exit with nothing parseable (unexpected); still surface as a single violation so it is not lost.
            GAZELLE_VIOLATION_COUNT=1
            FILES_WITH_GAZELLE_VIOLATIONS=1
        fi

        if [[ $GAZELLE_PROBLEM_COUNT -gt 0 ]]; then
            echo "✗ Gazelle could not resolve the BUILD dependency graph (see 'gazelle: ERROR' output above)." >&2
            echo "  Not auto-fixable: split multi-file targets, add a '# gazelle:resolve' directive, or fix the imports."
        fi
        if [[ $GAZELLE_DIFF_COUNT -gt 0 ]]; then
            echo "  $GAZELLE_DIFF_COUNT file(s) with stale BUILD dependency metadata — run 'dazel fix' to refresh."
        fi
        # Distinct "Gazelle total:" wording (not "Total:") so the two-pass wrapper's ^Total: parser ignores this line.
        echo "Gazelle total: $FILES_WITH_GAZELLE_VIOLATIONS file(s) with $GAZELLE_VIOLATION_COUNT problem(s)"
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + GAZELLE_VIOLATION_COUNT))
    else
        echo "✓ No Gazelle violations found!"
    fi
fi

# Overall summary
echo ""
echo "$SEPARATOR"
echo "SUMMARY"
echo "$SEPARATOR"
echo ""

TOTAL_FILES_WITH_VIOLATIONS=$((FILES_WITH_RUFF_VIOLATIONS + FILES_WITH_MYPY_VIOLATIONS + FILES_WITH_BUILDIFIER_VIOLATIONS + FILES_WITH_GAZELLE_VIOLATIONS))

if [[ $TOTAL_VIOLATIONS -eq 0 ]]; then
    printf '%b✓ All linting checks passed!%b\n' "${GREEN}" "${NC}"
else
    printf '%bTotal: %d file(s) with %d violation(s)%b\n' "${RED}" "$TOTAL_FILES_WITH_VIOLATIONS" "$TOTAL_VIOLATIONS" "${NC}"
    if [[ $RUFF_VIOLATIONS -gt 0 ]]; then
        printf '%b  - ruff: %d file(s), %d violation(s)%b\n' "${RED}" "$FILES_WITH_RUFF_VIOLATIONS" "$RUFF_VIOLATIONS" "${NC}"
    fi
    if [[ "$MYPY" == "true" && $MYPY_VIOLATIONS -gt 0 ]]; then
        printf '%b  - mypy: %d file(s), %d violation(s)%b\n' "${RED}" "$FILES_WITH_MYPY_VIOLATIONS" "$MYPY_VIOLATIONS" "${NC}"
    fi
    if [[ "$BUILDIFIER" == "true" && $BUILDIFIER_VIOLATION_COUNT -gt 0 ]]; then
        printf '%b  - buildifier: %d file(s), %d violation(s)%b\n' "${RED}" "$FILES_WITH_BUILDIFIER_VIOLATIONS" "$BUILDIFIER_VIOLATION_COUNT" "${NC}"
    fi
    if [[ "$GAZELLE" == "true" && $GAZELLE_VIOLATION_COUNT -gt 0 ]]; then
        printf '%b  - gazelle: %d file(s), %d violation(s)%b\n' "${RED}" "$FILES_WITH_GAZELLE_VIOLATIONS" "$GAZELLE_VIOLATION_COUNT" "${NC}"
    fi
fi
echo ""

# A Gazelle failure in fix mode is unfixable by construction (`gazelle update` already ran), so fail the lint even
# though fix mode otherwise tolerates residual (auto-fixable) violations. The violation is already reflected in the
# SUMMARY above; this only propagates the non-zero exit that the pre-refactor gazelle-update step used to emit.
if [[ "$APPLY_PATCHES" == "true" && "$GAZELLE" == "true" && $GAZELLE_EXIT_CODE -ne 0 ]]; then
    echo "✗ Failed to run Gazelle update" >&2
    exit 1
fi

# Exit with error only if FAIL_ON_VIOLATION is true and there were build failures or violations. BUILDIFIER_EXIT_CODE
# is included explicitly (in addition to feeding TOTAL_VIOLATIONS above) so a buildifier failure always fails the lint
# even if a future refactor changes how its violations are counted.
if [[ "$FAIL_ON_VIOLATION" == "true" && ( $BAZEL_EXIT_CODE -ne 0 || $BUILDIFIER_EXIT_CODE -ne 0 || $GAZELLE_EXIT_CODE -ne 0 || $TOTAL_VIOLATIONS -gt 0 ) ]]; then
    exit 1
fi

exit 0
