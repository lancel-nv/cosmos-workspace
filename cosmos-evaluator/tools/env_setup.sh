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

# Cosmos Evaluator build-environment setup.
#
# Sourced from a developer shell or automated build to install host requirements (Python, Docker, git, NVIDIA
# driver/toolkit), set up the dazel CLI + tab completion, and verify the dazel container can access the GPU.
#
# Supports both bash and zsh shells.
#
# Environment variables (optional):
#   COSMOS_EVAL_NO_BANNER  When set to a non-empty value, suppresses the trailing "Cosmos Evaluator build environment
#                          configured" help banner. The host-requirement check messages and warnings/errors are still
#                          printed. Useful for automated builds, quiet sourcing, or when this script is sourced from
#                          another setup script that prints its own banner.

GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
NC='\033[0m' # No Color

# Minimum recommended versions
DOCKER_MIN_VERSION="28.1.1"
GIT_MIN_VERSION="2.49.0"
NV_DRIVER_MIN_VERSION="570.124.06"
NV_CONTAINER_TOOLKIT_MIN_VERSION="1.16.2"
PYTHON_MIN_VERSION="3.10.0"

# Detect which shell is being used
if [[ -n "$ZSH_VERSION" ]]; then
    _IS_ZSH=true
    _IS_BASH=false
elif [[ -n "$BASH_VERSION" ]]; then
    _IS_ZSH=false
    _IS_BASH=true
else
    _IS_ZSH=false
    _IS_BASH=false
fi

# Locate this script (works for both bash and zsh, sourced or executed).
if $_IS_BASH; then
    _SCRIPT_SOURCE=${BASH_SOURCE[0]}

    if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
        echo "ERROR: This script should be sourced into the current shell.  Use the following syntax:"
        echo ""
        echo "    . tools/env_setup.sh"
        echo ""
        exit 1
    fi
elif $_IS_ZSH; then
    _SCRIPT_SOURCE=${(%):-%N}

    # When sourced, ZSH_EVAL_CONTEXT ends with ":file" (e.g., "toplevel:file"); otherwise it's just "toplevel".
    if [[ "${ZSH_EVAL_CONTEXT:-}" != *:file ]]; then
        echo "ERROR: This script should be sourced into the current shell.  Use the following syntax:"
        echo ""
        echo "    . tools/env_setup.sh"
        echo ""
        exit 1
    fi
else
    _SCRIPT_SOURCE=${0}
fi

# _COSMOS_EVAL_TOOLS_ROOT is referenced by the `dazel` shell function below and therefore preserved past the cleanup at
# the bottom of this file. It must be `export`ed so that subshells inheriting the (also-exported) `dazel` function can
# resolve it; without the export, calling `dazel` from a subshell with `set -u` (e.g. tools/lint/linter.sh) raises an
# unbound-variable error. Assign and export on separate lines so the cd/dirname/pwd exit status is preserved (SC2155).
_COSMOS_EVAL_TOOLS_ROOT="$(cd "$(dirname "${_SCRIPT_SOURCE}")" && pwd)"
export _COSMOS_EVAL_TOOLS_ROOT
_COSMOS_EVAL_DOCS_URL="https://github.com/NVIDIA/cosmos-evaluator"

# Tab completion script paths derived from _COSMOS_EVAL_TOOLS_ROOT.
# Note: Don't use BAZEL_ prefix as it gets unset by the cleanup loop in _dazel_installation_check
COMPLETION_SCRIPT_BAZEL="$_COSMOS_EVAL_TOOLS_ROOT/dazel/bazel_complete.sh"
COMPLETION_SCRIPT_DAZEL="$_COSMOS_EVAL_TOOLS_ROOT/dazel/dazel_complete.sh"

# `dazel` shell function: invokes the dazel.py wrapper at $_COSMOS_EVAL_TOOLS_ROOT/dazel/dazel.py.
function dazel {
    "python3" "$_COSMOS_EVAL_TOOLS_ROOT/dazel/dazel.py" "$@"
}

if $_IS_BASH; then
    # Exporting functions is a bash-only feature; for zsh, ensure availability in subshells via fpath or autoload.
    export -f dazel
fi

function _dazel_installation_check() {
    if ! command -v dazel &> /dev/null; then
        errors+=("Dazel installation unsuccessful.")
    else
        # Clear dazel commands cache to force refresh (works in both bash and zsh)
        unset _DAZEL_COMMANDS_CACHE 2>/dev/null || true

        if $_IS_BASH; then
            # Enable bash tab completion for dazel
            if [[ -f "$COMPLETION_SCRIPT_BAZEL" && -r "$COMPLETION_SCRIPT_BAZEL" ]]; then
                # Remove any existing completion registrations to allow clean reload
                complete -r dazel 2>/dev/null || true
                complete -r bazel 2>/dev/null || true
                complete -r ibazel 2>/dev/null || true

                # Unset all bazel completion functions to force reload (so completion script updates are picked up).
                for func in $(declare -F | grep -o '_bazel[^ ]*' 2>/dev/null); do
                    unset -f "$func" 2>/dev/null || true
                done

                # Unset bazel completion variables to force reload
                for var in $(compgen -v | grep '^BAZEL_' 2>/dev/null); do
                    unset "$var" 2>/dev/null || true
                done

                # Always re-source the completion script to pick up any updates
                if ! source "$COMPLETION_SCRIPT_BAZEL"; then
                    warnings+=("Failed to source bazel tab completion script.")
                fi

                # Source the custom dazel completion wrapper
                if [[ -f "$COMPLETION_SCRIPT_DAZEL" && -r "$COMPLETION_SCRIPT_DAZEL" ]]; then
                    if ! source "$COMPLETION_SCRIPT_DAZEL"; then
                        warnings+=("Failed to source dazel completion wrapper.")
                    fi
                fi

                # Register completion for dazel using our custom wrapper
                if declare -F _dazel_complete_wrapper >/dev/null 2>&1; then
                    complete -F _dazel_complete_wrapper -o nospace dazel
                elif declare -F _bazel__complete >/dev/null 2>&1; then
                    # Fallback to standard Bazel completion if wrapper not available
                    complete -F _bazel__complete -o nospace dazel
                else
                    warnings+=("Tab completion function not found.")
                fi
            else
                warnings+=("Tab completion script $COMPLETION_SCRIPT_BAZEL not found.")
            fi
        elif $_IS_ZSH; then
            # Enable zsh tab completion for dazel
            # Check if zsh completion system is initialized
            if ! typeset -f compdef >/dev/null 2>&1; then
                # Initialize zsh completion system if not already done
                autoload -Uz +X compinit 2>/dev/null && compinit -u 2>/dev/null
                autoload -Uz +X bashcompinit 2>/dev/null && bashcompinit -u 2>/dev/null
            fi

            # Source zsh-specific completion script if it exists
            if [[ -f "$COMPLETION_SCRIPT_BAZEL" && -r "$COMPLETION_SCRIPT_BAZEL" ]]; then
                # Remove any existing completion registrations to allow clean reload
                compdef -d dazel 2>/dev/null
                compdef -d bazel 2>/dev/null
                compdef -d ibazel 2>/dev/null

                # Unset all bazel completion functions to force reload
                for func in ${(k)functions[(I)_bazel*]}; do
                    unset -f "$func" 2>/dev/null
                done

                # Unset bazel completion variables to force reload
                for var in ${(k)parameters[(I)BAZEL_*]}; do
                    unset "$var" 2>/dev/null
                done

                if ! source "$COMPLETION_SCRIPT_BAZEL"; then
                    warnings+=("Failed to source bazel completion script for zsh.")
                fi
            fi
        fi

        echo -e "${GREEN}\u2714 Dazel successfully set up.${NC}"
    fi
}

function _compare_versions() {
    local min_version="$1"
    local current_version="$2"

    # Split versions into arrays (portable across bash/zsh)
    local -a min_version_parts current_version_parts
    if [[ -n "$ZSH_VERSION" ]]; then
        # zsh: use read -A and split on IFS
        IFS='.' read -r -A min_version_parts <<< "$min_version"
        IFS='.' read -r -A current_version_parts <<< "$current_version"
    else
        # bash: use read -a
        IFS='.' read -r -a min_version_parts <<< "$min_version"
        IFS='.' read -r -a current_version_parts <<< "$current_version"
    fi

    for i in 0 1 2; do
        local min=${min_version_parts[$i]:-0}
        local current=${current_version_parts[$i]:-0}
        if (( 10#$current > 10#$min )); then
            return 0  # current > min, version is OK
        elif (( 10#$current < 10#$min )); then
            return 1  # current < min, version is not OK
        fi
    done
    return 0  # versions are equal, OK
}

function _git_installation_check() {
    if ! command -v git &> /dev/null; then
        errors+=("Git is not installed. Please install Git per instructions at $_COSMOS_EVAL_DOCS_URL")
    else
        git_version=$(git --version | awk '{print $3}')

        if _compare_versions "$GIT_MIN_VERSION" "$git_version"; then
            version_ok=true
        else
            version_ok=false
        fi

        if ! $version_ok; then
            warnings+=("Git version $git_version is less than the recommended version $GIT_MIN_VERSION")
        else
            echo -e "${GREEN}\u2714 Git version $git_version meets recommended requirement.${NC}"
        fi
    fi
}

function _docker_installation_check() {
    if ! command -v docker &> /dev/null; then
        errors+=("Docker is not installed. Please install Docker per instructions at $_COSMOS_EVAL_DOCS_URL")
    else
        docker_version=$(docker --version | awk '{print $3}' | sed 's/,//g')

        if _compare_versions "$DOCKER_MIN_VERSION" "$docker_version"; then
            version_ok=true
        else
            version_ok=false
        fi
        if ! $version_ok; then
            warnings+=("Docker version $docker_version is less than the recommended version $DOCKER_MIN_VERSION")
        else
            echo -e "${GREEN}\u2714 Docker version $docker_version meets recommended requirement.${NC}"
        fi
    fi
}

function _nvidia_driver_installation_check() {
    if ! command -v nvidia-smi &> /dev/null; then
        errors+=("Nvidia GPU driver is not installed. Please install it per instructions at $_COSMOS_EVAL_DOCS_URL")
    else
         nv_driver_version=$(modinfo --field=version nvidia)

         if _compare_versions "$NV_DRIVER_MIN_VERSION" "$nv_driver_version"; then
             version_ok=true
         else
             version_ok=false
         fi

         if ! $version_ok; then
             local msg="Nvidia GPU driver version $nv_driver_version is less than the recommended version"
             msg+=" $NV_DRIVER_MIN_VERSION, running GPU-enabled tests/binaries locally may fail"
             warnings+=("$msg")
         else
             echo -e "${GREEN}\u2714 Nvidia GPU driver version $nv_driver_version meets recommended requirement.${NC}"
         fi
    fi
}

function _nvidia_container_toolkit_installation_check() {
    if ! command -v nvidia-container-toolkit &> /dev/null; then
        local msg="Nvidia container toolkit is not installed. Please install it per instructions at"
        errors+=("$msg $_COSMOS_EVAL_DOCS_URL")
    else
        # Get the version of nvidia-container-toolkit assuming it is the last part of the output line
        toolkit_version=$(nvidia-container-toolkit --version 2>/dev/null | head -n1 | awk '{print $NF}')
        if [[ -z "$toolkit_version" ]]; then
            warnings+=("Could not determine nvidia-container-toolkit version.")
        else
            if _compare_versions "$NV_CONTAINER_TOOLKIT_MIN_VERSION" "$toolkit_version"; then
                version_ok=true
            else
                version_ok=false
            fi

            if ! $version_ok; then
                local msg="Nvidia container toolkit version $toolkit_version is less than the recommended version"
                msg+=" $NV_CONTAINER_TOOLKIT_MIN_VERSION, running GPU-enabled tests/binaries locally may fail"
                warnings+=("$msg")
            else
                local msg="Nvidia container toolkit version $toolkit_version meets recommended requirement."
                echo -e "${GREEN}\u2714 ${msg}${NC}"
            fi
        fi
    fi
}

function _nvidia_container_runtime_check() {
    # Verify that the dazel-managed container, if already running, can access the NVIDIA GPU. The host nvidia-smi
    # check above only proves the host driver is healthy; a container that was started before a host driver or
    # nvidia-container-toolkit upgrade can lose GPU access (NVML returns "Failed to initialize NVML: Unknown Error",
    # which produces cudaErrorNoDevice for any CUDA workload inside).
    #
    # When such a stale container is detected we remove it automatically so that dazel can recreate a fresh one with
    # healthy GPU access on the next invocation. A hard error is only raised if the removal itself fails. Only the
    # container this environment would use is inspected/removed, so concurrent builds do not interfere with each other.
    #
    # The container name to inspect can be passed as the first argument. It defaults to the name .dazelrc derives for
    # this module, suffix included: PROJECT_DAZEL_INSTANCE_SUFFIX scopes concurrent instances to their own container
    # (`cosmos-eval-0`, `cosmos-eval-1`, ...). Drop it and the `docker ps` lookup below matches nothing whenever a
    # suffix is set, so the check reports success while leaving a stale container in place for the next invocation.
    local container_name="${1:-cosmos-eval${PROJECT_DAZEL_INSTANCE_SUFFIX:+-$PROJECT_DAZEL_INSTANCE_SUFFIX}}"

    if ! command -v docker &> /dev/null; then
        return
    fi

    # Skip if the docker daemon is not reachable; _docker_installation_check only validates the CLI version, not
    # connectivity, so don't conflate "docker unreachable" with "container has no GPU".
    if ! docker info >/dev/null 2>&1; then
        return
    fi

    # `grep -qxF` compares the name as a literal whole line, rather than through docker's `name=` filter or a bare grep
    # pattern - both of which are regexes over an interpolated name. A `.` is legal in a container name, so a suffixed
    # `cosmos-eval-1.2` would otherwise match a running `cosmos-eval-1x2`, and the checks below would then run on a name
    # that doesn't exist: `docker exec` fails, `docker rm -f` exits 0 regardless, and we report having removed a stale
    # container while never inspecting ours.
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qxF "$container_name"; then
        # No running container yet; dazel will create a fresh one on first invocation, which will see the GPU correctly.
        return
    fi

    if docker exec "$container_name" nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}\u2714 Dazel container '$container_name' can access the NVIDIA GPU.${NC}"
        return
    fi

    local msg="Dazel container '$container_name' cannot access the NVIDIA GPU."
    msg+=" Removing it so a fresh one can be created on next dazel invocation..."
    echo -e "${YELLOW}${msg}${NC}"
    if docker rm -f "$container_name" >/dev/null 2>&1; then
        local msg="Removed stale dazel container '$container_name'"
        msg+=" (dazel will recreate it with GPU access on next invocation)."
        echo -e "${GREEN}\u2714 ${msg}${NC}"
    else
        local msg="Failed to remove stale dazel container '$container_name' that cannot access the NVIDIA GPU."
        msg+=" Remove it manually and retry: docker rm -f $container_name"
        errors+=("$msg")
    fi
}

function _python_installation_check() {
    # Dazel uses #!/usr/bin/env python3, so we MUST have a python3 executable
    if ! command -v python3 &> /dev/null; then
        if command -v python &> /dev/null; then
            # Check if 'python' is actually Python 3
            local python_major
            python_major=$(python -c 'import sys; print(sys.version_info[0])' 2>/dev/null)
            if [[ "$python_major" == "3" ]]; then
                local msg="'python3' executable not found, but 'python' points to Python 3."
                msg+=" Dazel requires a 'python3' executable (its shebang is #!/usr/bin/env python3)."
                msg+=" Please create a symlink: sudo ln -s \$(which python) /usr/local/bin/python3"
                errors+=("$msg")
            else
                local msg="'python3' executable not found. Only Python 2.x is available."
                msg+=" Dazel requires Python $PYTHON_MIN_VERSION or later with a 'python3' executable."
                errors+=("$msg")
            fi
        else
            local msg="Python 3 is not installed. Dazel requires Python $PYTHON_MIN_VERSION"
            errors+=("$msg or later with a 'python3' executable.")
        fi
        return
    fi

    # Get Python version (e.g., "3.10.12")
    local python_version
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null)

    if [[ -z "$python_version" ]]; then
        errors+=("Could not determine Python version. Dazel requires Python $PYTHON_MIN_VERSION or later.")
    elif _compare_versions "$PYTHON_MIN_VERSION" "$python_version"; then
        echo -e "${GREEN}\u2714 Python version $python_version meets minimum requirement.${NC}"
    else
        local msg="Python version $python_version is less than the minimum required version"
        errors+=("$msg $PYTHON_MIN_VERSION. Dazel will not work.")
    fi
}

function _cosmos_eval_help() {
    cat <<EOF

Cosmos Evaluator build environment configured.

The following functions have been added to your environment:

  dazel                 Run bazel inside a Docker container

  Examples:

  dazel build //...     - Build all targets
  dazel fix //...       - Fix lint and formatting violations
  dazel test //...      - Run all tests

EOF
}

function _cosmos_eval_local_db_image_check() {
    # Bootstrap the database-postgres:<VERSION> image used by the cosmos-eval-local-db DAZEL_RUN_DEPS entry in .dazelrc.
    # .dazelrc gates the sidecar on the image already being loaded into Docker (otherwise a chicken-and-egg blocks
    # first-time setup), so this function loads it once via `dazel run //database:image_load` if absent. Subsequent
    # sources are no-ops as long as the image version doesn't change. Skipped entirely when COSMOS_EVAL_NO_LOCAL_DB is
    # set, which suppresses the local-DB image bootstrap.
    if ! command -v docker &> /dev/null; then
        return
    fi

    # The CLI can be present while the daemon is down (e.g. Docker Desktop not started); probe it so we don't attempt
    # a doomed `dazel run //database:image_load` that fails with an opaque connection error.
    if ! docker info &> /dev/null; then
        echo -e "${YELLOW}WARNING: Docker daemon not reachable; skipping cosmos-eval-local-db image bootstrap.${NC}"
        return
    fi

    local db_version
    db_version=$(awk -F'"' '/^VERSION/ { print $2 }' "$_COSMOS_EVAL_TOOLS_ROOT/../database/version.bzl" 2>/dev/null)
    if [[ -z "$db_version" ]]; then
        echo -e "${YELLOW}WARNING: Could not parse database/version.bzl; skipping cosmos-eval-local-db image bootstrap.${NC}"
        return
    fi

    if docker image inspect "database-postgres:$db_version" >/dev/null 2>&1; then
        echo -e "${GREEN}\u2714 cosmos-eval-local-db image database-postgres:$db_version already loaded.${NC}"
        return
    fi

    echo -e "${YELLOW}Loading cosmos-eval-local-db image database-postgres:$db_version (one-time per version bump)...${NC}"
    # Stream dazel/bazel/docker output live to stderr so cold-start progress is visible in interactive shells and in
    # captured build logs (otherwise users stare at "Loading..." for minutes wondering if it's hung).
    if DAZEL_GPUS="" dazel run //database:image_load >&2; then
        echo -e "${GREEN}\u2714 Loaded database-postgres:$db_version.${NC}"
    else
        echo -e "${YELLOW}WARNING: Failed to load database-postgres:$db_version. Run 'dazel run //database:image_load' manually to enable the local DB.${NC}"
    fi
}

function _cosmos_eval_install_build_requirements() {
    local -a warnings=()
    local -a errors=()

    _python_installation_check
    _docker_installation_check
    _git_installation_check

    # Only run NVIDIA checks if a GPU exists on the machine
    if command -v lspci &> /dev/null && lspci 2>/dev/null | grep -iq nvidia; then
        _nvidia_driver_installation_check
        _nvidia_container_toolkit_installation_check
        _nvidia_container_runtime_check
    else
        if command -v lspci &> /dev/null; then
            echo -e "${YELLOW}WARNING: No NVIDIA GPU detected. Skipping GPU-related checks.${NC}"
        else
            echo -e "${YELLOW}WARNING: lspci not available. Skipping GPU detection and related checks.${NC}"
        fi
    fi

    # Only set up dazel if there are no critical errors (like missing Python)
    if [[ ${#errors[@]} -eq 0 ]]; then
        _dazel_installation_check
    else
        echo -e "${RED}\u2718 Dazel was NOT installed due to missing requirements.${NC}"
    fi

    for e in "${errors[@]}"; do
        echo -e "${RED}ERROR: $e${NC}"
    done

    if [[ ${#errors[@]} -ne 0 ]]; then
        return 1
    fi

    # Bootstrap the local-DB image so the cosmos-eval-local-db sidecar in .dazelrc can auto-start.
    if [[ -z "${COSMOS_EVAL_NO_LOCAL_DB:-}" ]]; then
        _cosmos_eval_local_db_image_check
    fi

    if [[ -z "${COSMOS_EVAL_NO_BANNER:-}" ]]; then
        _cosmos_eval_help
    fi

    for w in "${warnings[@]}"; do
        echo -e "${YELLOW}WARNING: $w${NC}"
    done

    return 0
}

_cosmos_eval_install_build_requirements
_cosmos_eval_setup_rc=$?
if [[ $_cosmos_eval_setup_rc -ne 0 ]]; then
    echo -e "${RED}Build environment setup failed.${NC}"
fi

# Cleanup: unset script-internal variables. Function definitions remain so they can be reused
# (e.g. _nvidia_container_runtime_check is intentionally callable post-source). The dazel function depends on
# _COSMOS_EVAL_TOOLS_ROOT, so that variable is preserved.
unset GREEN YELLOW RED NC \
    DOCKER_MIN_VERSION GIT_MIN_VERSION NV_DRIVER_MIN_VERSION NV_CONTAINER_TOOLKIT_MIN_VERSION PYTHON_MIN_VERSION \
    COMPLETION_SCRIPT_BAZEL COMPLETION_SCRIPT_DAZEL \
    _COSMOS_EVAL_DOCS_URL \
    _IS_BASH _IS_ZSH _SCRIPT_SOURCE

if [[ $_cosmos_eval_setup_rc -ne 0 ]]; then
    unset _cosmos_eval_setup_rc
    return 1
fi
unset _cosmos_eval_setup_rc
