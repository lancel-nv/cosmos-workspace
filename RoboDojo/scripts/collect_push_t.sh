#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
if [[ -z "${COLLECTION_PYTHON:-}" ]]; then
    COLLECTION_PYTHON="$("${PIPELINE_PYTHON:-python3}" -c 'import json, os; print(os.path.abspath(os.path.expanduser(json.load(open("configs/push_t_collection.json"))["collection_python"])))')"
fi
if [[ ! -x "$COLLECTION_PYTHON" ]]; then
    printf 'Simulation Python not executable: %s\n' "$COLLECTION_PYTHON" >&2
    exit 1
fi
unset __EGL_VENDOR_LIBRARY_DIRS __EGL_VENDOR_LIBRARY_FILENAMES CUDA_VISIBLE_DEVICES PYTHONHOME
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$ROOT/XPolicyLab"
export PATH="$(dirname "$COLLECTION_PYTHON"):$PATH"
if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
if [[ ! -f Assets/Robots/x5/curobo.yml ]]; then
    "$COLLECTION_PYTHON" utils/update_embodiment_config_path.py
fi
exec "$COLLECTION_PYTHON" -u -m src.push_t_data.collect "$@"
