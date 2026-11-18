#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
exec "${PIPELINE_PYTHON:-python3}" -m src.push_t_data.workflow "$@"
