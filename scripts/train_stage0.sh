#!/usr/bin/env bash
set -euo pipefail
WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
: "${BUNDLE_ROOT:?Set BUNDLE_ROOT to the audited head-only bundle root}"
: "${COSMOS_SIM_BIMANUAL_GA_CHECKPOINT:?Set the original Cosmos3-Nano DCP initializer}"
: "${VIDEO_VAE_PATH:?Set the Wan2.2 VAE checkpoint path}"
: "${IMAGINAIRE_OUTPUT_ROOT:?Set a writable output directory}"
export COSMOS_SIM_BIMANUAL_LOCAL_BUNDLE="$BUNDLE_ROOT/train"
export COSMOS_INTERNAL=0 COSMOS_TRAINING=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH="$WORKSPACE/cosmos-framework"
FRAMEWORK_PYTHON=${FRAMEWORK_PYTHON:-$WORKSPACE/cosmos-framework/.venv/bin/python}
cd "$WORKSPACE/cosmos-framework"
"$FRAMEWORK_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["BUNDLE_ROOT"])
readiness = json.loads((root / "readiness.json").read_text())
assert readiness["data_ready_for_sft"]
assert readiness["export_receipt_sha256"] == hashlib.sha256((root / "export_receipt.json").read_bytes()).hexdigest()
for split in ("train", "validation"):
    manifest = json.loads((root / split / "manifest.json").read_text())
    assert manifest["visual_input"] == "cam_head_only"
    assert all(config["entry"]["dataset"]["viewpoint"] == "ego_view" for config in manifest["transforms"].values())
PY
RUN_NAME=${RUN_NAME:-push-t-head-stage0}
args=(
  -m cosmos_framework.scripts.train
  --sft-toml=examples/toml/sft_config/sim_bimanual_stage0_bidir.toml
)
overrides=(
  "job.name=$RUN_NAME"
  "model.config.tokenizer.vae_path=$VIDEO_VAE_PATH"
  model.config.num_embodiment_domains=34
  "dataloader_train.max_samples_per_batch=${BATCH_SIZE:-1}"
  dataloader_train.num_workers=0
  trainer.seed=42
  dataloader_train.seed=42
  trainer.callbacks.compile_tokenizer.enabled=False
  ++trainer.callbacks.grad_clip.error_if_nonfinite=True
  "trainer.max_iter=${MAX_ITER:-35000}"
  "checkpoint.save_iter=${SAVE_ITER:-100}"
)
case "${1:-dryrun}" in
  dryrun) exec "$FRAMEWORK_PYTHON" "${args[@]}" --dryrun -- "${overrides[@]}" ;;
  train)
    test -s "$COSMOS_SIM_BIMANUAL_GA_CHECKPOINT/model/.metadata"
    test -s "$VIDEO_VAE_PATH"
    test ! -e "$IMAGINAIRE_OUTPUT_ROOT/cosmos3_simulation/bimanual/$RUN_NAME"
    mkdir -p "$IMAGINAIRE_OUTPUT_ROOT"
    exec 9>"$IMAGINAIRE_OUTPUT_ROOT/.push-t-training.lock"
    flock -n 9
    if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
      echo 'GPU jobs exist; refusing to share or stop them.' >&2
      exit 1
    fi
    exec "$FRAMEWORK_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 "${args[@]}" -- "${overrides[@]}"
    ;;
  *) echo 'Usage: train_stage0.sh dryrun|train' >&2; exit 2 ;;
esac
