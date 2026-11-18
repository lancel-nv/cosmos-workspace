---
name: push-t-workflow
description: 在 cosmos-workspace 中复现 RoboDojo ARX-X5 Push-T 的采集、raw 数据校验、LeRobot V3/Curator 处理与单主相机 Bimanual Stage 0 数据和训练配置。适用于迁移集群、重新处理数据或准备训练，不用于自动启动未授权的长训练。
---

# Push-T reproducible workflow

Read [workflow.md](references/workflow.md) for executable steps and environment requirements. Work from the repository root; use its scripts instead of reconstructing the historical experiments.

## Route by requested task

- Existing data: download the pinned raw release, verify it, run `scripts/process_raw.py` first without `--execute`, then execute only the requested conversion. Collection/Isaac Sim is not required to process existing raw data.
- New collection: bootstrap the pinned simulator dependencies and assets, configure a new run, inspect the smoke episodes, then collect to the requested quota. Use the final collection configuration; do not import archived experiments.
- Training: require an audited native head-only bundle and a compatible original Cosmos3-Nano initializer. Check `scripts/train_stage0.sh dryrun`; only run `train` when the user asks. Measure the new cluster before choosing maximum batch size or a time budget.
- Publishing: confirm destination/visibility, scan the Git file list for credentials and data, verify the raw release, then publish only the requested artifacts. A workflow invocation is not permission to change visibility or start training.

## Non-obvious invariants

- Raw episodes contain T actions and T+1 measured observations with three synchronized 25 FPS cameras. Keep the final observation. Terminal-row placeholder actions in LeRobot must never become training controls.
- Distinguish measured state from commanded EEF. ARX wrist is `link6`, not the fingertip; rotations are wxyz in raw, column-based rot6d after adaptation. Keep calibrated gripper dead-zone/metric width and 16-action anchoring. Do not relabel ARX EEF as the existing joint-control domain.
- Native RGB is only `cam_head` (`ego_view`). Wrist cameras remain in raw/LeRobot for provenance but are excluded from model samples. Do not treat the framework's optional `concat_view` support as the Bimanual paper recipe.
- RGB condition is frame 0; all aligned actions are conditions; future RGB is a training target, never a leaked evaluation condition. This Stage0 is bidirectional, not the final streaming/distilled simulator.
- Split by the whole initialization layout hash before creating windows. Preserve success and failure episodes; 191/200 successes in this release concentrate in two layouts, so do not claim broad generalization.
- Curator does not preserve every custom transition column. The augmentation step restores the aligned numeric fields from LeRobot; the staging and native audits verify them. Do not omit that step.
- Fresh directories and content hashes protect accepted runs. If a stage fails, inspect its log and partial outputs; never invent a successful receipt or blindly overwrite/restart an active job.
- Never stop unrelated GPU jobs. Keep credentials in runtime authentication, not files committed to Git or dataset metadata.

## Completion evidence

Report which gates actually ran: raw hash/semantic/video checks, LeRobot roundtrip, Curator clip counts/errors, numeric backfill, native static/materialized audits, model smoke test, evaluation. Data readiness alone is not proof of trained model quality. Cite paths/versions and state untested stages explicitly.
