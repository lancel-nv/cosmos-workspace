"""Verify portable ARX bundles, all action windows, and representative native batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from cosmos_framework.data.generator.action.robodojo_eef import commanded_eef_actions
from cosmos_framework.data.generator.action.utils.unified_action_schema import UnifiedActionCodec
from cosmos_framework.data.generator.local_simulation.core import local_path
from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset
from cosmos_framework.scripts.validate_bimanual_data import materialize_batch, validate_bundle


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(root):
    report_path = root / "readiness.json"
    if report_path.exists():
        raise ValueError("Readiness report already exists; preserve the accepted report")
    summaries, split_layouts, split_episodes = {}, {}, {}
    for split in ("train", "validation"):
        bundle = root / split
        receipt = json.loads((bundle / "receipts/dataset-receipt.json").read_text())
        for relative, expected in receipt["files_sha256"].items():
            if digest(local_path(bundle, relative)) != expected:
                raise ValueError(f"Bundle bytes changed: {relative}")
        dataset = LocalSimulationDataset(bundle, model="bimanual", seed=42)
        pipeline = dataset._pipeline("robodojo_arx_x5_eef")
        codec = UnifiedActionCodec(
            native_normalizer=None,
            mapping=pipeline.dataset.get_unified_action_mapping(),
            unified_normalizer=pipeline._normalizer({}),
        )
        buckets, coverage, terminals = defaultdict(list), defaultdict(set), defaultdict(int)
        lengths, outcomes, shapes = {}, {}, {}
        split_layouts[split], split_episodes[split] = set(), set()
        maximum_roundtrip_error, maximum_normalized_abs = 0.0, 0.0
        current_path, data = None, None
        for index, record in enumerate(dataset.records):
            provenance = record["provenance"]
            if provenance["split"] != split:
                raise ValueError("Wrong episode split")
            episode = provenance["source_episode_id"]
            split_layouts[split].add(provenance["layout_hash"])
            split_episodes[split].add(episode)
            buckets[provenance["episode_outcome"]].append(index)
            outcomes[episode] = provenance["episode_outcome"]
            first = record["source_rows"]["cam_head"]
            path = local_path(bundle, record["assets"][first["numeric_uri"]])
            if current_path != path:
                with np.load(path, allow_pickle=False) as payload:
                    data = {name: payload[name] for name in payload.files}
                current_path = path
                if len(data) != 23 or not all(np.isfinite(value).all() for value in data.values()):
                    raise ValueError("Numeric schema or finiteness differs")
            indices = record["request"]["source_frame_indices"]
            start, end = indices[0], indices[-1]
            maximum_actions = dataset.manifest.get("max_window_actions", 32)
            if indices != list(range(start, end + 1)) or not 16 <= end - start <= maximum_actions or (end - start) % 16:
                raise ValueError("Wrong native window length or indices")
            if end >= len(data["action"]) or not data["action_valid"][start:end].all():
                raise ValueError("Invalid action slice")
            coverage[episode].update(range(start, end))
            lengths[episode] = len(data["action"]) - 1
            terminal_count = int(data["next.done"][start:end].sum())
            if provenance["contains_terminal_transition"] != bool(terminal_count):
                raise ValueError("Terminal provenance differs from numeric data")
            terminals[episode] += terminal_count
            encoded = codec.encode(
                torch.from_numpy(
                    commanded_eef_actions(
                        data["observation.state"][start],
                        data["action"][start:end],
                    )
                )
            )
            if not torch.isfinite(encoded.normalized).all() or encoded.valid_mask.sum() != 20:
                raise ValueError("Bad canonical action or mask")
            if encoded.normalized[:, ~encoded.valid_mask].count_nonzero() != 0:
                raise ValueError("Invalid semantic slots are not zero")
            recovered = codec.decode_to_unified(encoded.normalized)
            error = float((recovered - encoded.raw).abs().max())
            maximum_roundtrip_error = max(maximum_roundtrip_error, error)
            maximum_normalized_abs = max(maximum_normalized_abs, float(encoded.normalized.abs().max()))
            if error > 2e-5:
                raise ValueError("Normalization roundtrip failed")
        for episode, covered in coverage.items():
            if covered != set(range(lengths[episode])) or terminals[episode] != 1:
                raise ValueError("Lost action or duplicated/missing terminal transition")
        selected = sorted(
            {index for bucket in buckets.values() for index in (bucket[0], bucket[len(bucket) // 2], bucket[-1])}
            | {
                min(
                    range(len(dataset)),
                    key=lambda index: len(dataset.records[index]["request"]["source_frame_indices"]),
                ),
                max(
                    range(len(dataset)),
                    key=lambda index: len(dataset.records[index]["request"]["source_frame_indices"]),
                ),
            }
        )
        for index in selected:
            sample = dataset[index]
            action, video, mask = sample["action"], sample["video"], sample["action_valid_mask"]
            frames = len(dataset.records[index]["request"]["source_frame_indices"])
            if action.shape != (frames - 1, 64) or video.shape[:2] != (3, frames):
                raise ValueError("Wrong materialized video/action shape")
            if not torch.isfinite(action).all() or not torch.isfinite(video).all():
                raise ValueError("Nonfinite materialized sample")
            if mask.shape != (64,) or mask.sum() != 20 or action[:, ~mask].count_nonzero() != 0:
                raise ValueError("Invalid packed semantic mask")
            if int(sample["domain_id"]) != 33 or sample["mode"] != "forward_dynamics":
                raise ValueError("Wrong domain/objective")
            if sample["sequence_plan"].condition_frame_indexes_action != list(range(frames - 1)):
                raise ValueError("Forward dynamics must condition on all commanded actions")
            if sample["sequence_plan"].condition_frame_indexes_vision != [0]:
                raise ValueError("Only the initial RGB may be conditioning")
            shapes[dataset.records[index]["identity"]] = {
                "video": list(video.shape),
                "action": list(action.shape),
                "valid_channels": int(mask.sum()),
            }
        summaries[split] = {
            "static": validate_bundle(bundle),
            "all_action_windows_verified": len(dataset),
            "unique_actions_covered": sum(lengths.values()),
            "episodes": len(lengths),
            "representative_materialized_samples": shapes,
            "maximum_normalization_roundtrip_error": maximum_roundtrip_error,
            "maximum_normalized_abs_without_clipping": maximum_normalized_abs,
            "native_stages": {str(stage): materialize_batch(bundle, stage) for stage in range(4)},
        }
        print(
            f"{split}: all {len(dataset)} action windows and {len(selected)} representative samples passed", flush=True
        )
    if split_layouts["train"] & split_layouts["validation"] or split_episodes["train"] & split_episodes["validation"]:
        raise ValueError("Train/validation leakage")
    report = {
        "status": "native_bimanual_data_preflight_passed",
        "data_ready_for_sft": True,
        "scope": "Data materialization/packing only; no model forward/backward or checkpoint loading tested",
        "created_at": datetime.now(UTC).isoformat(),
        "bundles": summaries,
        "required_num_embodiment_domains": 34,
        "model_training_started": False,
        "training_prerequisites": [
            "Use the patched Framework adapter and num_embodiment_domains=34 for the new domain 33",
            "Choose/init a compatible checkpoint; released 32-domain projectors cannot be loaded unchanged",
            "Validate GPU topology/memory, model weights and VAE, then run a short optimizer-step smoke test",
        ],
        "export_receipt_sha256": digest(root / "export_receipt.json"),
        "auditor_sha256": digest(__file__),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.root.resolve())
