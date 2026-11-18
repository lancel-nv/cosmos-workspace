"""Export verified Curator staging windows into portable native Bimanual bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as element_tree
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import trimesh
import yaml

from cosmos_framework.data.generator.action.lance.lance_action_dataset import LanceActionSampleRequest
from cosmos_framework.data.generator.action.robodojo_eef import ADAPTER_TARGET, CONTRACT, commanded_eef_actions
from cosmos_framework.data.generator.lance.action_window_index import ActionWindowLocation
from cosmos_framework.scripts.validate_bimanual_data import validate_bundle

FRAMEWORK = Path(__file__).resolve().parents[1]
STATS = (
    FRAMEWORK
    / "cosmos_framework/data/generator/action/normalizers/global_asinh_v1_backward_anchored_gripper_metric.json"
)
STATS_SHA256 = "ae695e2c643a5626b3c3564f00705aa3a83ab0da7640549c7340db71172594b0"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def calibration_from_assets(robot_assets):
    config = yaml.safe_load((robot_assets / "robot_config.yml").read_text())
    if config["ee_link"] != "link6" or config["gripper_scale"] != [-0.01, 0.044]:
        raise ValueError("ARX control calibration differs from versioned adapter")
    robot = element_tree.parse(robot_assets / "X5A.urdf").getroot()
    inner_faces = []
    for number, sign in [(7, 1), (8, -1)]:
        joint = robot.find(f"joint[@name='joint{number}']")
        limits = joint.find("limit").attrib
        if float(limits["lower"]) != 0 or float(limits["upper"]) != 0.044:
            raise ValueError("URDF finger limits differ")
        if joint.find("parent").attrib["link"] != "link6":
            raise ValueError("Finger parent is not link6")
        if joint.find("axis").attrib["xyz"] != f"0 {sign} 0":
            raise ValueError("Finger slide axis differs")
        mesh = trimesh.load(robot_assets / f"meshes/link{number}.STL", force="mesh")
        offset = np.fromstring(joint.find("origin").attrib["xyz"], sep=" ")
        vertices = np.asarray(mesh.vertices) + offset
        fingertips = vertices[vertices[:, 0] > 0.12]
        inner_faces.append(float((fingertips[:, 1] * sign).min()))
    gap = sum(inner_faces)
    files = ["robot_config.yml", "X5A.urdf", "ARX.usd", "meshes/link7.STL", "meshes/link8.STL"]
    return {
        "contract": CONTRACT,
        "wrist_frame": "link6",
        "link6_to_wrist": np.eye(4).tolist(),
        "pose_units": "meter",
        "source_quaternion": "wxyz",
        "rot6d": "first two columns",
        "source_command_joint_endpoints_m": [-0.01, 0.044],
        "physical_joint_limits_m": [0.0, 0.044],
        "effective_openness": "clip(-0.01 + 0.054 * command_fraction, 0, 0.044) / 0.044",
        "closed_inner_surface_gap_m": gap,
        "fully_open_gripper_width_m": gap + 0.088,
        "width_evidence": "URDF finger transforms and distal collision mesh inner surfaces (link6 x > 0.12m)",
        "width_limitations": "Nominal rigid CAD jaw separation; not measured contact deformation. Closed fraction denotes joint stop.",
        "source_sha256": {name: digest(robot_assets / name) for name in files},
        "anchoring": "P0=initial measured link6; P[i+1]=command[i]; inverse(P[16*floor(i/16)]) @ P[i+1]",
        "future_measured_states_used_as_action": False,
    }


def export(staging, output, tokenizer, robot_assets):
    if output.exists():
        raise ValueError("Output already exists; use a fresh directory")
    receipt = json.loads((staging / "receipt.json").read_text())
    if receipt["status"] != "validated_staging_index":
        raise ValueError("Input staging has not passed validation")
    source = Path(receipt["source_root"])
    if output.is_relative_to(source) or output.is_relative_to(staging):
        raise ValueError("Do not publish inside a frozen input")
    if digest(source / receipt["source_receipt_file"]) != receipt["source_release_sha256"]:
        raise ValueError("Source release changed")
    for name, expected in receipt["source_evidence_sha256"].items():
        if digest(source / name) != expected:
            raise ValueError(f"Source evidence changed: {name}")
    for name, expected in receipt["output_sha256"].items():
        if digest(staging / name) != expected:
            raise ValueError(f"Staging file changed: {name}")
    if digest(STATS) != STATS_SHA256:
        raise ValueError("Pinned normalization statistics changed")
    for name in ("vocab.json", "merges.txt", "tokenizer_config.json"):
        if not (tokenizer / name).is_file():
            raise ValueError(f"Missing local tokenizer file: {name}")
    windows = [json.loads(line) for line in (staging / "windows.jsonl").read_text().splitlines()]
    numeric_hashes = json.loads((staging / "numeric_sha256.json").read_text())
    calibration = calibration_from_assets(robot_assets)
    output.mkdir(parents=True)
    summaries = {}
    source_hashes = {}
    split_layouts = {}
    for split in ("train", "validation"):
        selected = [row for row in windows if row["split"] == split]
        split_layouts[split] = {row["layout_hash"] for row in selected}
        bundle = output / split
        resources = bundle / "resources"
        resources.mkdir(parents=True)
        shutil.copytree(tokenizer, resources / "tokenizer")
        shutil.copy2(STATS, resources / "global_asinh.json")
        write_json(resources / "arx_calibration.json", calibration)
        for name in calibration["source_sha256"]:
            destination = resources / "robot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(robot_assets / name, destination)
        transforms = {
            "entry": {
                "dataset": {
                    "_target_": ADAPTER_TARGET,
                    "calibration_path": "bundle://resources/arx_calibration.json",
                    "viewpoint": "ego_view",
                },
                "ratio": 1.0,
            },
            "action_schema": "unified_v1",
            "global_asinh": True,
            "metric_gripper": True,
            "global_asinh_stats_path": "bundle://resources/global_asinh.json",
            "resolution": "480",
            "max_action_dim": 64,
            "video_temporal_downsample": 4,
            "cfg_dropout_rate": 0.1,
            "enable_mode_specific_prompt": False,
            "append_viewpoint_info": False,
            "append_duration_fps_timestamps": False,
            "append_resolution_info": False,
            "append_idle_frames": False,
            "format_prompt_as_json": False,
            "format_prompt_float_seconds": False,
        }
        records, copied, episodes = [], {}, {}
        current_path, data = None, None
        for row in selected:
            episode = row["episode_index"]
            episodes[episode] = row
            assets, source_rows, row_ids = {}, {}, {}
            numeric = Path(row["transition_npz"])
            head_views = [view for view in row["video_views"] if view["view_name"] == "observation.images.cam_head"]
            if len(head_views) != 1:
                raise ValueError("Expected exactly one head camera")
            paths = [numeric] + [Path(view["video_path"]) for view in head_views]
            for path in paths:
                key = str(path)
                if key not in copied:
                    expected = source_hashes.setdefault(key, digest(path))
                    if path == numeric and numeric_hashes[key] != expected:
                        raise ValueError(f"Numeric source changed: {path}")
                    relative = f"assets/episode_{row['source_episode_id']:06d}/{path.name}"
                    target = bundle / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    if digest(target) != expected:
                        raise ValueError(f"Copied bytes differ: {path}")
                    copied[key] = relative
                assets[f"source:{path}"] = copied[key]
            if current_path != numeric:
                with np.load(numeric, allow_pickle=False) as payload:
                    data = {name: payload[name] for name in payload.files}
                current_path = numeric
            start, end = row["action_start"], row["action_end_exclusive"]
            actions = commanded_eef_actions(data["observation.state"][start], data["action"][start:end])
            if actions.shape != (end - start, 20) or not np.isfinite(actions).all():
                raise ValueError("Invalid mapped command window")
            for camera_index, view in enumerate(head_views):
                name = view["view_name"].removeprefix("observation.images.")
                row_id = episode * 3 + camera_index
                row_ids[name] = row_id
                source_rows[name] = {
                    "row_id": row_id,
                    "episode_index": episode,
                    "video_uri": f"source:{view['video_path']}",
                    "numeric_uri": f"source:{numeric}",
                    "first_pts_seconds": view["first_pts_seconds"],
                    "curator_clip_id": view["clip_id"],
                }
            request = LanceActionSampleRequest(
                draw_count=len(records),
                global_index=len(records),
                aux_seed=42,
                location=ActionWindowLocation(episode, row_ids["cam_head"], start, end - start + 1),
                source_frame_indices=tuple(range(start, end + 1)),
                row_ids_by_view=row_ids,
                source_version=receipt["source_lance_version"],
                source_index_version=1,
                dataset_length=len(selected),
                index_space_fingerprint=receipt["output_sha256"]["windows.jsonl"],
                frame_count=len(data["action"]),
                source_fps=25.0,
                shard_id=split,
                episode_index=episode,
                resolution="480",
                target_fps=25.0,
                caption=row["prompt"],
            )
            records.append(
                {
                    "identity": row["identity"],
                    "stratum": "robodojo_arx_x5_eef",
                    "request": asdict(request),
                    "source_rows": source_rows,
                    "assets": assets,
                    "provenance": {
                        key: row[key]
                        for key in (
                            "source_episode_id",
                            "layout_hash",
                            "layout_index",
                            "split",
                            "episode_success",
                            "episode_outcome",
                            "contains_terminal_transition",
                        )
                    },
                }
            )
        manifest = {
            "format_version": 1,
            "model": "bimanual",
            "complete": True,
            "sample_count": len(records),
            "transforms": {"robodojo_arx_x5_eef": transforms},
            "adapter_contract": CONTRACT,
            "required_num_embodiment_domains": 34,
            "visual_input": "cam_head_only",
        }
        schema_receipt = json.loads((FRAMEWORK / "resources/unified-v1-action-schema-receipt.json").read_text())
        schema_receipt["normalization"]["statistics_path"] = "resources/global_asinh.json"
        schema_receipt["domains"]["robodojo_arx_x5_eef"] = {
            "domain_id": 33,
            "native_dim": 20,
            "native_layout": "left9,left_open,right9,right_open",
            "valid_indices": list(range(9, 18)) + [33] + list(range(34, 43)) + [58],
            "fully_open_width_m": calibration["fully_open_gripper_width_m"],
            "calibration": calibration,
        }
        write_json(bundle / "receipts/unified-v1-action-schema-receipt.json", schema_receipt)
        write_json(bundle / "samples.json", records)
        write_json(bundle / "manifest.json", manifest)
        validation = validate_bundle(bundle)
        evidence = {
            "created_at": datetime.now(UTC).isoformat(),
            "split": split,
            "source_staging_receipt_sha256": digest(staging / "receipt.json"),
            "source_fingerprint": receipt["source_fingerprint"],
            "calibration": calibration,
            "episodes": len(episodes),
            "windows": len(records),
            "outcomes": dict(Counter(row["episode_outcome"] for row in episodes.values())),
            "numeric_fields_preserved": 23,
            "view_composition": "head above half-resolution left/right wrists",
            "action_mode": "forward_dynamics",
            "normalization_applied_offline": False,
            "normalization_sha256": STATS_SHA256,
            "static_preflight": validation,
            "materialized_preflight": "not_run",
            "training_started": False,
            "required_training_override": "model.config.num_embodiment_domains=34",
            "checkpoint_warning": "Domain 33 is new. Do not strictly load an unchanged 32-domain released action projector. Stage-0 fresh action modules or explicit audited expansion required.",
            "exporter_sha256": digest(__file__),
            "adapter_sha256": digest(FRAMEWORK / "cosmos_framework/data/generator/action/robodojo_eef.py"),
            "files_sha256": {
                str(path.relative_to(bundle)): digest(path) for path in sorted(bundle.rglob("*")) if path.is_file()
            },
        }
        write_json(bundle / "receipts/dataset-receipt.json", evidence)
        summaries[split] = {key: evidence[key] for key in ("episodes", "windows", "outcomes", "static_preflight")}
        print(split, summaries[split], flush=True)
    if split_layouts["train"] & split_layouts["validation"]:
        raise ValueError("Layout split leakage")
    write_json(
        output / "export_receipt.json",
        {
            "status": "exported_static_validated",
            "framework_ready": False,
            "model_training_started": False,
            "bundles": summaries,
            "materialize_required": True,
            "calibration": calibration,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--robot-assets", type=Path, required=True)
    args = parser.parse_args()
    export(args.staging.resolve(), args.output.resolve(), args.tokenizer.resolve(), args.robot_assets.resolve())


if __name__ == "__main__":
    main()
