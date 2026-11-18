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

"""
Library for serializing objects results.

This module provides functions for saving and loading objects results to and from JSON files.
"""

import base64
import gzip
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

STATIC_SIGNAL_GROUPS: tuple[str, ...] = (
    "lane",
    "boundary",
    "traffic_light",
    "traffic_sign",
    "crosswalk",
    "wait_line",
)

STATIC_OBJECT_TYPE_TO_SIGNAL_GROUP: dict[str, str] = {
    "LaneLine": "lane",
    "RoadBoundary": "boundary",
    "TrafficLight": "traffic_light",
    "TrafficSign": "traffic_sign",
    "Crosswalk": "crosswalk",
    "WaitLine": "wait_line",
}

TRAFFIC_LIGHT_TRACK_METRICS: tuple[str, ...] = (
    "traffic_light_state_score",
    "traffic_light_multi_color_score",
    "traffic_light_state_confidence",
)


def save_results_to_json(
    results: dict[str, Any],
    clip_id: str,
    output_dir: str,
    matrix_format: str = "sparse",
    output_file_prefix: str | None = None,
    importance_matrix: np.ndarray | None = None,
    projection_area_matrix: np.ndarray | None = None,
) -> str:
    """
    Save processing results to a JSON file.

    Args:
        results: Results dictionary from process_clip
        clip_id: Clip ID for filename
        output_dir: Output directory path
        matrix_format: How to store the score matrix ("sparse", "dense", "compressed", or "summary")
        importance_matrix: Optional (frames, tracks) int matrix; emitted as the ``importance`` column in
            ``sparse_compact``. Used by dynamic objects.
        projection_area_matrix: Optional (frames, tracks) float matrix of per-frame visible-projection area in pixels
            (NaN where the track was not scored); emitted as the ``projection_area_px`` column in ``sparse_compact``.
            Used by static objects.

    Returns:
        Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Prepare results for JSON serialization
    json_results: dict[str, Any] = {}

    # Copy basic results (these are JSON-serializable)
    for key, value in results.items():
        if key == "score_matrix":
            continue  # Handle score matrix separately
        elif key == "track_ids":
            # Convert set to sorted list for consistent ordering
            json_results[key] = sorted(value) if value else []
        elif key == "processed_frame_ids":
            json_results[key] = list(value) if value else []
        else:
            json_results[key] = value

    # Handle score matrix based on format
    if results.get("score_matrix") is not None:
        score_matrix = results["score_matrix"]

        if matrix_format == "sparse":
            # Store only non-NaN values with their indices
            sparse_data = []
            for frame_idx in range(score_matrix.shape[0]):
                for track_idx in range(score_matrix.shape[1]):
                    score = score_matrix[frame_idx, track_idx]
                    if not np.isnan(score):
                        sparse_data.append(
                            {"frame_idx": int(frame_idx), "track_idx": int(track_idx), "score": float(score)}
                        )
            json_results["score_matrix"] = {"format": "sparse", "shape": list(score_matrix.shape), "data": sparse_data}

        elif matrix_format == "sparse_compact":
            # Compact 2D array. Columns are declared in ``columns`` and vary by
            # processor type: dynamic objects emit ``importance``; static objects
            # emit ``projection_area_px``. ``frame_idx``, ``track_idx``, and
            # ``score`` are always present.
            columns: list[str] = ["frame_idx", "track_idx"]
            if importance_matrix is not None:
                columns.append("importance")
            columns.append("score")
            if projection_area_matrix is not None:
                columns.append("projection_area_px")

            compact_data: list[list] = []
            for frame_idx in range(score_matrix.shape[0]):
                for track_idx in range(score_matrix.shape[1]):
                    score = score_matrix[frame_idx, track_idx]
                    if np.isnan(score):
                        continue
                    row: list = [int(frame_idx), int(track_idx)]
                    if importance_matrix is not None:
                        if frame_idx < importance_matrix.shape[0] and track_idx < importance_matrix.shape[1]:
                            row.append(int(importance_matrix[frame_idx, track_idx]))
                        else:
                            row.append(1)
                    row.append(round(float(score), 4))
                    if projection_area_matrix is not None:
                        if frame_idx < projection_area_matrix.shape[0] and track_idx < projection_area_matrix.shape[1]:
                            area = projection_area_matrix[frame_idx, track_idx]
                            row.append(0.0 if np.isnan(area) else float(area))
                        else:
                            row.append(0.0)
                    compact_data.append(row)
            json_results["score_matrix"] = {
                "format": "sparse_compact",
                "shape": list(score_matrix.shape),
                "columns": columns,
                "data": compact_data,
            }

        elif matrix_format == "dense":
            # Store full matrix as nested lists (larger file size)
            json_results["score_matrix"] = {
                "format": "dense",
                "shape": list(score_matrix.shape),
                "data": score_matrix.tolist(),
            }

        elif matrix_format == "compressed":
            # Store as base64-encoded compressed data
            # Convert NaN to a special value for compression
            matrix_copy = score_matrix.copy()
            matrix_copy[np.isnan(matrix_copy)] = -999.0  # Special value for NaN

            # Compress and encode
            compressed_data = gzip.compress(matrix_copy.tobytes())
            encoded_data = base64.b64encode(compressed_data).decode("utf-8")

            json_results["score_matrix"] = {
                "format": "compressed",
                "shape": list(score_matrix.shape),
                "nan_value": -999.0,
                "data": encoded_data,
            }

        elif matrix_format == "summary":
            # Store only summary statistics (smallest file size)
            valid_scores = score_matrix[~np.isnan(score_matrix)]
            json_results["score_matrix"] = {
                "format": "summary",
                "shape": list(score_matrix.shape),
                "statistics": {
                    "valid_count": len(valid_scores),
                    "nan_count": int(np.isnan(score_matrix).sum()),
                    "total_elements": int(score_matrix.size),
                    "mean": float(np.nanmean(score_matrix)),
                    "std": float(np.nanstd(score_matrix)),
                    "min": float(np.nanmin(score_matrix)),
                    "max": float(np.nanmax(score_matrix)),
                },
            }
        else:
            raise ValueError(f"Unknown matrix_format: {matrix_format}")
    else:
        json_results["score_matrix"] = None

    # Add metadata
    json_results["metadata"] = {"matrix_format": matrix_format, "version": "2.0"}

    # Save to file with custom formatting for compact arrays
    output_file = output_path / f"{clip_id}.objects.results.json"
    if output_file_prefix:
        output_file = output_path / f"{clip_id}.{output_file_prefix}.objects.results.json"

    def format_json_with_compact_arrays(obj: Any, indent: int = 2) -> str:
        """Format JSON with compact arrays for numeric lists."""
        json_str = json.dumps(obj, indent=indent)
        lines = json_str.split("\n")
        result_lines: list[str] = []
        is_num = re.compile(r"[+-]?\d+(?:\.\d+)?$").match
        i = 0
        while i < len(lines):
            stripped = lines[i].rstrip()
            if stripped.endswith("["):
                numbers: list[str] = []
                j = i + 1
                all_numeric = True
                while j < len(lines):
                    inner = lines[j].strip()
                    if inner in ("]", "],"):
                        break
                    value = inner.rstrip(",")
                    if not is_num(value):
                        all_numeric = False
                        break
                    numbers.append(value)
                    j += 1
                if all_numeric and numbers and j < len(lines):
                    trailing = "," if lines[j].strip().endswith(",") else ""
                    result_lines.append(stripped + ",".join(numbers) + "]" + trailing)
                    i = j + 1
                    continue
            result_lines.append(lines[i])
            i += 1
        return "\n".join(result_lines)

    with open(output_file, "w") as f:
        f.write(format_json_with_compact_arrays(json_results))

    return str(output_file)


def _correspondence_score_stats(values: np.ndarray) -> dict[str, float | int]:
    """Return aggregate correspondence statistics for a flat array of scores."""
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return {}
    return {
        "mean_score": float(np.mean(valid)),
        "std_score": float(np.std(valid)),
        "min_score": float(np.min(valid)),
        "max_score": float(np.max(valid)),
        "num_scored": int(valid.size),
    }


def compute_signal_group_scores(
    score_matrix: np.ndarray,
    sorted_track_ids: list[int],
    tracks: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Compute per-signal-group correspondence score statistics for static objects.

    Groups lane markings, road boundaries, traffic lights/signs, crosswalks, and wait lines separately. Traffic-light
    groups also include aggregated state-check metrics when present on individual tracks.

    Args:
        score_matrix: Trimmed (processed_frames, num_tracks) correspondence matrix.
        sorted_track_ids: Track IDs aligned with score_matrix columns.
        tracks: Per-track summary entries with ``object_type`` and optional TL metrics.

    Returns:
        Mapping of signal-group name to score statistics. Groups with no scored entries are omitted.
    """
    if score_matrix.size == 0 or not sorted_track_ids:
        return {}

    track_id_to_idx = {track_id: idx for idx, track_id in enumerate(sorted_track_ids)}
    group_to_track_idxs: dict[str, list[int]] = {group: [] for group in STATIC_SIGNAL_GROUPS}
    group_to_tracks: dict[str, list[dict[str, Any]]] = {group: [] for group in STATIC_SIGNAL_GROUPS}

    for track in tracks:
        object_type = track.get("object_type")
        track_id = track.get("track_id")
        if object_type is None or track_id is None:
            continue
        group = STATIC_OBJECT_TYPE_TO_SIGNAL_GROUP.get(str(object_type))
        if group is None:
            continue
        track_idx = track_id_to_idx.get(int(track_id))
        if track_idx is None:
            continue
        group_to_track_idxs[group].append(track_idx)
        group_to_tracks[group].append(track)

    signal_group_scores: dict[str, dict[str, float | int]] = {}
    for group in STATIC_SIGNAL_GROUPS:
        track_idxs = group_to_track_idxs[group]
        if not track_idxs:
            continue
        group_matrix = score_matrix[:, track_idxs]
        stats = _correspondence_score_stats(group_matrix.reshape(-1))
        if not stats:
            continue
        stats["num_tracks"] = len(track_idxs)
        if group == "traffic_light":
            for metric_key in TRAFFIC_LIGHT_TRACK_METRICS:
                metric_values = [
                    float(track[metric_key])
                    for track in group_to_tracks[group]
                    if metric_key in track and not np.isnan(track[metric_key])
                ]
                if metric_values:
                    stats[metric_key] = float(np.mean(metric_values))
        signal_group_scores[group] = stats

    return signal_group_scores
