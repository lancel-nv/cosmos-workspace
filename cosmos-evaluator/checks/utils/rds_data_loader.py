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

from __future__ import annotations

import copy
import logging
from pathlib import Path
import re
from typing import Any, ClassVar, Literal

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.ftheta import FThetaCamera
from webdataset import WebDataset  # type: ignore[attr-defined]

from checks.utils.cuboid import Cuboid
from checks.utils.frame_rates import RDS_INDEX_STRIDE
from checks.utils.object_class_names import OBJECT_CLASS_TO_RDS_TYPES
from checks.utils.polyline import Polyline
from checks.utils.rds_geometry import RDSGeometry
from checks.utils.static_object_tracker import assign_static_track_ids
from checks.utils.surface import Surface

logger = logging.getLogger(__name__)

# Per-frame static label tar entries are keyed like ``NNNNNN.<type>.json`` (six or more leading digits). The canonical
# clip-level entry is keyed without the leading frame number, e.g. ``crosswalks.json``. The first capture group
# recovers the frame index for callers that need it (e.g. ``get_per_frame_labels``); ``get_labels`` only uses the
# pattern as a matcher to skip per-frame entries when picking the canonical key.
_PER_FRAME_LABEL_KEY_PATTERN = re.compile(r"^(\d{6,})\.[^.]+\.json$")

# State a traffic light reads at RDS indices no per-frame label covers, when its state is built from per-frame labels.
# ``"unknown"`` is the status vocabulary's no-determinable-color value, so such an index asserts no color while keeping
# every list entry a string, as in a ``3d_traffic_lights_status`` list.
_UNLABELED_TRAFFIC_LIGHT_STATE = "unknown"


class RdsDataLoader:
    # Camera whose pose stream is used as the ego-vehicle reference frame (camera-to-vehicle extrinsic ~= identity).
    EGO_REFERENCE_CAMERA: ClassVar[str] = "camera_front_wide_120fov"

    # Mapping from our high-level classes to dataset object types (for filtering objects)
    CLASS_TO_OBJECT_TYPES: ClassVar[dict[str, list[str]]] = {
        class_name: list(object_types) for class_name, object_types in OBJECT_CLASS_TO_RDS_TYPES.items()
    }

    def __init__(self, dataset_dir: str | Path, clip_id: str, preserve_raw: bool = False) -> None:
        """Initialize the RDS data loader.

        Args:
            dataset_dir: Base directory containing the RDS dataset assets.
            clip_id: Clip identifier (e.g., "<session_uuid>_<...>").
            preserve_raw: When True, retain side-tables mapping each parsed item back to its raw on-disk payload
                so that downstream tools (e.g. the RDS writeback CLI) can emit byte-identical data for unchanged
                objects. Off by default because most callers do not need it and it roughly doubles the memory
                cost of label data.
        """
        self.OBJECT_FOLDER = "all_object_info"
        self.POSE_FOLDER = "pose"
        self.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        self.TRAFFIC_LIGHT_STATUS_FOLDER = "3d_traffic_lights_status"
        self.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        self.LANE_LINE_FOLDER = "3d_lanelines"
        self.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        self.WAIT_LINE_FOLDER = "3d_wait_lines"
        self.CROSSWALK_FOLDER = "3d_crosswalks"
        self.PINHOLE_INTRINSICS_FOLDER = "pinhole_intrinsic"
        self.FTHETA_INTRINSICS_FOLDER = "ftheta_intrinsic"
        self.dataset_dir = dataset_dir
        self.clip_id = clip_id
        self.session_uuid = clip_id.split("_")[0]
        self.preserve_raw = preserve_raw
        # Raw payload side-tables (populated when preserve_raw=True).
        # Dynamic tar: frame index -> raw tar-entry key (e.g. "0.all_object_info.json").
        self._raw_dynamic_frame_keys: dict[int, str] = {}
        # Static label tars: folder -> (raw top-level JSON dict, tar-entry key). The top-level dict is the full JSON as
        # read; its "labels" list is replaced at write time. Any sibling keys ride along verbatim.
        self._raw_static_containers: dict[str, tuple[dict[str, Any], str]] = {}
        # Static object id -> (folder, index within that folder's raw "labels" list). Used to look up the raw
        # label_entry dict for verbatim re-emission.
        self._raw_static_label_index: dict[str, tuple[str, int]] = {}
        # Traffic light status tar: (raw top-level JSON dict, tar-entry key).
        self._raw_tl_status_container: tuple[dict[str, Any], str] | None = None
        # Cached frame-index axis for canonical-only static types. Populated on first access
        # (during _get_static_objects) and reused by callers that synthesize extra static objects so they don't
        # re-read the all_object_info tar each time.
        self._cached_canonical_static_frame_axis: list[int] | None = None
        # Cached per-frame static labels per folder. Populated lazily by ``get_per_frame_labels`` so the same tar isn't
        # decoded twice when both ``_frame_indices_from_per_frame_static`` and the per-type loop in
        # ``_get_static_objects`` query the same folder.
        self._cached_per_frame_labels: dict[str, dict[int, list[dict[str, Any]]]] = {}
        # Per-frame per-object ego ground distance (m), recorded lazily by _stamp_ego_distances (the single source of
        # truth for object->ego distance) and stamped onto each get_object_data_for_frame view as ``ego_distance_m``.
        self._object_ego_distances: dict[int, dict[Any, float | None]] = {}
        # Cached ego pose map, loaded once by _stamp_ego_distances (get_camera_poses re-reads the pose tar per call).
        self._ego_poses_cache: dict[int, np.ndarray] | None = None
        self.all_dynamic_objects, self.object_count = self._get_dynamic_object_data_all_frames()
        self.all_static_objects = self._get_static_objects()
        self.CAMERA_RESCALED_RESOLUTION_WIDTH = 1280
        self.CAMERA_RESCALED_RESOLUTION_HEIGHT = 720

    @staticmethod
    def get_sample(url: str | Path) -> dict[str, Any]:
        """Get a single WebDataset sample from a URL with basic auto-decoding.

        Args:
            url: Path or string URL to a WebDataset tar shard.

        Returns:
            A mapping representing the sample's decoded items keyed by filename-like strings.
        """
        if isinstance(url, Path):
            url = url.as_posix()

        # Use shardshuffle=False to suppress warning and since we only read single samples
        dataset = WebDataset(url, shardshuffle=False).decode()
        try:
            return next(iter(dataset))  # type: ignore[call-overload]
        finally:
            if hasattr(dataset, "close"):
                dataset.close()

    def _get_camera_intrinsics_array(
        self,
        camera_name: str,
        model_type: Literal["pinhole", "ftheta"],
    ) -> np.ndarray:
        """Load camera intrinsics as a numpy array for the given camera and model type.

        Args:
            camera_name: Name of the camera.
            model_type: Camera model type, one of "pinhole" or "ftheta".

        Returns:
            Numpy array containing the camera intrinsics for the specified camera.

        Raises:
            ValueError: If ``model_type`` is not supported.
            FileNotFoundError: If the intrinsics tar file is not found or cannot be opened.
            KeyError: If the intrinsics entry for ``camera_name`` is missing in the sample.
        """
        if model_type not in ["pinhole", "ftheta"]:
            raise ValueError(f"Invalid model_type '{model_type}'. Must be 'pinhole' or 'ftheta'")

        # Determine intrinsics folder based on model type
        intrinsics_folder = self.PINHOLE_INTRINSICS_FOLDER if model_type == "pinhole" else self.FTHETA_INTRINSICS_FOLDER
        intrinsics_file = Path(self.dataset_dir) / intrinsics_folder / f"{self.clip_id}.tar"

        # Check if intrinsics file exists
        if not intrinsics_file.exists():
            raise FileNotFoundError(f"Camera intrinsics file not found: {intrinsics_file}")

        # Load intrinsics data
        try:
            intrinsics_data = self.get_sample(intrinsics_file)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load camera intrinsics file {intrinsics_file}: {e}") from e

        # Check if camera intrinsics data exists
        camera_key = f"{model_type}_intrinsic.{camera_name}.npy"
        if camera_key not in intrinsics_data:
            available_keys = list(intrinsics_data.keys())
            raise KeyError(
                f"Camera intrinsics not found for camera '{camera_name}' with model '{model_type}'. "
                f"Available keys: {available_keys}"
            )

        intrinsic_this_cam = intrinsics_data[camera_key]
        return intrinsic_this_cam

    def get_camera_intrinsics(
        self,
        camera_name: str,
        model_type: Literal["pinhole", "ftheta"],
        rescaled: bool = True,
    ) -> FThetaCamera | None:
        """Build a camera model for the given camera name.

        Args:
            camera_name: Name of the camera.
            model_type: Camera model type, one of "pinhole" or "ftheta".
            rescaled: Whether to rescale camera intrinsics to the rescaled resolution.

        Returns:
            A ``FThetaCamera`` instance when ``model_type`` is "ftheta"; otherwise ``None``.

        Raises:
            ValueError: If ``model_type`` is invalid.
            FileNotFoundError: If the intrinsics file is missing.
            KeyError: If the camera intrinsics data is not found.
            RuntimeError: If building the camera model fails.
        """
        intrinsic_this_cam = self._get_camera_intrinsics_array(camera_name, model_type)

        camera_model = None

        if model_type == "ftheta":
            try:
                camera_model = FThetaCamera.from_numpy(intrinsic_this_cam, device="cpu")
                if rescaled:
                    rescale_ratio = self.CAMERA_RESCALED_RESOLUTION_HEIGHT / camera_model.height
                    camera_model.rescale(rescale_ratio)
            except Exception as e:
                raise RuntimeError(f"Failed to create FTheta camera model for '{camera_name}': {e}") from e

        return camera_model

    def get_camera_poses(self, camera_name: str) -> dict[int, np.ndarray]:
        """Get camera poses for a given camera.

        Note that camera poses are sampled at 30 FPS.

        Args:
            camera_name: Name of the camera.

        Returns:
            Mapping from pose frame index to 4x4 camera-to-world pose matrices.
        """
        pose_file = Path(self.dataset_dir) / self.POSE_FOLDER / f"{self.clip_id}.tar"
        camera_key = f"pose.{camera_name}.npy"
        pose_data = self.get_sample(pose_file)
        pose_data_this_cam = {int(k.split(".")[0]): v for k, v in pose_data.items() if camera_key in k}
        return pose_data_this_cam

    def load_camera_data(self, camera_name: str) -> tuple[dict[int, np.ndarray], FThetaCamera | None]:
        """Load camera poses and the ftheta intrinsics model for a camera.

        Args:
            camera_name: Name of the camera stream to load.

        Returns:
            Tuple of ``(camera_poses, camera_model)`` where ``camera_poses`` maps frame index to a 4x4 pose matrix and
            ``camera_model`` is the ftheta intrinsics model (``None`` if unavailable for the model type).
        """
        camera_poses = self.get_camera_poses(camera_name)
        camera_model = self.get_camera_intrinsics(camera_name, "ftheta")
        return camera_poses, camera_model

    def load_ego_poses(self) -> dict[int, np.ndarray]:
        """Load ego-vehicle reference poses from the front-wide camera pose stream.

        The front-wide camera pose stream is used as the ego-vehicle reference frame because its camera-to-vehicle
        extrinsic is approximately identity.

        Returns:
            Mapping from frame index to a 4x4 pose matrix.

        Raises:
            Propagates any error from ``get_camera_poses`` (e.g. missing pose file) so callers decide how to degrade.
        """
        return self.get_camera_poses(self.EGO_REFERENCE_CAMERA)

    def _get_dynamic_object_data_all_frames(self) -> tuple[dict[int, dict], int]:
        """Load object annotations for all frames in the clip.

        Returns:
            The mapping from frame index to per-frame object annotation structures, and the first ID *past*
            every dynamic object's — the seed for ``object_count``, which the static-object allocator and any
            other ID allocator consume as "next free ID". Returning the maximum rather than one
            past it made the first allocated static reuse the highest dynamic object's ID; because
            ``get_object_data_for_frame(include_static=True)`` merges statics over dynamics under a shared
            key space, that static then silently replaced the dynamic object for every occlusion-aware
            consumer (``SceneRasterizer`` and everything reading through it).

            A clip with no dynamic objects returns ``0``: there is nothing to clear, so the statics start
            where they always did rather than being shifted past an ID nobody holds.
        """
        # -1 so the "+1" below lands on 0 for a clip with no dynamic objects, and on max+1 otherwise. A
        # 0 start would make the empty case return 1 and skip a free ID.
        max_object_id = -1
        object_file = Path(self.dataset_dir) / self.OBJECT_FOLDER / f"{self.clip_id}.tar"
        object_data_raw = self.get_sample(object_file)
        object_data: dict[int, dict] = {}
        for tar_key, frame_objects in object_data_raw.items():
            if "all_object_info" not in tar_key:
                continue
            frame_idx = int(tar_key.split(".")[0])
            object_data[frame_idx] = frame_objects
            if getattr(self, "preserve_raw", False):
                self._raw_dynamic_frame_keys[frame_idx] = tar_key
        for frame_objects in object_data.values():
            for object_id, object_info in frame_objects.items():
                max_object_id = max(max_object_id, int(object_id))
                if "object_to_world" not in object_info or "object_lwh" not in object_info:
                    continue
                object_info["geometry"] = Cuboid(object_info["object_to_world"], object_info["object_lwh"])
                object_info["is_static"] = False
        return object_data, max_object_id + 1

    def get_object_ego_distances(self) -> dict[int, dict[Any, float | None]]:
        """Per-frame ego ground-plane distance (m) for every object (dynamic and static) in the clip.

        Recorded by the loader — the single source of truth for object->ego distance — so downstream consumers
        (importance filtering, distance-bucketed reports, KPIs) read the distance instead of each recomputing the
        geometry. For each frame and object the distance is ``RDSGeometry.ego_ground_distance_m`` against that frame's
        ego pose (``Cuboid`` returns the ground-plane hypotenuse; centroid-less geometry such as lanelines yields
        ``None``). ``get_object_data_for_frame`` stamps the same values onto each per-frame object view as
        ``ego_distance_m``; this method forces the full map by visiting every annotated frame.

        Returns:
            ``{frame_id: {object_id: distance_m_or_None}}`` covering every frame that carries object annotations.
        """
        # Union of dynamic frames and every static object's per-frame frame IDs, so static-only frames (a static
        # object present on a frame with no dynamic objects) are visited too, not just frames in all_dynamic_objects.
        frame_ids = set(self.all_dynamic_objects)
        for static_obj in self.all_static_objects.values():
            geometry_per_frame = static_obj.get("geometry_per_frame")
            if geometry_per_frame:
                frame_ids.update(geometry_per_frame)
        for frame_id in sorted(frame_ids):
            # Side effect: _stamp_ego_distances populates self._object_ego_distances[frame_id] (dynamic + static).
            self.get_object_data_for_frame(frame_id, include_static=True)
        return self._object_ego_distances

    def _stamp_ego_distances(self, frame_id: int, objects: dict[Any, dict[str, Any]]) -> None:
        """Record each object's ego ground distance for ``frame_id`` and stamp it onto the per-frame views.

        Each (frame, object) distance is computed once via ``RDSGeometry.ego_ground_distance_m`` against the frame's
        ego pose and cached in ``self._object_ego_distances``; later calls reuse the cache (and extend it when a call
        includes objects — e.g. static ones — not seen before for the frame). Every object view gets an
        ``ego_distance_m`` field (``None`` for centroid-less geometry or a missing ego pose) so the importance filter
        and other consumers read the recorded distance rather than recomputing it.
        """
        frame_cache = self._object_ego_distances.setdefault(frame_id, {})
        ego_pose: np.ndarray | None = None
        ego_pose_loaded = False
        for object_id, info in objects.items():
            if object_id not in frame_cache:
                geometry = info.get("geometry")
                if isinstance(geometry, RDSGeometry):
                    # Look up the frame's ego pose lazily and only when a real geometry needs it (centroid-less
                    # geometry and non-geometry sentinels take the None branch without touching the pose map). The
                    # pose map is loaded once and cached, so stamping every frame reads one tar, not one per frame.
                    if not ego_pose_loaded:
                        if self._ego_poses_cache is None:
                            self._ego_poses_cache = self.load_ego_poses()
                        ego_pose = self._ego_poses_cache.get(frame_id)
                        ego_pose_loaded = True
                    frame_cache[object_id] = geometry.ego_ground_distance_m(ego_pose) if ego_pose is not None else None
                else:
                    frame_cache[object_id] = None
            info["ego_distance_m"] = frame_cache[object_id]

    @staticmethod
    def get_objects_by_class(frame_objects: dict[str, Any], class_name: str) -> dict[str, Any]:
        """
        Filter objects of a specific high-level class from frame objects.

        Args:
            frame_objects: Mapping of track_id -> object dict
            class_name: High-level class name

        Returns:
            Subset of frame_objects containing only objects of the requested class
        """
        object_types = RdsDataLoader.CLASS_TO_OBJECT_TYPES.get(class_name, [class_name])
        return {track_id: obj for track_id, obj in frame_objects.items() if obj.get("object_type") in object_types}

    def _get_static_objects(self) -> dict[str, dict[str, Any]]:
        """Build the clip-level static-object dict from canonical and per-frame label files.

        Returns:
            Dict mapping static object id to a dictionary with keys:
            - ``object_type``: Object type (e.g., ``LaneLine``, ``RoadBoundary``).
            - ``geometry``: Frame-independent canonical geometry instance, parsed from the non-frame-numbered label
              file. Reserved for callers that genuinely need a single clip-level representation (e.g. lane-class
              summaries, ego-relevancy distance checks). Per-frame consumers should go through ``geometry_per_frame`` /
              ``get_object_data_for_frame`` instead.
            - ``attributes``: Frame-independent canonical attributes dict (only present when the source label carries
              attributes).
            - ``geometry_per_frame``: Mapping of frame index to per-frame geometry instance. Sourced from per-frame
              label files when the type's tar carries them; falls back to the canonical instance replicated across the
              clip-wide structural axis when the type is canonical-only.
            - ``attributes_per_frame``: Mapping of frame index to per-frame attributes dict (only present when the
              source data carries attributes). Same key set as ``geometry_per_frame``.
            - ``state_per_frame``: (TrafficLight only) list of states indexed by RDS frame index. Read from the
              light's ``3d_traffic_lights_status`` entry; a light without one has it built from the ``state``
              attribute of its per-frame labels (see ``_state_per_frame_from_labels``).

        Per-frame matching: when a type's tar contains per-frame label files, per-frame detections are matched to
        canonical objects via the static-object tracker (``assign_static_track_ids``). The tracker allocates canonical
        IDs first (in canonical-list order, preserving the existing per-clip ID scheme), then assigns fresh IDs to
        per-frame detections that don't match any canonical (synthesized tracks). Synthesized tracks adopt the geometry
        of their first appearance as their canonical ``geometry`` field.
        """
        # Parallel lists keyed by iteration order so we can thread the source folder through to _raw_static_label_index.
        obj_type_sources = [
            ("LaneLine", self.LANE_LINE_FOLDER, self.get_lanelines()),
            ("RoadBoundary", self.ROAD_BOUNDARY_FOLDER, self.get_road_boundaries()),
            ("WaitLine", self.WAIT_LINE_FOLDER, self.get_wait_lines()),
            ("TrafficLight", self.TRAFFIC_LIGHT_FOLDER, self.get_traffic_lights()),
            ("TrafficSign", self.TRAFFIC_SIGN_FOLDER, self.get_traffic_signs()),
            ("Crosswalk", self.CROSSWALK_FOLDER, self.get_crosswalks()),
        ]

        all_traffic_light_states = self._get_all_traffic_light_status()
        ego_poses = self._ego_poses_for_synthesized_canonical()
        label_state_num_frames = self._label_state_num_frames(ego_poses)

        # Phase 1 fallback axis used when a type has no per-frame label files.
        canonical_frame_axis = self.get_canonical_static_frame_axis()

        # Build canonical descriptors (flat, in the order LaneLine, RoadBoundary, WaitLine, TrafficLight, TrafficSign,
        # Crosswalk). The tracker walks this list to allocate canonical IDs upfront, so this ordering pins the existing
        # per-clip ID scheme.
        canonical_descriptors: list[dict[str, Any]] = []
        canonical_meta: list[dict[str, Any]] = []
        for obj_type, folder, label_data in obj_type_sources:
            for obj_type_idx, label in enumerate(label_data):
                geom = self._instantiate_label_geometry(label, obj_type)
                if geom is None:
                    continue
                canonical_descriptors.append({"object_type": obj_type, "geometry": geom})
                canonical_meta.append(
                    {
                        "object_type": obj_type,
                        "object_type_index": obj_type_idx,
                        "geometry": geom,
                        "attributes": label.get("attributes"),
                        "folder": folder,
                        "raw_index": label.get("_raw_index"),
                    }
                )

        # Build per-frame detections across all types that have per-frame label files. The tracker partitions matching
        # by ``object_type`` internally, so a flat detection list per frame is sufficient.
        per_frame_detections: dict[int, list[dict[str, Any]]] = {}
        types_with_per_frame: set[str] = set()
        for obj_type, folder, _label_data in obj_type_sources:
            per_frame_labels = self.get_per_frame_labels(folder)
            if not per_frame_labels:
                continue
            types_with_per_frame.add(obj_type)
            for fi, labels_at_frame in per_frame_labels.items():
                bucket = per_frame_detections.setdefault(fi, [])
                for label in labels_at_frame:
                    geom = self._instantiate_label_geometry(label, obj_type)
                    if geom is None:
                        continue
                    detection: dict[str, Any] = {"object_type": obj_type, "geometry": geom}
                    if "attributes" in label:
                        detection["attributes"] = label["attributes"]
                    bucket.append(detection)

        # ID allocator that mints sequential numerical IDs from ``self.object_count``. The tracker calls it
        # ``len(canonical_descriptors)`` times upfront for canonical IDs, then once per unmatched per-frame detection.
        canonical_track_ids: list[str] = []
        canonical_count = len(canonical_descriptors)

        def allocator() -> str:
            nid = str(self.object_count)
            self.object_count += 1
            if len(canonical_track_ids) < canonical_count:
                canonical_track_ids.append(nid)
            return nid

        assignment = assign_static_track_ids(
            canonical_descriptors,
            per_frame_detections,
            allocator,
        )
        tracker_result = assignment.per_frame

        static_objects: dict[str, dict[str, Any]] = {}

        # 1) Canonical tracks. Per-frame entries come from tracker matches for types whose tar carries per-frame data;
        # canonical-only types fall back to replicating the canonical across the structural axis.
        canonical_id_set: set[str] = set(canonical_track_ids)
        for i, meta in enumerate(canonical_meta):
            track_id = canonical_track_ids[i]
            # Connectivity grouping puts every fragment of one physical line into a single track, so the
            # fragments that did not win the ID are already represented by the one that did. Emitting them
            # would add objects with no per-frame geometry that still look enumerable to callers reading
            # ``all_static_objects`` directly (KPI target selection does exactly that). A canonical that is
            # merely never observed is *not* in this set and is still emitted -- it is the only record of
            # its geometry.
            if track_id in assignment.absorbed_canonical_ids:
                continue
            obj: dict[str, Any] = {
                "object_type": meta["object_type"],
                "is_static": True,
                "object_type_index": meta["object_type_index"],
                # For a grouped polyline track the clip-level stand-in is the greatest-coverage observation of
                # the whole line rather than this one fragment, so ``geometry`` and ``geometry_per_frame``
                # describe the same object.
                "geometry": assignment.clip_geometry.get(track_id, meta["geometry"]),
            }
            if meta["attributes"] is not None:
                obj["attributes"] = meta["attributes"]

            if meta["object_type"] in types_with_per_frame:
                geometry_per_frame: dict[int, Any] = {}
                attributes_per_frame: dict[int, dict] = {}
                for fi, frame_data in tracker_result.items():
                    det = frame_data.get(track_id)
                    if det is None:
                        continue
                    geometry_per_frame[fi] = det["geometry"]
                    if "attributes" in det:
                        attributes_per_frame[fi] = det["attributes"]
                obj["geometry_per_frame"] = geometry_per_frame
                if attributes_per_frame:
                    obj["attributes_per_frame"] = attributes_per_frame
            else:
                # Canonical-only fallback: replicate canonical across the clip's structural axis so the per-frame API
                # surfaces the object at every frame.
                obj["geometry_per_frame"] = {fi: meta["geometry"] for fi in canonical_frame_axis}
                if meta["attributes"] is not None:
                    obj["attributes_per_frame"] = {fi: meta["attributes"] for fi in canonical_frame_axis}

            if meta["object_type"] == "TrafficLight":
                status_entry = all_traffic_light_states.get(str(meta["object_type_index"]), {})
                state_per_frame = status_entry.get("state")
                if state_per_frame is None and label_state_num_frames is not None:
                    state_per_frame = self._state_per_frame_from_labels(obj, label_state_num_frames)
                if state_per_frame is not None:
                    obj["state_per_frame"] = state_per_frame

            static_objects[track_id] = obj
            if getattr(self, "preserve_raw", False) and meta["raw_index"] is not None:
                self._raw_static_label_index[track_id] = (meta["folder"], meta["raw_index"])

        # 2) Synthesized tracks: per-frame detections that didn't match any canonical. Group observations per track
        # first, then pick the observation whose geometry was closest to the ego vehicle (in BEV / XY space) as the
        # canonical ``geometry``/``attributes``. The closest-to-ego observation is the most reliable representative
        # of a clip-level summary because perception quality drops with range — the near-ego frame's labeling is the
        # cleanest snapshot.
        synthesized_observations: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for fi in sorted(tracker_result):
            for tid, det in tracker_result[fi].items():
                if tid in canonical_id_set:
                    continue
                synthesized_observations.setdefault(tid, []).append((fi, det))

        for tid, observations in synthesized_observations.items():
            _canonical_fi, canonical_det = self._pick_closest_to_ego_observation(observations, ego_poses)
            new_obj: dict[str, Any] = {
                "object_type": canonical_det["object_type"],
                "is_static": True,
                "object_type_index": None,
                "geometry": assignment.clip_geometry.get(tid, canonical_det["geometry"]),
                "geometry_per_frame": {fi: det["geometry"] for fi, det in observations},
            }
            if "attributes" in canonical_det:
                new_obj["attributes"] = canonical_det["attributes"]
            attrs_per_frame = {fi: det["attributes"] for fi, det in observations if "attributes" in det}
            if attrs_per_frame:
                new_obj["attributes_per_frame"] = attrs_per_frame
            # A synthesized light has no ordinal in the canonical label list, so no status entry can key to it.
            if new_obj["object_type"] == "TrafficLight" and label_state_num_frames is not None:
                state_per_frame = self._state_per_frame_from_labels(new_obj, label_state_num_frames)
                if state_per_frame is not None:
                    new_obj["state_per_frame"] = state_per_frame
            static_objects[tid] = new_obj

        return static_objects

    # Class-level constant: camera used to resolve ego pose when picking the closest-to-ego observation for
    # synthesized-track canonicals. Clips that lack this camera fall back to first-seen.
    SYNTHESIZED_CANONICAL_CAMERA: ClassVar[str] = "camera_front_wide_120fov"

    def _ego_poses_for_synthesized_canonical(self) -> dict[int, np.ndarray]:
        """Return ego poses keyed by frame index for synthesized-canonical selection.

        Reads from ``self.SYNTHESIZED_CANONICAL_CAMERA``. Returns ``{}`` when the camera's pose tar is missing or
        unreadable; the caller treats that as "no closest-to-ego signal available" and falls back to first-seen.
        """
        try:
            return self.get_camera_poses(self.SYNTHESIZED_CANONICAL_CAMERA)
        except Exception:  # noqa: BLE001 - missing/unreadable pose tar yields no signal
            return {}

    @staticmethod
    def _pick_closest_to_ego_observation(
        observations: list[tuple[int, dict[str, Any]]],
        ego_poses: dict[int, np.ndarray],
    ) -> tuple[int, dict[str, Any]]:
        """Pick the (frame_idx, detection) whose geometry is closest to ego in XY.

        BEV space is the ground plane, so we measure horizontal (XY) distance and ignore Z. For polylines and surfaces
        we use the minimum vertex distance (their world extent matters); cuboids use their pose translation. When
        ``ego_poses`` is empty or no observation has a matching ego pose, the first-seen observation is returned as a
        fallback.
        """
        if not observations:
            raise ValueError("observations must be non-empty")
        if not ego_poses:
            return observations[0]
        best: tuple[int, dict[str, Any]] | None = None
        best_dist = float("inf")
        for fi, det in observations:
            ego_pose = ego_poses.get(fi)
            if ego_pose is None:
                continue
            ego_xy = np.asarray(ego_pose, dtype=float)[:2, 3]
            d = RdsDataLoader._min_xy_distance(det["geometry"], ego_xy)
            if d < best_dist:
                best_dist = d
                best = (fi, det)
        return best if best is not None else observations[0]

    @staticmethod
    def _min_xy_distance(geom: Any, point_xy: np.ndarray) -> float:
        """Minimum XY distance from a geometry to a 2D point.

        For Polyline / Surface (vertex-based geometries) the minimum vertex distance is used, so a long polyline that
        passes near ``point_xy`` scores small even if its centroid is far. For Cuboid (pose-based) the pose translation
        is used. Falls back to ``+inf`` for unrecognized geometry shapes.
        """
        if hasattr(geom, "vertices_world"):
            diffs = np.asarray(geom.vertices_world, dtype=float)[:, :2] - point_xy
            return float(np.linalg.norm(diffs, axis=1).min())
        if hasattr(geom, "object_to_world_pose"):
            center_xy = np.asarray(geom.object_to_world_pose, dtype=float)[:2, 3]
            return float(np.linalg.norm(center_xy - point_xy))
        if hasattr(geom, "corners"):
            diffs = np.asarray(geom.corners, dtype=float)[:, :2] - point_xy
            return float(np.linalg.norm(diffs, axis=1).min())
        return float("inf")

    @staticmethod
    def _instantiate_label_geometry(label: dict[str, Any], obj_type: str) -> Polyline | Cuboid | Surface | None:
        """Build an :class:`RDSGeometry` from a parsed label entry.

        Returns ``None`` when the label is missing required fields, has an unknown ``geom_type``, or fails to
        instantiate (e.g. a malformed cuboid). Failures are logged at warning level so they're visible without crashing
        the loader; the caller skips the entry on ``None``.
        """
        if "vertices" not in label or "geom_type" not in label:
            return None
        try:
            if label["geom_type"] == "polyline":
                return Polyline(label["vertices"])
            if label["geom_type"] == "cuboid":
                pose, lwh = Cuboid.compute_pose_and_lwh_from_corners(label["vertices"])
                return Cuboid(pose, lwh)
            if label["geom_type"] == "surface":
                return Surface(label["vertices"])
        except Exception as e:  # noqa: BLE001 - malformed geometry label returns None
            logger.warning("Failed to instantiate %s for %s: %s", label["geom_type"], obj_type, e)
            return None
        return None

    def get_canonical_static_frame_axis(self) -> list[int]:
        """Return the sorted list of frame indices used for canonical-only static types.

        Phase 1 fallback for static types that do not have per-frame label files yet (poles, road_markings,
        traffic_lights, traffic_signs): their canonical geometry is replicated across this axis so the per-frame API
        can surface them.

        Source order (each step is independent of dynamic object content):
            1. Per-frame indices already discovered in any other static label tar (preferred — keeps the static frame
               axis grounded in static data).
            2. Frame-index keys present in the all_object_info tar. These are structural to the clip and exist even
               when every frame's object set is empty.
            3. Empty list — last resort for clips with no all_object_info tar at all (test fixtures).

        TODO(static-per-frame): step 1 becomes a no-op until ``get_labels`` starts emitting per-frame indices; today
        only step 2 contributes.

        The result is cached on the loader so repeated calls (e.g. from callers synthesizing extra static objects)
        do not re-read the tar.
        """
        if self._cached_canonical_static_frame_axis is not None:
            return self._cached_canonical_static_frame_axis
        per_frame_axis = self._frame_indices_from_per_frame_static()
        if per_frame_axis:
            self._cached_canonical_static_frame_axis = per_frame_axis
        else:
            self._cached_canonical_static_frame_axis = self._frame_indices_from_object_info_tar()
        return self._cached_canonical_static_frame_axis

    def _frame_indices_from_per_frame_static(self) -> list[int]:
        """Return the union of per-frame indices observed across all static label tars.

        Each static label folder may contribute zero or more per-frame ``NNNNNN.<type>.json`` entries; this method
        aggregates their frame indices so canonical-only static types (e.g. poles, road markings, traffic lights/signs)
        inherit the same frame axis as the types whose tars do carry per-frame data. Returns ``[]`` when no static
        type in the dataset has per-frame entries; ``get_canonical_static_frame_axis`` then falls back to the
        all_object_info tar's structural axis.
        """
        folders = (
            self.LANE_LINE_FOLDER,
            self.ROAD_BOUNDARY_FOLDER,
            self.WAIT_LINE_FOLDER,
            self.TRAFFIC_LIGHT_FOLDER,
            self.TRAFFIC_SIGN_FOLDER,
            self.CROSSWALK_FOLDER,
        )
        indices: set[int] = set()
        for folder in folders:
            indices.update(self.get_per_frame_labels(folder).keys())
        return sorted(indices)

    def _frame_indices_from_object_info_tar(self) -> list[int]:
        """Return the sorted frame indices found in the all_object_info tar.

        Reads the structural axis directly from the tar's keys so the result is independent of the parsed
        dynamic-object content. Returns ``[]`` when the tar is missing or unreadable, which leaves canonical-only
        static types with empty ``geometry_per_frame`` (consistent with "no clip frames known").
        """
        object_file = Path(self.dataset_dir) / self.OBJECT_FOLDER / f"{self.clip_id}.tar"
        if not object_file.exists():
            return []
        try:
            sample = self.get_sample(object_file)
        except Exception:  # noqa: BLE001 - unreadable object tar yields empty result
            return []
        indices: set[int] = set()
        for tar_key in sample:
            if "all_object_info" not in tar_key:
                continue
            try:
                indices.add(int(tar_key.split(".")[0]))
            except (ValueError, IndexError):
                continue
        return sorted(indices)

    def _get_all_traffic_light_status(self) -> dict[str, Any]:
        """Load traffic light status for the clip from the RDS traffic-light-status tar.

        Reads the clip's traffic_light_states from the dataset tar (keyed by traffic light object index). Used when
        building static objects to attach state_per_frame to TrafficLight instances.

        Returns:
            Mapping from traffic light object index (str) to per-light data (e.g. "state" per frame). Empty dict if the
            tar is missing or contains no JSON.
        """
        tar_path = Path(self.dataset_dir) / self.TRAFFIC_LIGHT_STATUS_FOLDER / f"{self.clip_id}.tar"
        if not tar_path.exists():
            return {}

        sample = self.get_sample(tar_path)
        json_key = [k for k in sample if k.endswith(".json")]
        if not json_key:
            return {}

        top_level = sample[json_key[0]]
        if getattr(self, "preserve_raw", False) and isinstance(top_level, dict):
            self._raw_tl_status_container = (top_level, json_key[0])
        all_traffic_light_states = top_level.get("traffic_light_states", {}) if isinstance(top_level, dict) else {}
        return all_traffic_light_states

    def _label_state_num_frames(self, poses: dict[int, np.ndarray]) -> int | None:
        """Length of a ``state_per_frame`` list built from per-frame traffic-light labels.

        One entry per pose index, the length of a ``3d_traffic_lights_status`` list. Without poses the list ends where
        the hold of the last labeled frame ends.

        Returns:
            The list length, or ``None`` when no list may be built: the clip has no per-frame traffic-light labels, or
            a label sits off the ``RDS_INDEX_STRIDE`` grid. A canonical-only light carries its clip-level ``state``
            attribute on every frame, which is a summary rather than a per-frame observation, and an off-grid label
            held forward would cover RDS indices other than the ones it was recorded at.
        """
        label_frames = sorted(self.get_per_frame_labels(self.TRAFFIC_LIGHT_FOLDER))
        if not label_frames:
            return None
        off_grid = [frame for frame in label_frames if frame % RDS_INDEX_STRIDE != 0]
        if off_grid:
            logger.warning(
                "Clip %s: %d per-frame traffic-light label(s) are off the every-%d RDS index grid (first: %s); "
                "traffic-light state is not read from per-frame labels",
                self.clip_id,
                len(off_grid),
                RDS_INDEX_STRIDE,
                off_grid[:5],
            )
            return None
        return max(poses) + 1 if poses else label_frames[-1] + RDS_INDEX_STRIDE

    @staticmethod
    def _state_per_frame_from_labels(static_obj: dict[str, Any], num_frames: int) -> list[str] | None:
        """Build a traffic light's ``state_per_frame`` from the ``state`` attribute of its per-frame labels.

        Labels are recorded every ``RDS_INDEX_STRIDE`` RDS indices. The state labeled at index ``f`` is held forward
        over ``f`` .. ``f + RDS_INDEX_STRIDE - 1``, so the entry at every labeled index is that label's own state.
        An index no label covers, including a 10 Hz slot in which the light went unlabeled, reads
        ``_UNLABELED_TRAFFIC_LIGHT_STATE``; a state never carries across such a gap.

        Args:
            static_obj: Static-object dict with ``geometry_per_frame`` and ``attributes_per_frame``.
            num_frames: Length of the returned list; labels at or beyond it are dropped.

        Returns:
            The state per RDS index, or ``None`` when the light has no per-frame observation or any observation lacks
            a non-empty string ``state``. A partially labeled light is left without a state rather than read as
            ``"unknown"`` at the observations whose state was not recorded.
        """
        geometry_per_frame = static_obj.get("geometry_per_frame") or {}
        attributes_per_frame = static_obj.get("attributes_per_frame") or {}
        if not geometry_per_frame:
            return None
        labeled_states: dict[int, str] = {}
        for frame in geometry_per_frame:
            state = (attributes_per_frame.get(frame) or {}).get("state")
            if not isinstance(state, str) or not state.strip():
                return None
            labeled_states[frame] = state
        state_per_frame = [_UNLABELED_TRAFFIC_LIGHT_STATE] * num_frames
        for frame, state in labeled_states.items():
            for index in range(frame, min(frame + RDS_INDEX_STRIDE, num_frames)):
                state_per_frame[index] = state
        return state_per_frame

    def get_object_data_for_frame(self, frame_id: int, include_static: bool = False) -> dict:
        """Get raw object annotations for a frame.

        Args:
            frame_id: Frame index.
            include_static: When True, merge static objects whose ``geometry_per_frame`` contains ``frame_id``. Each
                static object is presented as a flat per-frame dict: the ``geometry_per_frame`` and
                ``attributes_per_frame`` indexed dicts are stripped out and replaced with single ``geometry`` and
                (optionally) ``attributes`` fields selected for ``frame_id``. Static objects whose
                ``geometry_per_frame`` does not include ``frame_id`` are omitted entirely.

        Returns:
            A dictionary containing the per-frame object annotations. Each object view carries an ``ego_distance_m``
            field (the loader-recorded ego ground-plane distance for this frame, ``None`` for centroid-less geometry).
        """
        all_objects = copy.deepcopy(self.all_dynamic_objects.get(frame_id, {}))
        if include_static:
            for tid, static_obj in self.all_static_objects.items():
                per_frame_view = self._static_object_view_for_frame(static_obj, frame_id)
                if per_frame_view is None:
                    continue
                # Dynamic and static objects share one key space here, so a duplicate ID means the static
                # view replaces a real dynamic object and every occlusion-aware consumer silently measures
                # the wrong thing. That is always an ID-allocation bug upstream, never legitimate, so say so
                # rather than letting it corrupt results quietly.
                if tid in all_objects:
                    logger.warning(
                        "Frame %s: static object %s (%s) shadows a dynamic object (%s) sharing its ID; "
                        "the dynamic object will be invisible to occlusion-aware consumers",
                        frame_id,
                        tid,
                        static_obj.get("object_type"),
                        all_objects[tid].get("object_type"),
                    )
                all_objects[tid] = per_frame_view
        self._stamp_ego_distances(frame_id, all_objects)
        return all_objects

    @staticmethod
    def _static_object_view_for_frame(static_obj: dict[str, Any], frame_id: int) -> dict[str, Any] | None:
        """Build the flat per-frame view of a static object for ``frame_id``.

        The clip-level ``all_static_objects`` dict carries both the canonical ``geometry``/``attributes``
        (frame-independent) and the per-frame ``geometry_per_frame``/``attributes_per_frame`` indexed dicts. Per-frame
        consumers expect a single ``geometry`` (and optionally ``attributes``) field selected for the frame, so this
        helper:

        - drops the indexed dicts and the canonical ``geometry``/``attributes`` (the per-frame view replaces them
          with the entry for ``frame_id``);
        - sets ``geometry`` from ``geometry_per_frame[frame_id]``;
        - sets ``attributes`` from ``attributes_per_frame[frame_id]`` when present.

        Returns ``None`` when ``geometry_per_frame`` does not include ``frame_id`` so the caller can omit the object at
        that frame.
        """
        geometry_per_frame = static_obj.get("geometry_per_frame")
        if not geometry_per_frame or frame_id not in geometry_per_frame:
            return None

        view: dict[str, Any] = {}
        for key, value in static_obj.items():
            if key in ("geometry", "attributes", "geometry_per_frame", "attributes_per_frame"):
                continue
            view[key] = copy.deepcopy(value)
        view["geometry"] = copy.deepcopy(geometry_per_frame[frame_id])

        attributes_per_frame = static_obj.get("attributes_per_frame")
        if attributes_per_frame and frame_id in attributes_per_frame:
            view["attributes"] = copy.deepcopy(attributes_per_frame[frame_id])
        return view

    def get_fps(self, data_type: str) -> float:
        """Get the FPS for a given data type.

        Args:
            data_type: Type of data ('object' or 'pose')

        Returns:
            FPS value (10.0 for 'object', 30.0 for 'pose')

        Raises:
            ValueError: If data_type is not 'object' or 'pose'
        """
        if data_type == "object":
            return 10.0
        elif data_type == "pose":
            return 30.0
        else:
            raise ValueError(f"Invalid data_type '{data_type}'. Must be 'object' or 'pose'")

    def get_time_range(self) -> tuple[int, int]:
        """Get the start and end timestamps for the clip in microseconds.

        The start timestamp is extracted from the clip_id (part after first '_'). The end timestamp is calculated based
        on the number of frames in all_object_info, which is sampled at 10 FPS.

        Returns:
            Tuple of (start_timestamp_us, end_timestamp_us)
        """
        # Extract start timestamp from clip_id (format: uuid_timestamp)
        parts = self.clip_id.split("_")
        if len(parts) < 2:
            raise ValueError(f"Invalid clip_id format: {self.clip_id}. Expected format: uuid_timestamp")

        try:
            start_timestamp_us = int(parts[1])
        except ValueError as e:
            raise ValueError(f"Failed to parse timestamp from clip_id '{self.clip_id}': {e}") from e

        # Calculate end timestamp using number of object frames and FPS
        num_frames = len(self.all_dynamic_objects)
        object_fps = self.get_fps("object")
        frame_duration_us = int(1.0 / object_fps * 1e6)
        end_timestamp_us = start_timestamp_us + (num_frames * frame_duration_us)

        return start_timestamp_us, end_timestamp_us

    @staticmethod
    def _extract_shape_attributes(shape3d: dict[str, Any]) -> dict[str, Any]:
        attributes: dict[str, Any] = {}

        for attr in shape3d.get("attributes", []):
            attr_name = attr.get("name")
            if not attr_name:
                continue

            value = None

            enums_list_container = attr.get("enumsList")
            if isinstance(enums_list_container, dict):
                enums_list = enums_list_container.get("enumsList", [])
                if enums_list:
                    value = enums_list

            if value is None and "text" in attr:
                text_value = attr.get("text")
                if text_value not in (None, ""):
                    value = text_value

            if value is not None:
                attributes[attr_name] = value

        return attributes

    @staticmethod
    def _parse_shape3d_entry(shape3d: Any) -> dict[str, Any] | None:
        if not isinstance(shape3d, dict):
            return None

        geom_type = None
        if "polyline3d" in shape3d:
            geom_data = shape3d["polyline3d"]
            geom_type = "polyline"
        elif "surface" in shape3d:
            geom_data = shape3d["surface"]
            geom_type = "surface"
        elif "cuboid3d" in shape3d:
            geom_data = shape3d["cuboid3d"]
            geom_type = "cuboid"
        else:
            return None

        vertices = geom_data.get("vertices", []) if isinstance(geom_data, dict) else []
        if not vertices:
            return None

        label_entry = {
            "vertices": np.asarray(vertices, dtype=np.float32),
            "geom_type": geom_type,
        }

        attributes = RdsDataLoader._extract_shape_attributes(shape3d)
        if attributes:
            label_entry["attributes"] = attributes

        return label_entry

    def get_labels(self, folder: str) -> list[dict[str, Any]]:
        """Load canonical (clip-level) label data from a specific folder.

        Args:
            folder: Folder name for label data (e.g., "3d_lanelines", "3d_wait_lines")

        Returns:
            List of dictionaries containing:
                - vertices: numpy array (N, 3) of 3D coordinates
                - attributes: dict of attribute_name -> list of per-vertex values. For label types without attributes,
                  this will be an empty dict.

        When the tar contains both a canonical (non-frame-numbered) entry and per-frame entries
        (``NNNNNN.<type>.json``), this method returns only the canonical's labels. Use ``get_per_frame_labels`` for the
        per-frame entries.
        """
        tar_path = Path(self.dataset_dir) / folder / f"{self.clip_id}.tar"
        if not tar_path.exists():
            return []

        sample = self.get_sample(tar_path)
        canonical_key = self._find_canonical_label_key(sample)
        if canonical_key is None:
            return []

        data = sample[canonical_key]

        if not isinstance(data, dict) or "labels" not in data:
            # Fallback to old vertex extraction for backward compatibility
            return []

        preserve_raw = getattr(self, "preserve_raw", False)
        if preserve_raw:
            self._raw_static_containers[folder] = (data, canonical_key)

        return RdsDataLoader._extract_labels_from_top_level(data, with_raw_index=preserve_raw)

    def get_per_frame_labels(self, folder: str) -> dict[int, list[dict[str, Any]]]:
        """Load per-frame label data from a static label tar.

        Args:
            folder: Folder name for label data (e.g. ``"3d_crosswalks"``).

        Returns:
            Mapping ``{frame_index: [label_entry, ...]}``. Each entry has the same shape as those returned by
            ``get_labels``: ``vertices``, ``geom_type``, optional ``attributes``. The dict is empty when the tar
            contains only a canonical (non-frame-numbered) entry, is missing, or is unreadable.

        Phase 2: ``_get_static_objects`` consumes this to populate ``geometry_per_frame`` / ``attributes_per_frame`` on
        each static object dict, replacing the Phase 1 canonical-replication fallback for types whose tars carry
        per-frame entries.
        """
        # Lazy-init cache so test fixtures using ``RdsDataLoader.__new__`` to
        # bypass ``__init__`` don't need to seed it.
        if not hasattr(self, "_cached_per_frame_labels"):
            self._cached_per_frame_labels = {}
        if folder in self._cached_per_frame_labels:
            return self._cached_per_frame_labels[folder]

        per_frame: dict[int, list[dict[str, Any]]] = {}
        tar_path = Path(self.dataset_dir) / folder / f"{self.clip_id}.tar"
        if tar_path.exists():
            sample = self.get_sample(tar_path)
            for key, value in sample.items():
                match = _PER_FRAME_LABEL_KEY_PATTERN.match(key)
                if match is None:
                    continue
                if not isinstance(value, dict) or "labels" not in value:
                    continue
                frame_idx = int(match.group(1))
                per_frame[frame_idx] = RdsDataLoader._extract_labels_from_top_level(value, with_raw_index=False)

        self._cached_per_frame_labels[folder] = per_frame
        return per_frame

    @staticmethod
    def _find_canonical_label_key(sample: dict[str, Any]) -> str | None:
        """Pick the canonical (non-frame-numbered) JSON key from a sample dict.

        WebDataset groups every static label tar entry into a single sample keyed by its filename suffix (e.g.
        ``crosswalks.json`` for the canonical, ``000003.crosswalks.json`` for per-frame entries).  This helper returns
        the canonical key — the one whose filename does not start with a 6+ digit frame index — or ``None`` if no
        canonical entry exists in the sample.

        Without this disambiguation a sample-dict-iteration-order pick would silently treat per-frame data (typically
        the first entry in tar order) as if it were the canonical clip-level summary.
        """
        for key in sample:
            if not isinstance(key, str) or not key.endswith(".json"):
                continue
            if _PER_FRAME_LABEL_KEY_PATTERN.match(key):
                continue
            return key
        return None

    @staticmethod
    def _extract_labels_from_top_level(data: dict[str, Any], with_raw_index: bool) -> list[dict[str, Any]]:
        """Extract label entries from a top-level static-label JSON dict.

        Walks ``data["labels"]``, parses each entry's ``shape3d`` via :meth:`_parse_shape3d_entry`, and returns the
        list of ``{vertices, geom_type, attributes?}`` dicts. When ``with_raw_index`` is True, each parsed entry is
        annotated with ``_raw_index`` (its position in the source list) so callers like ``get_labels`` can thread the
        index through to ``preserve_raw`` bookkeeping.
        """
        if not isinstance(data, dict) or "labels" not in data:
            return []
        extracted: list[dict[str, Any]] = []
        for raw_index, label_entry in enumerate(data["labels"]):
            if "labelData" not in label_entry:
                continue
            label_data = label_entry["labelData"]
            if "shape3d" not in label_data:
                continue
            parsed = RdsDataLoader._parse_shape3d_entry(label_data["shape3d"])
            if parsed is None:
                continue
            if with_raw_index:
                parsed["_raw_index"] = raw_index
            extracted.append(parsed)
        return extracted

    def get_lanelines(self) -> list[dict[str, Any]]:
        """Get 3D laneline data with attributes (colors, styles, etc.).

        Returns:
            List of lane lines, each is a dictionary containing:
                - vertices: numpy array (N, 3) of 3D coordinates
                - attributes: dict with keys like 'colors', 'styles', storing
                  per-vertex attribute values when available

        Example:
            [
                {
                    "vertices": array([[x1, y1, z1], [x2, y2, z2], ...]),
                    "attributes": {
                        "colors": [...],
                        "styles": [...]
                    }
                },
                ...
            ]
        """
        return self.get_labels(self.LANE_LINE_FOLDER)

    def get_road_boundaries(self) -> list[dict[str, Any]]:
        """Get 3D road boundary polylines without per-lane attributes."""
        return self.get_labels(self.ROAD_BOUNDARY_FOLDER)

    def get_wait_lines(self) -> list[dict[str, Any]]:
        """Get 3D wait line polylines without per-lane attributes."""
        return self.get_labels(self.WAIT_LINE_FOLDER)

    def get_crosswalks(self) -> list[dict[str, Any]]:
        """Get 3D crosswalk surfaces."""
        return self.get_labels(self.CROSSWALK_FOLDER)

    def get_traffic_lights(self) -> list[dict[str, Any]]:
        """Get 3D traffic light cuboids with attributes when available."""
        return self.get_labels(self.TRAFFIC_LIGHT_FOLDER)

    def get_traffic_signs(self) -> list[dict[str, Any]]:
        """Get 3D traffic sign cuboids with attributes when available."""
        return self.get_labels(self.TRAFFIC_SIGN_FOLDER)

    def dump_camera_intrinsics_to_csv(self, camera_name: str, output_dir: str) -> dict[str, bool]:
        """Dump the camera intrinsics for all available models to a CSV file for a given camera name.
        Args:
            camera_name: The camera name.
            output_dir: Directory where the CSV files will be written.

        Returns:
            Mapping from camera type ("pinhole"/"ftheta") to a boolean indicating success for that type.
        """
        camera_types: list[Literal["pinhole", "ftheta"]] = ["pinhole", "ftheta"]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        success: dict[str, bool] = {camera_type: True for camera_type in camera_types}
        for camera_type in camera_types:
            try:
                intrinsic_this_cam = self._get_camera_intrinsics_array(camera_name, camera_type)
                output_file = output_path / f"{camera_type}_intrinsic.{camera_name}.csv"
                # Flatten the array and save as a single row
                np.savetxt(output_file, intrinsic_this_cam.flatten()[np.newaxis, :], delimiter=",")
            except Exception:  # noqa: BLE001 - per-camera intrinsics dump failure recorded as False
                success[camera_type] = False
        return success

    def dump_camera_frame_egomotion_to_csv(
        self,
        camera_name: str,
        video_fps: float,
        output_dir: str | Path,
    ) -> None:
        """Write camera egomotion for a camera to a CSV aligned to a target FPS.

        Args:
            camera_name: The camera name.
            video_fps: Target video FPS. Poses are sampled at 30 FPS; ``video_fps`` must be in ``(0, 30]``
                and cleanly divide 30.
            output_dir: Directory where the CSV will be written.

        Raises:
            ValueError: If ``video_fps`` is not in the valid range or does not cleanly divide 30.
        """
        # Validate FPS: must be > 0 and <= 30
        if video_fps <= 0.0 or video_fps > 30.0:
            raise ValueError("Invalid video FPS, must be greater than 0 and at most 30")
        pose_data_this_cam = self.get_camera_poses(camera_name)
        sorted_pose_data_this_cam = sorted(pose_data_this_cam.items())

        # Prepare data for CSV: each row is [frame_number, timestamp_us, pose.flatten()]; rotation matrix is
        # flattened in column-major order
        rows = []
        delta_time_us = 1.0 / video_fps * 1e6
        # Must cleanly divide 30 FPS
        ratio = 30.0 / video_fps
        frame_skip = round(ratio)
        if abs(ratio - frame_skip) > 1e-6:
            raise ValueError("Invalid video FPS, must cleanly divide 30")
        for frame_num, pose in sorted_pose_data_this_cam:
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            # Only dump the camera poses for the video frames, and re-number the frames such that they
            # correspond to the video frames.
            if frame_num % frame_skip == 0:
                video_frame_num = int(frame_num / frame_skip)
                row = (
                    [video_frame_num]
                    + [int(video_frame_num * delta_time_us)]
                    + rotation.flatten(order="F").tolist()
                    + translation.flatten(order="F").tolist()
                )
                rows.append(row)

        # Write to CSV
        output_file = Path(output_dir) / f"pose.{camera_name}.csv"
        # Format: first two values as integer, rest as float
        fmt = ["%d"] + ["%d"] + ["%.8f"] * (len(rows[0]) - 2) if rows else ["%d"]
        header = "frame,timestamp_us,R00,R10,R20,R01,R11,R21,R02,R12,R22,t0,t1,t2"
        np.savetxt(output_file, rows, delimiter=",", fmt=fmt, header=header, comments="")
