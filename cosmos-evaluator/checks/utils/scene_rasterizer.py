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
Scene rasterizer for computing occlusion-aware visibility masks.

This module provides a SceneRasterizer class that projects multiple RDSGeometry
objects into image space and computes visibility masks accounting for occlusions
using depth buffer compositing.
"""

import logging
from typing import Any

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.utils.rds_geometry import RDSGeometry


class SceneRasterizer:
    """
    Rasterizes a collection of RDSGeometry objects into image space with occlusion handling.

    The SceneRasterizer takes a collection of geometry objects and projects them
    all into image space, computing per-object visibility masks that account for
    occlusions based on depth buffer compositing.

    The visibility masks are computed in the constructor, so a new SceneRasterizer
    should be created for each frame. This ensures that the camera pose and model
    are always synced to the object data.

    Usage:
        # Get objects from RdsDataLoader
        frame_objects = data_loader.get_object_data_for_frame(frame_id, include_static=True)
        vehicles = RdsDataLoader.get_objects_by_class(frame_objects, "vehicle")

        # Create rasterizer with objects and camera parameters - masks are computed immediately
        rasterizer = SceneRasterizer(
            objects=vehicles,
            camera_to_world_pose=camera_pose,
            camera_model=camera_model,
            image_width=width,
            image_height=height,
        )

        # Get visibility mask for specific object (O(1) lookup)
        mask = rasterizer.get_visibility_mask("track_123")
    """

    def __init__(
        self,
        objects: dict[str, dict[str, Any]],
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        logger: logging.Logger,
        depth_tolerance: float = 0.05,
    ) -> None:
        """
        Initialize the SceneRasterizer and compute visibility masks.

        The visibility masks are computed immediately in the constructor.
        This ensures that the camera pose and model are always synced to the object data.

        Args:
            objects: Dictionary mapping object IDs (e.g., track_id) to object info dicts. Each object dict must have a
                     "geometry" key containing an RDSGeometry instance. This format matches the output of
                     RdsDataLoader.get_objects_by_class().
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np() method.
            image_width: Output image width in pixels.
            image_height: Output image height in pixels.
            logger: Logger instance for debug output.
            depth_tolerance: Tolerance for depth comparison when determining visibility. A pixel is considered visible
                            if its depth is within this tolerance of the minimum scene depth at that pixel. Default is
                            0.05 meters.
        """
        self._logger = logger
        self._objects = objects
        self._depth_tolerance = depth_tolerance
        self._image_width = image_width
        self._image_height = image_height
        # Retained so in_frame_fraction() can re-project lazily (see that method).
        self._camera_to_world_pose = camera_to_world_pose
        self._camera_model = camera_model

        # Cached projection results (populated by _project())
        self._visibility_masks: dict[str, np.ndarray] = {}
        self._projected_masks: dict[str, np.ndarray] = {}
        self._depth_masks: dict[str, np.ndarray] = {}
        self._bboxes: dict[str, tuple[int, int, int, int] | None] = {}
        self._projection_metadata: dict[str, dict[str, int]] = {}
        self._scene_depth_buffer: np.ndarray = np.array([])
        # Memoized in_frame_fraction() results — computed on demand, not by _project().
        self._in_frame_fractions: dict[str, float] = {}
        # Memoized get_visible_spine_mask() results — also on demand. ``None`` records "this geometry has no
        # centerline", which is the common case, so the miss is cached as cheaply as the hit.
        self._visible_spine_masks: dict[tuple[str, float], np.ndarray | None] = {}

        # Compute visibility masks immediately
        self._project(camera_to_world_pose, camera_model, image_width, image_height)

    @property
    def objects(self) -> dict[str, dict[str, Any]]:
        """Return the collection of objects."""
        return self._objects

    @property
    def image_width(self) -> int:
        """Return the image width."""
        return self._image_width

    @property
    def image_height(self) -> int:
        """Return the image height."""
        return self._image_height

    def has_object(self, object_id: str) -> bool:
        """Return whether the object is in the collection.

        Args:
            object_id: The ID of the object.

        Returns:
            True if the object is in the collection, False otherwise.
        """
        return object_id in self._objects

    @staticmethod
    def _compute_bbox(
        mask: np.ndarray,
    ) -> tuple[int, int, int, int] | None:
        """
        Compute the bounding box of True pixels in a boolean mask.

        Args:
            mask: 2D boolean array.

        Returns:
            Tuple (min_x, min_y, max_x, max_y) or None if the mask is empty.
        """
        row_any = np.any(mask, axis=1)
        if not np.any(row_any):
            return None
        col_any = np.any(mask, axis=0)
        rows = np.where(row_any)[0]
        cols = np.where(col_any)[0]
        return (int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1]))

    def _project(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> None:
        """
        Project all objects into image space and compute occlusion-aware visibility masks.

        This method:
        1. Projects each object to get its binary mask and depth mask
        2. Composites all depth masks to create a global scene depth buffer using per-object bounding boxes so only
           relevant pixels are processed
        3. Computes visibility masks by comparing each object's depth to the scene depth, again restricted to each
           object's bounding box

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np() method.
            image_width: Output image width in pixels.
            image_height: Output image height in pixels.
        """
        # Initialize scene depth buffer with infinity
        self._scene_depth_buffer = np.full((image_height, image_width), np.inf, dtype=np.float32)

        # Step 1: Project all objects, collect masks and bounding boxes
        for obj_id, obj_info in self._objects.items():
            geometry = obj_info.get("geometry")
            if geometry is None or not isinstance(geometry, RDSGeometry):
                # Object has no valid geometry, create empty masks
                self._projected_masks[obj_id] = np.zeros((image_height, image_width), dtype=bool)
                self._depth_masks[obj_id] = np.full((image_height, image_width), np.inf, dtype=np.float32)
                self._bboxes[obj_id] = None
                continue

            # Get projected mask and depth mask from geometry
            proj_mask, depth_mask = geometry.get_projected_mask(
                camera_to_world_pose, camera_model, image_width, image_height
            )

            self._projected_masks[obj_id] = proj_mask
            self._depth_masks[obj_id] = depth_mask

            # Compute bounding box from projected mask for efficient compositing
            bbox = self._compute_bbox(proj_mask)
            self._bboxes[obj_id] = bbox

            # Update scene depth buffer (minimum depth wins) — only within the bbox
            if bbox is not None:
                min_x, min_y, max_x, max_y = bbox
                scene_roi = self._scene_depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
                depth_roi = depth_mask[min_y : max_y + 1, min_x : max_x + 1]
                np.minimum(scene_roi, depth_roi, out=scene_roi)

        # Step 2: Compute visibility masks using depth comparison — only within each bbox
        for obj_id in self._objects:
            bbox = self._bboxes[obj_id]

            if bbox is None:
                # No visible pixels — empty visibility mask
                self._visibility_masks[obj_id] = np.zeros((image_height, image_width), dtype=bool)
                self._projection_metadata[obj_id] = {"bbox_width": 0, "bbox_height": 0, "visible_area_px": 0}
                continue

            min_x, min_y, max_x, max_y = bbox

            proj_roi = self._projected_masks[obj_id][min_y : max_y + 1, min_x : max_x + 1]
            depth_roi = self._depth_masks[obj_id][min_y : max_y + 1, min_x : max_x + 1]
            scene_roi = self._scene_depth_buffer[min_y : max_y + 1, min_x : max_x + 1]

            # A pixel is visible if:
            # 1. It's in the object's projected mask, AND
            # 2. Its depth is within tolerance of the scene minimum depth
            is_front_roi = (depth_roi - scene_roi) <= self._depth_tolerance
            vis_roi = proj_roi & is_front_roi

            visibility_mask = np.zeros((image_height, image_width), dtype=bool)
            visibility_mask[min_y : max_y + 1, min_x : max_x + 1] = vis_roi

            vis_bbox = self._compute_bbox(visibility_mask)
            if vis_bbox is not None:
                vmin_x, vmin_y, vmax_x, vmax_y = vis_bbox
                self._projection_metadata[obj_id] = {
                    "bbox_width": vmax_x - vmin_x + 1,
                    "bbox_height": vmax_y - vmin_y + 1,
                    "visible_area_px": int(np.sum(visibility_mask)),
                }
            else:
                self._projection_metadata[obj_id] = {
                    "bbox_width": 0,
                    "bbox_height": 0,
                    "visible_area_px": 0,
                }

            self._visibility_masks[obj_id] = visibility_mask

    def get_visibility_mask(self, object_id: str) -> np.ndarray:
        """
        Get the occlusion-aware visibility mask for a specific object.

        This method has O(1) time complexity as masks are pre-computed during construction.

        Args:
            object_id: The ID of the object to get the visibility mask for.

        Returns:
            Boolean numpy array of shape (image_height, image_width) where True indicates visible pixels of the object.

        Raises:
            KeyError: If object_id is not in the collection.
        """
        if object_id not in self._visibility_masks:
            raise KeyError(f"Object ID '{object_id}' not found in the collection.")

        return self._visibility_masks[object_id]

    def get_projected_mask(self, object_id: str) -> np.ndarray:
        """
        Get the raw projected mask for a specific object (without occlusion handling).

        This method has O(1) time complexity as masks are pre-computed during construction.

        Args:
            object_id: The ID of the object to get the projected mask for.

        Returns:
            Boolean numpy array of shape (image_height, image_width) where True indicates pixels covered by the
            object's projection (ignoring occlusions).

        Raises:
            KeyError: If object_id is not in the collection.
        """
        if object_id not in self._projected_masks:
            raise KeyError(f"Object ID '{object_id}' not found in the collection.")

        return self._projected_masks[object_id]

    def get_depth_mask(self, object_id: str) -> np.ndarray:
        """
        Get the depth mask for a specific object.

        This method has O(1) time complexity as masks are pre-computed during construction.

        Args:
            object_id: The ID of the object to get the depth mask for.

        Returns:
            Float32 numpy array of shape (image_height, image_width) containing per-pixel depth values in camera space.
            Pixels not covered by the object have value inf.

        Raises:
            KeyError: If object_id is not in the collection.
        """
        if object_id not in self._depth_masks:
            raise KeyError(f"Object ID '{object_id}' not found in the collection.")

        return self._depth_masks[object_id]

    def get_scene_depth_buffer(self) -> np.ndarray:
        """
        Get the composited scene depth buffer.

        This is the minimum depth at each pixel across all objects.

        Returns:
            Float32 numpy array of shape (image_height, image_width) containing the minimum depth at each pixel. Pixels
            not covered by any object have value inf.
        """
        return self._scene_depth_buffer

    def get_all_visibility_masks(self) -> dict[str, np.ndarray]:
        """
        Get visibility masks for all objects.

        Returns:
            Dictionary mapping object IDs to their visibility masks.
        """
        return self._visibility_masks.copy()

    def get_visibility_ratio(self, object_id: str) -> float:
        """
        Get the visibility ratio for a specific object.

        The visibility ratio is the fraction of the object's projected area
        that is visible (not occluded by other objects).

        Args:
            object_id: The ID of the object.

        Returns:
            Float in range [0.0, 1.0] representing the fraction of visible pixels.
            Returns 0.0 if the object has no projected pixels.

        Raises:
            KeyError: If object_id is not in the collection.
        """
        proj_mask = self.get_projected_mask(object_id)
        vis_mask = self.get_visibility_mask(object_id)

        total_pixels = np.sum(proj_mask)
        if total_pixels == 0:
            return 0.0

        visible_pixels = np.sum(vis_mask)
        return float(visible_pixels) / float(total_pixels)

    def bottom_occluded_frac(self, object_id: str) -> float:
        """Fraction of the object's projected height, measured up from its ground-contact bottom, with no
        visible pixels — i.e. how far up the cuboid is occluded by nearer objects.

        High values mean only the TOP of the cuboid pokes out (a nearer object occludes its base), so the
        object is "top-only" visible. This differs from ``get_visibility_ratio`` (a pure area fraction):
        side or scattered occlusion can lower the visibility ratio without lifting the visible bottom,
        whereas this metric isolates bottom-up occlusion specifically.

        Args:
            object_id: The ID of the object.

        Returns:
            Float in range [0.0, 1.0]. Returns 0.0 if the object has no projected pixels and 1.0 if it is
            fully occluded (no visible pixels).

        Raises:
            KeyError: If object_id is not in the collection.
        """
        proj_rows = np.where(self.get_projected_mask(object_id).any(axis=1))[0]
        if proj_rows.size == 0:
            return 0.0
        proj_top, proj_bottom = int(proj_rows.min()), int(proj_rows.max())
        height = proj_bottom - proj_top + 1
        vis_rows = np.where(self.get_visibility_mask(object_id).any(axis=1))[0]
        if vis_rows.size == 0:
            return 1.0
        visible_bottom = int(vis_rows.max())
        return max(0.0, min(1.0, (proj_bottom - visible_bottom) / height))

    def get_projection_metadata(self, object_id: str) -> dict[str, int]:
        """
        Get projection metadata for a specific object.

        Returns the post-occlusion visibility bounding box dimensions and visible pixel count, describing how large
        this object appears in the image after accounting for occlusions.

        Args:
            object_id: The ID of the object.

        Returns:
            Dict with keys:
                - bbox_width: visibility bbox width in pixels (0 if fully occluded / off-screen)
                - bbox_height: visibility bbox height in pixels
                - visible_area_px: count of visible pixels

        Raises:
            KeyError: If object_id is not in the collection.
        """
        if object_id not in self._projection_metadata:
            raise KeyError(f"Object ID '{object_id}' not found in the collection.")
        return self._projection_metadata[object_id]

    def in_frame_fraction(self, object_id: str) -> float:
        """
        Fraction of the object's projected bounding box that lies inside the image.

        Unlike every other accessor on this class, this value is **occlusion-free**. It is computed from the
        object's own projected geometry (see :meth:`checks.utils.cuboid.Cuboid.compute_in_frame_fraction`) and
        ignores the scene depth buffer entirely: it answers "is this cuboid truncated by the image edge?", not
        "how much of it can be seen?". A cuboid completely hidden behind a nearer object still reports ``1.0``
        so long as its projection falls inside the frame. For occlusion-aware measures use
        :meth:`get_visibility_ratio`, :meth:`bottom_occluded_frac`, or :meth:`get_projection_metadata`.

        Computed on first request and memoized, so callers that never gate on frame truncation pay nothing and
        the consumers that do (correspondence scoring and the traffic-light state check) share one projection.

        Args:
            object_id: The ID of the object.

        Returns:
            ``1.0`` when the projection lies fully inside the frame, a smaller value when the frame edge
            truncates it, and ``0.0`` when it does not project at all. Geometries with no meaningful
            frame-edge concept (e.g. polylines and surfaces, which inherit the ``RDSGeometry`` default) and
            objects carrying no geometry both report ``1.0``, so they are never gated.

        Raises:
            KeyError: If object_id is not in the collection.
        """
        cached = self._in_frame_fractions.get(object_id)
        if cached is not None:
            return cached

        if object_id not in self._objects:
            raise KeyError(f"Object ID '{object_id}' not found in the collection.")

        geometry = self._objects[object_id].get("geometry")
        fraction = (
            geometry.in_frame_fraction(
                self._camera_to_world_pose, self._camera_model, self._image_width, self._image_height
            )
            if isinstance(geometry, RDSGeometry)
            else 1.0
        )
        self._in_frame_fractions[object_id] = fraction
        return fraction

    def get_visible_spine_mask(self, object_id: str, min_ribbon_width_px: float = 0.0) -> np.ndarray | None:
        """
        Get the occlusion-aware mask of the object's projected centerline.

        The spine counterpart of :meth:`get_visibility_mask`: it traces
        :meth:`checks.utils.rds_geometry.RDSGeometry.get_projected_spine_mask` and then hides the pixels a
        nearer object covers, using the same scene depth buffer and tolerance as the area masks. Only geometry
        with a meaningful centerline (today, :class:`checks.utils.polyline.Polyline`) returns a mask; cuboids
        and surfaces return ``None``, and callers fall back to an area-based metric for those.

        Occlusion is tested against ``min(spine depth, this object's own ribbon depth)``. The spine and the
        ribbon are two rasterizations of one surface whose interpolated depths agree only to within rounding,
        and the object's ribbon is itself in the scene depth buffer — so without the ``min`` a spine pixel a
        hair behind its own ribbon would read as self-occluded. Taking the nearer of the two removes that
        without weakening real occlusion: a genuine occluder sits far in front of both.

        Computed on first request and memoized, so a frame that scores no polyline class pays nothing.

        Args:
            object_id: The ID of the object.
            min_ribbon_width_px: Drop traced centerline samples whose local geometry projects narrower than
                this many pixels. Shortens only what is scored; the object's area mask and its contribution to
                the scene depth buffer are unchanged, so it still occludes whatever it did before.

        Returns:
            Boolean mask of shape (image_height, image_width) with the unoccluded centerline pixels set, or
            ``None`` when the object's geometry has no centerline (or carries no geometry at all).

        Raises:
            KeyError: If object_id is not in the collection.
        """
        cache_key = (object_id, min_ribbon_width_px)
        if cache_key in self._visible_spine_masks:
            return self._visible_spine_masks[cache_key]

        if object_id not in self._objects:
            raise KeyError(f"Object ID '{object_id}' not found in the collection.")

        geometry = self._objects[object_id].get("geometry")
        spine = (
            geometry.get_projected_spine_mask(
                self._camera_to_world_pose,
                self._camera_model,
                self._image_width,
                self._image_height,
                min_ribbon_width_px,
            )
            if isinstance(geometry, RDSGeometry)
            else None
        )
        if spine is None:
            self._visible_spine_masks[cache_key] = None
            return None

        spine_mask, spine_depth = spine
        own_depth = self._depth_masks.get(object_id)
        effective_depth = spine_depth if own_depth is None else np.minimum(spine_depth, own_depth)

        # Compare only on the traced pixels: everywhere else both depths are inf, and inf - inf would be a NaN
        # (with a RuntimeWarning) for no gain.
        ys, xs = np.nonzero(spine_mask)
        visible_spine = np.zeros_like(spine_mask)
        if ys.size > 0:
            is_front = (effective_depth[ys, xs] - self._scene_depth_buffer[ys, xs]) <= self._depth_tolerance
            visible_spine[ys[is_front], xs[is_front]] = True

        self._visible_spine_masks[cache_key] = visible_spine
        return visible_spine

    def get_projection_bbox(self, object_id: str) -> tuple[int, int, int, int] | None:
        """Get the object's projected image-space bbox as ``(min_x, min_y, max_x, max_y)``.

        This is the full projected extent (pre-occlusion), suitable for matching the object's image
        footprint against detections. Returns ``None`` when the object does not project into the image this
        frame (off-screen / behind the camera) or is not in the collection.
        """
        return self._bboxes.get(object_id)

    def get_projected_spine_polylines(self, object_id: str) -> list[np.ndarray]:
        """
        Get the object's projected centerline as image-space polylines, pre-occlusion.

        The point-list counterpart of :meth:`get_visible_spine_mask`, and the spine analogue of
        :meth:`get_projection_bbox`: like the bbox it reports the object's full projected footprint with no
        depth test, because a consumer matching a detection against the object has to allow the detection to
        land anywhere on it. Geometry with no centerline (cuboids, surfaces) returns ``[]``.

        Args:
            object_id: The ID of the object.

        Returns:
            A list of ``(N, 2)`` float arrays of image-space points, empty when the object carries no
            centerline geometry.

        Raises:
            KeyError: If object_id is not in the collection.
        """
        if object_id not in self._objects:
            raise KeyError(f"Object ID '{object_id}' not found in the collection.")
        geometry = self._objects[object_id].get("geometry")
        if not isinstance(geometry, RDSGeometry):
            return []
        return geometry.get_projected_spine_polylines(self._camera_to_world_pose, self._camera_model)

    def get_visible_pixel_count(self, object_id: str) -> int:
        """
        Get the number of visible pixels for a specific object.

        Args:
            object_id: The ID of the object.

        Returns:
            Integer count of visible pixels.

        Raises:
            KeyError: If object_id is not in the collection.
        """
        vis_mask = self.get_visibility_mask(object_id)
        return int(np.sum(vis_mask))
