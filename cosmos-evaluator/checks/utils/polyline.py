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
Polyline geometry module for projection-based masking.

This module defines a Polyline that takes an Nx3 array of world-space points
and provides utilities to project a dilated ribbon of the polyline into an
image given a camera pose and model.
"""

from contextlib import suppress
import math

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.utils.rasterization import clip_and_project_polygon, rasterize_polygon_with_depth
from checks.utils.rds_geometry import RDSGeometry

# Spacing (pixels) between successive samples when tracing a projected spine segment. Anything <= 1.0 makes the
# rounded samples 8-connected; 0.5 leaves margin so a segment that is nearly axis-aligned does not alias into a
# dotted trace.
_SPINE_SAMPLE_SPACING_PX: float = 0.5
# Two clipped segments continue one another when the first's end and the second's start are the same camera-space
# point. They are the same vertex when neither side was clipped, so the tolerance only absorbs float noise.
_SPINE_JOIN_ATOL_M: float = 1e-9

# Upper bound on samples per segment. A segment passing close to the camera can project to a pixel length far
# larger than the image, and an f-theta model can amplify that further; the cap keeps a pathological projection
# from allocating an unbounded array. Sized off the image so a full-diagonal segment is never truncated.
_MAX_SPINE_SAMPLES_PER_IMAGE_SPAN: int = 4

# Floor for a measured projected ribbon width, in pixels. A ribbon seen exactly edge-on projects to zero width;
# clamping here keeps the 1/width interpolation finite so such a sample is filtered out rather than producing a
# divide-by-zero.
_MIN_MEASURABLE_WIDTH_PX: float = 1e-6


class Polyline(RDSGeometry):
    """
    Represents a 3D polyline with projection capabilities.

    A polyline is defined by an ordered set of Nx3 vertices in world coordinates.
    For image masking, each adjacent pair of points [Pi, Pi+1] forms a segment.
    The segment is dilated in the local ground plane (XY-plane, preserving each
    endpoint's z) by a configurable half-width to form a 3D quad which is then
    clipped to the camera near plane and projected to pixels.
    """

    # Default tracker gate (meters). Polyline observations of the same physical
    # lane re-sample their visible window each frame, so the gate has to absorb
    # ~1 m of arc-length drift while still discriminating lanes that are
    # typically ~3 m apart.
    MATCH_GATE_METERS: float = 1.5

    def __init__(
        self,
        vertices_world: np.ndarray,
        half_width_meters: float = 0.20,
        min_cutoff_distance: float = 3.0,
        max_cutoff_distance: float = 50.0,
    ):
        """
        Initialize a polyline.

        Args:
            vertices_world: Nx3 array of polyline vertices in world coordinates.
            half_width_meters: Half of the ribbon thickness (meters) used to dilate
                               each segment normal in the ground plane.
        """
        verts = np.asarray(vertices_world, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 3 or verts.shape[0] < 2:
            raise ValueError(f"vertices_world must be of shape (N, 3) with N>=2, got {verts.shape}")
        self.vertices_world = np.array(verts)
        self.half_width_m = float(max(0.0, half_width_meters))
        self.min_cutoff_distance = min_cutoff_distance
        self.max_cutoff_distance = max_cutoff_distance
        if self.min_cutoff_distance >= self.max_cutoff_distance:
            raise ValueError("min_cutoff_distance must be less than max_cutoff_distance")

    def get_projected_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a binary mask and depth mask of the projected, dilated polyline in image space.
        For each segment, a 3D quad is built by offsetting the endpoints along
        the segment's ground-plane normal by ±half_width_m, then clipped to near
        plane and projected to pixels. The per-segment masks are unioned.

        Rasterization is performed into bounding-box-sized local arrays for better
        cache locality and reduced initialization cost, then embedded into full-size
        output arrays.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np(points_cam) method.
            image_width: Output image width.
            image_height: Output image height.

        Returns:
            Tuple of:
                - Binary mask of shape (image_height, image_width), dtype=bool.
                - Depth mask of shape (image_height, image_width), dtype=float32,
                  containing per-pixel depth in camera space (z coordinate).
                  Pixels outside the mask have value inf.
        """
        if camera_to_world_pose.shape != (4, 4):
            raise ValueError("camera_to_world_pose must be 4x4")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")

        empty_mask = np.zeros((image_height, image_width), dtype=bool)
        empty_depth = np.full((image_height, image_width), np.inf, dtype=np.float32)

        world_to_camera_pose = np.linalg.inv(camera_to_world_pose)

        # First pass: build, clip, and project each segment quad to collect
        # 2D vertices for overall bounding box computation.
        verts = self.vertices_world
        segment_data = []
        all_2d_points = []

        for i in range(len(verts) - 1):
            p0 = verts[i]
            p1 = verts[i + 1]
            # Ground-plane direction in XY
            d = p1[:2] - p0[:2]
            norm = float(np.hypot(d[0], d[1]))
            if norm < 1e-6:
                continue
            t = d / norm
            # Left-hand normal in XY (rotate t by +90 deg)
            n = np.array([-t[1], t[0]], dtype=float)
            hw = self.half_width_m

            # Build 4 corners in world coordinates; preserve each endpoint's z
            left0 = np.array([p0[0] + n[0] * hw, p0[1] + n[1] * hw, p0[2]], dtype=float)
            left1 = np.array([p1[0] + n[0] * hw, p1[1] + n[1] * hw, p1[2]], dtype=float)
            right1 = np.array([p1[0] - n[0] * hw, p1[1] - n[1] * hw, p1[2]], dtype=float)
            right0 = np.array([p0[0] - n[0] * hw, p0[1] - n[1] * hw, p0[2]], dtype=float)
            quad_world = np.stack([left0, left1, right1, right0], axis=0)  # (4,3)

            # Transform to camera coordinates, clip behind-camera geometry,
            # and project to pixels in one step.
            quad_cam = CameraBase.transform_points_np(quad_world, world_to_camera_pose)
            with suppress(Exception):
                result = clip_and_project_polygon(
                    quad_cam,
                    camera_model.ray2pixel_np,
                    near_z=self.min_cutoff_distance,
                    far_z=self.max_cutoff_distance,
                )
                if result is None:
                    continue
                pts, pts_z = result
                segment_data.append((pts, pts_z))
                all_2d_points.append(pts)

        if not segment_data:
            return empty_mask, empty_depth

        # Compute overall bounding box from all projected vertices, clamped to image
        all_pts = np.vstack(all_2d_points)
        min_x = max(0, int(np.floor(np.min(all_pts[:, 0]))))
        max_x = min(image_width - 1, int(np.ceil(np.max(all_pts[:, 0]))))
        min_y = max(0, int(np.floor(np.min(all_pts[:, 1]))))
        max_y = min(image_height - 1, int(np.ceil(np.max(all_pts[:, 1]))))

        if min_x > max_x or min_y > max_y:
            return empty_mask, empty_depth

        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1

        # Allocate local bbox-sized arrays for rasterization
        local_mask = np.zeros((bbox_h, bbox_w), dtype=np.uint8)
        local_depth = np.full((bbox_h, bbox_w), np.inf, dtype=np.float32)

        # Second pass: rasterize each segment into local arrays with offset coordinates
        for pts, pts_z in segment_data:
            offset_pts = pts.copy()
            offset_pts[:, 0] -= min_x
            offset_pts[:, 1] -= min_y
            rasterize_polygon_with_depth(offset_pts, pts_z, local_mask, local_depth, bbox_w, bbox_h)

        # Embed local results into full-size output arrays
        empty_mask[min_y : max_y + 1, min_x : max_x + 1] = local_mask.astype(bool)
        empty_depth[min_y : max_y + 1, min_x : max_x + 1] = local_depth

        return empty_mask, empty_depth

    def get_projected_spine_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        min_ribbon_width_px: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a binary mask and depth mask of the projected centerline, with no ground-plane dilation.

        Where :meth:`get_projected_mask` inflates every segment to a ``half_width_m`` ribbon, this traces the
        vertex path itself as a single-pixel-wide curve. Each segment is transformed to camera space, clipped to
        ``[min_cutoff_distance, max_cutoff_distance]`` (the same window as the ribbon, so the two agree on what
        part of the line is in range), projected, and sampled at sub-pixel spacing so the trace is 8-connected.
        Per-pixel depth is interpolated linearly along the segment in screen space, matching how
        :func:`checks.utils.rasterization.rasterize_polygon_with_depth` interpolates across the ribbon's
        triangles, so the two representations of the same surface agree to within rounding.

        All segments are projected in a single ``ray2pixel_np`` call: polylines regularly carry many short
        segments, and per-segment projection dominates the cost otherwise.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np(points_cam) method.
            image_width: Output image width.
            image_height: Output image height.
            min_ribbon_width_px: Drop centerline samples where the ribbon projects narrower than this many
                pixels. Beyond some range a lane line is thinner than the segmentation mask's native
                granularity, so the mask cannot represent it and the centerline there can only score 0 —
                which then averages into the same per-frame scalar as the near field, penalising long lines
                for their length. Expressed in pixels rather than meters because resolvability is a property
                of the mask, not of the world: the cutoff has to move with focal length, image size and
                ``half_width_meters`` rather than being pinned to one camera. The ribbon itself is unaffected,
                so the object still projects and occludes over its full extent; only the scored part shortens.
                ``0`` traces the full window. Ignored for a zero-width ribbon, which has no width to measure.

        Returns:
            Tuple of:
                - Binary mask of shape (image_height, image_width), dtype=bool, True on the traced centerline.
                - Depth mask of shape (image_height, image_width), dtype=float32, holding camera-space depth on
                  the traced pixels and ``inf`` elsewhere.
        """
        if camera_to_world_pose.shape != (4, 4):
            raise ValueError("camera_to_world_pose must be 4x4")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")

        mask = np.zeros((image_height, image_width), dtype=bool)
        depth = np.full((image_height, image_width), np.inf, dtype=np.float32)

        world_to_camera_pose = np.linalg.inv(camera_to_world_pose)
        verts_cam = CameraBase.transform_points_np(self.vertices_world, world_to_camera_pose)
        # Rotation only: the ribbon offset is a direction, so it must not pick up the pose translation.
        world_to_camera_rotation = world_to_camera_pose[:3, :3]
        measure_width = min_ribbon_width_px > 0.0 and self.half_width_m > 0.0

        # Clip every segment, then project all surviving points in one call. Each segment contributes its two
        # endpoints plus, when a width cutoff is in force, the ribbon's two edge points at each endpoint — six
        # points per segment, from which the projected ribbon width at either end follows directly.
        points_cam: list[np.ndarray] = []
        stride = 6 if measure_width else 2
        for i in range(len(verts_cam) - 1):
            segment = _clip_segment_to_z_range(
                verts_cam[i], verts_cam[i + 1], self.min_cutoff_distance, self.max_cutoff_distance
            )
            if segment is None:
                continue
            # Skip degenerate segments before either path. The ribbon drops them too, so tracing one would put
            # a pixel in the spine that has no counterpart in the area mask — inflating the denominator the
            # spine score divides by, for a segment of zero length.
            direction = self.vertices_world[i + 1][:2] - self.vertices_world[i][:2]
            norm = float(np.hypot(direction[0], direction[1]))
            if norm < 1e-6:
                continue
            if not measure_width:
                points_cam.append(segment)
                continue
            normal_world = np.array([-direction[1] / norm, direction[0] / norm, 0.0]) * self.half_width_m
            offset_cam = world_to_camera_rotation @ normal_world
            points_cam.append(
                np.stack(
                    [
                        segment[0],
                        segment[1],
                        segment[0] + offset_cam,
                        segment[0] - offset_cam,
                        segment[1] + offset_cam,
                        segment[1] - offset_cam,
                    ]
                )
            )

        if not points_cam:
            return mask, depth

        stacked_cam = np.concatenate(points_cam, axis=0)
        with suppress(Exception):
            stacked_2d = np.asarray(camera_model.ray2pixel_np(stacked_cam), dtype=float)
            if stacked_2d.shape != (stacked_cam.shape[0], 2):
                return mask, depth
            max_samples = _MAX_SPINE_SAMPLES_PER_IMAGE_SPAN * (image_width + image_height)
            for i in range(0, stacked_2d.shape[0], stride):
                widths = None
                if measure_width:
                    widths = (
                        float(np.linalg.norm(stacked_2d[i + 2] - stacked_2d[i + 3])),
                        float(np.linalg.norm(stacked_2d[i + 4] - stacked_2d[i + 5])),
                    )
                _trace_segment_with_depth(
                    stacked_2d[i : i + 2],
                    stacked_cam[i : i + 2, 2],
                    mask,
                    depth,
                    max_samples,
                    widths,
                    min_ribbon_width_px,
                )

        return mask, depth

    def get_projected_spine_polylines(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
    ) -> list[np.ndarray]:
        """
        Project the centerline to image space as open polylines, one per contiguous in-range span.

        Shares :meth:`get_projected_spine_mask`'s clipping so the two agree on which part of the line is in
        range, but stops before rasterizing: the result is the curve's vertices, which is what a consumer
        measuring *distance to the line* needs. See
        :meth:`checks.utils.rds_geometry.RDSGeometry.get_projected_spine_polylines` for why the spans are kept
        separate and why no width cutoff or occlusion test is applied.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np(points_cam) method.

        Returns:
            A list of ``(N, 2)`` float arrays of image-space points, ``N >= 2`` each; empty when no segment
            survives clipping or the projection fails.
        """
        if camera_to_world_pose.shape != (4, 4):
            raise ValueError("camera_to_world_pose must be 4x4")

        world_to_camera_pose = np.linalg.inv(camera_to_world_pose)
        verts_cam = CameraBase.transform_points_np(self.vertices_world, world_to_camera_pose)

        segments: list[np.ndarray] = []
        for i in range(len(verts_cam) - 1):
            # Same degenerate-segment test as the mask, so a segment the mask drops cannot appear here.
            direction = self.vertices_world[i + 1][:2] - self.vertices_world[i][:2]
            if float(np.hypot(direction[0], direction[1])) < 1e-6:
                continue
            clipped = _clip_segment_to_z_range(
                verts_cam[i], verts_cam[i + 1], self.min_cutoff_distance, self.max_cutoff_distance
            )
            if clipped is not None:
                segments.append(clipped)
        if not segments:
            return []

        stacked_cam = np.concatenate(segments, axis=0)
        try:
            stacked_2d = np.asarray(camera_model.ray2pixel_np(stacked_cam), dtype=float)
        except Exception:  # noqa: BLE001 - an unprojectable camera model yields no centerline, not an error
            return []
        if stacked_2d.shape != (stacked_cam.shape[0], 2):
            return []

        pieces: list[np.ndarray] = []
        current: list[np.ndarray] = []
        previous_end_cam: np.ndarray | None = None
        for index, segment in enumerate(segments):
            if previous_end_cam is None or not np.allclose(segment[0], previous_end_cam, atol=_SPINE_JOIN_ATOL_M):
                if len(current) >= 2:
                    pieces.append(np.asarray(current, dtype=float))
                current = [stacked_2d[2 * index]]
            current.append(stacked_2d[2 * index + 1])
            previous_end_cam = segment[1]
        if len(current) >= 2:
            pieces.append(np.asarray(current, dtype=float))
        return pieces

    def world_distance(self, other: RDSGeometry) -> float:
        """
        Asymmetric median distance from this polyline's vertices to the
        closest point on ``other`` (point-to-segment, not point-to-vertex).

        Asymmetry is intentional: per-frame polyline observations of the same
        physical lane often extend further than the canonical (or vice versa)
        because the perception model resamples the visible window each frame.
        Penalizing only the side that is being matched (``self``) keeps the
        metric stable when the two polylines cover overlapping but unequal
        arc-lengths. Distance is measured to the nearest segment of ``other``
        so that a self-vertex lying on (but between) ``other``'s vertices
        scores zero rather than picking up sparse-sampling jitter. Returns
        ``+inf`` for non-Polyline inputs.
        """
        if not isinstance(other, Polyline):
            return float("inf")
        nearest = _nearest_distances_to_polyline(self.vertices_world, other.vertices_world)
        return float(np.median(nearest))


def _clip_segment_to_z_range(p0_cam: np.ndarray, p1_cam: np.ndarray, near_z: float, far_z: float) -> np.ndarray | None:
    """
    Clip a camera-space segment to the depth slab ``near_z <= z <= far_z``.

    The segment equivalent of :func:`checks.utils.coord_transforms.clip_polygon_to_z_planes`, which needs at
    least three vertices and so cannot be reused here. Both bounds are applied to the single line parameter
    ``t in [0, 1]``, narrowing it until either the surviving span is empty (the segment misses the slab) or both
    endpoints sit inside it.

    Args:
        p0_cam: Segment start in camera coordinates, shape (3,).
        p1_cam: Segment end in camera coordinates, shape (3,).
        near_z: Near plane; points with ``z < near_z`` are clipped away.
        far_z: Far plane; points with ``z > far_z`` are clipped away.

    Returns:
        A (2, 3) array of the clipped endpoints, or ``None`` when the segment lies wholly outside the slab or
        collapses to a point.
    """
    p0 = np.asarray(p0_cam, dtype=float)
    p1 = np.asarray(p1_cam, dtype=float)
    z0 = float(p0[2])
    dz = float(p1[2]) - z0

    t_min, t_max = 0.0, 1.0
    for limit, keep_above in ((float(near_z), True), (float(far_z), False)):
        if abs(dz) < 1e-12:
            # Segment is parallel to the plane: wholly inside or wholly outside, no parameter to narrow.
            if (z0 < limit) if keep_above else (z0 > limit):
                return None
            continue
        t_limit = (limit - z0) / dz
        # The half-space the plane keeps is "t above t_limit" exactly when the segment crosses it in the
        # direction the constraint permits, i.e. when advancing t moves z the same way the constraint does.
        if (dz > 0.0) == keep_above:
            t_min = max(t_min, t_limit)
        else:
            t_max = min(t_max, t_limit)

    if t_min >= t_max:
        return None
    return np.stack([p0 + t_min * (p1 - p0), p0 + t_max * (p1 - p0)], axis=0)


def _visible_t_span(
    x0: float, y0: float, x1: float, y1: float, image_width: int, image_height: int
) -> tuple[float | None, float | None]:
    """The sub-interval of ``t in [0, 1]`` where the segment ``(x0, y0) -> (x1, y1)`` is inside the image.

    Liang-Barsky against the four image edges, in the segment's own parameterization so a caller can keep
    interpolating depth and width at the original ``t``. Returns ``(None, None)`` when the segment misses the
    image entirely. The bounds are the rounding limits a sample must satisfy to index the array, so they are
    half-open on the far side and widened by half a pixel on each end - a sample rounding to pixel 0 or
    ``image_width - 1`` is in range even when its exact coordinate sits just outside.
    """
    t_lo, t_hi = 0.0, 1.0
    for delta, start, lower, upper in (
        (x1 - x0, x0, -0.5, image_width - 0.5),
        (y1 - y0, y0, -0.5, image_height - 0.5),
    ):
        for p, q in ((-delta, start - lower), (delta, upper - start)):
            if abs(p) < 1e-12:
                # Parallel to this pair of edges: inside for every t, or outside for every t.
                if q < 0.0:
                    return None, None
                continue
            r = q / p
            if p < 0.0:
                t_lo = max(t_lo, r)
            else:
                t_hi = min(t_hi, r)
    return (t_lo, t_hi) if t_lo <= t_hi else (None, None)


def _trace_segment_with_depth(
    endpoints_2d: np.ndarray,
    endpoints_z: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    max_samples: int,
    endpoint_widths_px: tuple[float, float] | None = None,
    min_width_px: float = 0.0,
) -> None:
    """
    Trace one projected segment into ``mask`` and ``depth`` in place, one pixel wide.

    Samples the segment at :data:`_SPINE_SAMPLE_SPACING_PX` intervals and rounds each sample to a pixel, which
    yields an 8-connected run because consecutive samples are under a pixel apart. Depth is written with a
    minimum so a polyline that folds back on itself in image space keeps its nearest branch, matching the
    z-buffer semantics of the ribbon rasterizer.

    When ``endpoint_widths_px`` is given, samples whose local ribbon width falls below ``min_width_px`` are
    dropped. Width is interpolated as a linear ramp in ``1 / width`` rather than in width: a ribbon's projected
    width goes as ``1 / depth``, and depth is already interpolated linearly in screen space here (matching
    :func:`checks.utils.rasterization.rasterize_polygon_with_depth`), so ``1 / width`` is the quantity that is
    actually linear along the segment. Interpolating width directly would misplace the cutoff on any segment
    spanning a wide depth range — exactly the long segments the cutoff exists for.

    Args:
        endpoints_2d: (2, 2) array of the segment's projected pixel coordinates.
        endpoints_z: (2,) array of the corresponding camera-space depths.
        mask: (H, W) bool array to set traced pixels on; modified in place.
        depth: (H, W) float32 array to write interpolated depth into; modified in place.
        max_samples: Cap on samples for this segment, bounding the work a pathological projection can request.
        endpoint_widths_px: Projected ribbon width at each endpoint, or ``None`` to trace the whole segment.
        min_width_px: Width below which a sample is dropped; ignored without ``endpoint_widths_px``.
    """
    if not np.all(np.isfinite(endpoints_2d)):
        return

    image_height, image_width = mask.shape
    (x0, y0), (x1, y1) = endpoints_2d

    # Sample only the part of the segment that can land on a pixel. ``max_samples`` bounds the work a
    # pathological projection can request, but spreading that budget over a segment that is mostly off-screen
    # coarsens the visible part: past ``max_samples * _SPINE_SAMPLE_SPACING_PX`` of projected length the trace
    # stops being 8-connected, so it breaks into dots and ``OverlapDetector._score_spine`` divides by fewer
    # centerline pixels than the object actually has. Narrowing ``t`` to the on-screen span first keeps the
    # spacing at half a pixel wherever the budget can cover it. ``t`` stays in the segment's own
    # parameterization, so the depth and width ramps below are unaffected.
    t_lo, t_hi = _visible_t_span(x0, y0, x1, y1, image_width, image_height)
    if t_lo is None or t_hi is None:
        return
    length_px = math.hypot(x1 - x0, y1 - y0) * (t_hi - t_lo)
    num_samples = min(math.ceil(length_px / _SPINE_SAMPLE_SPACING_PX) + 1, max_samples)

    t = np.linspace(t_lo, t_hi, num_samples)
    xs = np.rint(x0 + t * (x1 - x0)).astype(np.int64)
    ys = np.rint(y0 + t * (y1 - y0)).astype(np.int64)
    zs = (endpoints_z[0] + t * (endpoints_z[1] - endpoints_z[0])).astype(np.float32)

    keep = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)

    if endpoint_widths_px is not None and min_width_px > 0.0:
        # Clamp away from zero so a degenerate endpoint yields a huge 1/width (hence a filtered sample)
        # rather than a division warning.
        w0 = max(float(endpoint_widths_px[0]), _MIN_MEASURABLE_WIDTH_PX)
        w1 = max(float(endpoint_widths_px[1]), _MIN_MEASURABLE_WIDTH_PX)
        inverse_width = (1.0 - t) / w0 + t / w1
        keep &= inverse_width <= 1.0 / min_width_px

    if not keep.any():
        return
    xs, ys, zs = xs[keep], ys[keep], zs[keep]

    np.minimum.at(depth, (ys, xs), zs)
    mask[ys, xs] = True


def _nearest_distances_to_polyline(points: np.ndarray, polyline_vertices: np.ndarray) -> np.ndarray:
    """
    For each point in ``points``, compute the distance to the nearest point on
    the polyline whose vertices are ``polyline_vertices``. The polyline is
    interpreted as the union of segments between consecutive vertices.

    Args:
        points: (N, 3) array of query points.
        polyline_vertices: (M, 3) array of polyline vertices, M >= 2.

    Returns:
        (N,) array of nearest-segment distances.
    """
    a = polyline_vertices[:-1]  # (M-1, 3) segment starts
    b = polyline_vertices[1:]  # (M-1, 3) segment ends
    ab = b - a  # (M-1, 3)
    ab_sq = (ab * ab).sum(axis=1)  # (M-1,)
    # Project each point onto each segment's parameter line, clamp to [0, 1].
    ap = points[:, None, :] - a[None, :, :]  # (N, M-1, 3)
    t = (ap * ab[None, :, :]).sum(axis=2) / np.maximum(ab_sq[None, :], 1e-12)  # (N, M-1)
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]  # (N, M-1, 3)
    seg_distances = np.linalg.norm(points[:, None, :] - closest, axis=2)  # (N, M-1)
    return seg_distances.min(axis=1)  # (N,)
