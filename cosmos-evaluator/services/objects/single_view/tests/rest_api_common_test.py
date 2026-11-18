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

"""Unit tests for the Objects Single-View REST API.

Scoped to what this service contributes on top of the shared transport: its request model, the inputs it fetches, the
four-artifact storage layout it writes, the processed-frame fraction it reports, and the CUDA guard it fails startup
on. Everything else -- lifespan, concurrency, error mapping, the job protocol -- lives in
``services.framework.checker_app`` and is tested there.
"""

from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import zipfile

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.framework.auth import install_api_key_auth
from services.framework.protocols.storage_provider import StorageUrls
from services.framework.storage_providers.config import S3Config
from services.framework.storage_providers.local_storage_provider import LocalStorageProvider
from services.framework.storage_providers.s3_storage_provider import S3StorageProvider
from services.objects.single_view.models import ObjectsCloudRequest, ObjectsCloudResult, ObjectsResult
from services.objects.single_view.rest_api_common import (
    _build_response_data,
    _probe_gpu,
    _process_request,
    _processing_rate,
    _verify_model_devices,
    app,
    settings,
)
from services.utils import get_contents_from_runfile

# Mock get_git_sha because //utils:git_sha is not in this test's runfiles. Stopped via
# ``addModuleCleanup`` so the patch does not leak into other modules in the same test run.
_git_sha_patcher = patch("services.utils.get_git_sha", return_value="test_sha")
_git_sha_patcher.start()
unittest.addModuleCleanup(_git_sha_patcher.stop)

CLIP_ID = "test_clip_12345"


def _cloud_request(
    *,
    camera_name: str = "camera_front_wide_120fov",
    output_storage_prefix: str | None = None,
    trial_frames: int | None = 10,
    verbose: str = "INFO",
    world_model_video_url: str | None = "https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
) -> ObjectsCloudRequest:
    return ObjectsCloudRequest(
        rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
        augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4?presigned",
        world_model_video_url=world_model_video_url,
        camera_name=camera_name,
        config={"av.objects_single_view": {"overlap_check": {"vehicle": {"method": "ratio"}}}},
        verbose=verbose,
        trial_frames=trial_frames,
        output_storage_prefix=output_storage_prefix,
    )


def _result(output_dir: Path, *, processed_frames: int = 10, total_video_frames: int = 10) -> ObjectsResult:
    return ObjectsResult(
        processed_frames=processed_frames,
        total_video_frames=total_video_frames,
        mean_score=0.75,
        std_score=0.12,
        min_score=0.45,
        max_score=0.95,
        unique_track_ids=[1, 2, 3],
        processed_frame_ids=list(range(processed_frames)),
        config_summary={"av.objects_single_view": {}},
        clip_id=CLIP_ID,
        output_dir=str(output_dir),
    )


class TestObjectsCloudRequest(unittest.TestCase):
    """The three URLs and the camera are mandatory; everything else has a default."""

    def test_defaults_are_applied_when_only_required_fields_are_given(self) -> None:
        request = ObjectsCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4?presigned",
            world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            camera_name="front_camera",
        )

        self.assertEqual(request.verbose, "INFO")
        self.assertIsNone(request.trial_frames)
        self.assertIsNone(request.output_storage_prefix)

    def test_every_required_field_is_reported_when_absent(self) -> None:
        # world_model_video_url is optional (omitted in camera-only mode), so it is not in the required set.
        required = ("rds_hq_url", "augmented_video_url", "camera_name")
        complete: dict[str, Any] = {
            "rds_hq_url": "https://s3.amazonaws.com/bucket/rds_hq.zip",
            "augmented_video_url": "https://s3.amazonaws.com/bucket/video.mp4",
            "world_model_video_url": "https://s3.amazonaws.com/bucket/world_video.mp4",
            "camera_name": "front_camera",
        }

        for field in required:
            with self.subTest(field=field):
                payload = {key: value for key, value in complete.items() if key != field}
                with self.assertRaises(ValueError) as ctx:
                    ObjectsCloudRequest(**payload)
                self.assertIn(field, str(ctx.exception))

    def test_config_must_be_a_mapping(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            ObjectsCloudRequest(
                rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
                augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
                world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
                camera_name="front_camera",
                config="not_a_dict",
            )

        self.assertIn("Input should be a valid dictionary", str(ctx.exception))

    def test_a_nonsensical_trial_frame_count_is_left_to_the_service(self) -> None:
        """The model is a transport contract, not a validator; the service decides what is runnable."""
        request = ObjectsCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
            world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
            camera_name="front_camera",
            trial_frames=-5,
        )

        self.assertEqual(request.trial_frames, -5)


def _rds_hq_storage(*entries: str) -> AsyncMock:
    """A storage provider whose ``.zip`` download writes a real archive and whose other downloads write bytes."""
    storage = AsyncMock()

    async def _download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.suffix == ".zip":
            with zipfile.ZipFile(dest, "w") as archive:
                for entry in entries:
                    archive.writestr(entry, "payload")
        else:
            dest.write_text(f"downloaded from {url}")

    storage.download_from_url.side_effect = _download
    return storage


class TestProcessRequest(unittest.IsolatedAsyncioTestCase):
    """What this checker fetches into the temp directory the transport hands it."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))

    async def test_all_three_inputs_are_downloaded_and_the_archive_is_extracted(self) -> None:
        storage = _rds_hq_storage(f"{CLIP_ID}/pose.tar")

        with patch("services.objects.single_view.rest_api_common.utils.extract_clip_id", return_value=CLIP_ID):
            request = await _process_request(_cloud_request(), storage, self.temp_dir)

        self.assertEqual(storage.download_from_url.await_count, 3)
        self.assertTrue(Path(request.video_path).exists())
        # Two-stream request: the world path is always resolved (world_video_path is optional only in camera-only).
        self.assertIsNotNone(request.world_video_path)
        assert request.world_video_path is not None
        self.assertTrue(Path(request.world_video_path).exists())
        self.assertTrue((self.temp_dir / "rds_hq_extracted" / CLIP_ID / "pose.tar").exists())

    async def test_missing_world_model_url_runs_camera_only(self) -> None:
        """A cloud request without a world-model URL runs camera-only: the world-model download is skipped and the
        local request carries no world video, rather than raising."""
        storage = _rds_hq_storage(f"{CLIP_ID}/pose.tar")

        with patch("services.objects.single_view.rest_api_common.utils.extract_clip_id", return_value=CLIP_ID):
            request = await _process_request(_cloud_request(world_model_video_url=None), storage, self.temp_dir)

        # Only the RDS zip and the augmented video are downloaded; the world-model video is skipped. An absent
        # world_video_path is what makes the downstream static check run camera-only.
        self.assertEqual(storage.download_from_url.await_count, 2)
        self.assertIsNone(request.world_video_path)

    async def test_the_local_request_carries_the_resolved_clip_and_configured_device(self) -> None:
        """The device comes from settings, not the request: the deployment decides what hardware it has."""
        storage = _rds_hq_storage(f"{CLIP_ID}/pose.tar")

        with patch("services.objects.single_view.rest_api_common.utils.extract_clip_id", return_value=CLIP_ID):
            request = await _process_request(
                _cloud_request(camera_name="front_camera", verbose="DEBUG", trial_frames=7), storage, self.temp_dir
            )

        self.assertEqual(request.clip_id, CLIP_ID)
        self.assertEqual(request.camera_name, "front_camera")
        self.assertEqual(request.model_device, settings.model_device)
        self.assertEqual(request.verbose, "DEBUG")
        self.assertEqual(request.trial_frames, 7)

    async def test_a_download_failure_propagates(self) -> None:
        """The transport owns the temp directory, so this only has to fail loudly rather than clean up."""
        storage = AsyncMock()
        storage.download_from_url.side_effect = RuntimeError("Download failed")

        with self.assertRaisesRegex(RuntimeError, "Download failed"):
            await _process_request(_cloud_request(), storage, self.temp_dir)

    async def test_an_unresolvable_clip_id_propagates(self) -> None:
        storage = _rds_hq_storage("empty/file.txt")

        with (
            patch(
                "services.objects.single_view.rest_api_common.utils.extract_clip_id",
                side_effect=ValueError("No .tar files found"),
            ),
            self.assertRaisesRegex(ValueError, "No .tar files found"),
        ):
            await _process_request(_cloud_request(), storage, self.temp_dir)

    async def test_a_corrupt_archive_propagates(self) -> None:
        storage = AsyncMock()

        async def _download(url: str, dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("not a zip")

        storage.download_from_url.side_effect = _download

        with self.assertRaises(zipfile.BadZipFile):
            await _process_request(_cloud_request(), storage, self.temp_dir)


def _output_provider(urls: StorageUrls | None = None, flavor: str = "s3") -> MagicMock:
    """A stand-in output StorageProvider that records upload keys and returns fixed URLs.

    ``make_key_prefix`` delegates to a real backend (S3 by default; ``flavor='local'`` selects
    ``LocalStorageProvider``) so the tests exercise the production prefix-joining semantics rather than a
    re-implementation of them.
    """
    provider = MagicMock()
    provider.store_file = AsyncMock(return_value=urls or StorageUrls())
    backend = LocalStorageProvider("") if flavor == "local" else S3StorageProvider(S3Config(bucket_name="test"))
    provider.make_key_prefix = MagicMock(side_effect=backend.make_key_prefix)
    return provider


def _keys(provider: MagicMock) -> list[str]:
    """The storage keys the checker asked the provider to write, in order."""
    return [call.args[1] for call in provider.store_file.await_args_list]


def _content_types(provider: MagicMock) -> set[str]:
    return {call.args[2] for call in provider.store_file.await_args_list}


class TestBuildResponseData(unittest.IsolatedAsyncioTestCase):
    """The four-artifact upload layout:

    ``[<output_storage_prefix>/]objects_single_view/<camera_name>/<file_stem>.<uuid>.<ext>``
    """

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self.temp_dir, ignore_errors=True))
        self.output_dir = self.temp_dir / "out"
        self.output_dir.mkdir()
        for filename in (
            f"{CLIP_ID}.dynamic.objects.results.json",
            f"{CLIP_ID}.static.objects.results.json",
            f"{CLIP_ID}.dynamic.objects.mp4",
            f"{CLIP_ID}.static.objects.mp4",
        ):
            (self.output_dir / filename).write_text("payload")

    async def test_default_prefix_is_the_checker_and_camera(self) -> None:
        provider = _output_provider()

        data = await _build_response_data(_cloud_request(), _result(self.output_dir), provider)

        self.assertEqual(ObjectsCloudResult(**data).clip_id, CLIP_ID)
        self.assertTrue(_keys(provider))
        for key in _keys(provider):
            self.assertTrue(key.startswith("objects_single_view/camera_front_wide_120fov/"), key)

    async def test_empty_prefix_adds_no_leading_segment(self) -> None:
        provider = _output_provider()

        await _build_response_data(
            _cloud_request(camera_name="camera_cross_left_120fov", output_storage_prefix=""),
            _result(self.output_dir),
            provider,
        )

        for key in _keys(provider):
            self.assertTrue(key.startswith("objects_single_view/camera_cross_left_120fov/"), key)

    async def test_explicit_prefix_leads_the_key(self) -> None:
        provider = _output_provider()

        await _build_response_data(
            _cloud_request(camera_name="camera_rear_left_70fov", output_storage_prefix="my-team/run-42/"),
            _result(self.output_dir),
            provider,
        )

        for key in _keys(provider):
            self.assertTrue(key.startswith("my-team/run-42/objects_single_view/camera_rear_left_70fov/"), key)

    async def test_all_four_files_share_one_uuid(self) -> None:
        """One uuid per invocation, so the four artifacts of a run can be correlated by eye in the bucket."""
        provider = _output_provider()

        await _build_response_data(_cloud_request(), _result(self.output_dir), provider)

        self.assertEqual(len(_keys(provider)), 4, _keys(provider))
        uuids = set()
        for key in _keys(provider):
            match = re.search(r"\.([0-9a-f]{32})\.[^.]+$", key)
            self.assertIsNotNone(match, key)
            assert match is not None
            uuids.add(match.group(1))
        self.assertEqual(len(uuids), 1, uuids)

        stems = {key.rsplit("/", 1)[-1].rsplit(".", 2)[0] for key in _keys(provider)}
        self.assertEqual(
            stems,
            {
                f"{CLIP_ID}.dynamic.objects.results",
                f"{CLIP_ID}.static.objects.results",
                f"{CLIP_ID}.dynamic.objects",
                f"{CLIP_ID}.static.objects",
            },
        )

    async def test_two_cameras_of_one_clip_do_not_collide(self) -> None:
        """Regression guard: per-camera uploads once shared a key and overwrote each other."""
        front = _output_provider()
        cross = _output_provider()

        await _build_response_data(
            _cloud_request(camera_name="camera_front_wide_120fov"), _result(self.output_dir), front
        )
        await _build_response_data(
            _cloud_request(camera_name="camera_cross_left_120fov"), _result(self.output_dir), cross
        )

        self.assertTrue(all(key.startswith("objects_single_view/camera_front_wide_120fov/") for key in _keys(front)))
        self.assertTrue(all(key.startswith("objects_single_view/camera_cross_left_120fov/") for key in _keys(cross)))
        self.assertEqual(set(_keys(front)) & set(_keys(cross)), set())

    async def test_upload_urls_reach_the_response(self) -> None:
        provider = _output_provider(
            StorageUrls(raw="s3://bucket/results.json", presigned="https://presigned.url/results.json")
        )

        data = await _build_response_data(_cloud_request(), _result(self.output_dir), provider)

        cloud_result = ObjectsCloudResult(**data)
        self.assertEqual(set(cloud_result.results_json_urls), {"objects.dynamic", "objects.static"})
        self.assertEqual(set(cloud_result.visualizations_presigned_urls), {"objects.dynamic", "objects.static"})

    async def test_a_provider_that_returns_no_urls_still_produces_a_result(self) -> None:
        """Uploads that yield neither a raw nor a presigned URL are dropped rather than failing the request."""
        provider = _output_provider(StorageUrls())

        data = await _build_response_data(_cloud_request(), _result(self.output_dir), provider)

        cloud_result = ObjectsCloudResult(**data)
        self.assertEqual(cloud_result.visualizations_urls, {})
        self.assertEqual(cloud_result.visualizations_presigned_urls, {})
        self.assertEqual(cloud_result.clip_id, CLIP_ID)

    async def test_missing_artifacts_are_skipped(self) -> None:
        """A run that produced nothing (or exited early) uploads nothing and reports no URLs."""
        empty_dir = self.temp_dir / "empty"
        empty_dir.mkdir()
        provider = _output_provider()

        data = await _build_response_data(_cloud_request(), _result(empty_dir), provider)

        self.assertEqual(provider.store_file.await_count, 0)
        cloud_result = ObjectsCloudResult(**data)
        self.assertEqual(cloud_result.results_json_urls, {})
        self.assertEqual(cloud_result.visualizations_urls, {})

    async def test_a_debug_tree_is_tarred_and_its_url_returned(self) -> None:
        """With ``debug_dump`` on, the per-frame masks the JSON omits travel as a tar beside the results."""
        debug_dir = self.output_dir / "debug"
        debug_dir.mkdir()
        (debug_dir / "cwip_compact.npz").write_text("mock")
        provider = _output_provider(
            StorageUrls(raw="s3://bucket/debug.tar", presigned="https://presigned.url/debug.tar")
        )

        data = await _build_response_data(_cloud_request(), _result(self.output_dir), provider)

        cloud_result = ObjectsCloudResult(**data)
        self.assertEqual(cloud_result.debug_artifacts_url, "s3://bucket/debug.tar")
        self.assertEqual(cloud_result.debug_artifacts_presigned_url, "https://presigned.url/debug.tar")
        self.assertIn("application/x-tar", _content_types(provider))

    async def test_no_debug_tree_means_no_debug_urls(self) -> None:
        provider = _output_provider(StorageUrls(raw="s3://bucket/x", presigned="https://presigned.url/x"))

        data = await _build_response_data(_cloud_request(), _result(self.output_dir), provider)

        cloud_result = ObjectsCloudResult(**data)
        self.assertIsNone(cloud_result.debug_artifacts_url)
        self.assertIsNone(cloud_result.debug_artifacts_presigned_url)

    async def test_local_absolute_prefix_yields_absolute_keys(self) -> None:
        provider = _output_provider(flavor="local")

        await _build_response_data(_cloud_request(output_storage_prefix="/tmp/x"), _result(self.output_dir), provider)

        for key in _keys(provider):
            self.assertTrue(key.startswith("/tmp/x/objects_single_view/camera_front_wide_120fov/"), key)

    async def test_local_relative_prefix_remains_relative(self) -> None:
        provider = _output_provider(flavor="local")

        await _build_response_data(_cloud_request(output_storage_prefix="my/dir"), _result(self.output_dir), provider)

        for key in _keys(provider):
            self.assertTrue(key.startswith("my/dir/objects_single_view/camera_front_wide_120fov/"), key)
            self.assertFalse(key.startswith("/"), key)

    async def test_s3_absolute_prefix_strips_leading_slash(self) -> None:
        provider = _output_provider(flavor="s3")

        await _build_response_data(_cloud_request(output_storage_prefix="/tmp/x"), _result(self.output_dir), provider)

        for key in _keys(provider):
            self.assertFalse(key.startswith("/"), key)
            self.assertTrue(key.startswith("tmp/x/objects_single_view/camera_front_wide_120fov/"), key)


class TestProcessingRate(unittest.TestCase):
    """The response metadata this checker adds: how much of the video it actually looked at."""

    def test_full_processing_reports_one_hundred_percent(self) -> None:
        rate = _processing_rate(_result(Path("/tmp/out"), processed_frames=100, total_video_frames=100))

        self.assertEqual(rate, {"processing_rate": "100.0%"})

    def test_partial_processing_reports_the_fraction(self) -> None:
        """Trial mode and early exits are the normal reasons this is below 100%."""
        rate = _processing_rate(_result(Path("/tmp/out"), processed_frames=50, total_video_frames=200))

        self.assertEqual(rate, {"processing_rate": "25.0%"})

    def test_an_unknown_frame_count_reports_null_rather_than_dividing_by_zero(self) -> None:
        rate = _processing_rate(_result(Path("/tmp/out"), processed_frames=0, total_video_frames=0))

        self.assertEqual(rate, {"processing_rate": "null"})

    def test_the_rate_is_reported_to_one_decimal(self) -> None:
        rate = _processing_rate(_result(Path("/tmp/out"), processed_frames=1, total_video_frames=3))

        self.assertEqual(rate, {"processing_rate": "33.3%"})


class TestVerifyModelDevices(unittest.TestCase):
    """The startup guard: configured for CUDA but missing it must crash, not silently run 10-100x slower.

    torch and onnxruntime are imported at module scope but are not test deps, so the CUDA paths patch the module-level
    ``torch`` / ``ort`` names rather than importing the real packages.
    """

    def test_cpu_device_is_a_noop(self) -> None:
        # Passes as long as nothing is raised: CPU-only dev deployments must start normally.
        _verify_model_devices("cpu")

    def test_cuda_passes_when_both_backends_have_it(self) -> None:
        torch_stub = MagicMock()
        torch_stub.cuda.is_available.return_value = True
        ort_stub = MagicMock()
        ort_stub.get_available_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        with (
            patch("services.objects.single_view.rest_api_common.torch", torch_stub),
            patch("services.objects.single_view.rest_api_common.ort", ort_stub),
        ):
            _verify_model_devices("cuda")

    def test_cuda_raises_when_torch_cannot_see_the_gpu(self) -> None:
        torch_stub = MagicMock()
        torch_stub.cuda.is_available.return_value = False

        with (
            patch("services.objects.single_view.rest_api_common.torch", torch_stub),
            self.assertRaisesRegex(RuntimeError, "CWIP"),
        ):
            _verify_model_devices("cuda")

    def test_cuda_raises_when_onnxruntime_lacks_the_cuda_provider(self) -> None:
        """Two independent CUDA stacks, so torch reporting CUDA does not mean SegFormer is on the GPU."""
        torch_stub = MagicMock()
        torch_stub.cuda.is_available.return_value = True
        ort_stub = MagicMock()
        ort_stub.get_available_providers.return_value = ["CPUExecutionProvider"]

        with (
            patch("services.objects.single_view.rest_api_common.torch", torch_stub),
            patch("services.objects.single_view.rest_api_common.ort", ort_stub),
            self.assertRaisesRegex(RuntimeError, "SegFormer"),
        ):
            _verify_model_devices("cuda")


class TestProbeGpu(unittest.TestCase):
    """The runtime check: has this process's CUDA context died, as opposed to was it ever there.

    Same stubbing as the startup guard above -- torch is imported at module scope but is not a test dep, so these
    patch the module-level name. ``settings`` is patched too rather than taken as it comes, since the probe branches
    on the configured device and these tests should not change meaning with the environment they run in.
    """

    @staticmethod
    def _settings_for(device: str) -> MagicMock:
        settings_stub = MagicMock()
        settings_stub.model_device = device
        return settings_stub

    def test_the_probe_actually_touches_the_device(self) -> None:
        """Asking the driver is not enough, which is the entire point of this probe.

        A CUDA context poisoned within this process keeps answering True to ``torch.cuda.is_available()``, so a probe
        built on that would clear a GPU that cannot run anything. It has to allocate, and it has to synchronize --
        CUDA is asynchronous enough that the allocation alone can succeed against a device that will not run work.
        """
        torch_stub = MagicMock()

        with (
            patch("services.objects.single_view.rest_api_common.torch", torch_stub),
            patch("services.objects.single_view.rest_api_common.settings", self._settings_for("cuda")),
        ):
            _probe_gpu()

        torch_stub.zeros.assert_called_once()
        torch_stub.cuda.synchronize.assert_called_once()

    def test_a_dead_context_is_reported_by_raising(self) -> None:
        torch_stub = MagicMock()
        torch_stub.cuda.synchronize.side_effect = RuntimeError("CUDA error: unspecified launch failure")

        with (
            patch("services.objects.single_view.rest_api_common.torch", torch_stub),
            patch("services.objects.single_view.rest_api_common.settings", self._settings_for("cuda")),
            self.assertRaisesRegex(RuntimeError, "unspecified launch failure"),
        ):
            _probe_gpu()

    def test_a_cpu_deployment_has_no_gpu_to_have_lost(self) -> None:
        """A GPU failure reported by a CPU deployment is about something else, and must not retire the replica."""
        torch_stub = MagicMock()

        with (
            patch("services.objects.single_view.rest_api_common.torch", torch_stub),
            patch("services.objects.single_view.rest_api_common.settings", self._settings_for("cpu")),
        ):
            _probe_gpu()

        torch_stub.zeros.assert_not_called()


class TestSingleViewApp(unittest.TestCase):
    """The app is assembled from this service's spec.

    ``TestClient`` is used without its context manager: entering it would run the real lifespan, which loads models.
    Collaborators go onto ``app.state`` directly, which is where the transport reads them from.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.service = MagicMock()
        self.service.get_default_config = AsyncMock(return_value={"av.objects_single_view": {"overlap_check": {}}})
        app.state.service = self.service
        self.addCleanup(self._reset_state)

    def _reset_state(self) -> None:
        app.state.service = None
        app.state.input_storage_provider = None
        app.state.output_storage_provider = None

    def test_health_reports_this_services_identity(self) -> None:
        data = self.client.get("/health").json()["data"]

        self.assertEqual(data["service"], "Objects Single-View Service API")
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], get_contents_from_runfile("services/objects/single_view/version.txt"))
        self.assertEqual(data["git_sha"], "test_sha")
        self.assertEqual(data["slots"], {"active": 0, "limit": app.state.slots.limit})
        self.assertEqual(data["checker"], "av.objects_single_view")

    def test_metrics_are_labeled_for_this_checker(self) -> None:
        response = self.client.get("/metrics")

        self.assertIn(f'checker_slots_limit{{checker="av.objects_single_view"}} {app.state.slots.limit}', response.text)

    def test_config_returns_the_services_defaults(self) -> None:
        data = self.client.get("/config").json()["data"]

        self.assertIn("av.objects_single_view", data["default_config"])

    def test_a_request_runs_this_services_pipeline_end_to_end(self) -> None:
        """One pass through the assembled app, with this checker's real download and upload steps in the path."""
        temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        (temp_dir / f"{CLIP_ID}.dynamic.objects.results.json").write_text("{}")

        self.service.validate_input = AsyncMock(return_value=True)
        self.service.process = AsyncMock(return_value=_result(temp_dir, processed_frames=5, total_video_frames=10))
        self.service.cleanup = AsyncMock()
        app.state.input_storage_provider = _rds_hq_storage(f"{CLIP_ID}/pose.tar")
        app.state.output_storage_provider = _output_provider(
            StorageUrls(raw="s3://bucket/results.json", presigned="https://presigned.url/results.json")
        )

        with patch("services.objects.single_view.rest_api_common.utils.extract_clip_id", return_value=CLIP_ID):
            response = self.client.post("/process", json=_cloud_request().model_dump())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"]["clip_id"], CLIP_ID)
        self.assertEqual(body["metadata"]["processing_rate"], "50.0%")
        self.assertEqual(body["data"]["results_json_urls"], {"objects.dynamic": "s3://bucket/results.json"})
        # The request the service saw came from this checker's own prepare step, not from the test.
        local_request = self.service.process.await_args.args[0]
        self.assertEqual(local_request.clip_id, CLIP_ID)
        self.assertEqual(local_request.camera_name, "camera_front_wide_120fov")
        self.service.cleanup.assert_awaited_once()


class TestApiKeyAuthWiring(unittest.TestCase):
    """The framework API key middleware is wired into this service's app.

    The live ``app`` runs under ``COSMOS_EVALUATOR_ENV=local`` (``active_api_key is None``) so it is unprotected. To
    exercise enforcement we build a probe app that reuses the real routes and installs the middleware with a key,
    mirroring how the app behaves in staging/prod.
    """

    API_KEY = "test-api-key"

    def _probe_client(self) -> TestClient:
        probe = FastAPI()
        probe.router.routes = list(app.router.routes)
        install_api_key_auth(probe, self.API_KEY)
        return TestClient(probe)

    def test_health_is_exempt(self) -> None:
        self.assertEqual(self._probe_client().get("/health").status_code, 200)

    def test_protected_path_requires_key(self) -> None:
        self.assertEqual(self._probe_client().get("/config").status_code, 401)

    def test_protected_path_rejects_invalid_key(self) -> None:
        self.assertEqual(self._probe_client().get("/config", headers={"X-API-Key": "bad"}).status_code, 401)

    def test_protected_path_accepts_valid_key(self) -> None:
        # Stub the service so /config reaches a successful handler; this proves the valid key passes the middleware
        # AND lands on a working route (a bare assertNotEqual(401) would also accept a 500/503).
        service_stub = MagicMock()
        service_stub.get_default_config = AsyncMock(return_value={})
        app.state.service = service_stub
        try:
            response = self._probe_client().get("/config", headers={"X-API-Key": self.API_KEY})
        finally:
            app.state.service = None
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
