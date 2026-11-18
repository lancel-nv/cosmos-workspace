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

"""FastAPI application for the Objects Single-View Service.

Only the checker-specific pieces live here -- the GPU startup check, the per-request heap trim, fetching this
checker's inputs and uploading its outputs. The endpoints, lifespan, concurrency limiting and durable job protocol
come from ``services.framework.checker_app.build_checker_app``.
"""

import argparse
import asyncio
import ctypes
import ctypes.util
import logging
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import onnxruntime as ort
import torch
import uvicorn

from checks.objects.single_view.object_processor_static import load_cwip_model
from checks.utils.segformer import load_segformer_session
from services import utils
from services.framework.checker_app import CheckerSpec, build_checker_app
from services.framework.protocols.storage_provider import StorageProvider, StorageUrls
from services.objects.single_view.models import (
    ObjectsCloudRequest,
    ObjectsCloudResult,
    ObjectsRequest,
    ObjectsResult,
)
from services.objects.single_view.service import ObjectsService
from services.objects.single_view.settings import get_settings
from services.settings_base import LOG_FORMAT, LogLevel, build_log_config
from utils.cancellation import shielded_to_thread

settings = get_settings()

logger = logging.getLogger(__name__)


def _verify_model_devices(device: str) -> None:
    """Fail fast at startup when configured for CUDA but the GPU stack is unavailable.

    Enforced only when ``device == "cuda"`` (the production default). CWIP runs on torch and SegFormer on onnxruntime -
    two independent CUDA stacks - so torch reporting CUDA does not imply onnxruntime was built with a CUDA provider at
    all, which is why both backends are checked here rather than one standing in for the other.

    This is a check on *capability*, and it stays even though ``get_inference_session`` now raises rather than falling
    back to the CPU. The two catch different things at different times: this reports a misbuilt or misconfigured image
    before any model load is attempted, naming the backend at fault, while the raise covers a provider that is present
    and still refuses to initialize. Skipped when ``device == "cpu"`` so CPU-only dev/test deployments
    (``OBJECTS_MODEL_DEVICE=cpu``) start normally.
    """
    if device != "cuda":
        return

    if not torch.cuda.is_available():
        raise RuntimeError("OBJECTS_MODEL_DEVICE=cuda but CUDA is unavailable for CWIP (torch).")

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "OBJECTS_MODEL_DEVICE=cuda but CUDAExecutionProvider is unavailable for SegFormer (onnxruntime)."
        )


def _release_freed_heap() -> None:
    """Best-effort return of freed heap memory to the OS via glibc ``malloc_trim``.

    Each request allocates large transient CPU buffers (CWIP patch logits, native-resolution index maps, etc.). glibc
    keeps freed chunks in its per-thread arenas instead of returning them to the OS, which manifests as a steady RSS
    climb across requests in this long-lived process. Calling ``malloc_trim(0)`` after each request - together with
    ``MALLOC_ARENA_MAX`` set in the service image env - hands that memory back. This is a no-op on non-glibc platforms
    (e.g. macOS dev machines), where the C library has no ``malloc_trim`` symbol.
    """
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        return
    try:
        libc = ctypes.CDLL(libc_name)
        malloc_trim = libc.malloc_trim
    except (OSError, AttributeError):
        # Non-glibc libc (musl, macOS) has no malloc_trim symbol; nothing to do.
        return
    try:
        malloc_trim(0)
    except OSError as exc:
        logger.debug("malloc_trim(0) failed: %s", exc)


async def _trim_heap() -> None:
    """Return this request's freed buffers to the OS, off the event loop.

    Off the loop because ``malloc_trim`` can walk the heap for a few milliseconds after a large clip.
    """
    await asyncio.to_thread(_release_freed_heap)


def _probe_gpu() -> None:
    """Confirm this process can still reach its GPU, after a job reported that it could not.

    Asked of the GPU rather than of the driver, which is the distinction that makes this useful: the failure being
    tested for is a CUDA *context* that has been poisoned within this process, and ``torch.cuda.is_available()`` goes
    on answering True through exactly that. So this allocates, and then synchronizes to force the allocation to be
    real -- CUDA is asynchronous enough that an allocation alone can succeed against a device that will not run work.

    Torch stands in for onnxruntime here even though the failure that prompted this arrived through onnxruntime. Both
    sit on one CUDA context per process, so a context that cannot serve torch cannot serve ONNX either, and torch is
    the one that can be asked without building a session -- which is the expensive, risky operation this exists to
    keep off the failure path.

    A no-op when configured for CPU: there is no GPU to have lost, so a failure reported on a CPU deployment is about
    something else and should not retire the replica.

    Raises:
        Exception: Whatever torch raises when the context is unusable. Left untranslated: the caller treats any
            failure as confirmation, and the original error is the more informative thing to log.
    """
    if settings.model_device != "cuda":
        return
    probe = torch.zeros(1, device=settings.model_device)
    torch.cuda.synchronize()
    del probe


async def _process_request(
    api_request: ObjectsCloudRequest, input_storage_provider: StorageProvider, temp_dir: Path
) -> ObjectsRequest:
    """Download the request's inputs into ``temp_dir`` and return the local request.

    Args:
        api_request: Request object containing cloud storage URLs
        input_storage_provider: Provider used to resolve those URLs
        temp_dir: Scratch directory owned by the caller, which also deletes it

    Returns:
        The local-path request the service processes
    """
    rds_hq_zip_path = temp_dir / "rds_hq.zip"
    await input_storage_provider.download_from_url(api_request.rds_hq_url, rds_hq_zip_path)

    # Extraction and the subsequent tar-glob are blocking CPU+I/O work, so run them off the event loop. Shielded
    # because neither has a cancellation checkpoint: abandoning the thread would let the caller's cleanup delete
    # temp_dir while extraction is still writing into it.
    rds_hq_extracted_path = temp_dir / "rds_hq_extracted"
    logger.info("Extracting RDS HQ zip to %s", rds_hq_extracted_path)
    await shielded_to_thread(utils.extract_rds_hq_zip, rds_hq_zip_path, rds_hq_extracted_path)

    clip_id = await shielded_to_thread(utils.extract_clip_id, rds_hq_extracted_path)

    video_path = temp_dir / "generated_video.mp4"
    await input_storage_provider.download_from_url(api_request.augmented_video_url, video_path)

    # An absent world-model video is a supported camera-only request, not an error: the static check then runs on the
    # camera stream alone, so the world-model download is skipped. To run camera-only, omit ``world_model_video_url``.
    world_video_path: str | None = None
    if api_request.world_model_video_url is not None:
        world_video_local = temp_dir / "world_video.mp4"
        await input_storage_provider.download_from_url(api_request.world_model_video_url, world_video_local)
        world_video_path = str(world_video_local)

    return ObjectsRequest(
        input_data_path=str(rds_hq_extracted_path),
        clip_id=clip_id,
        camera_name=api_request.camera_name,
        video_path=str(video_path),
        world_video_path=world_video_path,
        config=api_request.config,
        model_device=settings.model_device,
        verbose=api_request.verbose,
        trial_frames=api_request.trial_frames,
        output_storage_prefix=api_request.output_storage_prefix,
    )


async def _build_response_data(
    request: ObjectsCloudRequest, result: ObjectsResult, output_storage_provider: StorageProvider
) -> dict[str, Any]:
    """Upload outputs to storage and return the serialized cloud result.

    The storage layout is::

        [<output_storage_prefix>/]objects_single_view/<camera_name>/<file_stem>.<uuid>.<ext>

    A single uuid4 is minted per invocation so all four uploaded files (dynamic + static results JSON + dynamic +
    static MP4) share the same id, making it easy to correlate them when browsing the bucket. The ``camera_name``
    segment guarantees per-camera uniqueness within a single orchestration request; the ``uuid`` segment guarantees
    uniqueness across re-runs. Cross-request grouping (e.g. by analysis id) is the caller's responsibility via
    ``output_storage_prefix``.

    The provider's ``make_key_prefix`` owns the per-backend slash semantics: in cloud mode ``output_storage_prefix``
    is treated as a key prefix and leading slashes are stripped; in local mode it may be an absolute filesystem path
    that anchors uploads at the filesystem root.
    """
    uid = uuid.uuid4().hex
    prefix = output_storage_provider.make_key_prefix(
        request.output_storage_prefix,
        "objects_single_view",
        request.camera_name,
    )
    output_dir_path = Path(result.output_dir)

    results_json_outputs: dict[str, tuple[Path, str]] = {
        "objects.dynamic": (
            output_dir_path / f"{result.clip_id}.dynamic.objects.results.json",
            f"{prefix}{utils.add_uuid_to_filename(f'{result.clip_id}.dynamic.objects.results.json', uid)}",
        ),
        "objects.static": (
            output_dir_path / f"{result.clip_id}.static.objects.results.json",
            f"{prefix}{utils.add_uuid_to_filename(f'{result.clip_id}.static.objects.results.json', uid)}",
        ),
    }

    viz_video_outputs: dict[str, tuple[Path, str]] = {
        "objects.dynamic": (
            output_dir_path / f"{result.clip_id}.dynamic.objects.mp4",
            f"{prefix}{utils.add_uuid_to_filename(f'{result.clip_id}.dynamic.objects.mp4', uid)}",
        ),
        "objects.static": (
            output_dir_path / f"{result.clip_id}.static.objects.mp4",
            f"{prefix}{utils.add_uuid_to_filename(f'{result.clip_id}.static.objects.mp4', uid)}",
        ),
    }

    results_json_storage_urls: dict[str, StorageUrls] = {}
    viz_video_storage_urls: dict[str, StorageUrls] = {}

    for output_key, (local_path, storage_key) in results_json_outputs.items():
        if local_path.exists():
            results_json_storage_urls[output_key] = await output_storage_provider.store_file(
                local_path, storage_key, "application/json"
            )
        else:
            logger.warning("Results JSON not found at: %s\nSkipping upload.", local_path)

    for output_key, (local_path, storage_key) in viz_video_outputs.items():
        if local_path.exists():
            viz_video_storage_urls[output_key] = await output_storage_provider.store_file(
                local_path, storage_key, "video/mp4"
            )
        else:
            logger.warning("Visualization video not found at: %s\nSkipping upload.", local_path)

    # Opt-in debug artifacts (static processor's ``debug_dump``): the results JSON omits the per-frame CWIP +
    # hallucination masks, so when the ``<output_dir>/debug`` tree exists, tar it and upload it under the same key
    # prefix as the results. Its URL is returned in the response (``debug_artifacts_url`` /
    # ``debug_artifacts_presigned_url``). Best-effort: a failure here never fails the request.
    debug_dir = output_dir_path / "debug"
    debug_urls = None
    if debug_dir.is_dir():
        try:
            tar_name = utils.add_uuid_to_filename(f"{result.clip_id}.static.debug.tar", uid)
            tar_path = output_dir_path / tar_name
            # Use shutil.make_archive (not tarfile) to tar the local "debug" subdir we just wrote — avoids the
            # tarfile archive-safety review and is equivalent here. make_archive appends the ".tar" suffix, so
            # pass the base name without it. Offload the blocking tar build to a thread so it never stalls the
            # event loop (the debug tree can be large), shielded so a cancellation here cannot leave the tar
            # writing into an output dir that the cleanup is about to delete.
            await shielded_to_thread(
                shutil.make_archive,
                str(tar_path)[: -len(".tar")],
                "tar",
                root_dir=str(output_dir_path),
                base_dir="debug",
            )
            debug_urls = await output_storage_provider.store_file(tar_path, f"{prefix}{tar_name}", "application/x-tar")
            logger.info("Uploaded debug artifacts: %s", debug_urls.raw or debug_urls.presigned)
        except Exception:  # debug upload is best-effort; never fail the request
            logger.warning("Failed to tar/upload debug artifacts from %s", debug_dir, exc_info=True)

    cloud_result = ObjectsCloudResult(
        **result.model_dump(),
        results_json_presigned_urls={
            key: urls.presigned for key, urls in results_json_storage_urls.items() if urls.presigned
        },
        results_json_urls={key: urls.raw for key, urls in results_json_storage_urls.items() if urls.raw},
        visualizations_presigned_urls={
            key: urls.presigned for key, urls in viz_video_storage_urls.items() if urls.presigned
        },
        visualizations_urls={key: urls.raw for key, urls in viz_video_storage_urls.items() if urls.raw},
        debug_artifacts_presigned_url=debug_urls.presigned if debug_urls else None,
        debug_artifacts_url=debug_urls.raw if debug_urls else None,
    )

    return cloud_result.model_dump(mode="json")


def _build_service() -> ObjectsService:
    """Construct the service with both models already loaded.

    Each load takes seconds, so both happen once at startup and every request reuses them rather than rebuilding them
    inside a fresh processor.

    For SegFormer the cost is the lesser reason. Building its ONNX session initializes the CUDA execution provider,
    which allocates a cuBLAS handle and a device memory arena, and doing that per request meant a pod claiming a batch
    of jobs ran several of those initializations at once -- which is how a healthy replica ended up failing
    ``cublasCreate`` and then every request after it. One session, built here, is shared by every job; each job still
    gets its own ``IOBinding``, which is the part that cannot be shared.
    """
    cwip_model = load_cwip_model(settings.model_device)
    segformer_session = load_segformer_session(settings.model_device)
    logger.info("Pre-loaded CWIP model and SegFormer session once for reuse across requests")
    return ObjectsService(seg_model=cwip_model, seg_session=segformer_session)


def _processing_rate(result: ObjectsResult) -> dict[str, Any]:
    """Report what fraction of the video was actually processed, which trial mode and early exits change."""
    rate = (
        f"{(100.0 * result.processed_frames / result.total_video_frames):.1f}%"
        if result.total_video_frames > 0
        else "null"
    )
    return {"processing_rate": rate}


app = build_checker_app(
    CheckerSpec(
        name="objects_single_view",
        checker_name="av.objects_single_view",
        title="Objects Single-View Service API",
        description="REST API for objects single-view analysis using computer vision and world model data",
        version=utils.get_contents_from_runfile("services/objects/single_view/version.txt"),
        config_description="Default configuration for object processing",
        settings=settings,
        concurrency_limit=settings.process_concurrency_limit,
        cloud_request_model=ObjectsCloudRequest,
        build_service=_build_service,
        prepare_request=_process_request,
        build_response_data=_build_response_data,
        # Fail before any other startup work if the configured device is CUDA but the GPU stack is unavailable:
        # silently degrading to CPU is far worse than crashing, at 10-100x slower.
        verify_startup=lambda: _verify_model_devices(settings.model_device),
        # The same question asked at the other end of the replica's life: startup decides whether to come up at all,
        # this decides whether a GPU failure mid-flight was that job's problem or the pod's.
        probe_gpu=_probe_gpu,
        response_metadata=_processing_rate,
        after_request=_trim_heap,
    )
)


def main() -> None:
    """Parse CLI arguments and run the Objects Single-View API under uvicorn.

    Shared entry point for both the local (:rest_api) and container (:rest_api_deployment) binaries; the only
    difference between them is whether //utils:git_sha is on the binary's runfiles, which is controlled in the BUILD
    file rather than here.
    """
    parser = argparse.ArgumentParser(
        description="Run the Objects Single-View API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=os.getenv("UVICORN_HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("UVICORN_PORT", "8080")),
        choices=range(1, 65536),
        help="Port to bind to",
        metavar="[1-65535]",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level.value,
        choices=[log_level.value for log_level in LogLevel],
        help="Log level",
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)
    logger.info("Starting Objects Single-View API on %s:%s", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        log_config=build_log_config(args.log_level),
    )
