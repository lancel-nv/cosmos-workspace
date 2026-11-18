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

import asyncio
from collections.abc import Sequence
import threading
import unittest
from unittest.mock import patch

from checks.physical_plausibility.attention import contract_fingerprint
from checks.physical_plausibility.processor import PhysicalPlausibilityProcessor, PhysicalPlausibilityResult
from services.arbitrator.dependency_graph import DependencyGraph
from services.physical_plausibility.models import PhysicalPlausibilityRequest
from services.physical_plausibility.service import PhysicalPlausibilityService


class FakeProcessor:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def process_batch(
        self,
        items: Sequence[tuple[str, str, float | None]],
    ) -> list[PhysicalPlausibilityResult]:
        self.batch_sizes.append(len(items))
        return [
            PhysicalPlausibilityResult(
                clip_id=clip_id,
                passed=True,
                bad_score=0.2,
                plausibility_score=0.8,
                bad_threshold=0.4 if threshold is None else threshold,
                prediction="good",
                fingerprint=contract_fingerprint(),
            )
            for clip_id, _, threshold in items
        ]


class BlockingProcessor(FakeProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def process_batch(
        self,
        items: Sequence[tuple[str, str, float | None]],
    ) -> list[PhysicalPlausibilityResult]:
        self.started.set()
        self.release.wait()
        return super().process_batch(items)


class PhysicalPlausibilityServiceTest(unittest.TestCase):
    def test_remote_validation_is_deferred_to_download(self) -> None:
        async def run() -> None:
            service = PhysicalPlausibilityService(processor=FakeProcessor())
            request = PhysicalPlausibilityRequest(clip_id="clip", video_path="s3://bucket/video.mp4")
            with patch("services.physical_plausibility.service.validate_uri") as validate_uri:
                self.assertTrue(await service.validate_input(request))
            validate_uri.assert_not_called()

        asyncio.run(run())

    def test_local_validation_checks_for_a_file(self) -> None:
        async def run() -> None:
            service = PhysicalPlausibilityService(processor=FakeProcessor())
            request = PhysicalPlausibilityRequest(clip_id="clip", video_path="/videos/clip.mp4")
            with patch("services.physical_plausibility.service.validate_uri", return_value=True) as validate_uri:
                self.assertTrue(await service.validate_input(request))
            validate_uri.assert_called_once_with("/videos/clip.mp4", True)

        asyncio.run(run())

    def test_config_directory_is_used_for_processor_and_defaults(self) -> None:
        with patch("services.physical_plausibility.service.PhysicalPlausibilityProcessor") as processor:
            PhysicalPlausibilityService(checkpoint_uri="/models/checker.pt", config_dir="/configs")
        processor.assert_called_once_with(checkpoint_uri="/models/checker.pt", config_dir="/configs")

        async def run() -> None:
            service = PhysicalPlausibilityService(processor=FakeProcessor(), config_dir="/configs")
            defaults = {"metropolis.physical_plausibility": {"bad_threshold": 0.4}}
            with patch.object(
                PhysicalPlausibilityProcessor, "get_default_config", return_value=defaults
            ) as get_default_config:
                self.assertEqual(
                    await service.get_default_config(),
                    defaults["metropolis.physical_plausibility"],
                )
            get_default_config.assert_called_once_with("/configs")

        asyncio.run(run())

    def test_service_dynamically_batches_concurrent_requests(self) -> None:
        async def run() -> None:
            processor = FakeProcessor()
            service = PhysicalPlausibilityService(processor=processor, max_batch_size=4, max_batch_wait_ms=20)
            requests = [
                PhysicalPlausibilityRequest(
                    clip_id=f"clip-{index}",
                    video_path=f"video-{index}.mp4",
                    config={"bad_threshold": 0.3},
                )
                for index in range(3)
            ]
            results = await asyncio.gather(*(service.process(request) for request in requests))
            await service.close()
            self.assertEqual(processor.batch_sizes, [3])
            self.assertEqual([result.clip_id for result in results], ["clip-0", "clip-1", "clip-2"])
            self.assertTrue(all(result.bad_threshold == 0.3 for result in results))

        asyncio.run(run())

    def test_arbitrator_fans_out_physical_plausibility_per_camera(self) -> None:
        graph = DependencyGraph({"metropolis.physical_plausibility": {"deps": []}})
        queue = graph.build_queue(
            ["metropolis.physical_plausibility"],
            augmented_video_urls={"front": "msc://videos/front.mp4"},
        )
        job = queue.get_job("metropolis.physical_plausibility[front]")
        self.assertEqual(
            job.params,
            {
                "camera_name": "front",
                "augmented_video_url": "msc://videos/front.mp4",
            },
        )

    def test_close_cancels_in_flight_batch_futures(self) -> None:
        async def run() -> None:
            processor = BlockingProcessor()
            service = PhysicalPlausibilityService(processor=processor)
            request = PhysicalPlausibilityRequest(clip_id="clip", video_path="video.mp4")
            process_task = asyncio.create_task(service.process(request))
            try:
                await asyncio.wait_for(asyncio.to_thread(processor.started.wait), timeout=1)

                await service.close()
                with self.assertRaises(asyncio.CancelledError):
                    await process_task
            finally:
                processor.release.set()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
