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

"""Database credentials shared by checker services that coordinate their own job rows.

Mixed into each checker's ``Settings`` so the fields pick up that service's ``env_prefix`` (``OBJECTS_``,
``EGOMOTION_``, ...), matching how the arbitrator already scopes ``ARBITRATOR_DATABASE_API_KEY``. Per-service
prefixes are what let one pod's credential be rotated or revoked without touching the others.

A checker holds a *worker* key, never the read or admin key: its entire database surface is the job rows it owns.
"""

from pydantic import Field, field_validator, model_validator

from services.settings_base import SettingsBase


class WorkerDatabaseSettings(SettingsBase):
    """Settings base supplying the Database REST API endpoint and worker credential for a checker service.

    Both fields default to empty. A checker with no configured endpoint simply does not coordinate through the
    database, which is what keeps local CLI runs and existing deployments working before the key is provisioned.

    Extends ``SettingsBase`` rather than sitting beside it so that "a checker's settings" is a single type. The shared
    transport needs the storage and environment fields from ``SettingsBase`` and the credentials from here in the same
    object, and there is no useful configuration that has one half without the other.
    """

    database_api_url: str = Field(default="", description="Base URL of the Database REST API endpoint")
    database_worker_key: str = Field(default="", description="Worker-scoped API key for checker job coordination")

    @field_validator("database_api_url", "database_worker_key", mode="after")
    @classmethod
    def _blank_is_empty(cls, value: str) -> str:
        """Trim both credentials, so a whitespace-only value counts as unset everywhere below.

        A deployment manifest that ends up supplying ``DATABASE_WORKER_KEY=" "`` -- a blank YAML value, a trailing line
        continuation -- has not provisioned a key. Left as-is it would pass the all-or-nothing check below and be sent
        as a credential, which is the silent never-claiming pod that check exists to prevent.
        """
        return value.strip()

    @model_validator(mode="after")
    def _coordination_is_all_or_nothing(self) -> "WorkerDatabaseSettings":
        """Refuse to start on half a credential, rather than starting and quietly never claiming anything.

        Configuring one of the pair is always a mistake -- a typo in an environment variable name, or a key that was
        never provisioned. Coordination would then be disabled, and the pod would come up healthy, answer ``/health``,
        and take no work at all. That failure is invisible until somebody notices the queue is not draining, so it is
        turned into a startup error instead.
        """
        if bool(self.database_api_url) != bool(self.database_worker_key):
            missing = "database_worker_key" if self.database_api_url else "database_api_url"
            raise ValueError(
                f"Database job coordination is half-configured: {missing} is empty. Set both "
                "database_api_url and database_worker_key to coordinate, or neither to run standalone."
            )
        return self

    @property
    def database_coordination_enabled(self) -> bool:
        """Whether this replica has everything it needs to claim and report jobs through the database."""
        return bool(self.database_api_url and self.database_worker_key)
