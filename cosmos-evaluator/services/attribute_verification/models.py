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
Attribute Verification Request Models.

Three-tier request layering mirroring the canonical AV checker pattern (see ``services/objects/single_view/models.py``):

* :class:`AttributeVerificationRequestBase` - transport-independent fields (clip_id, config, verbose,
  output_storage_prefix).
* :class:`AttributeVerificationRequest` - local-path variant consumed by the service/processor.
* :class:`AttributeVerificationCloudRequest` - URL variant constructed by the arbitrator and accepted by the REST
  transport.

This module intentionally depends only on pydantic so the arbitrator can import
:class:`AttributeVerificationCloudRequest` without pulling in the checker's heavy runtime. Remote inputs are downloaded
by the processor via ``multistorageclient`` (``MULTISTORAGECLIENT_CONFIGURATION``); the transport simply maps the URL
field onto the local-path field, so the cloud variant's URL flows straight into the processor.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class AttributeVerificationRequestBase(BaseModel):
    """Transport-independent base fields shared by the local and cloud request variants."""

    clip_id: str = Field(..., description="Identifier for the clip being processed (echoed in the result).")
    config: dict[str, Any] | None = Field(default=None, description="Configuration dictionary for processing")
    verbose: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Logging level")
    output_storage_prefix: str | None = Field(
        default=None,
        description=(
            "Optional caller-supplied prefix for output storage keys. Accepted for parity with other checkers; the "
            "attribute verification check produces only metrics (no uploaded artifacts), so it is currently a no-op."
        ),
    )


class AttributeVerificationRequest(AttributeVerificationRequestBase):
    """Local-path request consumed by the service and processor."""

    augmented_video_path: str = Field(
        ...,
        description="Path (or URL) to the augmented image or video; the field name is retained for compatibility",
    )


class AttributeVerificationCloudRequest(AttributeVerificationRequestBase):
    """URL request constructed by the arbitrator and accepted by the REST transport.

    The processor downloads remote inputs itself via ``multistorageclient``, so the transport maps this URL field
    straight onto the local-path field of :class:`AttributeVerificationRequest` without a separate download step.
    """

    augmented_video_url: str = Field(
        ...,
        description="URL or path to the augmented image or video; the field name is retained for compatibility",
    )
