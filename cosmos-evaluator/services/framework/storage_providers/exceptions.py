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

from typing import Any


class StorageError(Exception):
    """Base exception for storage operations."""

    def __init__(self, message: str, key: str | None = None, operation: str | None = None, details: Any | None = None):
        super().__init__(message)
        self.message = message
        self.key = key
        self.operation = operation
        self.details = details


class S3StorageError(StorageError):
    """Exception for S3-specific storage errors."""

    def __init__(
        self,
        message: str,
        key: str | None = None,
        operation: str | None = None,
        bucket: str | None = None,
        aws_error_code: str | None = None,
        details: Any | None = None,
    ):
        super().__init__(message, key, operation, details)
        self.bucket = bucket
        self.aws_error_code = aws_error_code


class AzureBlobStorageError(StorageError):
    """Exception for Azure Blob Storage-specific storage errors."""

    def __init__(
        self,
        message: str,
        key: str | None = None,
        operation: str | None = None,
        container: str | None = None,
        error_code: str | None = None,
        details: Any | None = None,
    ):
        super().__init__(message, key, operation, details)
        self.container = container
        self.error_code = error_code


class StorageConnectionError(StorageError):
    """Exception for storage connection failures."""


class StorageAuthenticationError(StorageError):
    """Exception for storage authentication failures."""


class StorageNotFoundError(StorageError):
    """Exception for storage item not found."""


class StoragePermissionError(StorageError):
    """Exception for storage permission errors."""


class StorageQuotaExceededError(StorageError):
    """Exception for storage quota exceeded."""
