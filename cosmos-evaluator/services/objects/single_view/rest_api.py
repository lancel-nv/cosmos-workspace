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

"""Local entry point for the Objects Single-View API (carries //utils:git_sha in its runfiles for dev).

The runnable logic lives in ``rest_api_common.main`` so this entry point and the container entry point
(``rest_api_deployment``) share one implementation; the two differ only in BUILD-level runfiles (git_sha).
"""

from services.objects.single_view.rest_api_common import main

if __name__ == "__main__":
    main()
