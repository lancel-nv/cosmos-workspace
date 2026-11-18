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

"""Container entry point for the client-agnostic Arbitrator API.

Identical to ``rest_api`` except that its BUILD target omits //utils:git_sha from runfiles so the app_layer cache is
not invalidated on every commit; the running container reads the SHA from /app/git_sha.txt supplied by the image's
git_sha layer. Having its own source file (rather than reusing rest_api.py) lets Gazelle own and manage this binary's
dependencies instead of skipping a second target that shares one source.
"""

from services.arbitrator.rest_api_common import main

if __name__ == "__main__":
    main()
