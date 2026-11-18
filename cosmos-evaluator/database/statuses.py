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

"""Status vocabulary of the orchestration tables, shared by everything that reads or writes it.

These sets are a property of the schema rather than of any one caller, and three layers need the same answer to
"is this row finished?": the arbitrator, which stops tracking a job once its row is terminal; the database queries
layer, which refuses to write a finished row backwards; and the snapshot endpoint, which counts finished checkers.
A copy per layer is a copy that can drift, and the failure mode of drift is silent -- a status one layer treats as
terminal and another does not is exactly the gap a row gets stuck in.

Deliberately dependency-free so both the server-side query modules and the client-side arbitrator can import it
without either pulling in the other's dependencies.
"""

# ``checker_executions.status`` values that mean one checker's job is over. A superset of the parent set below:
# ``timeout`` and ``skipped`` are terminal for a checker but never surface as a parent status.
CHECKER_TERMINAL_STATUSES = frozenset({"success", "fail", "timeout", "skipped"})

# ``arbitrator_executions.status`` values that mean the whole run is over and clients should stop polling. Mirrors
# what ``ExecutionRecord.complete()`` / ``.fail()`` write.
EXECUTION_TERMINAL_STATUSES = frozenset({"success", "fail"})
