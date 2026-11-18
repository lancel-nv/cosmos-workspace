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

"""Statement-building helpers shared by the query modules.

Table-specific SQL belongs in the per-domain modules next to this one; what lives here is the mechanical part
several of them would otherwise each reimplement.
"""

from collections.abc import Sequence
from typing import Any


class PartialUpdate:
    """Accumulates the ``SET`` list of an UPDATE that writes only the columns a caller actually supplied.

    Every patch endpoint in this service takes its columns as optional arguments and must leave an omitted one
    alone rather than null it, so each one ends up assembling a ``SET`` list and a parameter list in lockstep.
    Done by hand that is two structures a call site has to keep in agreement, and the agreement is positional:
    ``status = $2`` is correct only because ``status`` happens to be the second thing appended to ``params``.
    Adding a column in the wrong place writes the right SQL against the wrong values, which no type checker
    catches and which a mocked connection reports as a passing test. Owning both structures here makes that
    class of mistake unrepresentable.

    Assignments are emitted in the order they are made, so a caller that cares which column lands on ``$1``
    controls it by call order.
    """

    def __init__(self, *unconditional: str) -> None:
        """Start a ``SET`` list, optionally led by assignments that take no bind parameter.

        ``unconditional`` is for the self-contained SQL a patch always applies regardless of what the caller
        supplied, such as ``"updated_at = CURRENT_TIMESTAMP"``.
        """
        self._assignments: list[str] = list(unconditional)
        self._params: list[Any] = []
        self._bound_columns = 0

    def set(self, column: str, value: Any) -> None:
        """Assign ``column`` to ``value``, unless ``value`` is ``None`` -- in which case the column is left alone."""
        if value is None:
            return
        self._params.append(value)
        self._assignments.append(f"{column} = ${len(self._params)}")
        self._bound_columns += 1

    def set_always(self, column: str, value: Any) -> None:
        """Assign ``column`` even when ``value`` is ``None``, writing SQL NULL.

        :meth:`set` reads ``None`` as "leave this column alone", which is what a patch wants. A write that replaces
        a row's whole state needs the opposite: a column the caller did not supply has to be cleared, or the row
        keeps a value describing something that is no longer true -- a retried job inheriting the previous attempt's
        ``error_message`` onto its success, for instance.
        """
        self._params.append(value)
        self._assignments.append(f"{column} = ${len(self._params)}")
        self._bound_columns += 1

    def set_all(self, columns: Sequence[tuple[str, Any]]) -> None:
        """Apply :meth:`set` to each ``(column, value)`` pair, in order."""
        for column, value in columns:
            self.set(column, value)

    def parts(self) -> tuple[str, list[Any]]:
        """Render the ``SET`` list and a copy of its bind parameters, for a caller composing its own predicate.

        Use this instead of :meth:`statement` when the UPDATE needs a predicate this class cannot express -- a
        guard on the row's current contents, say -- so the caller can append its own parameters after these. The
        parameter list is copied rather than handed over, so composing a statement here cannot disturb the
        accumulated state the way :meth:`statement` does.
        """
        return ", ".join(self._assignments), list(self._params)

    def has_columns(self) -> bool:
        """Whether any caller-supplied column was assigned, ignoring the unconditional assignments.

        A patch with nothing to write is a no-op the caller should skip rather than send: an UPDATE whose only
        assignment is ``updated_at`` would touch the row's timestamp on a request that changed nothing.
        """
        return self._bound_columns > 0

    def statement(self, table: str, *, key: str, key_value: Any, returning: str = "") -> tuple[str, list[Any]]:
        """Render the UPDATE for a single-row key, with its bind parameters in matching order.

        The key is bound last so it lands after every assignment, which is what lets assignments be added in any
        order without renumbering the predicate. Single-use: the key parameter is appended on the way out.
        """
        self._params.append(key_value)
        statement = f"UPDATE {table} SET {', '.join(self._assignments)} WHERE {key} = ${len(self._params)}"
        if returning:
            statement += f" RETURNING {returning}"
        return statement, self._params
