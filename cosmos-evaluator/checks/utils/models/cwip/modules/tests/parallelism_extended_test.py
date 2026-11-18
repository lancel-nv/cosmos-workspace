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

"""Extended tests for parallelism module - exercises dist-active and Megatron-active branches via mocks.

The baseline ``parallelism_test.py`` covers the unmocked single-process (TP=1, FP=1) paths. This file targets the
high-line-count branches that only execute when ``torch.distributed`` is initialized, when Megatron is installed, or
when an autograd Function backward runs across a multi-rank group — all of which are impossible to reach in a normal
unit-test process and so were uncovered in the baseline lcov report.
"""

import sys
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch
from torch import nn

from checks.utils.models.cwip.modules import parallelism as P

# =============================================================================
# initialize_model_parallel — dist-active branches
# =============================================================================


class TestInitializeModelParallelDistActive(unittest.TestCase):
    """initialize_model_parallel branches that require dist.is_initialized()=True."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_divisibility_error_raises(self, mock_dist: Mock) -> None:
        """world_size not divisible by tp*cp -> ValueError."""
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 5
        mock_dist.get_rank.return_value = 0
        with self.assertRaisesRegex(ValueError, "must be divisible"):
            P.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_custom_groups_branch_no_megatron(self, mock_dist: Mock) -> None:
        """When megatron.core import fails the function builds custom dist groups."""
        # Force the megatron.core import to raise ImportError inside the function.
        # We do this by ensuring ``megatron.core`` is not importable from inside the function:
        # since megatron is not installed in test env, the except ImportError branch is hit.
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.side_effect = lambda ranks: MagicMock(name=f"group_{ranks}")

        # Make sure megatron.core is not importable
        with patch.dict(sys.modules, {"megatron.core": None}):
            P.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)

        self.assertTrue(P._MODEL_PARALLEL_STATE["initialized"])
        self.assertEqual(P._MODEL_PARALLEL_STATE["tensor_parallel_size"], 2)
        self.assertEqual(P._MODEL_PARALLEL_STATE["context_parallel_size"], 2)
        self.assertIsNotNone(P._MODEL_PARALLEL_STATE["tensor_parallel_group"])
        self.assertIsNotNone(P._MODEL_PARALLEL_STATE["context_parallel_group"])
        self.assertIsNotNone(P._MODEL_PARALLEL_STATE["frame_parallel_group"])

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_megatron_branch(self, mock_dist: Mock) -> None:
        """When megatron.core imports succeed, that branch is used."""
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4
        mock_dist.get_rank.return_value = 0

        # Build a fake megatron.core.parallel_state module
        fake_ps = MagicMock()
        fake_ps.is_initialized.return_value = True
        fake_ps.get_tensor_model_parallel_group.return_value = MagicMock(name="tp_group")
        fake_ps.get_context_parallel_group.return_value = MagicMock(name="cp_group")
        fake_ps.get_data_parallel_group.return_value = MagicMock(name="dp_group")
        with patch.object(P, "HAS_MEGATRON", True), patch.object(P, "parallel_state", fake_ps):
            P.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2, sequence_parallel=True)

        self.assertTrue(P._MODEL_PARALLEL_STATE["initialized"])
        self.assertEqual(P._MODEL_PARALLEL_STATE["tensor_parallel_size"], 1)
        self.assertEqual(P._MODEL_PARALLEL_STATE["context_parallel_size"], 2)
        self.assertTrue(P._MODEL_PARALLEL_STATE["sequence_parallel"])
        fake_ps.initialize_model_parallel.assert_called()

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_megatron_branch_already_initialized_destroys(self, mock_dist: Mock) -> None:
        """If parallel state was already initialized, the Megatron branch destroys it first."""
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 2
        mock_dist.get_rank.return_value = 0

        fake_ps = MagicMock()
        fake_ps.get_tensor_model_parallel_group.return_value = MagicMock()
        fake_ps.get_context_parallel_group.return_value = MagicMock()
        fake_ps.get_data_parallel_group.return_value = MagicMock()
        P._MODEL_PARALLEL_STATE["initialized"] = True

        with patch.object(P, "HAS_MEGATRON", True), patch.object(P, "parallel_state", fake_ps):
            P.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=2)

        fake_ps.destroy_model_parallel.assert_called_once()


# =============================================================================
# sync_from_megatron — Megatron-active branches
# =============================================================================


class TestSyncFromMegatron(unittest.TestCase):
    """Exercise sync_from_megatron with a mocked megatron.core.parallel_state."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def _make_fake_megatron(
        self, cp_world_size: int = 1, tp_world_size: int = 1
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        fake_ps = MagicMock()
        fake_ps.is_initialized.return_value = True
        fake_ps.get_tensor_model_parallel_world_size.return_value = tp_world_size
        fake_ps.get_tensor_model_parallel_group.return_value = MagicMock(name="tp_group")
        fake_ps.get_context_parallel_world_size.return_value = cp_world_size
        fake_ps.get_context_parallel_group.return_value = MagicMock(name="cp_group")
        fake_ps.get_data_parallel_group.return_value = MagicMock(name="dp_group")
        fake_ps.sequence_parallel = False

        fake_core = MagicMock(name="megatron.core")
        fake_core.parallel_state = fake_ps
        fake_megatron = MagicMock(name="megatron")
        fake_megatron.core = fake_core
        return fake_megatron, fake_core, fake_ps

    def test_megatron_not_initialized_returns_early(self) -> None:
        """parallel_state.is_initialized() == False -> early return."""
        fake_megatron, fake_core, fake_ps = self._make_fake_megatron(cp_world_size=2)
        fake_ps.is_initialized.return_value = False

        with patch.dict(sys.modules, {"megatron": fake_megatron, "megatron.core": fake_core}):
            P.sync_from_megatron(frame_parallel_size=2)

        self.assertFalse(P._MODEL_PARALLEL_STATE["initialized"])

    def test_cp_size_one_uses_megatron_cp_as_fp(self) -> None:
        """With true_cp=1, the megatron CP group becomes the FP group directly."""
        _fake_megatron, _fake_core, fake_ps = self._make_fake_megatron(cp_world_size=4)

        with patch.object(P, "HAS_MEGATRON", True), patch.object(P, "parallel_state", fake_ps):
            P.sync_from_megatron(frame_parallel_size=4, true_context_parallel_size=1)

        self.assertTrue(P._MODEL_PARALLEL_STATE["initialized"])
        self.assertEqual(P._MODEL_PARALLEL_STATE["frame_parallel_size"], 4)
        self.assertEqual(P._MODEL_PARALLEL_STATE["true_context_parallel_size"], 1)
        self.assertIs(
            P._MODEL_PARALLEL_STATE["frame_parallel_group"],
            fake_ps.get_context_parallel_group.return_value,
        )
        self.assertIsNone(P._MODEL_PARALLEL_STATE["true_context_parallel_group"])

    def test_fp_cp_product_mismatch_raises(self) -> None:
        """fp_size * cp_size != megatron_cp_size -> ValueError."""
        _fake_megatron, _fake_core, fake_ps = self._make_fake_megatron(cp_world_size=4)

        with (
            patch.object(P, "HAS_MEGATRON", True),
            patch.object(P, "parallel_state", fake_ps),
            self.assertRaisesRegex(ValueError, "Megatron CP size"),
        ):
            P.sync_from_megatron(frame_parallel_size=3, true_context_parallel_size=1)

    def test_fp_size_zero_uses_full_megatron_cp(self) -> None:
        """frame_parallel_size=0 (default) means 'use all of megatron CP as FP'."""
        _fake_megatron, _fake_core, fake_ps = self._make_fake_megatron(cp_world_size=2)

        with patch.object(P, "HAS_MEGATRON", True), patch.object(P, "parallel_state", fake_ps):
            P.sync_from_megatron(frame_parallel_size=0, true_context_parallel_size=1)

        self.assertEqual(P._MODEL_PARALLEL_STATE["frame_parallel_size"], 2)

    def test_megatron_import_error_no_op(self) -> None:
        """When megatron.core can't be imported, sync_from_megatron silently no-ops."""
        with patch.dict(sys.modules, {"megatron.core": None}):
            P.sync_from_megatron(frame_parallel_size=2)

        self.assertFalse(P._MODEL_PARALLEL_STATE["initialized"])


# =============================================================================
# _AllGatherCatBatchDim and *_optional_tensor* helpers — backward path
# =============================================================================


class TestAllGatherCatBatchDim(unittest.TestCase):
    """Forward / backward of _AllGatherCatBatchDim with mocked dist."""

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_forward_world_size_one_identity(self, mock_dist: Mock) -> None:
        mock_dist.get_world_size.return_value = 1
        group = MagicMock()
        x = torch.randn(2, 4)
        out = P._AllGatherCatBatchDim.apply(x, group)
        self.assertTrue(torch.equal(out, x))

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_forward_concat_along_batch(self, mock_dist: Mock) -> None:
        """world_size=2: forward all-gathers and cats along dim 0."""
        mock_dist.get_world_size.return_value = 2
        group = MagicMock()

        def fake_all_gather(tensor_list: list[torch.Tensor], tensor: torch.Tensor, group: object = None) -> None:
            for t in tensor_list:
                t.copy_(tensor)

        mock_dist.all_gather.side_effect = fake_all_gather

        x = torch.randn(2, 4, requires_grad=True)
        out = P._AllGatherCatBatchDim.apply(x, group)
        self.assertEqual(out.shape, (4, 4))

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_backward_returns_local_chunk(self, mock_dist: Mock) -> None:
        """Backward returns this rank's contiguous slice of grad_output."""
        mock_dist.get_world_size.return_value = 2
        mock_dist.get_rank.return_value = 1
        group = MagicMock()

        def fake_all_gather(tensor_list: list[torch.Tensor], tensor: torch.Tensor, group: object = None) -> None:
            for t in tensor_list:
                t.copy_(tensor)

        mock_dist.all_gather.side_effect = fake_all_gather

        x = torch.randn(2, 4, requires_grad=True)
        out = P._AllGatherCatBatchDim.apply(x, group)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        assert x.grad is not None
        self.assertEqual(x.grad.shape, (2, 4))

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_backward_world_size_one(self, mock_dist: Mock) -> None:
        """When world_size=1, backward returns grad_output unchanged."""
        mock_dist.get_world_size.return_value = 1
        group = MagicMock()
        x = torch.randn(3, 5, requires_grad=True)
        out = P._AllGatherCatBatchDim.apply(x, group)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_gather_optional_tensor_none(self) -> None:
        out = P.gather_optional_tensor(None, group=None)
        self.assertIsNone(out)

    def test_gather_optional_tensor_with_group_none(self) -> None:
        x = torch.randn(2, 4)
        out = P.gather_optional_tensor(x, group=None)
        assert out is not None
        self.assertTrue(torch.equal(out, x))

    def test_gather_optional_tensor_list_none(self) -> None:
        self.assertIsNone(P.gather_optional_tensor_list(None, group=None))

    def test_gather_optional_tensor_list_preserves_nones(self) -> None:
        lst = [torch.randn(2, 4), None, torch.randn(3, 4)]
        out = P.gather_optional_tensor_list(lst, group=None)
        assert out is not None
        self.assertEqual(len(out), 3)
        self.assertIsNone(out[1])


# =============================================================================
# split_tensor_for_fp — return_indices / padding branches
# =============================================================================


class TestSplitTensorForFpExtended(unittest.TestCase):
    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_fp_size_one_return_indices(self) -> None:
        """fp_size<=1 path with return_indices returns (t, 0, n)."""
        t = torch.randn(4, 8)
        out, lo, hi = P.split_tensor_for_fp(t, fp_size=1, return_indices=True)
        self.assertIs(out, t)
        self.assertEqual(lo, 0)
        self.assertEqual(hi, 4)

    @patch.object(P, "get_frame_parallel_rank", return_value=0)
    @patch.object(P, "get_frame_parallel_world_size", return_value=3)
    def test_padding_required(self, _mock_ws: Mock, _mock_rank: Mock) -> None:
        """Batch not divisible by fp_size -> right-pad before split."""
        t = torch.arange(5 * 2).reshape(5, 2).float()
        out = P.split_tensor_for_fp(t)
        # Padded to 6 frames, fp_size=3 -> 2 frames per rank.
        assert isinstance(out, torch.Tensor)
        self.assertEqual(out.shape, (2, 2))

    @patch.object(P, "get_frame_parallel_rank", return_value=2)
    @patch.object(P, "get_frame_parallel_world_size", return_value=3)
    def test_padding_with_indices(self, _mock_ws: Mock, _mock_rank: Mock) -> None:
        """Last rank with right-pad: real_end clamps to original batch size."""
        t = torch.arange(5 * 2).reshape(5, 2).float()
        out, lo, hi = P.split_tensor_for_fp(t, return_indices=True)
        self.assertEqual(out.shape, (2, 2))
        # Padded batch = 6, frames_per_rank = 2. Rank 2 covers [4,6) -> clamped to original 5.
        self.assertEqual(lo, 4)
        self.assertEqual(hi, 5)


# =============================================================================
# prepare_true_cp_sequence_shard / gather_trim_true_cp_sequence — true CP active
# =============================================================================


class TestTrueCpSequenceShard(unittest.TestCase):
    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_inactive_when_dist_not_initialized(self, mock_dist: Mock) -> None:
        mock_dist.is_initialized.return_value = False
        a = torch.randn(1, 4, 8)
        b = torch.randn(1, 4, 16)
        sa, sb, meta = P.prepare_true_cp_sequence_shard(a, b, grid_h=2, grid_w=2)
        self.assertIs(sa, a)
        self.assertIs(sb, b)
        self.assertIsNone(meta)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_inactive_when_cp_size_one(self, mock_dist: Mock) -> None:
        mock_dist.is_initialized.return_value = True
        a = torch.randn(1, 4, 8)
        b = torch.randn(1, 4, 16)
        sa, sb, meta = P.prepare_true_cp_sequence_shard(a, b, grid_h=2, grid_w=2)
        self.assertIs(sa, a)
        self.assertIs(sb, b)
        self.assertIsNone(meta)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    @patch.object(P, "get_true_context_parallel_rank", return_value=0)
    @patch.object(P, "get_true_context_parallel_world_size", return_value=2)
    @patch.object(P, "get_true_context_parallel_group")
    def test_active_shard_no_padding(self, mock_group: Mock, _mock_ws: Mock, _mock_rank: Mock, mock_dist: Mock) -> None:
        mock_dist.is_initialized.return_value = True
        mock_group.return_value = MagicMock(name="cp_group")
        a = torch.randn(1, 4, 8)
        b = torch.randn(1, 4, 16)
        sa, sb, meta = P.prepare_true_cp_sequence_shard(a, b, grid_h=2, grid_w=2)
        self.assertEqual(sa.shape, (1, 2, 8))
        self.assertEqual(sb.shape, (1, 2, 16))
        self.assertIsNotNone(meta)
        assert meta is not None
        self.assertEqual(meta["cp_size"], 2)
        self.assertEqual(meta["s_unpadded"], 4)
        self.assertEqual(meta["s_padded"], 4)
        self.assertEqual(meta["s_lo"], 0)
        self.assertEqual(meta["s_hi"], 2)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    @patch.object(P, "get_true_context_parallel_rank", return_value=1)
    @patch.object(P, "get_true_context_parallel_world_size", return_value=3)
    @patch.object(P, "get_true_context_parallel_group")
    def test_active_shard_with_padding(
        self, mock_group: Mock, _mock_ws: Mock, _mock_rank: Mock, mock_dist: Mock
    ) -> None:
        """S=4 with cp_size=3 -> pads to 6, each rank gets 2 tokens."""
        mock_dist.is_initialized.return_value = True
        mock_group.return_value = MagicMock(name="cp_group")
        a = torch.randn(1, 4, 8)
        b = torch.randn(1, 4, 16)
        sa, _sb, meta = P.prepare_true_cp_sequence_shard(a, b, grid_h=2, grid_w=2)
        self.assertEqual(sa.shape, (1, 2, 8))
        assert meta is not None
        self.assertEqual(meta["s_padded"], 6)
        self.assertEqual(meta["s_lo"], 2)
        self.assertEqual(meta["s_hi"], 4)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    @patch.object(P, "get_true_context_parallel_world_size", return_value=2)
    @patch.object(P, "get_true_context_parallel_group")
    def test_batch_mismatch_raises(self, mock_group: Mock, _mock_ws: Mock, mock_dist: Mock) -> None:
        mock_dist.is_initialized.return_value = True
        mock_group.return_value = MagicMock()
        a = torch.randn(1, 4, 8)
        b = torch.randn(2, 4, 16)  # batch mismatch
        with self.assertRaisesRegex(ValueError, "match in batch"):
            P.prepare_true_cp_sequence_shard(a, b, grid_h=2, grid_w=2)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    @patch.object(P, "get_true_context_parallel_world_size", return_value=2)
    @patch.object(P, "get_true_context_parallel_group")
    def test_grid_mismatch_raises(self, mock_group: Mock, _mock_ws: Mock, mock_dist: Mock) -> None:
        mock_dist.is_initialized.return_value = True
        mock_group.return_value = MagicMock()
        a = torch.randn(1, 5, 8)  # S=5 not equal to 2*2
        b = torch.randn(1, 5, 16)
        with self.assertRaisesRegex(ValueError, "grid_h"):
            P.prepare_true_cp_sequence_shard(a, b, grid_h=2, grid_w=2)

    def test_gather_trim_meta_none_identity(self) -> None:
        x = torch.randn(2, 4, 8)
        out = P.gather_trim_true_cp_sequence(x, meta=None)
        self.assertIs(out, x)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_gather_trim_active(self, mock_dist: Mock) -> None:
        cp_group = MagicMock(name="cp_group")

        def fake_all_gather(tensor_list: list[torch.Tensor], tensor: torch.Tensor, group: object = None) -> None:
            for t in tensor_list:
                t.copy_(tensor)

        mock_dist.all_gather.side_effect = fake_all_gather

        x = torch.randn(2, 3, 8)
        meta = {
            "active": True,
            "cp_group": cp_group,
            "cp_size": 2,
            "s_unpadded": 5,
            "s_padded": 6,
            "s_lo": 0,
            "s_hi": 3,
            "s_local": 3,
        }
        out = P.gather_trim_true_cp_sequence(x, meta)
        # gathered length 6, trimmed to s_unpadded=5
        self.assertEqual(out.shape, (2, 5, 8))


# =============================================================================
# slice_rope_freq_along_sequence — all ndim cases
# =============================================================================


class TestSliceRopeFreq(unittest.TestCase):
    def test_ndim_4(self) -> None:
        freq = torch.randn(1, 1, 10, 4)
        out = P.slice_rope_freq_along_sequence(freq, s_lo=2, s_hi=5)
        self.assertEqual(out.shape, (1, 1, 3, 4))

    def test_ndim_3(self) -> None:
        freq = torch.randn(1, 10, 4)
        out = P.slice_rope_freq_along_sequence(freq, s_lo=1, s_hi=7)
        self.assertEqual(out.shape, (1, 6, 4))

    def test_ndim_2(self) -> None:
        freq = torch.randn(10, 4)
        out = P.slice_rope_freq_along_sequence(freq, s_lo=0, s_hi=4)
        self.assertEqual(out.shape, (4, 4))

    def test_invalid_ndim_raises(self) -> None:
        freq = torch.randn(2, 2, 2, 2, 2)
        with self.assertRaisesRegex(ValueError, "unsupported ndim"):
            P.slice_rope_freq_along_sequence(freq, s_lo=0, s_hi=1)


# =============================================================================
# sdpa_or_true_cp_attention — CP active branches
# =============================================================================


class TestSdpaOrTrueCpAttention(unittest.TestCase):
    def test_no_cp_group_falls_back_to_sdpa(self) -> None:
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        out = P.sdpa_or_true_cp_attention(q, k, v, scale=0.125, attn_mask=None, dropout_p=0.0, cp_group=None)
        self.assertEqual(out.shape, (1, 2, 4, 8))

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_cp_group_world_size_one_falls_back(self, mock_dist: Mock) -> None:
        """cp_group present but world_size<=1 -> still uses plain SDPA."""
        mock_dist.get_world_size.return_value = 1
        cp_group = MagicMock()
        q = torch.randn(1, 2, 4, 8)
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        out = P.sdpa_or_true_cp_attention(q, k, v, scale=0.125, attn_mask=None, dropout_p=0.0, cp_group=cp_group)
        self.assertEqual(out.shape, (1, 2, 4, 8))

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_cp_active_gathers_kv(self, mock_dist: Mock) -> None:
        """cp_size=2 -> gather k/v along seq, then SDPA."""
        mock_dist.get_world_size.return_value = 2

        def fake_all_gather(tensor_list: list[torch.Tensor], tensor: torch.Tensor, group: object = None) -> None:
            for t in tensor_list:
                t.copy_(tensor)

        mock_dist.all_gather.side_effect = fake_all_gather

        q = torch.randn(1, 2, 8, 4)
        k = torch.randn(1, 2, 4, 4)
        v = torch.randn(1, 2, 4, 4)
        out = P.sdpa_or_true_cp_attention(q, k, v, scale=0.5, attn_mask=None, dropout_p=0.0, cp_group=MagicMock())
        self.assertEqual(out.shape, (1, 2, 8, 4))


# =============================================================================
# Tensor-parallel collective autograd Functions — group-active branches
# =============================================================================


class TestModelParallelCollectivesActive(unittest.TestCase):
    """Exercise the 'group is not None and world_size > 1' branches of the four ``_*ToModelParallelRegion`` autograd
    Functions.
    """

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_copy_to_tp_active_all_reduce_backward(self, mock_dist: Mock) -> None:
        """_CopyToModelParallelRegion forward is identity; backward all-reduces grads."""
        mock_dist.get_world_size.return_value = 2
        group = MagicMock()
        P._MODEL_PARALLEL_STATE["tensor_parallel_group"] = group

        x = torch.randn(2, 4, requires_grad=True)
        out = P.copy_to_tensor_parallel_region(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        mock_dist.all_reduce.assert_called_once()

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_reduce_from_tp_active_all_reduce_forward(self, mock_dist: Mock) -> None:
        mock_dist.get_world_size.return_value = 2
        group = MagicMock()
        P._MODEL_PARALLEL_STATE["tensor_parallel_group"] = group

        x = torch.randn(2, 4, requires_grad=True)
        out = P.reduce_from_tensor_parallel_region(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        mock_dist.all_reduce.assert_called_once()

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_gather_from_tp_active(self, mock_dist: Mock) -> None:
        mock_dist.get_world_size.return_value = 2
        mock_dist.get_rank.return_value = 0

        def fake_all_gather(tensor_list: list[torch.Tensor], tensor: torch.Tensor, group: object = None) -> None:
            for t in tensor_list:
                t.copy_(tensor)

        mock_dist.all_gather.side_effect = fake_all_gather
        group = MagicMock()
        P._MODEL_PARALLEL_STATE["tensor_parallel_group"] = group

        x = torch.randn(2, 4, requires_grad=True)
        out = P.gather_from_tensor_parallel_region(x, dim=-1)
        # gather along dim -1 -> output last-dim = 2 * 4 = 8
        self.assertEqual(out.shape, (2, 8))
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_gather_from_tp_world_size_one(self, mock_dist: Mock) -> None:
        """world_size=1 branch returns identity in forward AND backward."""
        mock_dist.get_world_size.return_value = 1
        group = MagicMock()
        P._MODEL_PARALLEL_STATE["tensor_parallel_group"] = group

        x = torch.randn(2, 4, requires_grad=True)
        out = P.gather_from_tensor_parallel_region(x, dim=-1)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_scatter_to_tp_active(self, mock_dist: Mock) -> None:
        mock_dist.get_world_size.return_value = 2
        mock_dist.get_rank.return_value = 0

        def fake_all_gather(tensor_list: list[torch.Tensor], tensor: torch.Tensor, group: object = None) -> None:
            for t in tensor_list:
                t.copy_(tensor)

        mock_dist.all_gather.side_effect = fake_all_gather
        group = MagicMock()
        P._MODEL_PARALLEL_STATE["tensor_parallel_group"] = group

        x = torch.randn(2, 8, requires_grad=True)
        out = P.scatter_to_tensor_parallel_region(x, dim=-1)
        # scatter dim -1 with world_size=2 -> last-dim = 4
        self.assertEqual(out.shape, (2, 4))
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_scatter_to_tp_world_size_one(self, mock_dist: Mock) -> None:
        mock_dist.get_world_size.return_value = 1
        group = MagicMock()
        P._MODEL_PARALLEL_STATE["tensor_parallel_group"] = group

        x = torch.randn(2, 8, requires_grad=True)
        out = P.scatter_to_tensor_parallel_region(x, dim=-1)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_get_tensor_parallel_rank_with_group(self, mock_dist: Mock) -> None:
        mock_dist.get_rank.return_value = 1
        group = MagicMock()
        P._MODEL_PARALLEL_STATE["tensor_parallel_group"] = group
        self.assertEqual(P.get_tensor_parallel_rank(), 1)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_get_frame_parallel_rank_with_group(self, mock_dist: Mock) -> None:
        mock_dist.get_rank.return_value = 2
        group = MagicMock()
        P._MODEL_PARALLEL_STATE["frame_parallel_group"] = group
        self.assertEqual(P.get_frame_parallel_rank(), 2)

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    def test_get_true_cp_rank_with_group(self, mock_dist: Mock) -> None:
        mock_dist.get_rank.return_value = 3
        group = MagicMock()
        P._MODEL_PARALLEL_STATE["true_context_parallel_group"] = group
        self.assertEqual(P.get_true_context_parallel_rank(), 3)


# =============================================================================
# get_tensor_parallel_world_size_and_rank_for_inference_viz
# =============================================================================


class TestTpWorldSizeRankForVis(unittest.TestCase):
    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def _fake_megatron(
        self, initialized: bool, world_size: int = 2, rank: int = 1
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        fake_ps = MagicMock()
        fake_ps.is_initialized.return_value = initialized
        fake_ps.get_tensor_model_parallel_world_size.return_value = world_size
        fake_ps.get_tensor_model_parallel_rank.return_value = rank
        fake_core = MagicMock(name="megatron.core")
        fake_core.parallel_state = fake_ps
        fake_megatron = MagicMock(name="megatron")
        fake_megatron.core = fake_core
        return fake_megatron, fake_core, fake_ps

    def test_megatron_not_initialized_uses_local_state(self) -> None:
        fake_megatron, fake_core, _ = self._fake_megatron(initialized=False)
        with patch.dict(sys.modules, {"megatron": fake_megatron, "megatron.core": fake_core}):
            ws, rk = P.get_tensor_parallel_world_size_and_rank_for_inference_viz()
        self.assertEqual(ws, 1)
        self.assertEqual(rk, 0)

    def test_megatron_initialized_uses_megatron(self) -> None:
        _fake_megatron, _fake_core, fake_ps = self._fake_megatron(initialized=True, world_size=4, rank=2)
        with patch.object(P, "parallel_state", fake_ps):
            ws, rk = P.get_tensor_parallel_world_size_and_rank_for_inference_viz()
        self.assertEqual(ws, 4)
        self.assertEqual(rk, 2)

    def test_megatron_query_raises_runtimeerror_falls_back(self) -> None:
        """Megatron's is_initialized raising RuntimeError falls back to local state."""
        fake_megatron, fake_core, fake_ps = self._fake_megatron(initialized=True)
        fake_ps.is_initialized.side_effect = RuntimeError("groups not built yet")
        with patch.dict(sys.modules, {"megatron": fake_megatron, "megatron.core": fake_core}):
            ws, rk = P.get_tensor_parallel_world_size_and_rank_for_inference_viz()
        self.assertEqual(ws, 1)
        self.assertEqual(rk, 0)

    def test_megatron_import_error_falls_back(self) -> None:
        """No megatron installed -> local state."""
        with patch.dict(sys.modules, {"megatron.core": None}):
            ws, rk = P.get_tensor_parallel_world_size_and_rank_for_inference_viz()
        self.assertEqual(ws, 1)
        self.assertEqual(rk, 0)


# =============================================================================
# get_megatron_config
# =============================================================================


class TestGetMegatronConfig(unittest.TestCase):
    """Exercises both branches of get_megatron_config: cached return + fresh construction."""

    def setUp(self) -> None:
        self._saved_config = P._MEGATRON_CONFIG
        self._saved_has_megatron = P.HAS_MEGATRON
        P._MEGATRON_CONFIG = None

    def tearDown(self) -> None:
        P._MEGATRON_CONFIG = self._saved_config
        P.HAS_MEGATRON = self._saved_has_megatron

    def test_returns_cached_when_already_built(self) -> None:
        """Pre-existing _MEGATRON_CONFIG is returned directly without rebuild."""
        sentinel = object()
        P._MEGATRON_CONFIG = sentinel
        P.HAS_MEGATRON = True
        self.assertIs(P.get_megatron_config(), sentinel)

    def test_builds_when_has_megatron_and_none(self) -> None:
        """When HAS_MEGATRON is True and cache is None, ModelParallelConfig is constructed once."""
        fake_config_cls = MagicMock(name="ModelParallelConfig")
        fake_config_cls.return_value = MagicMock(name="config_instance")
        with patch.object(P, "HAS_MEGATRON", True), patch.object(P, "ModelParallelConfig", fake_config_cls):
            cfg = P.get_megatron_config()
        self.assertIs(cfg, fake_config_cls.return_value)
        fake_config_cls.assert_called_once()


# =============================================================================
# TPColumnParallelLinear with _use_megatron=True (mocked)
# =============================================================================


def _make_fake_megatron_linear(out_features: int, in_features: int, has_bias: bool = True) -> nn.Module:
    """A drop-in stand-in for Megatron's ColumnParallelLinear / RowParallelLinear.

    Exposes ``weight`` (Parameter) and ``bias`` (Parameter or None) attributes plus a ``forward(x)`` that returns
    ``(F.linear(x, weight, bias), None)`` — matching the (output, bias) tuple return shape Megatron uses.
    """

    class FakeMegatronLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            if has_bias:
                self.bias = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("bias", None)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
            return torch.nn.functional.linear(x, self.weight, self.bias), None

    return FakeMegatronLinear()


class TestTPColumnParallelLinearMegatronPaths(unittest.TestCase):
    """Exercise TPColumnParallelLinear branches that require ``_use_megatron=True``."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def _make_megatron_layer(
        self, out_features: int = 8, in_features: int = 4, bias: bool = True
    ) -> P.TPColumnParallelLinear:
        layer = P.TPColumnParallelLinear(in_features, out_features, bias=bias)
        layer._use_megatron = True
        # Replace its weight/bias parameters with a fake _megatron_layer sub-module. __getattr__ proxies ``weight`` /
        # ``bias`` to ``_megatron_layer`` when _use_megatron is True, so we need to delete the direct nn.Parameter
        # attributes first to avoid Module's own __getattr__ shadowing.
        del layer._parameters["weight"]
        if "bias" in layer._parameters:
            del layer._parameters["bias"]
        layer._megatron_layer = _make_fake_megatron_linear(out_features, in_features, has_bias=bias)
        return layer

    def test_out_features_not_divisible_raises(self) -> None:
        """Constructor raises when out_features not divisible by tp_size."""
        # tp_size defaults to 1 -> any out_features is divisible. Force tp_size=2 via mock.
        with (
            patch.object(P, "get_tensor_parallel_world_size", return_value=2),
            self.assertRaisesRegex(ValueError, "must be divisible"),
        ):
            P.TPColumnParallelLinear(4, 7)

    def test_getattr_proxy_weight(self) -> None:
        layer = self._make_megatron_layer()
        # ``weight`` access should route through __getattr__ -> _megatron_layer.weight
        self.assertEqual(layer.weight.shape, (8, 4))

    def test_getattr_proxy_bias(self) -> None:
        layer = self._make_megatron_layer()
        self.assertEqual(layer.bias.shape, (8,))

    def test_save_to_state_dict_megatron(self) -> None:
        layer = self._make_megatron_layer()
        dst: dict = {}
        layer._save_to_state_dict(dst, prefix="ll.", keep_vars=False)
        self.assertIn("ll.weight", dst)
        self.assertIn("ll.bias", dst)
        self.assertFalse(isinstance(dst["ll.weight"], nn.Parameter))

    def test_save_to_state_dict_keep_vars(self) -> None:
        layer = self._make_megatron_layer()
        dst: dict = {}
        layer._save_to_state_dict(dst, prefix="ll.", keep_vars=True)
        self.assertIsInstance(dst["ll.weight"], nn.Parameter)

    def test_save_to_state_dict_no_bias(self) -> None:
        layer = self._make_megatron_layer(bias=False)
        dst: dict = {}
        layer._save_to_state_dict(dst, prefix="ll.", keep_vars=False)
        self.assertIn("ll.weight", dst)
        self.assertNotIn("ll.bias", dst)

    def test_state_dict_megatron_skips_child_traversal(self) -> None:
        layer = self._make_megatron_layer()
        sd = layer.state_dict()
        # Only flat keys; no ``_megatron_layer.weight`` duplicate.
        self.assertIn("weight", sd)
        self.assertNotIn("_megatron_layer.weight", sd)

    def test_load_from_state_dict_matching_shape(self) -> None:
        layer = self._make_megatron_layer(out_features=8, in_features=4)
        weight = torch.randn(8, 4)
        bias = torch.randn(8)
        sd = {"weight": weight, "bias": bias}
        missing: list = []
        unexpected: list = []
        errs: list = []
        layer._load_from_state_dict(
            sd,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=missing,
            unexpected_keys=unexpected,
            error_msgs=errs,
        )
        self.assertTrue(torch.equal(layer._megatron_layer.weight.data, weight))
        self.assertTrue(torch.equal(layer._megatron_layer.bias.data, bias))
        self.assertIn("_megatron_layer.weight", sd)
        self.assertEqual(missing, [])

    def test_load_from_state_dict_shard_full_weights(self) -> None:
        """When the incoming weight is larger, it's sliced along dim 0 for this rank."""
        layer = self._make_megatron_layer(out_features=8, in_features=4)
        # "Full" weights with tp_size=2 would be (16, 4). We feed that and expect
        # rank 0 to copy the first 8 rows.
        full_w = torch.arange(16 * 4).reshape(16, 4).float()
        full_b = torch.arange(16).float()
        sd = {"weight": full_w, "bias": full_b}
        with patch.object(P, "get_tensor_parallel_rank", return_value=0):
            layer._load_from_state_dict(
                sd, prefix="", local_metadata={}, strict=True, missing_keys=[], unexpected_keys=[], error_msgs=[]
            )
        self.assertTrue(torch.equal(layer._megatron_layer.weight.data, full_w[:8]))
        self.assertTrue(torch.equal(layer._megatron_layer.bias.data, full_b[:8]))

    def test_load_from_state_dict_missing_keys_strict(self) -> None:
        layer = self._make_megatron_layer(out_features=8, in_features=4)
        missing: list = []
        layer._load_from_state_dict(
            {}, prefix="", local_metadata={}, strict=True, missing_keys=missing, unexpected_keys=[], error_msgs=[]
        )
        self.assertIn("weight", missing)
        self.assertIn("bias", missing)

    def test_forward_megatron(self) -> None:
        layer = self._make_megatron_layer(out_features=8, in_features=4)
        x = torch.randn(2, 4)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8))


class TestTPRowParallelLinearMegatronPaths(unittest.TestCase):
    """Exercise TPRowParallelLinear branches that require ``_use_megatron=True``."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def _make_megatron_layer(
        self, out_features: int = 4, in_features: int = 8, bias: bool = True
    ) -> P.TPRowParallelLinear:
        layer = P.TPRowParallelLinear(in_features, out_features, bias=bias)
        layer._use_megatron = True
        del layer._parameters["weight"]
        if "bias" in layer._parameters:
            del layer._parameters["bias"]
        layer._megatron_layer = _make_fake_megatron_linear(out_features, in_features, has_bias=bias)
        return layer

    def test_in_features_not_divisible_raises(self) -> None:
        with (
            patch.object(P, "get_tensor_parallel_world_size", return_value=2),
            self.assertRaisesRegex(ValueError, "must be divisible"),
        ):
            P.TPRowParallelLinear(7, 4)

    def test_getattr_proxy(self) -> None:
        layer = self._make_megatron_layer()
        self.assertEqual(layer.weight.shape, (4, 8))
        self.assertEqual(layer.bias.shape, (4,))

    def test_save_to_state_dict_megatron(self) -> None:
        layer = self._make_megatron_layer()
        dst: dict = {}
        layer._save_to_state_dict(dst, prefix="ll.", keep_vars=False)
        self.assertIn("ll.weight", dst)
        self.assertIn("ll.bias", dst)

    def test_save_to_state_dict_keep_vars(self) -> None:
        layer = self._make_megatron_layer()
        dst: dict = {}
        layer._save_to_state_dict(dst, prefix="ll.", keep_vars=True)
        self.assertIsInstance(dst["ll.weight"], nn.Parameter)

    def test_save_to_state_dict_no_bias(self) -> None:
        layer = self._make_megatron_layer(bias=False)
        dst: dict = {}
        layer._save_to_state_dict(dst, prefix="ll.", keep_vars=False)
        self.assertIn("ll.weight", dst)
        self.assertNotIn("ll.bias", dst)

    def test_state_dict_megatron_flat(self) -> None:
        layer = self._make_megatron_layer()
        sd = layer.state_dict()
        self.assertIn("weight", sd)
        self.assertNotIn("_megatron_layer.weight", sd)

    def test_load_from_state_dict_matching_shape(self) -> None:
        layer = self._make_megatron_layer(out_features=4, in_features=8)
        weight = torch.randn(4, 8)
        bias = torch.randn(4)
        sd = {"weight": weight, "bias": bias}
        layer._load_from_state_dict(
            sd, prefix="", local_metadata={}, strict=True, missing_keys=[], unexpected_keys=[], error_msgs=[]
        )
        self.assertTrue(torch.equal(layer._megatron_layer.weight.data, weight))
        self.assertTrue(torch.equal(layer._megatron_layer.bias.data, bias))

    def test_load_from_state_dict_shard_full_weights_dim1(self) -> None:
        """RowParallel auto-shards full weights along dim 1 (the in_features axis)."""
        layer = self._make_megatron_layer(out_features=4, in_features=8)
        # "Full" weights with tp_size=2 would be (4, 16); each rank should take dim-1 slice.
        full_w = torch.arange(4 * 16).reshape(4, 16).float()
        # Bias is NOT sharded for row-parallel; size must match expected (4,).
        bias = torch.arange(4).float()
        sd = {"weight": full_w, "bias": bias}
        with patch.object(P, "get_tensor_parallel_rank", return_value=1):
            layer._load_from_state_dict(
                sd, prefix="", local_metadata={}, strict=True, missing_keys=[], unexpected_keys=[], error_msgs=[]
            )
        self.assertTrue(torch.equal(layer._megatron_layer.weight.data, full_w[:, 8:16]))
        self.assertTrue(torch.equal(layer._megatron_layer.bias.data, bias))

    def test_load_from_state_dict_missing_strict(self) -> None:
        layer = self._make_megatron_layer(out_features=4, in_features=8)
        missing: list = []
        layer._load_from_state_dict(
            {}, prefix="", local_metadata={}, strict=True, missing_keys=missing, unexpected_keys=[], error_msgs=[]
        )
        self.assertIn("weight", missing)
        self.assertIn("bias", missing)

    def test_forward_megatron(self) -> None:
        layer = self._make_megatron_layer(out_features=4, in_features=8)
        x = torch.randn(2, 8)
        out = layer(x)
        self.assertEqual(out.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
