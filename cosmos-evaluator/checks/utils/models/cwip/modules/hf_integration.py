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
Hugging Face ``transformers`` integration for CWIP.

Defines :class:`HFModelConfig` (the :class:`~transformers.PretrainedConfig` companion) and :class:`CWIPPreTrainedModel`
(the :class:`~transformers.PreTrainedModel` base class). Concrete CWIP variants subclass
:class:`CWIPPreTrainedModel` and implement the three abstract hooks documented on it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from third_party.perception_models.core.vision_encoder.rope import Rope2D
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from checks.utils.models.cwip.modules.configs.defaults.model import ModelConfig

logger = logging.getLogger(__name__)


# Restrict Hub snapshots to files CWIP's loaders consume, so customers don't pay for unrelated assets (training logs,
# viz artifacts, ...) some model-card authors include alongside the weights.
_HUB_ALLOW_PATTERNS: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
)


def _resolve_to_local_dir(
    pretrained_model_name_or_path: str | os.PathLike,
    *,
    revision: str | None = None,
    cache_dir: str | os.PathLike | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> str:
    """Return a local directory; download from the Hub if given a repo ID.

    Args:
        pretrained_model_name_or_path: Existing local directory OR a Hub repo ID (``"org/repo"``).
        revision: Git ref (branch, tag, or commit SHA) on the Hub repo. Ignored for local directories.
        cache_dir: Override for the default ``~/.cache/huggingface/hub`` location used by :func:`snapshot_download`.
        token: HF access token, or ``True`` to use the cached login. Needed for private repos.
        local_files_only: If True, refuse to hit the network; raise :class:`FileNotFoundError` if the input isn't
            already on disk.

    Returns:
        Absolute path to the on-disk directory containing the checkpoint files.

    Raises:
        FileNotFoundError: The path is neither a local directory nor an accessible Hub repo, or the Hub download failed.
    """
    path = str(pretrained_model_name_or_path)
    if os.path.isdir(path):
        return path

    if local_files_only and not os.path.isdir(path):
        raise FileNotFoundError(f"local_files_only=True but {path!r} is not an existing local directory.")

    try:
        return snapshot_download(
            repo_id=path,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            token=token,
            allow_patterns=list(_HUB_ALLOW_PATTERNS),
            local_files_only=local_files_only,
        )
    except RepositoryNotFoundError as exc:
        raise FileNotFoundError(
            f"{path!r} is neither a local directory nor an accessible Hub repo. "
            "If this is a private repo, pass a token via the `token` kwarg or "
            "log in with `huggingface-cli login`."
        ) from exc
    except HfHubHTTPError as exc:
        raise FileNotFoundError(f"Failed to download {path!r} from the Hugging Face Hub: {exc}") from exc


class HFModelConfig(PretrainedConfig):
    """:class:`~transformers.PretrainedConfig` companion for CWIP models.

    Stores the raw ``config.json`` verbatim under :attr:`cwip_config_dict`;
    :meth:`CWIPPreTrainedModel._reconstruct_config_from_dict` turns it into a
    canonical :class:`ModelConfig` at construction time.
    """

    model_type = "cwip"

    def __init__(self, cwip_config_dict: dict | None = None, **kwargs: Any):
        """Store the raw ``config.json`` dict for later reconstruction.

        Args:
            cwip_config_dict: Parsed ``config.json``. ``None`` produces an empty config (used when HF instantiates from
                a blank state).
            **kwargs: Forwarded to :meth:`PretrainedConfig.__init__`.
        """
        super().__init__(**kwargs)
        self.cwip_config_dict: dict = cwip_config_dict or {}

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs: Any,
    ) -> HFModelConfig | tuple[HFModelConfig, dict[str, Any]]:
        """Load a CWIP config from a local directory or a Hub repo.

        Reads ``config.json`` verbatim into :attr:`cwip_config_dict` rather than going through HF's default parser
        (which would reject CWIP's nested fields). Pops the standard HF-internal kwargs so they're not forwarded
        downstream, and honors the ``return_unused_kwargs`` contract that :meth:`PreTrainedModel.from_pretrained`
        relies on.

        Args:
            pretrained_model_name_or_path: Local directory OR Hub repo ID.
            **kwargs: Standard HF kwargs (``cache_dir``, ``token``, ``revision``, ``local_files_only``,
                ``return_unused_kwargs``, ...). HF-internal kwargs are popped and discarded.

        Returns:
            The loaded :class:`HFModelConfig`. If ``return_unused_kwargs=True``, returns ``(config, unused_kwargs)``
            instead.

        Raises:
            FileNotFoundError: No ``config.json`` exists at the resolved directory.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        cache_dir = kwargs.pop("cache_dir", None)
        kwargs.pop("force_download", False)
        kwargs.pop("resume_download", None)
        kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("subfolder", "")
        kwargs.pop("_from_pipeline", None)
        kwargs.pop("_from_auto", False)
        kwargs.pop("_commit_hash", None)

        local_dir = _resolve_to_local_dir(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        config_path = os.path.join(local_dir, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"No config.json found in {local_dir}")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        config = cls(cwip_config_dict=raw)
        if return_unused_kwargs:
            return config, kwargs
        return config


class CWIPPreTrainedModel(PreTrainedModel):
    """Base class wiring CWIP models into the ``transformers`` runtime.

    Provides the HF capability flags, a two-branch ``__init__`` that accepts either an :class:`HFModelConfig` (HF path)
    or a :class:`ModelConfig` (direct construction), and a :meth:`from_pretrained` classmethod with a post-load
    :class:`Rope2D` re-materialisation step (see method docstring).

    Subclasses must implement three hooks:

    * :meth:`_reconstruct_config_from_dict` -- rebuild a :class:`ModelConfig` from a saved ``config.json`` dict.
    * :meth:`_instantiate_encoders_and_decoders` -- build all ``nn.Module`` children that consume :attr:`self.config`.
    * :meth:`_initialize_remaining_parameters` -- post-instantiation parameter setup (learnable temperatures, logit
      biases, ...).

    Instance attrs after construction:

    * :attr:`config` -- always the canonical :class:`ModelConfig` (both ``__init__`` branches end here, so subclass /
      external code can read ``self.config.X`` unconditionally).
    * :attr:`hf_config` -- the :class:`HFModelConfig` companion, populated only on the HF path; ``None`` for direct
      construction.
    * :attr:`inference_lock` -- serializes forward passes; see :meth:`__init__`.
    """

    # ---- HF PreTrainedModel class attrs -------------------------------------

    config_class = HFModelConfig
    # Empty prefix: the model body itself is the top-level class, so state_dict
    # keys carry no wrapper-class prefix (camera.X, world.X, ...).
    base_model_prefix = ""
    main_input_name = "camera_image"
    supports_gradient_checkpointing = False
    # CWIP uses F.scaled_dot_product_attention directly; this flag just
    # advertises that to callers that introspect it.
    _supports_sdpa = True
    _supports_flash_attn_2 = False

    hf_config: HFModelConfig | None = None

    inference_lock: threading.Lock

    # ---- Construction -------------------------------------------------------

    def __init__(self, config: ModelConfig | HFModelConfig):
        """Initialize from either an :class:`HFModelConfig` or a :class:`ModelConfig`.

        The HF branch runs :meth:`PreTrainedModel.__init__` + :meth:`post_init`; the direct-construction branch uses the
        bare :meth:`nn.Module.__init__` so callers don't need to fabricate a stub :class:`HFModelConfig`. Both branches
        end with ``self.config`` as a :class:`ModelConfig`.

        Also creates :attr:`inference_lock`, which forward passes must hold. A checker service loads the model once and
        hands the same instance to every job it runs concurrently, and a forward pass mutates instance state that is not
        guarded otherwise: the resolution-dependent ``image_size`` / ``image_transform`` pair, and the ``Rope2D``
        frequency grid inside each encoder. ``Rope2D.update_grid`` in particular publishes its ``grid_size`` guard
        before the ``freq`` tensor that guard protects, so a second thread arriving at the same resolution skips the
        build and dereferences ``freq`` while it is still ``None``. The lock is per instance, so independently loaded
        models never wait on each other.

        Args:
            config: Either an :class:`HFModelConfig` (HF / ``from_pretrained`` path; ``self.hf_config`` is set to the
                input and ``self.config`` to the reconstructed :class:`ModelConfig`) or a :class:`ModelConfig` (direct
                construction; ``self.hf_config`` stays ``None``).
        """
        if isinstance(config, PretrainedConfig):
            super().__init__(config)
            self.hf_config = config
            # CWIP deliberately replaces the HF ``PreTrainedConfig`` with its own ``ModelConfig``; HF's
            # base typing expects ``self.config`` to stay a ``PreTrainedConfig``.
            self.config = self._reconstruct_config_from_dict(config.cwip_config_dict)  # type: ignore[assignment]
            # ``PreTrainedModel`` and its loaders read ``model_type`` and the ``base_model_{tp,ep,pp,fsdp}_plan`` fields
            # off ``self.config`` directly (e.g. in ``conversion_mapping`` and ``post_init``).
            hf_runtime_config: Any = self.config
            hf_runtime_config.model_type = type(config).model_type
            hf_runtime_config.base_model_tp_plan = None
            hf_runtime_config.base_model_ep_plan = None
            hf_runtime_config.base_model_pp_plan = None
            hf_runtime_config.base_model_fsdp_plan = None
            _run_post_init = True
        else:
            nn.Module.__init__(self)
            self.config = config  # type: ignore[assignment]
            self.hf_config = None
            _run_post_init = False

        self.inference_lock = threading.Lock()

        self._instantiate_encoders_and_decoders()
        self._initialize_remaining_parameters()

        # Seeds all_tied_weights_keys, _tp_plan, _keep_in_fp32_modules, etc., which downstream HF loaders read off
        # ``self``. Skip on the direct path -- there's no HF machinery to wire up.
        if _run_post_init:
            self.post_init()

    def _init_weights(self, module: torch.nn.Module) -> None:
        """No-op. CWIP submodules self-initialize; weights come from safetensors.

        Args:
            module: Submodule visited by :meth:`PreTrainedModel.__init__`'s tree walk. Unused.
        """
        return

    # ---- Subclass hooks (must be implemented) -------------------------------

    @staticmethod
    def _reconstruct_config_from_dict(saved_config_dict: dict) -> ModelConfig:
        """Rebuild a :class:`ModelConfig` from a saved ``config.json`` dict.

        Args:
            saved_config_dict: Parsed ``config.json``, as stored under :attr:`HFModelConfig.cwip_config_dict`.

        Returns:
            A fully-materialised :class:`ModelConfig` ready to drive :meth:`_instantiate_encoders_and_decoders`.
        """
        raise NotImplementedError

    def _instantiate_encoders_and_decoders(self) -> None:
        """Build all ``nn.Module`` children that consume :attr:`self.config`.

        Called once at the end of :meth:`__init__`, after :attr:`self.config` is set on either code path.
        """
        raise NotImplementedError

    def _initialize_remaining_parameters(self) -> None:
        """Post-instantiation parameter setup (temperatures, logit biases, ...).

        Called once at the end of :meth:`__init__`, after :meth:`_instantiate_encoders_and_decoders`.
        """
        raise NotImplementedError

    # ---- HF Hub loading -----------------------------------------------------

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        *args: Any,
        **kwargs: Any,
    ) -> PreTrainedModel:
        """Load a CWIP checkpoint from a local directory or a Hub repo.

        Thin wrapper around :meth:`PreTrainedModel.from_pretrained` that adds a single CWIP-specific post-load step:
        **Rope2D re-materialisation**.

        Modern ``transformers`` builds every submodule on ``torch.device("meta")`` and then re-materialises each
        ``nn.Parameter`` / buffer from safetensors. ``Rope2D`` is a plain Python container (not an ``nn.Module``), so
        its inner ``RotaryEmbedding.freqs`` is invisible to HF's tree walk and stays on meta. The first forward pass
        would then fail in ``Rope2D.update_grid`` with ``"Cannot copy out of meta tensor"``. We walk the loaded model,
        find every ``Rope2D``, and rebuild its ``RotaryEmbedding`` from deterministic constants (bit-identical to what
        HF would have loaded).

        Older checkpoints needing QKV-layout normalization should be loaded via the subclass's ``from_hf_format``
        instead.

        For ``channels_last``: apply it yourself after the model is on CUDA::

            model = CWIPInference.from_pretrained(..., device_map="cuda")
            model.to(memory_format=torch.channels_last)
            model._use_channels_last = True

        Args:
            pretrained_model_name_or_path: Local directory OR Hub repo ID.
            *args, **kwargs: Forwarded to :meth:`PreTrainedModel.from_pretrained` (``revision``, ``cache_dir``,
                ``token``, ``torch_dtype``, ``device_map``, ``local_files_only``, ``low_cpu_mem_usage``, ...).

        Returns:
            A ``cls`` instance with ``self.config`` populated from ``config.json``, weights loaded from safetensors, and
            every ``Rope2D`` rotary embedding re-materialised.

        Hub example (see ``checks/utils/models/cwip/README.md`` for the canonical published repo IDs)::

            model = CWIPInference.from_pretrained(
                "<org>/<repo>",
                revision="v1.0.0",
                torch_dtype="bfloat16",
                device_map="cuda",
            )
        """
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        rope_count = 0
        for _sub in model.modules():
            _rope = _sub.__dict__.get("rope")
            if isinstance(_rope, Rope2D):
                _rope.init_tensors()
                rope_count += 1
        if rope_count:
            logger.debug(
                "from_pretrained: re-materialized %d orphan Rope2D rotary embeddings",
                rope_count,
            )

        return model
