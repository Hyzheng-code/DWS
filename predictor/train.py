#!/usr/bin/env python
"""Standalone Our-DWS training with pooled prompt states and two hazard heads.

Dependencies: torch, transformers, numpy. Run with --help for inputs.
Reads an existing token cache, workload labels and alignment/split arrays.
Stage A: BS128, 1 frozen + 20 joint epochs; head/encoder LR 3e-4/4e-5,
hold the first 50% then decay linearly to zero, without warm-up or early stop.
Supervise only T(h), h in {1,4,8,...,32}, with normalized SmoothL1 (beta=1).
Stage B: BS256, 4 frozen + 4 joint epochs, last four encoder layers;
encoder/surface LR 5e-6/2e-4, continuous cosine schedule with 3% warm-up.
Outputs best.pt, config.json, minilm/ and predict.py for the serving backend.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
MAX_DENOISING_STEPS = 32
MAX_BLOCKS = 33
MODEL_MAX_BLOCKS = 33
SOURCE_PADDED_BLOCKS = 129
DEFAULT_HORIZONS = (1, *range(4, MAX_DENOISING_STEPS + 1, 4))


class FactorizedEmbedding(nn.Module):
    """Low-rank word embedding, optionally initialized by a base-model SVD."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        padding_idx: int | None,
        pretrained_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = vocab_size
        self.embedding_dim = hidden_size
        self.padding_idx = padding_idx
        self.codes = nn.Embedding(vocab_size, rank, padding_idx=padding_idx)
        self.projection = nn.Linear(rank, hidden_size, bias=False)
        if pretrained_weight is not None:
            with torch.no_grad():
                weight = pretrained_weight.detach().float().cpu()
                u, singular, vh = torch.linalg.svd(weight, full_matrices=False)
                self.codes.weight.copy_(u[:, :rank] * singular[:rank])
                self.projection.weight.copy_(vh[:rank].transpose(0, 1))
                if padding_idx is not None:
                    self.codes.weight[padding_idx].zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.projection(self.codes(ids))


def build_encoder(
    model_path: Path,
    encoder_layers: int,
    factorized_embedding_rank: int,
    initialize_from_base: bool,
) -> nn.Module:
    """Build an encoder from a local generic base model or only its config."""

    if initialize_from_base:
        encoder = AutoModel.from_pretrained(str(model_path), local_files_only=True)
    else:
        config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        encoder = AutoModel.from_config(config)
    if factorized_embedding_rank:
        original = encoder.embeddings.word_embeddings
        if not 1 <= factorized_embedding_rank <= original.embedding_dim:
            raise ValueError("invalid factorized embedding rank")
        encoder.embeddings.word_embeddings = FactorizedEmbedding(
            original.num_embeddings,
            original.embedding_dim,
            factorized_embedding_rank,
            original.padding_idx,
            original.weight if initialize_from_base else None,
        )
    total_layers = len(encoder.encoder.layer)
    if encoder_layers:
        if not 1 <= encoder_layers <= total_layers:
            raise ValueError(
                f"encoder_layers must be in [1, {total_layers}], got {encoder_layers}"
            )
        selected = np.linspace(0, total_layers - 1, encoder_layers).round().astype(int)
        encoder.encoder.layer = nn.ModuleList(
            [encoder.encoder.layer[int(index)] for index in selected]
        )
        encoder.config.num_hidden_layers = encoder_layers
    if os.environ.get("V11_GRADIENT_CHECKPOINTING", "0") == "1":
        enable = getattr(encoder, "gradient_checkpointing_enable", None)
        if enable is None:
            raise TypeError(
                f"{type(encoder).__name__} does not support gradient checkpointing"
            )
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable()
    return encoder


class SemanticPooler(nn.Module):
    """Pool prompt states while keeping the encoder hidden width unchanged."""

    def __init__(self, hidden_size: int, pooling: str) -> None:
        super().__init__()
        self.pooling = pooling
        if pooling == "mean_max":
            self.gate_norm = nn.LayerNorm(2 * hidden_size)
            self.gate = nn.Linear(2 * hidden_size, hidden_size)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -2.1972245773362196)
        elif pooling in {
            "attention",
            "mean_attention",
            "mean_attention_max",
            "mean_attention_max_diag",
            "mean_attention_max_lr16",
            "mean_attention_max_lr32",
            "mean_attention_max_diag_linear",
            "mean_attention_max_lr16_linear",
        }:
            self.attention_norm = nn.LayerNorm(hidden_size)
            if pooling.endswith("_linear"):
                self.attention_score = nn.Linear(hidden_size, 1, bias=False)
                nn.init.zeros_(self.attention_score.weight)
            else:
                attention_dim = min(hidden_size, 128)
                self.attention_score = nn.Sequential(
                    nn.Linear(hidden_size, attention_dim),
                    nn.GELU(),
                    nn.Linear(attention_dim, 1),
                )
                nn.init.zeros_(self.attention_score[-1].weight)
                nn.init.zeros_(self.attention_score[-1].bias)
            if pooling in {"mean_attention", "mean_attention_max"}:
                views = 2 if pooling == "mean_attention" else 3
                self.view_fusion = nn.Linear(views * hidden_size, hidden_size)
                with torch.no_grad():
                    self.view_fusion.weight.zero_()
                    identity = torch.eye(hidden_size)
                    if views == 2:
                        coefficients = (0.5, 0.5)
                    else:
                        coefficients = (0.45, 0.45, 0.10)
                    for view, coefficient in enumerate(coefficients):
                        start = view * hidden_size
                        self.view_fusion.weight[:, start : start + hidden_size].copy_(
                            coefficient * identity
                        )
                    self.view_fusion.bias.zero_()
            elif pooling.startswith("mean_attention_max_"):
                self.view_scale = nn.Parameter(
                    torch.tensor((0.45, 0.45, 0.10))[:, None].repeat(1, hidden_size)
                )
                self.view_bias = nn.Parameter(torch.zeros(hidden_size))
                if "_diag" not in pooling:
                    rank = 16 if "_lr16" in pooling else 32
                    self.residual_down = nn.Linear(
                        3 * hidden_size, rank, bias=False
                    )
                    self.residual_up = nn.Linear(rank, hidden_size, bias=False)
                    nn.init.normal_(self.residual_down.weight, std=0.02)
                    nn.init.zeros_(self.residual_up.weight)
        elif pooling not in {"cls", "mean", "max"}:
            raise ValueError(f"unknown semantic pooling: {pooling}")

    @staticmethod
    def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(hidden.dtype)
        pooled = (hidden * weights[:, :, None]).sum(dim=1)
        return pooled / weights.sum(dim=1, keepdim=True).clamp_min(1)

    @staticmethod
    def masked_max(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        floor = torch.finfo(hidden.dtype).min
        return hidden.masked_fill(~mask[:, :, None], floor).amax(dim=1)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        mean = self.masked_mean(hidden, mask)
        if self.pooling == "mean":
            return mean
        if self.pooling in {
            "attention",
            "mean_attention",
            "mean_attention_max",
            "mean_attention_max_diag",
            "mean_attention_max_lr16",
            "mean_attention_max_lr32",
            "mean_attention_max_diag_linear",
            "mean_attention_max_lr16_linear",
        }:
            score = self.attention_score(self.attention_norm(hidden))[:, :, 0]
            score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
            weight = torch.softmax(score.float(), dim=1).to(hidden.dtype)
            attended = (hidden * weight[:, :, None]).sum(dim=1)
            if self.pooling == "attention":
                return attended
            views = [mean, attended]
            if self.pooling != "mean_attention":
                views.append(self.masked_max(hidden, mask))
            if hasattr(self, "view_fusion"):
                return self.view_fusion(torch.cat(views, dim=-1))
            stacked = torch.stack(views, dim=1)
            fused = (stacked * self.view_scale[None, :, :]).sum(dim=1)
            if hasattr(self, "residual_down"):
                fused = fused + self.residual_up(
                    self.residual_down(torch.cat(views, dim=-1))
                )
            return fused + self.view_bias
        maximum = self.masked_max(hidden, mask)
        if self.pooling == "max":
            return maximum
        features = torch.cat([mean, maximum], dim=-1)
        gate = torch.sigmoid(self.gate(self.gate_norm(features)))
        return (1.0 - gate) * mean + gate * maximum


class LayerMixer(nn.Module):
    """Learn a convex mix of final encoder layers without extra encoder FLOPs."""

    def __init__(self, last_n: int) -> None:
        super().__init__()
        self.last_n = last_n
        if last_n > 1:
            self.logits = nn.Parameter(torch.zeros(last_n))

    def forward(self, encoder_output) -> torch.Tensor:
        if self.last_n <= 1:
            return encoder_output.last_hidden_state
        states = torch.stack(encoder_output.hidden_states[-self.last_n :], dim=0)
        weights = torch.softmax(self.logits.float(), dim=0).to(states.dtype)
        return (states * weights[:, None, None, None]).sum(dim=0)


class CompactWorkloadCurvePredictor(nn.Module):
    """One pooled prompt representation feeding two continuation/hazard heads.

    The count head also receives Appendix H's prompt-alignment features.
    Token states reach both heads only through pooling; there is no decoder.
    """

    ARCHITECTURE = "pooled_hazard_heads_v1"

    @classmethod
    def validate_config(cls, config):
        if config.get("predictor_architecture") != cls.ARCHITECTURE:
            raise ValueError(
                "Expected pooled_hazard_heads_v1 predictor. Legacy token-memory/"
                "block-decoder checkpoints are incompatible; retrain Stage B "
                "with the pooled heads and re-export the predictor."
            )

    def __init__(
        self,
        model_path: Path,
        config: dict,
        initialize_encoder_from_base: bool = False,
    ) -> None:
        super().__init__()
        self.validate_config(config)
        self.encoder = build_encoder(
            model_path,
            int(config["encoder_layers"]),
            int(config.get("factorized_embedding_rank", 0)),
            initialize_from_base=initialize_encoder_from_base,
        )
        self.semantic_pooling = str(config.get("semantic_pooling", "mean"))
        self.layer_mixer = LayerMixer(
            int(config.get("layer_mix_last_n", 1))
        )
        hidden = int(self.encoder.config.hidden_size)
        head_dim = int(config.get("head_dim", 128))
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        dropout = float(config.get("dropout", 0.1))
        self.count_alignment_features = str(
            config.get("count_alignment_features", "none")
        )
        if self.count_alignment_features not in {
            "none",
            "length",
            "remainder",
            "length_remainder",
        }:
            raise ValueError(
                "count_alignment_features must be none, length, remainder, "
                "or length_remainder"
            )
        self.count_representation = str(
            config.get("count_representation", "continuation_hazard")
        )
        if self.count_representation != "continuation_hazard":
            raise ValueError(
                "This predictor requires "
                "count_representation=continuation_hazard"
            )
        self.cap_modeling = str(config.get("cap_modeling", "none"))
        if self.cap_modeling not in {"none", "auxiliary", "mixture"}:
            raise ValueError("cap_modeling must be none, auxiliary, or mixture")
        if (
            self.cap_modeling == "mixture"
            and "remainder" not in self.count_alignment_features
        ):
            raise ValueError("cap mixture requires a remainder alignment feature")
        self.continuation_hard_support_mask = bool(
            config.get("continuation_hard_support_mask", False)
        )
        self.prompt_length_scale = float(
            config.get("prompt_length_scale", 1.0)
        )
        if self.prompt_length_scale <= 0:
            raise ValueError("prompt_length_scale must be positive")
        count_input_dim = hidden
        if "remainder" in self.count_alignment_features:
            remainder_dim = int(config.get("remainder_embedding_dim", 16))
            if remainder_dim <= 0:
                raise ValueError("remainder_embedding_dim must be positive")
            self.remainder_embedding_dim = remainder_dim
            count_input_dim += remainder_dim
        if "length" in self.count_alignment_features:
            count_input_dim += 1
        self.max_physical_output_length = int(
            config.get("data_max_physical_output_length", 1024)
        )
        if self.max_physical_output_length <= 0:
            raise ValueError("data_max_physical_output_length must be positive")
        self.target_area_scale = float(config.get("target_area_scale", 1.0))
        self.semantic_pooler = SemanticPooler(hidden, self.semantic_pooling)
        if "remainder" in self.count_alignment_features:
            self.remainder_embedding = nn.Embedding(32, self.remainder_embedding_dim)

        # Both heads operate directly on the single pooled prompt state.
        # Each output coordinate has its own weights; no learned block queries
        # or token-memory cross-attention is needed to distinguish blocks.
        self.continuation_head = nn.Sequential(
            nn.LayerNorm(count_input_dim),
            nn.Linear(count_input_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, MAX_BLOCKS - 1),
        )
        self.step_continuation_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, MAX_BLOCKS * (MAX_DENOISING_STEPS - 1)),
        )
        if self.cap_modeling != "none":
            self.cap_head = nn.Sequential(
                nn.LayerNorm(count_input_dim),
                nn.Linear(count_input_dim, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, 1),
            )
            cap_prevalence = min(
                max(float(config.get("cap_prevalence", 0.15)), 1e-4), 1.0 - 1e-4
            )
            nn.init.zeros_(self.cap_head[-1].weight)
            nn.init.constant_(
                self.cap_head[-1].bias,
                math.log(cap_prevalence / (1.0 - cap_prevalence)),
            )

    @property
    def surface_parameters(self):
        for name, parameter in self.named_parameters():
            if not name.startswith("encoder."):
                yield parameter

    def pooled_state(self, ids: torch.Tensor, mask: torch.Tensor):
        encoder_output = self.encoder(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=self.layer_mixer.last_n > 1,
        )
        hidden = self.layer_mixer(encoder_output)
        return self.semantic_pooler(hidden, mask)

    def _count_input(
        self,
        global_state: torch.Tensor,
        prompt_remainder: torch.Tensor | None,
        target_prompt_length: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.count_alignment_features == "none":
            return global_state
        features = [global_state]
        if "remainder" in self.count_alignment_features:
            if prompt_remainder is None:
                raise ValueError(
                    "prompt_remainder is required by count_alignment_features"
                )
            prompt_remainder = prompt_remainder.to(
                device=global_state.device, dtype=torch.long
            )
            if prompt_remainder.shape != (global_state.shape[0],):
                raise ValueError("prompt_remainder must have shape [B]")
            if torch.any((prompt_remainder < 0) | (prompt_remainder >= 32)):
                raise ValueError("prompt_remainder values must lie in 0..31")
            features.append(
                self.remainder_embedding(prompt_remainder).to(global_state.dtype)
            )
        if "length" in self.count_alignment_features:
            if target_prompt_length is None:
                raise ValueError(
                    "target_prompt_length is required by count_alignment_features"
                )
            target_prompt_length = target_prompt_length.to(
                device=global_state.device, dtype=global_state.dtype
            )
            if target_prompt_length.shape != (global_state.shape[0],):
                raise ValueError("target_prompt_length must have shape [B]")
            features.append(
                (target_prompt_length / self.prompt_length_scale)[:, None]
            )
        return torch.cat(features, dim=-1)

    @staticmethod
    def _distribution_from_continuation_logits(
        continuation_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert K-1 continuation logits to a distribution on 1..K."""

        decisions = MAX_BLOCKS - 1
        if (
            continuation_logits.ndim != 2
            or continuation_logits.shape[1] != decisions
        ):
            raise ValueError(
                f"continuation_logits must have shape [B,{decisions}]"
            )
        logits = continuation_logits.float()
        log_continue = torch.nn.functional.logsigmoid(logits)
        log_stop = torch.nn.functional.logsigmoid(-logits)
        log_survival = torch.cat(
            [
                torch.zeros(
                    logits.shape[0], 1, device=logits.device, dtype=logits.dtype
                ),
                torch.cumsum(log_continue, dim=1),
            ],
            dim=1,
        )
        log_probability = torch.cat(
            [
                log_survival[:, :decisions] + log_stop,
                log_survival[:, decisions : decisions + 1],
            ],
            dim=1,
        )
        log_probability = log_probability - torch.logsumexp(
            log_probability, dim=1, keepdim=True
        )
        probability = log_probability.exp()
        survival = probability.flip(-1).cumsum(-1).flip(-1)
        return log_probability, probability, survival

    def _count_distribution(
        self,
        global_state: torch.Tensor,
        prompt_remainder: torch.Tensor | None,
        target_prompt_length: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        count_input = self._count_input(
            global_state, prompt_remainder, target_prompt_length
        )
        continuation_logits = self.continuation_head(count_input)
        if bool(getattr(self, "continuation_hard_support_mask", False)):
            if prompt_remainder is None:
                raise ValueError(
                    "prompt_remainder is required for continuation support mask"
                )
            maximum_count = torch.div(
                prompt_remainder.to(device=global_state.device, dtype=torch.long)
                + self.max_physical_output_length
                + 31,
                32,
                rounding_mode="floor",
            )
            continuation_count = torch.arange(
                2, MAX_BLOCKS + 1, device=global_state.device
            )
            continuation_logits = continuation_logits.masked_fill(
                continuation_count[None, :] > maximum_count[:, None],
                float("-inf"),
            )
        count_logits, probability, survival = (
            self._distribution_from_continuation_logits(continuation_logits)
        )
        return count_logits, probability, survival, continuation_logits

    @staticmethod
    def _step_distribution_from_continuation_logits(
        continuation_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert 31 step-continuation logits to T in 1..32 and survival."""

        if (
            continuation_logits.ndim != 3
            or continuation_logits.shape[1] != MAX_BLOCKS
            or continuation_logits.shape[2] != MAX_DENOISING_STEPS - 1
        ):
            raise ValueError(
                "step continuation logits must have shape "
                f"[B,{MAX_BLOCKS},{MAX_DENOISING_STEPS - 1}]"
            )
        logits = continuation_logits.float()
        log_continue = torch.nn.functional.logsigmoid(logits)
        log_stop = torch.nn.functional.logsigmoid(-logits)
        log_survival = torch.cat(
            [
                torch.zeros(
                    logits.shape[0],
                    logits.shape[1],
                    1,
                    device=logits.device,
                    dtype=logits.dtype,
                ),
                torch.cumsum(log_continue, dim=2),
            ],
            dim=2,
        )
        log_probability = torch.cat(
            [
                log_survival[:, :, : MAX_DENOISING_STEPS - 1] + log_stop,
                log_survival[:, :, MAX_DENOISING_STEPS - 1 :],
            ],
            dim=2,
        )
        # The construction is normalized analytically.  Renormalizing in log
        # space only removes floating-point drift and preserves the hazards.
        log_probability = log_probability - torch.logsumexp(
            log_probability, dim=2, keepdim=True
        )
        probability = log_probability.exp()
        survival = log_survival.exp()
        return log_probability, probability, survival, torch.sigmoid(logits)

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        prompt_remainder: torch.Tensor | None = None,
        target_prompt_length: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if ids.ndim != 2 or mask.ndim != 2 or ids.shape != mask.shape:
            raise ValueError("ids and mask must both have shape [B, L]")
        mask = mask.to(dtype=torch.bool)
        global_state = self.pooled_state(ids, mask)
        (
            count_logits,
            count_probability,
            block_survival,
            continuation_logits,
        ) = self._count_distribution(
            global_state,
            prompt_remainder,
            target_prompt_length,
        )
        cap_logit = None
        cap_probability = None
        body_count_probability = None
        body_block_survival = None
        if self.cap_modeling != "none":
            count_input = self._count_input(
                global_state, prompt_remainder, target_prompt_length
            )
            cap_logit = self.cap_head(count_input).squeeze(-1)
            cap_probability = torch.sigmoid(cap_logit.float())
            if self.cap_modeling == "mixture":
                if prompt_remainder is None:
                    raise ValueError("prompt_remainder is required for cap mixture")
                body_count_probability = count_probability
                body_block_survival = block_survival
                remainder = prompt_remainder.to(
                    device=global_state.device, dtype=torch.long
                )
                cap_count = torch.div(
                    self.max_physical_output_length + remainder + 31,
                    32,
                    rounding_mode="floor",
                ).clamp(1, MAX_BLOCKS)
                cap_atom = torch.nn.functional.one_hot(
                    cap_count - 1, num_classes=MAX_BLOCKS
                ).to(count_probability.dtype)
                count_probability = (
                    (1.0 - cap_probability[:, None]) * body_count_probability
                    + cap_probability[:, None] * cap_atom
                )
                count_logits = count_probability.clamp_min(1e-30).log()
                block_survival = count_probability.flip(-1).cumsum(-1).flip(-1)
        step_continuation_logits = self.step_continuation_head(global_state).reshape(
            ids.shape[0], MAX_BLOCKS, MAX_DENOISING_STEPS - 1
        )
        (
            step_logits,
            step_probability,
            step_survival,
            step_continuation_probability,
        ) = self._step_distribution_from_continuation_logits(
            step_continuation_logits
        )
        surface = block_survival[:, :, None] * step_survival
        surface_curve = surface.cumsum(dim=2).sum(dim=1)
        surface_area = surface_curve[:, -1]
        output = {
            "pooled": global_state,
            "count_logits": count_logits,
            "count_probability": count_probability,
            "step_logits": step_logits,
            "step_probability": step_probability,
            "step_continuation_logits": step_continuation_logits,
            "step_continuation_probability": step_continuation_probability,
            "block_survival": block_survival,
            "step_survival": step_survival,
            "surface": surface,
            "raw_surface_curve": surface_curve,
            "surface_area": surface_area,
            "workload_curve": surface_curve,
            "area": surface_area,
            "expected_blocks": block_survival.sum(dim=1),
            "expected_steps": step_survival.sum(dim=2),
        }
        if continuation_logits is not None:
            output["continuation_logits"] = continuation_logits
        if cap_logit is not None and cap_probability is not None:
            output["cap_logit"] = cap_logit
            output["cap_probability"] = cap_probability
        if body_count_probability is not None and body_block_survival is not None:
            output["body_count_probability"] = body_count_probability
            output["body_block_survival"] = body_block_survival
        return output

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_arrays(token_cache: Path, workload: Path) -> dict[str, np.ndarray]:
    meta = np.load(token_cache / "meta.npz")
    arrays = {
        "ids": np.load(token_cache / "input_ids.npy", mmap_mode="r"),
        "prompt_idx": meta["prompt_idx"].astype(np.int64),
        "lengths": meta["lengths"].astype(np.int64),
        "split": meta["split"].astype(np.int8),
        "total": meta["S_total"].astype(np.float32),
        "semantic_output_length": meta["L_star"].astype(np.float32),
        # Retain legacy length metadata; Stage A is supervised only by steps.
        "output_tokens": meta["L_star"].astype(np.float32),
        "steps": np.load(workload / "step_tokens.npy", mmap_mode="r"),
    }
    n = len(arrays["prompt_idx"])
    if arrays["ids"].shape[0] != n or arrays["steps"].shape != (
        n,
        SOURCE_PADDED_BLOCKS,
    ):
        raise ValueError("token cache and workload arrays are not row-aligned")
    for start in range(0, n, 8192):
        end = min(start + 8192, n)
        area = np.asarray(arrays["steps"][start:end], dtype=np.int64).sum(axis=1)
        truth = arrays["total"][start:end].astype(np.int64)
        if not np.array_equal(area, truth):
            raise ValueError(f"DWS area identity failed at rows {start}:{end}")
    return arrays


def load_alignment_metadata(
    arrays: dict[str, np.ndarray], path: Path
) -> None:
    """Load target-tokenizer alignment fields and enforce row identity."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as alignment:
        required = {
            "prompt_idx",
            "split",
            "target_prompt_length",
            "prompt_remainder",
            "physical_output_length",
            "semantic_output_length",
            "cap_flag",
            "censored_flag",
            "saw_eos",
            "active_block_count",
        }
        missing = sorted(required.difference(alignment.files))
        if missing:
            raise ValueError(f"alignment metadata is missing fields: {missing}")
        if not np.array_equal(alignment["prompt_idx"], arrays["prompt_idx"]):
            raise ValueError("alignment metadata is not row-aligned with token cache")
        if not np.array_equal(alignment["split"], arrays["split"]):
            raise ValueError("alignment split does not match token cache")
        for name in sorted(required.difference({"prompt_idx", "split"})):
            value = alignment[name]
            if value.shape != arrays["prompt_idx"].shape:
                raise ValueError(
                    f"alignment field {name!r} has shape {value.shape}, expected "
                    f"{arrays['prompt_idx'].shape}"
                )
            arrays[name] = value.copy()

    step_blocks = np.zeros(len(arrays["prompt_idx"]), dtype=np.int64)
    for start in range(0, len(step_blocks), 8192):
        end = min(start + 8192, len(step_blocks))
        step_blocks[start:end] = np.count_nonzero(
            np.asarray(arrays["steps"][start:end]), axis=1
        )
    if not np.array_equal(arrays["active_block_count"], step_blocks):
        raise ValueError("alignment active_block_count does not match workload labels")


def load_split_metadata(arrays: dict[str, np.ndarray], path: Path) -> None:
    """Replace the cache split with a row-aligned externally defined split."""

    with np.load(Path(path), allow_pickle=False) as split_meta:
        if not {"prompt_idx", "split"}.issubset(split_meta.files):
            raise ValueError("split metadata must contain prompt_idx and split")
        if not np.array_equal(split_meta["prompt_idx"], arrays["prompt_idx"]):
            raise ValueError("split metadata is not row-aligned")
        split = np.asarray(split_meta["split"], dtype=np.int8)
        if not np.isin(split, [0, 1, 2]).all():
            raise ValueError("split codes must lie in {0,1,2}")
        arrays["split"] = split.copy()


def split_indices(
    arrays: dict[str, np.ndarray], split: int, limit: int
) -> np.ndarray:
    indices = np.flatnonzero(arrays["split"] == split)
    if limit > 0:
        indices = indices[:limit]
    return indices


def select_train_rows(
    prompt_idx: np.ndarray,
    split: np.ndarray,
    prompt_ids_path: Path | None,
) -> np.ndarray:
    """Return cache rows for an explicit list of training prompt IDs."""

    prompt_idx = np.asarray(prompt_idx)
    split = np.asarray(split)
    if prompt_idx.ndim != 1 or split.ndim != 1 or prompt_idx.shape != split.shape:
        raise ValueError("prompt_idx and split must be aligned one-dimensional arrays")
    if len(np.unique(prompt_idx)) != len(prompt_idx):
        raise ValueError("cache prompt_idx values must be unique")
    if prompt_ids_path is None:
        rows = np.flatnonzero(split == 0)
        if len(rows) == 0:
            raise ValueError("training split is empty")
        return rows.astype(np.int64, copy=False)

    prompt_ids_path = Path(prompt_ids_path)
    requested = np.load(prompt_ids_path, allow_pickle=False)
    if requested.ndim != 1 or not np.issubdtype(requested.dtype, np.integer):
        raise ValueError(
            "training prompt IDs must be a one-dimensional integer array: "
            f"{prompt_ids_path}"
        )
    requested = requested.astype(np.int64, copy=False)
    if len(requested) == 0:
        raise ValueError(f"training prompt ID file is empty: {prompt_ids_path}")
    if len(np.unique(requested)) != len(requested):
        raise ValueError(f"training prompt IDs contain duplicates: {prompt_ids_path}")

    order = np.argsort(prompt_idx, kind="stable")
    sorted_prompt_idx = prompt_idx[order]
    positions = np.searchsorted(sorted_prompt_idx, requested)
    in_bounds = positions < len(sorted_prompt_idx)
    matched = np.zeros(len(requested), dtype=bool)
    matched[in_bounds] = sorted_prompt_idx[positions[in_bounds]] == requested[in_bounds]
    if not matched.all():
        missing = requested[~matched]
        raise ValueError(
            f"{len(missing)} requested prompt IDs are absent from this cache; "
            f"first missing IDs: {missing[:10].tolist()}"
        )
    rows = order[positions].astype(np.int64, copy=False)
    non_train = requested[split[rows] != 0]
    if len(non_train):
        raise ValueError(
            "requested prompt IDs must all belong to split=0; "
            f"first invalid IDs: {non_train[:10].tolist()}"
        )
    return rows


def make_batches(
    indices: np.ndarray,
    lengths: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    mode: str,
) -> list[np.ndarray]:
    if mode == "length":
        order = indices[np.argsort(lengths[indices], kind="stable")]
    elif mode == "random":
        order = indices.copy()
        rng.shuffle(order)
    else:
        raise ValueError(f"unknown batching mode: {mode}")
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def eval_order(indices: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    return indices[np.argsort(lengths[indices], kind="stable")]


def tensor_batch(
    arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, ...]:
    lengths_np = arrays["lengths"][indices]
    width = int(lengths_np.max())
    ids = torch.from_numpy(
        np.asarray(arrays["ids"][indices, :width], dtype=np.int64)
    ).to(device, non_blocking=True)
    lengths = torch.from_numpy(lengths_np).to(device, non_blocking=True)
    mask = torch.arange(width, device=device)[None, :] < lengths[:, None]
    steps = torch.from_numpy(
        np.asarray(arrays["steps"][indices], dtype=np.int64)
    ).to(device, non_blocking=True)
    total = torch.from_numpy(arrays["total"][indices]).to(
        device, non_blocking=True
    )
    output_tokens = torch.from_numpy(arrays["output_tokens"][indices]).to(
        device, non_blocking=True
    )
    return ids, mask, steps, total, output_tokens


def alignment_tensor_batch(
    arrays: dict[str, np.ndarray], indices: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    """Return request-time target-tokenizer metadata for compact models."""

    required = ("prompt_remainder", "target_prompt_length")
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"alignment arrays are unavailable: {missing}")
    return {
        "prompt_remainder": torch.from_numpy(
            np.asarray(arrays["prompt_remainder"][indices], dtype=np.int64)
        ).to(device, non_blocking=True),
        "target_prompt_length": torch.from_numpy(
            np.asarray(arrays["target_prompt_length"][indices], dtype=np.float32)
        ).to(device, non_blocking=True),
    }


def horizon_costs(steps: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
    return torch.minimum(steps[:, :, None], horizons[None, None, :]).sum(dim=1)


def compute_pretrain_scales(
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    horizons: tuple[int, ...],
) -> np.ndarray:
    """Training-set mean workload per horizon; no auxiliary target scales."""
    sums = np.zeros(len(horizons), dtype=np.float64)
    for start in range(0, len(train_idx), 8192):
        idx = train_idx[start : start + 8192]
        steps = np.asarray(arrays["steps"][idx], dtype=np.int64)
        for column, horizon in enumerate(horizons):
            sums[column] += np.minimum(steps, horizon).sum()
    count = max(len(train_idx), 1)
    scales = (sums / count).astype(np.float32)
    return np.maximum(scales, 1.0)


def selection_key(metrics: dict, config: dict) -> tuple[float, ...]:
    """Use the final recipe's workload MAE and count-distribution tie breakers."""
    if config["checkpoint_selection"] != "prediction_surface":
        raise ValueError("the final recipe requires prediction_surface selection")
    return (
        -float(metrics["workload"]["surface"]["mae"]),
        -float(metrics["count"]["crps_sum_thresholds"]),
        -float(metrics["count"]["mean"]["mae"]),
    )


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    schedule: str,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    if schedule == "constant":
        return None
    warmup_steps = int(round(total_steps * warmup_ratio))

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "cosine":
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif schedule == "linear":
            decay = 1.0 - progress
        else:
            raise ValueError(f"unknown LR schedule: {schedule}")
        return min_lr_ratio + (1.0 - min_lr_ratio) * decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def error_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Float64 validation errors; MAE selects the final checkpoint."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("target and prediction must be aligned one-dimensional arrays")
    error = prediction - target
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "bias": float(error.mean())}


def count_selection_metrics(target: np.ndarray, survival: np.ndarray) -> dict:
    """Retain the exact CRPS and mean-count MAE used for checkpoint ties."""
    target = np.asarray(target, dtype=np.int64)
    survival = np.asarray(survival, dtype=np.float64)
    if survival.ndim != 2 or target.shape != (len(survival),):
        raise ValueError("count target and survival must have shapes [N] and [N,K]")
    classes = survival.shape[1]
    if np.any((target < 1) | (target > classes)):
        raise ValueError(f"count target lies outside 1..{classes}")
    levels = np.arange(1, classes + 1, dtype=np.float64)
    target_survival = levels[None, :] <= target[:, None]
    crps_cells = np.square(survival[:, 1:] - target_survival[:, 1:])
    return {"crps_sum_thresholds": float(crps_cells.sum(axis=1).mean()),
            "mean": error_metrics(target, survival.sum(axis=1))}


def pretrain_targets(
    steps: torch.Tensor,
    horizons: torch.Tensor,
) -> torch.Tensor:
    """Paper Eq. (18): T(h) = sum_b min(s_b, h), including T(1) = n."""
    return horizon_costs(steps, horizons).to(torch.float32)


@torch.no_grad()
def evaluate_pretrainer(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    horizons: tuple[int, ...],
    scales: np.ndarray,
) -> dict[str, float]:
    """Validate Stage A's workload readout for best-checkpoint selection."""
    model.eval()
    ordered = eval_order(indices, arrays["lengths"])
    predictions = []
    scale_tensor = torch.tensor(scales, device=device)
    for start in range(0, len(ordered), batch_size):
        idx = ordered[start : start + batch_size]
        ids, mask, _, _, _ = tensor_batch(arrays, idx, device)
        with _autocast(device):
            output = model(ids, mask)
        predictions.append((output.float() * scale_tensor).cpu().numpy())
    prediction = np.concatenate(predictions)
    return error_metrics(arrays["total"][ordered],
                         np.maximum(prediction[:, len(horizons) - 1], 0.0))

def share_encoder_layers(encoder: nn.Module, groups: int) -> None:
    """Replace encoder blocks by contiguous, averaged, weight-shared groups."""

    if groups <= 0:
        return
    original = list(encoder.encoder.layer)
    if not 1 <= groups <= len(original):
        raise ValueError("shared-layer-groups must not exceed encoder layers")
    partitions = np.array_split(np.arange(len(original)), groups)
    shared_layers: list[nn.Module] = []
    for partition in partitions:
        indices = [int(index) for index in partition]
        layer = copy.deepcopy(original[indices[0]])
        averaged = layer.state_dict()
        source_states = [original[index].state_dict() for index in indices]
        for name, value in averaged.items():
            if value.is_floating_point():
                value.copy_(
                    torch.stack(
                        [state[name].detach().float() for state in source_states]
                    ).mean(dim=0).to(value.dtype)
                )
            else:
                value.copy_(source_states[0][name])
        layer.load_state_dict(averaged, strict=True)
        shared_layers.extend([layer] * len(indices))
    encoder.encoder.layer = nn.ModuleList(shared_layers)


class ExecutionPretrainer(nn.Module):
    """Stage-A encoder with a temporary workload-regression head."""

    def __init__(self, model_path: Path, config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = build_encoder(
            model_path,
            int(config["encoder_layers"]),
            int(config["factorized_embedding_rank"]),
            initialize_from_base=True,
        )
        share_encoder_layers(self.encoder, int(config.get("shared_layer_groups", 0)))
        hidden = int(self.encoder.config.hidden_size)
        self.semantic_pooler = SemanticPooler(hidden, str(config["semantic_pooling"]))
        self.layer_mixer = LayerMixer(int(config["layer_mix_last_n"]))
        head_dim = int(config["pretrain_head_dim"])
        outputs = len(config["pretrain_horizons"])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, head_dim),
            nn.GELU(),
            nn.Dropout(float(config["dropout"])),
            nn.Linear(head_dim, outputs),
        )

    def pooled_state(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=self.layer_mixer.last_n > 1,
        )
        hidden = self.layer_mixer(encoded)
        return self.semantic_pooler(hidden, mask)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled_state(ids, mask))

SCHEME = {
    "key": "OUR-DWS", "strategy": "uniform_all", "frozen_epochs": 1,
    "joint_epochs": 20, "encoder_lr": 4e-5, "head_lr": 3e-4,
    "stage_a_batch_size": 128, "stage_b_batch_size": 256,
}


def hold50_linear_multiplier(step: int, total_steps: int) -> float:
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    return 1.0 if progress <= 0.5 else max(0.0, 2.0 * (1.0 - progress))


def stage_a_lr_scheduler(optimizer, total_steps):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, [lambda step: hold50_linear_multiplier(step, total_steps)
                    for _ in optimizer.param_groups])


def _make_adamw(groups, weight_decay: float, device: torch.device):
    try:
        return torch.optim.AdamW(
            groups,
            weight_decay=weight_decay,
            fused=(device.type == "cuda"),
        )
    except TypeError:
        return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_encoder_stage(
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    output_dir: Path,
    device: torch.device,
):
    """Final Stage A: one frozen epoch, twenty joint epochs, workload MAE selection."""
    seed_everything(int(config["seed"]))
    model = ExecutionPretrainer(Path(config["model"]), config).to(device)
    seed_everything(int(config["seed"]))

    horizons = tuple(int(value) for value in config["pretrain_horizons"])
    scales = np.asarray(config["pretrain_target_scales"], dtype=np.float32)
    horizon_tensor = torch.tensor(horizons, device=device)
    scale_tensor = torch.tensor(scales, device=device)
    rng = np.random.default_rng(int(config["seed"]))

    config["stage_a_experiment"] = dict(SCHEME)

    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    started = time.time()


    stage_specs = [
        ("head_only", 1),
        ("joint", int(SCHEME["joint_epochs"])),
    ]

    joint_epoch_number = 0

    for stage_name, epochs in stage_specs:
        if epochs <= 0:
            continue

        is_joint = stage_name == "joint"

        if not is_joint:
            for parameter in model.encoder.parameters():
                parameter.requires_grad_(False)
            encoder_groups = []
        else:
            encoder_parameters = list(model.encoder.parameters())
            for parameter in encoder_parameters:
                parameter.requires_grad_(True)
            encoder_groups = [{"params": encoder_parameters, "lr": float(SCHEME["encoder_lr"])}]

        head_parameters = list(model.head.parameters())
        head_parameters.extend(model.semantic_pooler.parameters())
        head_parameters.extend(model.layer_mixer.parameters())

        groups = [
            {
                "params": head_parameters,
                "lr": float(config["pretrain_head_lr"]),
            }
        ]
        groups.extend(encoder_groups)

        optimizer = _make_adamw(
            groups,
            weight_decay=float(config["weight_decay"]),
            device=device,
        )

        batches_per_epoch = math.ceil(
            len(train_idx) / int(config["batch_size"])
        )
        scheduler = stage_a_lr_scheduler(optimizer, epochs * batches_per_epoch)

        for _ in range(epochs):
            epoch = len(history) + 1
            if is_joint:
                joint_epoch_number += 1

            model.train()

            if not is_joint:
                model.encoder.eval()

            totals = {"loss": 0.0, "regression": 0.0}
            batches = make_batches(
                train_idx,
                arrays["lengths"],
                int(config["batch_size"]),
                rng,
                str(config["batching_mode"]),
            )

            for indices in batches:
                ids, mask, steps, _total, _output_tokens = tensor_batch(
                    arrays, indices, device
                )
                optimizer.zero_grad(set_to_none=True)

                with _autocast(device):
                    output = model(ids, mask)
                    target = pretrain_targets(steps, horizon_tensor)
                    regression = F.smooth_l1_loss(
                        output,
                        target / scale_tensor[None, :],
                        beta=float(config.get("regression_beta", 1.0)),
                    )
                    loss = regression

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite Stage-A loss in "
                        f"{stage_name} epoch={epoch}"
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(config["grad_clip"]),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                totals["loss"] += float(loss.detach())
                totals["regression"] += float(regression.detach())

            validation = evaluate_pretrainer(
                model,
                arrays,
                val_idx,
                int(config["eval_batch_size"]),
                device,
                horizons,
                scales,
            )

            val_mae = float(validation["mae"])
            row = {
                "epoch": epoch,
                "stage": stage_name,
                "joint_epoch": joint_epoch_number if is_joint else 0,
                "train": {
                    key: value / max(len(batches), 1)
                    for key, value in totals.items()
                },
                "validation": validation,
                "optimizer_group_lrs_end": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
                "stage_a_experiment": SCHEME["key"],
                "elapsed_s": time.time() - started,
            }
            history.append(row)
            print(
                json.dumps(
                    {"encoder_pretrain": row},
                    ensure_ascii=False,
                ),
                flush=True,
            )

            # Exact baseline checkpoint selection:
            # smallest validation workload MAE wins.
            current_key = (-val_mae,)
            checkpoint = {
                "encoder_state_dict": model.encoder.state_dict(),
                "pooler_state_dict": model.semantic_pooler.state_dict(),
                "layer_mixer_state_dict": model.layer_mixer.state_dict(),
                "pretrainer_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "stage": stage_name,
                "validation": validation,
                "selection_key": current_key,
                "stage_a_experiment": dict(SCHEME),
            }

            if best_key is None or current_key > best_key:
                best_key = current_key
                best_epoch = epoch
                best_checkpoint = dict(checkpoint)
                best_checkpoint.pop("stage")
                torch.save(
                    best_checkpoint,
                    output_dir / "encoder_pretrain_best.pt",
                )

        del optimizer

    saved = torch.load(
        output_dir / "encoder_pretrain_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    report = {
        "experiment": dict(SCHEME),
        "best_epoch": best_epoch,
        "best_validation": saved["validation"],
        "best_selection_key": saved.get("selection_key"),
        "history": history,
        "joint_epochs_completed": int(joint_epoch_number),
        "elapsed_s": time.time() - started,
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (
        saved["encoder_state_dict"],
        saved["pooler_state_dict"],
        saved["layer_mixer_state_dict"],
        report,
    )


def validate_reduced_block_range(arrays: dict[str, np.ndarray]) -> None:
    """Ensure padded source columns never contain an active block."""

    steps = arrays["steps"]
    for start in range(0, len(steps), 8192):
        tail = np.asarray(steps[start : start + 8192, MAX_BLOCKS:])
        if np.any(tail > 0):
            raise ValueError(
                f"source workload has active blocks beyond reduced limit {MAX_BLOCKS}"
            )


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    from contextlib import nullcontext

    return nullcontext()


def masked_hazard_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    risk: torch.Tensor,
    normalization: str,
) -> torch.Tensor:
    """Stable BCE on observed risk-set states with explicit reduction semantics.

    ``trajectory_nll`` is the negative log-likelihood of the observed
    continuation trajectory: risk-set decisions are summed within each
    request and those request NLLs are averaged over the batch.  The legacy
    ``per_request`` mode averages decisions within each request instead and is
    retained as an alias of the more precise ``per_request_normalized`` name.
    """

    if logits.shape != target.shape or logits.shape != risk.shape:
        raise ValueError("hazard logits, target, and risk must share a shape")
    raw = F.binary_cross_entropy_with_logits(
        logits.float(), target.to(torch.float32), reduction="none"
    )
    risk_float = risk.to(raw.dtype)
    reduce_dims = tuple(range(1, raw.ndim))
    numerator = (raw * risk_float).sum(dim=reduce_dims)
    denominator = risk_float.sum(dim=reduce_dims)
    if torch.any(denominator <= 0):
        raise ValueError("every request must contribute at least one risk event")
    if normalization in {"per_request", "per_request_normalized"}:
        return (numerator / denominator).mean()
    if normalization == "trajectory_nll":
        return numerator.mean()
    if normalization == "global_risk":
        return numerator.sum() / denominator.sum()
    raise ValueError(f"unknown hazard normalization: {normalization}")


def hazard_loss_components(
    output: dict[str, torch.Tensor],
    steps: torch.Tensor,
    normalization: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return block/step continuation losses and their exact risk sets."""

    steps = steps[:, :MAX_BLOCKS]
    n_blocks = (steps > 0).sum(dim=1)

    # At block decision b in 1..K-1, the request is at risk iff N>=b and
    # continues iff N>b. N=K is right-censored at the configured boundary.
    block_decision = torch.arange(1, MAX_BLOCKS, device=steps.device)
    block_risk = block_decision[None, :] <= n_blocks[:, None]
    block_target = block_decision[None, :] < n_blocks[:, None]

    # At step decision s in 1..31, block b is at risk iff it exists and T_b>=s;
    # it continues iff T_b>s.  T_b=32 is right-censored at the boundary.
    alive = steps > 0
    step_decision = torch.arange(
        1, MAX_DENOISING_STEPS, device=steps.device
    )
    step_risk = alive[:, :, None] & (
        step_decision[None, None, :] <= steps[:, :, None]
    )
    step_target = alive[:, :, None] & (
        step_decision[None, None, :] < steps[:, :, None]
    )

    block_logits = output.get("continuation_logits")
    step_logits = output.get("step_continuation_logits")
    if block_logits is None or step_logits is None:
        raise KeyError("dual-hazard outputs are required by the Stage-B loss")
    block_loss = masked_hazard_bce(
        block_logits, block_target, block_risk, normalization
    )
    step_loss = masked_hazard_bce(
        step_logits, step_target, step_risk, normalization
    )
    return block_loss, step_loss, {
        "block_risk": block_risk,
        "block_target": block_target,
        "step_risk": step_risk,
        "step_target": step_target,
    }


def compact_loss(
    output: dict[str, torch.Tensor],
    steps: torch.Tensor,
    target_total: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    steps = steps[:, :MAX_BLOCKS]
    n_blocks = (steps > 0).sum(dim=1)
    block_position = torch.arange(MAX_BLOCKS, device=steps.device)
    step_position = torch.arange(1, MAX_DENOISING_STEPS + 1, device=steps.device)
    alive = block_position[None, :] < n_blocks[:, None]
    step_target = step_position[None, None, :] <= steps[:, :, None]
    joint_target = alive[:, :, None] & step_target
    block_hazard_bce, step_hazard_bce, risk = hazard_loss_components(
        output, steps, str(config["hazard_normalization"])
    )
    surface_brier = (
        output["surface"] - joint_target.to(torch.float32)
    ).square().mean()
    area_scale = float(config["target_area_scale"])
    area_loss = F.smooth_l1_loss(
        output["area"] / area_scale,
        target_total / area_scale,
        beta=float(config.get("regression_beta", 1.0)),
    )
    objective = str(config["stage_b_objective"])
    if objective == "dual_hazard":
        # Only the two continuation processes are optimized.  Surface Brier
        # and area Smooth-L1 below are diagnostics, not hidden objectives.
        loss = block_hazard_bce + step_hazard_bce
    elif objective == "surface_brier_control":
        # Loss-only control: identical dual-hazard heads, but the L4 joint
        # Surface Brier is the sole optimized objective.
        loss = surface_brier
    else:
        raise ValueError(f"unknown Stage-B objective: {objective}")
    values = {
        "loss": float(loss.detach()),
        "block_hazard_bce": float(block_hazard_bce.detach()),
        "step_hazard_bce": float(step_hazard_bce.detach()),
        "surface_brier_diagnostic": float(surface_brier.detach()),
        "area_smooth_l1_diagnostic": float(area_loss.detach()),
        "block_risk_events": float(risk["block_risk"].sum(dim=1).float().mean()),
        "step_risk_events": float(
            risk["step_risk"].sum(dim=(1, 2)).float().mean()
        ),
    }
    return loss, values


@torch.no_grad()
def evaluate_compact(
    model: CompactWorkloadCurvePredictor,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Validate the surface and count distribution needed to select weights."""
    model.eval()
    ordered = eval_order(indices, arrays["lengths"])
    surface_area = np.empty(len(ordered), dtype=np.float32)
    count_survival = np.empty((len(ordered), MAX_BLOCKS), dtype=np.float32)
    for start in range(0, len(ordered), batch_size):
        end = min(start + batch_size, len(ordered))
        idx = ordered[start:end]
        ids, mask, _, _, _ = tensor_batch(arrays, idx, device)
        alignment_inputs = alignment_tensor_batch(arrays, idx, device)
        with _autocast(device):
            output = model(ids, mask, **alignment_inputs)
        surface_area[start:end] = output["surface_area"].float().cpu().numpy()
        count_survival[start:end] = output["block_survival"].float().cpu().numpy()
    surface_metrics = error_metrics(arrays["total"][ordered], np.maximum(surface_area, 0.0))
    steps = np.asarray(arrays["steps"][ordered], dtype=np.int64)
    target_count = np.count_nonzero(steps[:, :MAX_BLOCKS], axis=1)
    if "active_block_count" in arrays:
        aligned_count = np.asarray(arrays["active_block_count"][ordered], dtype=np.int64)
        if not np.array_equal(target_count, aligned_count):
            raise ValueError("alignment active_block_count disagrees with step labels")
    return {**surface_metrics, "workload": {"surface": surface_metrics},
            "count": count_selection_metrics(target_count, count_survival)}


def configure_encoder_training(
    model: CompactWorkloadCurvePredictor,
    train_encoder: bool,
    last_n_layers: int,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Module]]:
    """Select all encoder parameters or only the final transformer layers."""

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    if not train_encoder:
        return [], []
    if last_n_layers <= 0:
        parameters = list(model.encoder.parameters())
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters, [model.encoder]

    layers = list(model.encoder.encoder.layer)
    if last_n_layers > len(layers):
        raise ValueError(
            f"cannot unfreeze last {last_n_layers} layers of {len(layers)}-layer encoder"
        )
    modules = layers[-last_n_layers:]
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            if id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters, modules


def train_compact_stage(
    config: dict,
    encoder_state: dict[str, torch.Tensor],
    pooler_state: dict[str, torch.Tensor],
    layer_mixer_state: dict[str, torch.Tensor],
    arrays: dict[str, np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    output_dir: Path,
    device: torch.device,
) -> dict:
    stage_seed = int(config["seed"]) + 1
    seed_everything(stage_seed)
    model = CompactWorkloadCurvePredictor(
        Path(config["model"]), config, initialize_encoder_from_base=False
    )
    model.encoder.load_state_dict(encoder_state, strict=True)
    model.semantic_pooler.load_state_dict(pooler_state, strict=True)
    model.layer_mixer.load_state_dict(layer_mixer_state, strict=True)

    model.to(device)

    rng = np.random.default_rng(stage_seed)
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    history: list[dict] = []
    global_epoch = 0
    started = time.time()
    protocol = str(config.get("compact_training_protocol", "continuous_joint"))
    if protocol != "continuous_joint":
        raise ValueError("the final recipe requires continuous_joint training")
    stages = [
        (
            "curve_head_only",
            int(config["compact_frozen_epochs"]),
            False,
        ),
        (
            "curve_joint",
            int(config["compact_full_epochs"]),
            True,
        ),
    ]

    shared_encoder_parameters, shared_encoder_train_modules = configure_encoder_training(
        model, True, int(config.get("compact_unfreeze_last_n_layers", 0)),
    )
    for parameter in shared_encoder_parameters:
        parameter.requires_grad_(False)
    groups = [{"params": list(model.surface_parameters),
               "lr": float(config["compact_surface_lr"])}]
    if shared_encoder_parameters:
        groups.append({"params": shared_encoder_parameters,
                       "lr": float(config["compact_encoder_lr"])})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config["weight_decay"]),
                                 fused=device.type == "cuda")
    batches_per_epoch = math.ceil(len(train_idx) / int(config["batch_size"]))
    scheduler = build_lr_scheduler(
        optimizer,
        (int(config["compact_frozen_epochs"]) + int(config["compact_full_epochs"])) * batches_per_epoch,
        str(config["lr_schedule"]), float(config["warmup_ratio"]), float(config["min_lr_ratio"]),
    )

    for stage, epochs, train_encoder in stages:
        if epochs <= 0:
            continue
        encoder_parameters = shared_encoder_parameters if train_encoder else []
        encoder_train_modules = shared_encoder_train_modules if train_encoder else []
        for parameter in shared_encoder_parameters:
            parameter.requires_grad_(train_encoder)
        for _ in range(epochs):
            global_epoch += 1
            model.train()
            if not encoder_parameters:
                model.encoder.eval()
            elif int(config.get("compact_unfreeze_last_n_layers", 0)) > 0:
                model.encoder.eval()
                for module in encoder_train_modules:
                    module.train()
            totals: dict[str, float] = {}
            batches = make_batches(
                train_idx,
                arrays["lengths"],
                int(config["batch_size"]),
                rng,
                str(config["batching_mode"]),
            )
            for idx in batches:
                ids, mask, steps, total, _ = tensor_batch(
                    arrays, idx, device
                )
                alignment_inputs = alignment_tensor_batch(
                    arrays, idx, device
                )
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device):
                    output = model(ids, mask, **alignment_inputs)
                    loss, values = compact_loss(
                        output,
                        steps,
                        total,
                        config,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(config["grad_clip"]),
                )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                for key, value in values.items():
                    totals[key] = totals.get(key, 0.0) + value

            validation = evaluate_compact(
                model,
                arrays,
                val_idx,
                int(config["eval_batch_size"]),
                device,
            )
            row = {
                "epoch": global_epoch,
                "stage": stage,
                "training_protocol": protocol,
                "optimizer_continuous_across_boundary": True,
                "optimizer_group_lrs": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
                "train": {key: value / max(len(batches), 1) for key, value in totals.items()},
                "validation": validation,
                "elapsed_s": time.time() - started,
            }
            history.append(row)
            print(json.dumps({"compact": row}, ensure_ascii=False), flush=True)
            current_key = selection_key(validation, config)
            if best_key is None or current_key > best_key:
                best_key = current_key
                best_epoch = global_epoch
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "config": config,
                        "epoch": global_epoch,
                        "validation": validation,
                        "selection_key": current_key,
                    },
                    output_dir / "best.pt",
                )
    del optimizer

    if not (output_dir / "best.pt").is_file():
        raise RuntimeError("compact Stage B produced no checkpoint; set compact epochs > 0")
    saved = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    validation = evaluate_compact(
        model,
        arrays,
        val_idx,
        int(config["eval_batch_size"]),
        device,
    )
    test = None
    if not bool(config["skip_test"]):
        test = evaluate_compact(
            model,
            arrays,
            test_idx,
            int(config["eval_batch_size"]),
            device,
        )
    return {
        "best_epoch": best_epoch,
        "best_validation": validation,
        "checkpoint_selection_validation": saved["validation"],
        "best_selection_key": saved.get("selection_key"),
        "history": history,
        "test": test,
        "elapsed_s": time.time() - started,
    }

class OurDWSPredictor:
    """Load one final package and predict its 33x32 workload surface."""

    def __init__(
        self,
        model_dir: str | Path = HERE,
        device: str = "cpu",
        cpu_threads: int = 16,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu"
        )
        if self.device.type == "cpu":
            torch.set_num_threads(int(cpu_threads))

        checkpoint = torch.load(
            self.model_dir / "best.pt", map_location="cpu", weights_only=False
        )
        self.config = dict(checkpoint["config"])
        CompactWorkloadCurvePredictor.validate_config(self.config)
        contract = self.config.get("training_contract", {})
        if contract.get("dllm_prefill_required") is not False:
            raise ValueError("checkpoint does not satisfy the pre-prefill contract")
        if self.config.get("count_alignment_features") != "remainder":
            raise ValueError("final Our-DWS requires remainder alignment")

        packaged_resources = self.model_dir / "minilm"
        model_resources = (
            packaged_resources
            if (packaged_resources / "config.json").is_file()
            else Path(self.config["model"])
        )
        tokenizer_metadata_path = model_resources / "tokenizer_config.json"
        tokenizer_metadata = (
            json.loads(tokenizer_metadata_path.read_text(encoding="utf-8"))
            if tokenizer_metadata_path.is_file()
            else {}
        )
        # The DeBERTa cache manifest was built by the SentencePiece backend.
        # Keep that backend at serving time as well; its byte-fallback behavior
        # is not reproduced exactly by the converted fast tokenizer.
        prefer_fast = tokenizer_metadata.get("vocab_type") != "spm"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_resources),
                local_files_only=True,
                use_fast=prefer_fast,
            )
        except (ModuleNotFoundError, ValueError) as fast_error:
            # Local model packages do not necessarily include a serialized
            # fast tokenizer. Falling back to the canonical slow tokenizer
            # preserves token IDs and avoids a hard dependency on tiktoken.
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_resources), local_files_only=True, use_fast=False
                )
            except Exception:
                raise fast_error
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"
        self.max_length = int(self.config.get("max_length", 512))

        self.model = CompactWorkloadCurvePredictor(
            model_resources, self.config, initialize_encoder_from_base=False
        )
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.to(self.device).eval()

    def _tokenize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Exactly reproduce the training cache's left-truncation contract."""
        body_limit = self.max_length - 2
        bodies = self.tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=body_limit,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        rows = [
            [
                int(self.tokenizer.cls_token_id),
                *map(int, body[-body_limit:]),
                int(self.tokenizer.sep_token_id),
            ]
            for body in bodies
        ]
        width = max(map(len, rows))
        ids = torch.full(
            (len(rows), width), int(self.tokenizer.pad_token_id), dtype=torch.long
        )
        mask = torch.zeros((len(rows), width), dtype=torch.bool)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[index, : len(row)] = True
        return ids.to(self.device), mask.to(self.device)

    def _amp_context(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @torch.inference_mode()
    def predict(
        self,
        texts: str | Iterable[str],
        target_prompt_lengths: int | Iterable[int],
        batch_size: int = 32,
        max_denoising_steps: int = 32,
        include_surface: bool = False,
    ) -> list[dict[str, object]]:
        """Predict workload; lengths must come from the serving dLLM tokenizer."""
        prompts = [texts] if isinstance(texts, str) else list(texts)
        lengths = (
            [target_prompt_lengths]
            if isinstance(target_prompt_lengths, int)
            else list(target_prompt_lengths)
        )
        if len(prompts) != len(lengths):
            raise ValueError("texts and target_prompt_lengths must have equal length")
        if not prompts:
            return []
        if any(length < 0 for length in lengths):
            raise ValueError("target prompt lengths must be non-negative")
        if not 1 <= max_denoising_steps <= 32:
            raise ValueError("max_denoising_steps must be in [1, 32]")

        results: list[dict[str, object]] = []
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_lengths = lengths[start : start + batch_size]
            ids, mask = self._tokenize(batch_prompts)
            target_length = torch.tensor(
                batch_lengths, dtype=torch.float32, device=self.device
            )
            remainder = torch.tensor(
                [length % 32 for length in batch_lengths],
                dtype=torch.long,
                device=self.device,
            )
            with self._amp_context():
                output = self.model(
                    ids,
                    mask,
                    prompt_remainder=remainder,
                    target_prompt_length=target_length,
                )
            curve = output["workload_curve"].float().cpu()
            surface = output["surface"].float().cpu()
            expected_blocks = output["expected_blocks"].float().cpu()
            expected_steps = output["expected_steps"].float().cpu()
            for index, prompt_length in enumerate(batch_lengths):
                row: dict[str, object] = {
                    "scheduler_score": float(curve[index, max_denoising_steps - 1]),
                    "predicted_workload_32": float(curve[index, -1]),
                    "max_denoising_steps": int(max_denoising_steps),
                    "expected_blocks": float(expected_blocks[index]),
                    "expected_steps_by_block": expected_steps[index].tolist(),
                    "workload_curve": curve[index].tolist(),
                    "target_prompt_length": int(prompt_length),
                    "prompt_remainder": int(prompt_length % 32),
                    "surface_shape": list(surface[index].shape),
                }
                if include_surface:
                    row["surface"] = surface[index].tolist()
                results.append(row)
        return results

DEFAULT_CONFIG = {'predictor_architecture': 'pooled_hazard_heads_v1',
 'encoder_layers': 12,
 'factorized_embedding_rank': 0,
 'semantic_pooling': 'mean_attention_max_lr32',
 'layer_mix_last_n': 1,
 'dropout': 0.1,
 'pretrain_head_dim': 256,
 'stage_a_targets': 'horizon_workload_only',
 'pretrain_horizons': list(DEFAULT_HORIZONS),
 'pretrain_frozen_epochs': 1,
 'pretrain_full_epochs': 20,
 'pretrain_encoder_lr': 4e-05,
 'pretrain_head_lr': 0.0003,
 'head_dim': 128,
 'count_alignment_features': 'remainder',
 'count_representation': 'continuation_hazard',
 'step_representation': 'continuation_hazard',
 'hazard_normalization': 'per_request_normalized',
 'stage_b_objective': 'surface_brier_control',
 'continuation_hard_support_mask': False,
 'cap_modeling': 'none',
 'remainder_embedding_dim': 16,
 'compact_frozen_epochs': 4,
 'compact_full_epochs': 4,
 'compact_training_protocol': 'continuous_joint',
 'compact_unfreeze_last_n_layers': 4,
 'compact_encoder_lr': 5e-06,
 'compact_surface_lr': 0.0002,
 'batching_mode': 'length',
 'checkpoint_selection': 'prediction_surface',
 'lr_schedule': 'cosine',
 'warmup_ratio': 0.03,
 'min_lr_ratio': 0.05,
 'weight_decay': 0.01,
 'regression_beta': 1.0,
 'grad_clip': 1.0,
 'training_contract': {'deployment_inputs': ['prompt token ids',
                                             'prompt attention mask',
                                             'target tokenizer prompt remainder'],
                       'dllm_prefill_required': False,
                       'count_logits': True,
                       'count_logits_shape': '[B,33]',
                       'count_representation': 'continuation_hazard',
                       'block_continuation_logits_shape': '[B,32]',
                       'cap_modeling': 'none',
                       'step_logits': True,
                       'step_logits_shape': '[B,33,32] derived log-probabilities',
                       'step_representation': 'continuation_hazard',
                       'step_continuation_logits_shape': '[B,33,31]',
                       'representation': 'single_pooled_prompt',
                       'token_memory': False,
                       'block_decoder': False,
                       'training_targets': 'ground_truth_only'},
 'seed': 42,
 'stage_a_batch_size': 128,
 'stage_b_batch_size': 256,
 'eval_batch_size': 1024,
 'skip_test': False,
 'max_length': 512,
 'max_blocks': 33,
 'max_denoising_steps': 32,
 'data_max_physical_output_length': 1024,
 'batch_size': 128,
 'scheduler_score_mode': 'surface',
 'stage_a_objective': 'normalized_multi_horizon_smooth_l1',
 'version': 'our_dws_pooled_hazard_20260926'}


def make_config(args, arrays, train_idx):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(model=str(args.model.resolve()), seed=args.seed, split=args.split,
                  eval_batch_size=args.eval_batch_size, skip_test=args.skip_test,
                  max_length=int(arrays["ids"].shape[1]), max_blocks=MAX_BLOCKS,
                  max_denoising_steps=MAX_DENOISING_STEPS,
                  model_type="compact_33block_dws")
    horizons = tuple(config["pretrain_horizons"])
    scales = compute_pretrain_scales(arrays, train_idx, horizons)
    config.update(pretrain_target_scales=scales.tolist(),
                  target_area_scale=float(scales[-1]),
                  prompt_length_scale=float(np.max(arrays["target_prompt_length"])),
                  cap_prevalence=float(np.mean(arrays["cap_flag"][train_idx])))
    return config


def package_model(output: Path, model_path: Path):
    resources = output / "minilm"
    resources.mkdir(exist_ok=True)
    AutoConfig.from_pretrained(str(model_path), local_files_only=True).save_pretrained(resources)
    AutoTokenizer.from_pretrained(str(model_path), local_files_only=True).save_pretrained(resources)
    (resources / "README.md").write_text(
        "Encoder configuration and tokenizer copied from the local training model. "
        "Encoder weights are stored in ../best.pt.\n", encoding="utf-8")
    # Export only the model/inference definitions, without training or experiment code.
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    runtime_names = ['CompactWorkloadCurvePredictor', 'FactorizedEmbedding', 'LayerMixer', 'OurDWSPredictor', 'SemanticPooler', 'build_encoder']
    runtime = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
               and node.name in runtime_names]
    preamble = 'from __future__ import annotations\nimport contextlib\nimport json\nimport math\nimport os\nfrom pathlib import Path\nfrom typing import Iterable\nimport numpy as np\nimport torch\nimport torch.nn as nn\nfrom transformers import AutoConfig, AutoModel, AutoTokenizer\nHERE = Path(__file__).resolve().parent\nMAX_DENOISING_STEPS = 32\nMAX_BLOCKS = 33\n'
    (output / "predict.py").write_text(preamble + "\n\n" +
        ast.unparse(ast.Module(body=runtime, type_ignores=[])) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local all-MiniLM-L12-v2 encoder/tokenizer")
    parser.add_argument("--token-cache", type=Path, required=True, help="input_ids.npy and meta.npz")
    parser.add_argument("--workload", type=Path, required=True, help="step_tokens.npy with 129 padded columns")
    parser.add_argument("--alignment-meta", type=Path, required=True, help="Row-aligned target-tokenizer metadata NPZ")
    parser.add_argument("--split-meta", type=Path, required=True, help="Row-aligned prompt_idx/split NPZ; 0=train, 1=val, 2=test")
    parser.add_argument("--train-prompt-ids", type=Path, help="Optional fixed training membership NPY")
    parser.add_argument("--split", choices=("default", "strict"), required=True, help="Label for the supplied split; recipe is identical")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output}")
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select an available device")
    required = [args.model / "config.json", args.token_cache / "input_ids.npy",
                args.token_cache / "meta.npz", args.workload / "step_tokens.npy",
                args.alignment_meta, args.split_meta]
    if args.train_prompt_ids is not None:
        required.append(args.train_prompt_ids)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing training inputs: {missing}")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    arrays = load_arrays(args.token_cache, args.workload)
    validate_reduced_block_range(arrays)
    load_alignment_metadata(arrays, args.alignment_meta)
    load_split_metadata(arrays, args.split_meta)
    arrays["output_tokens"] = np.asarray(arrays["physical_output_length"], dtype=np.float32)
    train_idx = (select_train_rows(arrays["prompt_idx"], arrays["split"], args.train_prompt_ids)
                 if args.train_prompt_ids is not None else split_indices(arrays, 0, 0))
    val_idx, test_idx = split_indices(arrays, 1, 0), split_indices(arrays, 2, 0)
    if not len(train_idx) or not len(val_idx) or (not args.skip_test and not len(test_idx)):
        raise ValueError("Training, validation and enabled test splits must be nonempty")
    config = make_config(args, arrays, train_idx)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "README.md").write_text(
        "Our-DWS predictor trained with the fixed two-stage recipe.\n"
        f"Local encoder: {args.model.resolve()}; seed: {args.seed}; split: {args.split}.\n"
        f"Token cache: {args.token_cache.resolve()}; labels: {args.workload.resolve()}.\n"
        f"Alignment: {args.alignment_meta.resolve()}; split metadata: {args.split_meta.resolve()}.\n"
        "best.pt contains the selected Stage-B weights; encoder_pretrain_best.pt contains Stage A.\n"
        "config.json records model parameters; metrics.json records training validation/test metrics.\n"
        "predict.py and minilm/ implement the serving interface.\n", encoding="utf-8")
    config["batch_size"] = 128
    encoder, pooler, mixer, stage_a_report = train_encoder_stage(
        config, arrays, train_idx, val_idx, args.output, device)
    config["batch_size"] = 256
    stage_b_report = train_compact_stage(config, encoder, pooler, mixer,
        arrays, train_idx, val_idx, test_idx, args.output, device)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(
        {"stage_a": stage_a_report, "stage_b": stage_b_report}, indent=2) + "\n", encoding="utf-8")
    package_model(args.output, args.model)


if __name__ == "__main__":
    main()
