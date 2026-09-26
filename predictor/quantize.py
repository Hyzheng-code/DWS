#!/usr/bin/env python
"""Standalone Our-DWS ONNX export and encoder quantization.

Dependencies: torch, transformers, numpy, onnx, onnxruntime.
--model-dir accepts the train.py output or a compatible finalized checkpoint.
INT8 is dynamic per-channel QInt8; INT4 uses weight-only MatMul/Gather;
the DWS heads remain floating point. FP16 keeps float32 graph inputs/outputs.
--verification-inputs optionally supplies real tokenized input arrays in NPZ.
Without it, verification checks synthetic inputs only, not prediction accuracy.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
MAX_DENOISING_STEPS = 32
MAX_BLOCKS = 33
MODEL_MAX_BLOCKS = 33
SOURCE_PADDED_BLOCKS = 129
DEFAULT_HORIZONS = (1, *range(4, MAX_DENOISING_STEPS + 1, 4))
CURVE_HORIZONS = tuple(range(1, MAX_DENOISING_STEPS + 1))

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.transformers.float16 import convert_float_to_float16

OUTPUT_NAMES = ("surface_area", "raw_surface_curve", "surface", "count_logits",
                "count_probability", "block_survival", "step_survival",
                "expected_blocks", "expected_steps")

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

def session_options(threads: int) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options

def encoder_weight_nodes(model_path: Path) -> tuple[list[str], list[str]]:
    model = onnx.load(model_path, load_external_data=False)
    initializers = {item.name for item in model.graph.initializer}
    supported = {"MatMul", "Gemm", "Gather"}
    included: list[str] = []
    excluded_int4: list[str] = []
    for node in model.graph.node:
        if node.op_type not in supported or not node.name:
            continue
        weight_input = node.input[0] if node.op_type == "Gather" else (
            node.input[1] if len(node.input) > 1 else ""
        )
        has_constant_weight = weight_input in initializers
        is_encoder = node.name.startswith("/model/encoder/")
        if has_constant_weight and is_encoder:
            included.append(node.name)
        elif node.op_type in {"MatMul", "Gather"}:
            excluded_int4.append(node.name)
    if not included:
        raise RuntimeError("no encoder weight-bearing ONNX nodes were found")
    return sorted(included), sorted(excluded_int4)


def quantize_encoder_int8(fp32_path: Path, int8_path: Path, included: list[str]) -> float:
    started = time.time()
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=included,
    )
    # DeBERTa exports contain If subgraphs that capture outer-scope tensors.
    # ONNX's standalone checker can reject the Identity nodes inserted by the
    # dynamic quantizer even though ORT resolves and executes the graph.  The
    # deployment runtime load is therefore the authoritative validation.
    try:
        onnx.checker.check_model(onnx.load(int8_path))
    except onnx.checker.ValidationError as error:
        print(f"ONNX checker warning for {int8_path.name}: {error}", flush=True)
    ort.InferenceSession(
        str(int8_path),
        sess_options=session_options(1),
        providers=["CPUExecutionProvider"],
    )
    return time.time() - started


def quantize_encoder_int4(
    fp32_path: Path,
    int4_path: Path,
    excluded: list[str],
    block_size: int,
    symmetric: bool,
) -> float:
    from onnxruntime.quantization import quant_utils
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )

    started = time.time()
    # GatherBlockQuantized uses ONNX INT4/UINT4 tensor types introduced in
    # opset 21.  ORT's quantizer otherwise only changes the opset number,
    # which leaves legacy Reduce* attributes invalid.  Run the actual ONNX
    # schema conversion first so every non-quantized node remains legal.
    converted_path = int4_path.with_name(int4_path.stem + "_opset21_input.onnx")
    converted = onnx.version_converter.convert_version(onnx.load(fp32_path), 21)
    onnx.checker.check_model(converted)
    onnx.save(converted, converted_path)
    config = DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=symmetric,
        accuracy_level=4,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=("MatMul", "Gather"),
        quant_axes=(("MatMul", 0), ("Gather", 1)),
        bits=4,
    )
    quantizer = MatMulNBitsQuantizer(
        str(converted_path),
        nodes_to_exclude=excluded,
        algo_config=config,
    )
    try:
        quantizer.process()
        quantizer.model.save_model_to_file(str(int4_path), use_external_data_format=False)
        try:
            onnx.checker.check_model(onnx.load(int4_path))
        except onnx.checker.ValidationError as error:
            print(f"ONNX checker warning for {int4_path.name}: {error}", flush=True)
        ort.InferenceSession(
            str(int4_path),
            sess_options=session_options(1),
            providers=["CPUExecutionProvider"],
        )
    finally:
        converted_path.unlink(missing_ok=True)
    return time.time() - started

class DeploymentWrapper(torch.nn.Module):
    """Expose only structured DWS deployment outputs in the ONNX graph."""

    def __init__(self, model: CompactWorkloadCurvePredictor) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        prompt_remainder: torch.Tensor,
        target_prompt_length: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.model(ids, mask, prompt_remainder, target_prompt_length)
        return tuple(outputs[name] for name in OUTPUT_NAMES)


def stable_topological_sort(graph: onnx.GraphProto) -> None:
    """Repair Cast placement produced by ORT's FP16 graph converter."""

    nodes = list(graph.node)
    producer = {
        output: index
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    dependencies = [
        {
            producer[name]
            for name in node.input
            if name in producer and producer[name] != index
        }
        for index, node in enumerate(nodes)
    ]
    completed: set[int] = set()
    ordered: list[onnx.NodeProto] = []
    while len(ordered) < len(nodes):
        ready = [
            index
            for index in range(len(nodes))
            if index not in completed and dependencies[index].issubset(completed)
        ]
        if not ready:
            raise RuntimeError("FP16 graph contains a dependency cycle")
        for index in ready:
            completed.add(index)
            ordered.append(nodes[index])
    del graph.node[:]
    graph.node.extend(ordered)


def export_fp16(fp32_path: Path, fp16_path: Path) -> float:
    started = time.time()
    graph = convert_float_to_float16(
        onnx.load(fp32_path),
        keep_io_types=True,
        disable_shape_infer=False,
    )
    # The converter appends output compatibility Casts after their consumers
    # for this graph.  Reordering is semantics-preserving and checker-clean.
    stable_topological_sort(graph.graph)
    onnx.save(graph, fp16_path)
    onnx.checker.check_model(onnx.load(fp16_path))
    ort.InferenceSession(
        str(fp16_path),
        sess_options=session_options(1),
        providers=["CPUExecutionProvider"],
    )
    return time.time() - started

class QuantizedOurDWSPredictor:
    def __init__(
        self,
        model_dir: str | Path = HERE,
        threads: int = 8,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = json.loads(
            (self.model_dir / "config.json").read_text(encoding="utf-8")
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_dir / "model.onnx"),
            sess_options=options,
            providers=[provider],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.output_names = [item.name for item in self.session.get_outputs()]
        resources = self.model_dir / "minilm"
        if not (resources / "config.json").is_file():
            resources = Path(self.config["model"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(resources), local_files_only=True, use_fast=True
        )
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "right"
        self.max_length = int(self.config.get("max_length", 512))

    def _tokenize(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
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
        ids = np.full(
            (len(rows), width), int(self.tokenizer.pad_token_id), dtype=np.int64
        )
        mask = np.zeros((len(rows), width), dtype=np.bool_)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = row
            mask[index, : len(row)] = True
        return ids, mask

    def predict(
        self,
        texts: str | Iterable[str],
        target_prompt_lengths: int | Iterable[int],
        max_denoising_steps: int = 32,
        include_surface: bool = False,
    ) -> list[dict[str, object]]:
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
        ids, mask = self._tokenize(prompts)
        feed = {
            "input_ids": ids,
            "attention_mask": mask,
            "prompt_remainder": np.asarray(
                [length % 32 for length in lengths], dtype=np.int64
            ),
            "target_prompt_length": np.asarray(lengths, dtype=np.int64),
        }
        values = self.session.run(
            self.output_names,
            {name: value for name, value in feed.items() if name in self.input_names},
        )
        output = dict(zip(self.output_names, values, strict=True))
        results = []
        for index, prompt_length in enumerate(lengths):
            curve = np.asarray(output["raw_surface_curve"][index], dtype=np.float32)
            row: dict[str, object] = {
                "scheduler_score": float(curve[max_denoising_steps - 1]),
                "predicted_workload_32": float(output["surface_area"][index]),
                "max_denoising_steps": int(max_denoising_steps),
                "expected_blocks": float(output["expected_blocks"][index]),
                "expected_steps_by_block": np.asarray(
                    output["expected_steps"][index], dtype=np.float32
                ).tolist(),
                "workload_curve": curve.tolist(),
                "target_prompt_length": int(prompt_length),
                "prompt_remainder": int(prompt_length % 32),
                "surface_shape": list(output["surface"][index].shape),
            }
            if include_surface:
                row["surface"] = np.asarray(
                    output["surface"][index], dtype=np.float32
                ).tolist()
            results.append(row)
        return results


def verification_inputs(model, config, path=None):
    if path is not None:
        with np.load(path, allow_pickle=False) as values:
            feed = {k: values[k].copy() for k in ("input_ids", "attention_mask", "prompt_remainder")}
            feed["target_prompt_length"] = (values["target_prompt_length"].copy()
                if "target_prompt_length" in values else feed["prompt_remainder"].copy())
        feed = {k: np.ascontiguousarray(v, dtype=np.bool_ if k == "attention_mask" else np.int64)
                for k, v in feed.items()}
    else:
        rng = np.random.default_rng(42)
        width = min(int(config.get("max_length", 512)), 16)
        feed = {"input_ids": rng.integers(0, model.encoder.config.vocab_size, size=(2, width), dtype=np.int64),
                "attention_mask": np.ones((2, width), dtype=np.bool_),
                "prompt_remainder": np.array([0, 7], dtype=np.int64),
                "target_prompt_length": np.array([32, 39], dtype=np.int64)}
        feed["attention_mask"][1, width//2:] = False
    ids, mask, remainder = (feed[k] for k in ("input_ids", "attention_mask", "prompt_remainder"))
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[0] < 1 or ids.shape[1] < 1:
        raise ValueError("Token IDs and attention mask must have the same nonempty [batch, sequence] shape")
    if remainder.shape != (len(ids),) or feed["target_prompt_length"].shape != remainder.shape:
        raise ValueError("Alignment inputs must contain one value per request")
    if np.any(remainder < 0) or np.any(remainder >= 32) or np.any(feed["target_prompt_length"] < 0):
        raise ValueError("Invalid target-tokenizer alignment inputs")
    if np.any(feed["target_prompt_length"] % 32 != remainder):
        raise ValueError("prompt_remainder must equal target_prompt_length modulo 32")
    if ids.shape[1] > int(config.get("max_length", 512)) or np.any(mask.sum(1) == 0):
        raise ValueError("Empty or overlong input sequences")
    if np.any(ids < 0) or np.any(ids >= model.encoder.config.vocab_size):
        raise ValueError("Input token IDs exceed the encoder vocabulary")
    return feed


def run_graph(path, feed, threads):
    session = ort.InferenceSession(str(path), sess_options=session_options(threads),
                                   providers=["CPUExecutionProvider"])
    names = {x.name for x in session.get_inputs()}
    values = session.run(list(OUTPUT_NAMES), {k: v for k, v in feed.items() if k in names})
    return dict(zip(OUTPUT_NAMES, values))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New or empty deployment directory")
    parser.add_argument("--precision", choices=("INT8", "INT4", "FP16"), default="INT8")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--int4-block-size", type=int, default=64)
    parser.add_argument("--verification-inputs", type=Path, help="NPZ containing token IDs, mask and prompt alignment")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output}")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.int4_block_size < 16 or args.int4_block_size & (args.int4_block_size-1):
        raise ValueError("--int4-block-size must be a power of two, at least 16")
    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint_path = args.model_dir / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint["config"])
    CompactWorkloadCurvePredictor.validate_config(config)
    if config.get("count_alignment_features") != "remainder":
        raise ValueError("Expected a finalized Our-DWS remainder-aligned checkpoint")
    resources = args.model_dir / "minilm"
    if not (resources / "config.json").is_file():
        resources = Path(config["model"])
    model = CompactWorkloadCurvePredictor(resources, config, initialize_encoder_from_base=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.float().eval()
    feed = verification_inputs(model, config, args.verification_inputs)
    tensors = tuple(torch.from_numpy(feed[k]) for k in
        ("input_ids", "attention_mask", "prompt_remainder", "target_prompt_length"))
    wrapper = DeploymentWrapper(model).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "README.md").write_text(
        f"Our-DWS {args.precision} ONNX deployment exported from {args.model_dir.resolve()}.\n"
        "model.onnx and config.json define the predictor; minilm/ contains encoder config/tokenizer.\n"
        "predict_onnx.py exports QuantizedOurDWSPredictor for SGLang.\n"
        "verification.json records numerical checks; synthetic inputs do not establish accuracy.\n",
        encoding="utf-8")
    fp32 = args.output / "model_fp32.onnx"
    with torch.inference_mode():
        expected = tuple(v.float().numpy() for v in wrapper(*tensors))
        torch.onnx.export(wrapper, tensors, fp32,
            input_names=("input_ids", "attention_mask", "prompt_remainder", "target_prompt_length"),
            output_names=OUTPUT_NAMES,
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"},
                          "prompt_remainder": {0: "batch"}, "target_prompt_length": {0: "batch"},
                          **{name: {0: "batch"} for name in OUTPUT_NAMES}},
            opset_version=17, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(fp32))
    reference = run_graph(fp32, feed, args.threads)
    for name, tensor in zip(OUTPUT_NAMES, expected):
        np.testing.assert_allclose(reference[name], tensor, rtol=2e-4, atol=2e-4, err_msg=name)
    included, excluded = encoder_weight_nodes(fp32)
    target = args.output / "model.onnx"
    if args.precision == "INT8":
        quantize_encoder_int8(fp32, target, included)
    elif args.precision == "INT4":
        quantize_encoder_int4(fp32, target, excluded, args.int4_block_size, False)
    else:
        export_fp16(fp32, target)
    actual = run_graph(target, feed, args.threads)
    differences = {}
    for name in OUTPUT_NAMES:
        if actual[name].shape != reference[name].shape or not np.isfinite(actual[name]).all():
            raise ValueError(f"Invalid quantized output: {name}")
        error = np.abs(actual[name].astype(np.float64) - reference[name])
        differences[name] = {"max_abs_error": float(error.max()), "mean_abs_error": float(error.mean())}
    surface_rtol = 2 * np.finfo(np.float16).eps if args.precision == "FP16" else 1e-4
    np.testing.assert_allclose(actual["surface"].sum((1, 2)), actual["surface_area"],
                               rtol=surface_rtol, atol=1e-3)
    verification = {"input_kind": "provided" if args.verification_inputs else "synthetic",
                    "precision": args.precision, "rows": len(feed["input_ids"]),
                    "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    "differences_from_fp32": differences}
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    (args.output / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    resource_output = args.output / "minilm"
    resource_output.mkdir()
    AutoConfig.from_pretrained(str(resources), local_files_only=True).save_pretrained(resource_output)
    AutoTokenizer.from_pretrained(str(resources), local_files_only=True).save_pretrained(resource_output)
    (resource_output / "README.md").write_text("Local encoder configuration and tokenizer for ../model.onnx.\n")
    source = Path(__file__).read_text(encoding="utf-8")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "QuantizedOurDWSPredictor")
    preamble = ("from __future__ import annotations\nimport json\nfrom pathlib import Path\n"
                "from typing import Iterable\nimport numpy as np\nimport onnxruntime as ort\n"
                "from transformers import AutoTokenizer\nHERE = Path(__file__).resolve().parent\n\n")
    (args.output / "predict_onnx.py").write_text(preamble+ast.unparse(node)+"\n", encoding="utf-8")
    fp32.unlink()
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
