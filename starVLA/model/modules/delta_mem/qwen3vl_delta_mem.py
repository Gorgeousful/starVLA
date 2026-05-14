from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3VLTextAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


VALID_DELTA_HEADS = ("q", "k", "v", "o")


def _as_plain(value):
    if hasattr(value, "to_container"):
        return value.to_container()
    return value


def _parse_layers(raw) -> tuple[int, ...]:
    raw = _as_plain(raw)
    if raw is None:
        return ()
    if isinstance(raw, str):
        stripped = raw.strip().lower()
        if stripped in {"", "all", "none", "null"}:
            return ()
        return tuple(int(piece.strip()) for piece in raw.split(",") if piece.strip())
    if isinstance(raw, Iterable):
        return tuple(int(item) for item in raw)
    raise TypeError(f"Unsupported adapter_layers value: {raw!r}")


def _parse_delta_heads(raw) -> tuple[str, ...]:
    raw = _as_plain(raw)
    if raw is None:
        return ("q", "o")
    if isinstance(raw, str):
        heads = tuple(piece.strip().lower() for piece in raw.split(",") if piece.strip())
    else:
        heads = tuple(str(piece).strip().lower() for piece in raw if str(piece).strip())
    invalid = [head for head in heads if head not in VALID_DELTA_HEADS]
    if invalid:
        raise ValueError(f"Unsupported delta_heads: {invalid}; expected subset of {VALID_DELTA_HEADS}")
    return heads


@dataclass(frozen=True)
class DeltaMemConfig:
    rank: int = 8
    alpha: float = 16.0
    adapter_layers: tuple[int, ...] = ()
    delta_heads: tuple[str, ...] = ("q", "o")
    beta_bias_init: float = -1.5
    normalize_qk: bool = True
    couple_lambda: bool = True
    state_update_mode: str = "standard"
    rankwise_gates: bool = True
    output_init: str = "base_slice_fixed"
    base_slice_ref_width: int = 8
    online_gain: float = 0.05
    history_write_mode: str = "frame_mean"

    @classmethod
    def from_config(cls, cfg) -> "DeltaMemConfig":
        return cls(
            rank=int(cfg.get("rank", 8)),
            alpha=float(cfg.get("alpha", 16.0)),
            adapter_layers=_parse_layers(cfg.get("adapter_layers", None)),
            delta_heads=_parse_delta_heads(cfg.get("delta_heads", ("q", "o"))),
            beta_bias_init=float(cfg.get("beta_bias_init", -1.5)),
            normalize_qk=bool(cfg.get("normalize_qk", True)),
            couple_lambda=bool(cfg.get("couple_lambda", True)),
            state_update_mode=str(cfg.get("state_update_mode", "standard")),
            rankwise_gates=bool(cfg.get("rankwise_gates", True)),
            output_init=str(cfg.get("output_init", "base_slice_fixed")),
            base_slice_ref_width=int(cfg.get("base_slice_ref_width", 8)),
            online_gain=float(cfg.get("online_gain", 0.05)),
            history_write_mode=str(cfg.get("history_write_mode", "frame_mean")),
        )

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.history_write_mode != "frame_mean":
            raise ValueError("The first QwenKI Delta-Mem path only supports history_write_mode='frame_mean'.")
        if self.state_update_mode not in {"standard", "lambda_outside", "no_lambda"}:
            raise ValueError("Unsupported state_update_mode.")
        if self.output_init not in {"zero", "random", "base_slice", "base_slice_fixed"}:
            raise ValueError("Unsupported output_init.")
        if self.base_slice_ref_width <= 0:
            raise ValueError("base_slice_ref_width must be positive")


class Qwen3VLDeltaMemAttention(nn.Module):
    def __init__(self, base: Qwen3VLTextAttention, config: DeltaMemConfig) -> None:
        super().__init__()
        self.base = base
        self.delta_config = config
        self.config = base.config
        self.layer_idx = base.layer_idx
        self.head_dim = base.head_dim
        self.num_key_value_groups = base.num_key_value_groups
        self.scaling = base.scaling
        self.attention_dropout = base.attention_dropout
        self.is_causal = base.is_causal

        hidden_size = base.q_proj.in_features
        self.hidden_size = hidden_size
        self.rank = config.rank
        self.delta_scaling = config.alpha / config.rank
        self.normalize_qk = config.normalize_qk
        self.couple_lambda = config.couple_lambda
        self.state_update_mode = config.state_update_mode
        self.rankwise_gates = config.rankwise_gates
        self.output_init = config.output_init
        self.base_slice_ref_width = config.base_slice_ref_width
        self.online_gain = config.online_gain
        self.gate_dim = self.rank if config.rankwise_gates else 1
        self.active_delta_heads = frozenset(config.delta_heads)

        self.memory_q_proj = nn.Parameter(torch.empty(self.rank, hidden_size))
        self.memory_k_proj = nn.Parameter(torch.empty(self.rank, hidden_size))
        self.memory_v_proj = nn.Parameter(torch.empty(self.rank, hidden_size))
        self.beta_proj = nn.Parameter(torch.empty(self.gate_dim, hidden_size))
        self.beta_bias = nn.Parameter(torch.full((self.gate_dim,), config.beta_bias_init))
        if not config.couple_lambda:
            self.lambda_proj = nn.Parameter(torch.empty(self.gate_dim, hidden_size))
            self.lambda_bias = nn.Parameter(torch.full((self.gate_dim,), -config.beta_bias_init))

        self.delta_q_proj = nn.Parameter(torch.empty(base.q_proj.out_features, self.rank))
        self.delta_k_proj = nn.Parameter(torch.empty(base.k_proj.out_features, self.rank))
        self.delta_v_proj = nn.Parameter(torch.empty(base.v_proj.out_features, self.rank))
        self.delta_o_proj = nn.Parameter(torch.empty(base.o_proj.out_features, self.rank))

        self.delta_state: torch.Tensor | None = None
        self.write_enabled = True
        self.reset_parameters()

    def _init_delta_head(self, head: nn.Parameter, base_weight: torch.Tensor) -> None:
        if self.output_init == "zero":
            nn.init.zeros_(head)
            return
        if self.output_init == "random":
            nn.init.kaiming_uniform_(head, a=math.sqrt(5))
            with torch.no_grad():
                head.mul_(self.online_gain)
            return
        with torch.no_grad():
            if self.output_init == "base_slice":
                slice_width = min(self.rank, base_weight.shape[1])
            else:
                slice_width = min(self.base_slice_ref_width, self.rank, base_weight.shape[1])
            head.zero_()
            if slice_width == 0:
                return
            base_slice = base_weight[:, :slice_width].detach().clone().float()
            base_slice = F.normalize(base_slice, dim=0, eps=1e-6)
            head[:, :slice_width].copy_((base_slice * self.online_gain).to(dtype=head.dtype))

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.memory_q_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.memory_k_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.memory_v_proj, a=math.sqrt(5))
        nn.init.zeros_(self.beta_proj)
        if not self.couple_lambda:
            nn.init.zeros_(self.lambda_proj)
        self._init_delta_head(self.delta_q_proj, self.base.q_proj.weight)
        self._init_delta_head(self.delta_k_proj, self.base.k_proj.weight)
        self._init_delta_head(self.delta_v_proj, self.base.v_proj.weight)
        self._init_delta_head(self.delta_o_proj, self.base.o_proj.weight)
        for head_name, param in (
            ("q", self.delta_q_proj),
            ("k", self.delta_k_proj),
            ("v", self.delta_v_proj),
            ("o", self.delta_o_proj),
        ):
            if head_name not in self.active_delta_heads:
                nn.init.zeros_(param)

    def reset_state(self) -> None:
        self.delta_state = None

    def set_write_enabled(self, enabled: bool) -> None:
        self.write_enabled = enabled

    def _ensure_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        state_shape = (batch_size, self.rank, self.rank)
        if self.delta_state is None or self.delta_state.shape != state_shape:
            self.delta_state = torch.zeros(*state_shape, device=device, dtype=dtype)
        elif self.delta_state.device != device or self.delta_state.dtype != dtype:
            self.delta_state = self.delta_state.to(device=device, dtype=dtype)
        return self.delta_state

    def _token_validity_mask(
        self,
        attention_mask,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            return attention_mask[:, -seq_len:].to(device=device).ne(0)
        if attention_mask.dim() == 4:
            if attention_mask.size(0) != batch_size:
                return None
            if attention_mask.size(-2) < seq_len or attention_mask.size(-1) < seq_len:
                return None
            query_mask = attention_mask[:, 0, -seq_len:, -seq_len:]
            diagonal = query_mask.diagonal(dim1=-2, dim2=-1)
            return diagonal.eq(0)
        return None

    def _summary_hidden(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if token_mask is None:
            return hidden_states.mean(dim=1)
        weights = token_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden_states * weights).sum(dim=1) / denom

    def _normalize_memory_projection(self, projected: torch.Tensor) -> torch.Tensor:
        projected = torch.tanh(projected)
        if self.normalize_qk:
            projected = F.normalize(projected, dim=-1, eps=1e-6)
        return projected

    def _gate_rows(self, summary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta = torch.sigmoid(F.linear(summary, self.beta_proj, self.beta_bias))
        if self.state_update_mode == "no_lambda":
            lam = torch.ones_like(beta)
        elif self.couple_lambda:
            lam = 1.0 - beta
        else:
            lam = torch.sigmoid(F.linear(summary, self.lambda_proj, self.lambda_bias))
        if beta.size(-1) == 1:
            beta = beta.expand(-1, self.rank)
        if lam.size(-1) == 1:
            lam = lam.expand(-1, self.rank)
        return beta.unsqueeze(-1), lam.unsqueeze(-1)

    def _update_coefficients(
        self,
        beta: torch.Tensor,
        lam: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.state_update_mode == "standard":
            return lam, beta, beta
        if self.state_update_mode == "lambda_outside":
            return lam, lam * beta, beta
        if self.state_update_mode == "no_lambda":
            ones = torch.ones_like(beta)
            return ones, beta, beta
        raise ValueError(f"Unsupported state_update_mode: {self.state_update_mode}")

    def _memory_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._normalize_memory_projection(F.linear(hidden_states, self.memory_q_proj))

    def _read_state(self, hidden_states: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        read_query = self._memory_q(hidden_states)
        return torch.einsum("bij,btj->bti", state, read_query)

    def _update_state(
        self,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        summary = self._summary_hidden(hidden_states, token_mask)
        memory_k = self._normalize_memory_projection(F.linear(summary, self.memory_k_proj))
        memory_v = F.linear(summary, self.memory_v_proj)
        beta, lam = self._gate_rows(summary)
        keep, erase, write = self._update_coefficients(beta, lam)

        pred = torch.einsum("bij,bj->bi", state, memory_k)
        pred_outer = pred.unsqueeze(-1) * memory_k.unsqueeze(1)
        write_outer = memory_v.unsqueeze(-1) * memory_k.unsqueeze(1)
        return keep * state - erase * pred_outer + write * write_outer

    def _delta_head(self, reads: torch.Tensor, head: nn.Parameter, name: str) -> torch.Tensor | None:
        if name not in self.active_delta_heads:
            return None
        return F.linear(reads, head) * self.delta_scaling

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        state = self._ensure_state(hidden_states.size(0), hidden_states.device, hidden_states.dtype)
        token_mask = self._token_validity_mask(
            attention_mask,
            batch_size=hidden_states.size(0),
            seq_len=hidden_states.size(1),
            device=hidden_states.device,
        )
        reads = self._read_state(hidden_states, state)
        if token_mask is not None:
            reads = reads * token_mask.unsqueeze(-1).to(dtype=reads.dtype)
        if self.write_enabled:
            state = self._update_state(hidden_states, state, token_mask)
            self.delta_state = state

        delta_q = self._delta_head(reads, self.delta_q_proj, "q")
        delta_k = self._delta_head(reads, self.delta_k_proj, "k")
        delta_v = self._delta_head(reads, self.delta_v_proj, "v")
        delta_o = self._delta_head(reads, self.delta_o_proj, "o")

        query_states = self.base.q_proj(hidden_states)
        key_states = self.base.k_proj(hidden_states)
        value_states = self.base.v_proj(hidden_states)
        if delta_q is not None:
            query_states = query_states + delta_q
        if delta_k is not None:
            key_states = key_states + delta_k
        if delta_v is not None:
            value_states = value_states + delta_v

        query_states = self.base.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.base.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.base.layer_idx,
                cache_kwargs,
            )

        attention_interface = eager_attention_forward
        if self.base.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.base.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self.base,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.base.attention_dropout,
            scaling=self.base.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.base.o_proj(attn_output)
        if delta_o is not None:
            attn_output = attn_output + delta_o.to(attn_output.dtype)
        return attn_output, attn_weights


def _get_parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def iter_delta_mem_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, Qwen3VLDeltaMemAttention):
            yield name, module


def attach_delta_mem_to_qwen3vl(model: nn.Module, config: DeltaMemConfig) -> list[str]:
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, Qwen3VLTextAttention):
            continue
        if config.adapter_layers and module.layer_idx not in config.adapter_layers:
            continue
        parent, attr = _get_parent_module(model, name)
        wrapped = Qwen3VLDeltaMemAttention(module, config).to(
            device=module.q_proj.weight.device,
            dtype=module.q_proj.weight.dtype,
        )
        setattr(parent, attr, wrapped)
        replaced.append(name)
    if not replaced:
        raise RuntimeError("No Qwen3VLTextAttention modules were replaced by Delta-Mem.")
    return replaced


def reset_delta_mem_states(model: nn.Module) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.reset_state()


def set_delta_mem_write_enabled(model: nn.Module, enabled: bool) -> None:
    for _, module in iter_delta_mem_modules(model):
        module.set_write_enabled(enabled)
