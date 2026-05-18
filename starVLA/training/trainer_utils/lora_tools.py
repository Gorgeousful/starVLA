from __future__ import annotations

import logging
from typing import Iterable

import torch.nn as nn

logger = logging.getLogger(__name__)

_DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
_DEFAULT_EXTRA_TRAINABLE_MODULES = ("project_layers", "action_model")


def _as_plain_list(value, default: Iterable[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _get_cfg_value(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_lora_cfg(cfg):
    trainer_cfg = getattr(cfg, "trainer", None)
    return _get_cfg_value(trainer_cfg, "lora", None)


def _find_target_module_counts(model: nn.Module, target_modules: Iterable[str]) -> dict[str, int]:
    targets = set(target_modules)
    counts = {target: 0 for target in targets}
    for name, module in model.named_modules():
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name in targets and hasattr(module, "weight"):
            counts[leaf_name] += 1
    return counts


def _unfreeze_delta_mem_direct_parameters(model: nn.Module) -> int:
    try:
        from starVLA.model.modules.delta_mem.qwen3vl_delta_mem import Qwen3VLDeltaMemAttention
    except Exception:
        return 0

    num_params = 0
    for module in model.modules():
        if not isinstance(module, Qwen3VLDeltaMemAttention):
            continue
        for param in module.parameters(recurse=False):
            param.requires_grad = True
            num_params += param.numel()
    return num_params


def _unfreeze_named_child_modules(model: nn.Module, module_names: Iterable[str]) -> dict[str, int]:
    unfrozen: dict[str, int] = {}
    for module_name in module_names:
        module = getattr(model, module_name, None)
        if module is None:
            logger.warning(f"LoRA extra trainable module `{module_name}` not found on {type(model).__name__}.")
            continue
        count = 0
        for param in module.parameters():
            param.requires_grad = True
            count += param.numel()
        unfrozen[module_name] = count
    return unfrozen


def apply_lora_if_enabled(model: nn.Module, cfg) -> nn.Module:
    """Apply PEFT LoRA to the VLM backbone while keeping VLA heads trainable.

    This intentionally targets only ``model.qwen_vl_interface.model``. PEFT freezes
    the wrapped backbone by default and leaves LoRA parameters trainable. Since
    QwenKI's Delta-Mem modules live inside that backbone, we explicitly re-enable
    their direct parameters after wrapping.
    """

    lora_cfg = _get_lora_cfg(cfg)
    if not lora_cfg or not _get_cfg_value(lora_cfg, "enabled", False):
        return model

    if not hasattr(model, "qwen_vl_interface") or not hasattr(model.qwen_vl_interface, "model"):
        raise AttributeError("LoRA is enabled, but model.qwen_vl_interface.model was not found.")

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("LoRA is enabled but `peft` is not installed. Install `peft>=0.14.0`.") from exc

    vlm_backbone = model.qwen_vl_interface.model
    target_modules = _as_plain_list(
        _get_cfg_value(lora_cfg, "target_modules", None),
        _DEFAULT_TARGET_MODULES,
    )
    target_counts = _find_target_module_counts(vlm_backbone, target_modules)
    missing_targets = [name for name, count in target_counts.items() if count == 0]
    if missing_targets:
        raise ValueError(
            "LoRA target_modules not found in qwen_vl_interface.model: "
            f"{missing_targets}. Available target counts: {target_counts}"
        )

    peft_config = LoraConfig(
        r=int(_get_cfg_value(lora_cfg, "rank", 16)),
        lora_alpha=float(_get_cfg_value(lora_cfg, "alpha", 32.0)),
        lora_dropout=float(_get_cfg_value(lora_cfg, "dropout", 0.05)),
        target_modules=target_modules,
        bias=str(_get_cfg_value(lora_cfg, "bias", "none")),
        task_type=TaskType.CAUSAL_LM,
    )
    model.qwen_vl_interface.model = get_peft_model(vlm_backbone, peft_config)

    delta_mem_params = _unfreeze_delta_mem_direct_parameters(model.qwen_vl_interface.model)
    extra_trainable_modules = _as_plain_list(
        _get_cfg_value(lora_cfg, "extra_trainable_modules", None),
        _DEFAULT_EXTRA_TRAINABLE_MODULES,
    )
    extra_unfrozen = _unfreeze_named_child_modules(model, extra_trainable_modules)

    logger.info(
        "Applied PEFT LoRA to qwen_vl_interface.model. "
        f"target_counts={target_counts}, delta_mem_params={delta_mem_params}, "
        f"extra_trainable={extra_unfrozen}"
    )
    return model
