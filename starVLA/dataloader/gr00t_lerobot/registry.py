"""
Centralized registry that loads the selected benchmark-specific data config from
``examples/<BENCH>/train_files/data_registry/`` and merges it with the base
registries defined in this package.

Three registries are maintained:

* ``DATASET_NAMED_MIXTURES``       – mixture_name → [(dataset, weight, robot_type)]
* ``ROBOT_TYPE_CONFIG_MAP``       – robot_type → DataConfig instance
* ``ROBOT_TYPE_TO_EMBODIMENT_TAG`` – robot_type → EmbodimentTag


Set ``STARVLA_DATA_REGISTRY_BENCH`` (for example, ``LIBERO``), or pass a
``--config_yaml`` path under ``examples/<BENCH>/...`` so the benchmark can be
inferred.

Usage::

    from starVLA.dataloader.gr00t_lerobot.registry import (
        ROBOT_TYPE_CONFIG_MAP,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
        DATASET_NAMED_MIXTURES,
    )
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

# Base registries (kept as fallback / seed values)
from starVLA.dataloader.gr00t_lerobot.data_config import (
    ROBOT_TYPE_CONFIG_MAP as _BASE_CONFIG_MAP,
)
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import (
    ROBOT_TYPE_TO_EMBODIMENT_TAG as _BASE_EMBODIMENT_MAP,
    EmbodimentTag,  # noqa: F401  – re-export for convenience
)
from starVLA.dataloader.gr00t_lerobot.mixtures import (
    DATASET_NAMED_MIXTURES as _BASE_MIXTURES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mutable copies – will be extended by discovered modules
# ---------------------------------------------------------------------------
ROBOT_TYPE_CONFIG_MAP: dict = dict(_BASE_CONFIG_MAP)
ROBOT_TYPE_TO_EMBODIMENT_TAG: dict = dict(_BASE_EMBODIMENT_MAP)
DATASET_NAMED_MIXTURES: dict = dict(_BASE_MIXTURES)

# ---------------------------------------------------------------------------
# Discovery logic
# ---------------------------------------------------------------------------
_REGISTRY_DIR_NAME = "data_registry"
_DISCOVERED = False


def _repo_root() -> Path:
    # Walk up from this file to the repo root
    # registry.py is at starVLA/dataloader/gr00t_lerobot/registry.py
    #   parents: [0]=gr00t_lerobot, [1]=dataloader, [2]=starVLA(pkg), [3]=repo root
    return Path(__file__).resolve().parents[3]


def _infer_bench_from_argv() -> str | None:
    """Infer examples/<bench> from a --config_yaml CLI argument when present."""
    argv = list(sys.argv)
    config_path = None
    for idx, arg in enumerate(argv):
        if arg == "--config_yaml" and idx + 1 < len(argv):
            config_path = argv[idx + 1]
            break
        if arg.startswith("--config_yaml="):
            config_path = arg.split("=", 1)[1]
            break

    if not config_path:
        return None

    parts = Path(config_path).parts
    for idx, part in enumerate(parts[:-1]):
        if part == "examples" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _selected_bench_name() -> str:
    bench_name = os.environ.get("STARVLA_DATA_REGISTRY_BENCH") or _infer_bench_from_argv()
    if bench_name:
        return bench_name

    raise RuntimeError(
        "Unable to select a data registry. Set STARVLA_DATA_REGISTRY_BENCH "
        "(for example, STARVLA_DATA_REGISTRY_BENCH=LIBERO) or pass "
        "--config_yaml with a path under examples/<BENCH>/..."
    )


def _find_registry_dirs() -> list[Path]:
    """Return only the selected ``examples/<bench>/train_files/data_registry`` directory."""
    repo_root = _repo_root()
    examples_dir = repo_root / "examples"
    if not examples_dir.is_dir():
        return []
    bench_name = _selected_bench_name()
    registry_dir = examples_dir / bench_name / "train_files" / _REGISTRY_DIR_NAME
    if not registry_dir.is_dir():
        raise RuntimeError(
            f"Selected data registry examples/{bench_name}/train_files/{_REGISTRY_DIR_NAME} "
            "does not exist."
        )
    return [registry_dir]


def _load_module_from_path(module_name: str, file_path: Path):
    """Import a Python file as a module with the given name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_and_merge() -> None:
    """Scan ``examples/*/train_files/data_registry/`` and merge into global registries."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    for registry_dir in _find_registry_dirs():
        bench_name = registry_dir.parents[1].name  # examples/<BenchName>/train_files/data_registry
        prefix = f"_data_registry_{bench_name}"

        # --- data_config.py (may contain all three registries) ---
        cfg_file = registry_dir / "data_config.py"
        if cfg_file.is_file():
            mod = _load_module_from_path(f"{prefix}.data_config", cfg_file)
            if mod:
                if hasattr(mod, "ROBOT_TYPE_CONFIG_MAP"):
                    ROBOT_TYPE_CONFIG_MAP.update(mod.ROBOT_TYPE_CONFIG_MAP)
                    logger.info(f"[registry] Loaded data_config from {bench_name}: {list(mod.ROBOT_TYPE_CONFIG_MAP.keys())}")
                if hasattr(mod, "ROBOT_TYPE_TO_EMBODIMENT_TAG"):
                    ROBOT_TYPE_TO_EMBODIMENT_TAG.update(mod.ROBOT_TYPE_TO_EMBODIMENT_TAG)
                    logger.info(f"[registry] Loaded embodiment_tags from {bench_name} (data_config): {list(mod.ROBOT_TYPE_TO_EMBODIMENT_TAG.keys())}")
                if hasattr(mod, "DATASET_NAMED_MIXTURES"):
                    DATASET_NAMED_MIXTURES.update(mod.DATASET_NAMED_MIXTURES)
                    logger.info(f"[registry] Loaded mixtures from {bench_name} (data_config): {list(mod.DATASET_NAMED_MIXTURES.keys())}")


# Run discovery on first import
discover_and_merge()
