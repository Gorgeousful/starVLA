from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _resolve_config_yaml(config_yaml: str | Path | None) -> Path | None:
    if config_yaml is None:
        return None

    config_path = Path(config_yaml).expanduser()
    if config_path.is_absolute():
        return config_path
    return Path.cwd() / config_path


def snapshot_data_registry(
    config_yaml: str | Path | None,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path | None:
    """Freeze the training data registry next to the run checkpoint files."""
    config_path = _resolve_config_yaml(config_yaml)
    if config_path is None:
        logger.warning("Skip data registry snapshot: cfg.config_yaml is missing.")
        return None

    src_file = config_path.parent / "data_registry" / "data_config.py"
    if not src_file.is_file():
        logger.warning("Skip data registry snapshot: %s does not exist.", src_file)
        return None

    dst_dir = Path(output_dir) / "data_registry"
    dst_file = dst_dir / "data_config.py"
    if dst_file.exists() and not overwrite:
        logger.info("Keeping existing data registry snapshot at %s", dst_file)
        return dst_file

    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    logger.info("Saved data registry snapshot: %s -> %s", src_file, dst_file)
    return dst_file
