"""Shared runtime defaults for command-line and Qt entry points."""

from __future__ import annotations

import os
from pathlib import Path


def configure_runtime_environment() -> None:
    """Set writable library defaults before imports with side effects."""

    repo_root = Path(__file__).resolve().parent
    configured_mpl_dir = os.environ.get("MPLCONFIGDIR")
    mpl_config_dir = Path(configured_mpl_dir).expanduser() if configured_mpl_dir else repo_root / ".mplconfig"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    try:
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Matplotlib can still fall back to its own temporary cache directory.
        pass
