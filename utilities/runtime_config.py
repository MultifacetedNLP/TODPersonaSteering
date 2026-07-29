from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
REPO_ROOT = Path(os.environ.get("REPO_ROOT", PROJECT_ROOT.parent)).resolve()


def require_env(name: str, purpose: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"{name} is required for {purpose}. Set it in the environment or in "
            f"{PROJECT_ROOT / '.env'}."
        )
    return value


def hf_cache_dir() -> Path:
    return Path(os.environ.get("HF_CACHE_DIR", PROJECT_ROOT / "hf_cache")).resolve()


def persona_repo_root() -> Path:
    return Path(os.environ.get("PERSONA_REPO_ROOT", REPO_ROOT / "persona")).resolve()


def model_vector_dir(model_name: str) -> Path:
    return persona_repo_root() / "vectors" / model_name.split("/")[-1]
