"""Project J-EI configuration, API key resolution, and local embedding.

Single-agent GPT-6-Astra architecture via AgentRouter Codex Gateway.
Local embeddings via BAAI/bge-small-en-v1.5 (sentence-transformers).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

# Silence benign third-party library warning banners
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import dotenv
import numpy as np
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

InputType = Literal["passage", "query"]


def parse_env_file_safely(env_path: str | Path = ".env") -> dict[str, str]:
    """Parse .env file supporting both '=' and ':' separators and 'Bearer ' prefixes.

    Does NOT modify the file and never exposes raw secrets.
    """
    path = Path(env_path)
    if not path.is_file():
        found = dotenv.find_dotenv(str(env_path))
        if found:
            path = Path(found)
        else:
            return {}

    entries: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sep = "=" if "=" in line else (":" if ":" in line else None)
            if sep:
                k, v = line.split(sep, 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if v.startswith("Bearer "):
                    v = v[7:].strip()
                if k:
                    entries[k] = v
    return entries


def resolve_agentrouter_api_key(
    env_dict: dict[str, str] | None = None,
    env_file: str | Path | None = ".env",
) -> str | None:
    """Resolve AgentRouter API key for docked Codex CLI from environment or .env."""
    clean_dict: dict[str, str] = {}
    if env_dict:
        for k, v in env_dict.items():
            val = v.strip().strip("\"'")
            if val.startswith("Bearer "):
                val = val[7:].strip()
            clean_dict[k] = val

    sources: dict[str, str] = {}
    if env_file:
        sources.update(parse_env_file_safely(env_file))
    sources.update(os.environ)

    search_dicts = [clean_dict, sources] if clean_dict else [sources]

    for d in search_dicts:
        for k, v in d.items():
            if not v:
                continue
            norm_k = re.sub(r"[-.]", "_", k.upper())
            if "AGENTROUTER" in norm_k or "CODEX_GATEWAY" in norm_k:
                return v

    return None


class JEIConfig(BaseSettings):
    """Centralized configuration for Project J-EI single-agent Astra architecture."""

    model_config = SettingsConfigDict(extra="ignore")

    # AgentRouter / Codex Gateway
    agentrouter_api_key: str | None = None
    agentrouter_base_url: str = Field(default="https://agentrouter.org/v1")
    codex_model: str = Field(default_factory=lambda: os.getenv("CODEX_MODEL", "gpt-5.6-sol"))
    codex_dock_dir: Path = Field(default=Path("/tmp/j_ei_codex_dock"))
    codex_timeout: int = Field(default=180)

    # Local embedding (BAAI/bge-small-en-v1.5 via sentence-transformers)
    embed_model_name: str = Field(default="BAAI/bge-small-en-v1.5")
    embed_dim: int = Field(default=384)

    # SearXNG
    searxng_url: str = Field(default="http://localhost:8888")

    def model_post_init(self, __context: Any) -> None:
        if not self.agentrouter_api_key:
            self.agentrouter_api_key = resolve_agentrouter_api_key()

    def get_api_key(self) -> str | None:
        return self.agentrouter_api_key


# Backward-compat alias
AppConfig = JEIConfig

_default_config: JEIConfig | None = None


def get_default_config() -> JEIConfig:
    global _default_config
    if _default_config is None:
        _default_config = JEIConfig()
    return _default_config


def reset_default_config() -> None:
    """Reset cached config (useful in tests)."""
    global _default_config
    _default_config = None


# ---------------------------------------------------------------------------
# Local embedding via sentence-transformers
# ---------------------------------------------------------------------------

_embed_model = None


def _get_embed_model(model_name: str = "BAAI/bge-small-en-v1.5"):
    """Lazy-load sentence-transformers model (cached after first call)."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer(model_name)
    return _embed_model


def normalize_mrl_vector(vec: list[float], target_dim: int = 384) -> list[float]:
    """Slice vector to target_dim and apply L2 normalization.

    Raises ValueError on zero-norm vectors or invalid dimensions.
    """
    if target_dim <= 0:
        raise ValueError(f"target_dim must be positive, got {target_dim}")
    if not vec:
        raise ValueError("Cannot normalize empty vector")

    arr = np.asarray(vec[:target_dim], dtype=np.float64)
    if arr.size < target_dim:
        raise ValueError(
            f"Vector dimension {arr.size} is smaller than requested target_dim {target_dim}"
        )

    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or np.isnan(norm) or norm < 1e-12:
        raise ValueError("Zero-norm or invalid vector cannot be normalized")

    return (arr / norm).tolist()


def embed_texts(
    texts: list[str],
    input_type: InputType,
    target_dim: int = 384,
    batch_size: int = 32,
    config: JEIConfig | None = None,
    **kwargs,
) -> list[list[float]]:
    """Embed texts using BAAI/bge-small-en-v1.5 with L2 normalization.

    For BGE models, queries should be prefixed with 'Represent this sentence: '
    but sentence-transformers handles this via the model's prompt configuration.
    """
    if not texts:
        return []
    if input_type not in ("passage", "query"):
        raise ValueError(f"input_type must be 'passage' or 'query', got {input_type!r}")

    cfg = config or get_default_config()
    model = _get_embed_model(cfg.embed_model_name)

    # BGE models benefit from query prefix for retrieval
    if input_type == "query":
        prefixed = [f"Represent this sentence: {t}" for t in texts]
    else:
        prefixed = texts

    embeddings = model.encode(
        prefixed,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    results: list[list[float]] = []
    for emb in embeddings:
        vec = emb.tolist()
        # Already L2-normed by sentence-transformers, but truncate to target_dim
        results.append(vec[:target_dim])

    return results
