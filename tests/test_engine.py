"""Test suite for single-shot GPT-6-Astra engine, retriever, and search modules."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from j_ei.config import JEIConfig
from j_ei.engine import (
    AGENTS_MD,
    SEARCH_CLI_PY,
    SYSTEM_PROMPT_TEMPLATE,
    EngineResult,
    _build_codex_prompt,
    _provision_workspace,
)
from j_ei.retriever import (
    load_vault_playbooks,
    playbook_to_context_chunk,
    playbook_to_text,
    retrieve_top_k,
)
from j_ei.schemas import ForwardThoughtTemplate
from j_ei.search import ddg_fallback_search, searxng_search, web_search


# =====================================================================
# 1. Search Module Tests
# =====================================================================

def test_ddg_fallback_search() -> None:
    """Verify DDG fallback returns structured snippets without crashing."""
    results = ddg_fallback_search("photosynthesis")
    assert isinstance(results, list)
    if results:
        assert "title" in results[0]
        assert "url" in results[0]
        assert "body" in results[0]


def test_searxng_search_offline_graceful() -> None:
    """Verify SearXNG search handles offline server gracefully (returns empty list)."""
    results = searxng_search("test query", searxng_url="http://localhost:99999", timeout=0.5)
    assert results == []


def test_web_search_fallback() -> None:
    """Verify web_search falls back to DDG when SearXNG is unavailable."""
    with patch("j_ei.search.searxng_search", return_value=[]):
        with patch("j_ei.search.ddg_fallback_search", return_value=[{"title": "T", "url": "U", "body": "B"}]):
            results = web_search("test query")
            assert len(results) == 1
            assert results[0]["title"] == "T"


# =====================================================================
# 2. Retriever Module Tests
# =====================================================================

def test_load_vault_playbooks() -> None:
    """Verify loading of the 26 Vault Alpha masters."""
    alpha_path = Path("data/templates/masters/vault_alpha_masters.jsonl")
    if alpha_path.exists():
        playbooks = load_vault_playbooks(alpha_path)
        assert len(playbooks) == 26
        assert all(isinstance(p, ForwardThoughtTemplate) for p in playbooks)


def test_playbook_to_text_and_chunk() -> None:
    """Verify playbook serialization for retrieval and context chunking."""
    alpha_path = Path("data/templates/masters/vault_alpha_masters.jsonl")
    if alpha_path.exists():
        playbooks = load_vault_playbooks(alpha_path)
        pb = playbooks[0]

        text = playbook_to_text(pb)
        assert pb.title in text
        assert pb.cognitive_function in text

        chunk = playbook_to_context_chunk(pb)
        assert f"### PLAYBOOK: {pb.title}" in chunk
        assert pb.retrieval.situation_signature in chunk


# =====================================================================
# 3. Workspace Provisioning & Isolation Tests
# =====================================================================

def test_provision_workspace_exact_four_files(tmp_path: Path) -> None:
    """Verify workspace provisioning creates exactly the 4 required files."""
    cfg = JEIConfig(codex_dock_dir=tmp_path / "dock")
    workspace = _provision_workspace(
        user_query="How to diagnose agricultural wilt?",
        playbook_chunks="### PLAYBOOK: test\nSome details",
        config=cfg,
    )

    assert workspace.exists()
    files = sorted(f.name for f in workspace.iterdir() if f.is_file())
    assert files == ["AGENTS.md", "query.txt", "search_cli.py", "vault_context.md"]

    # Verify AGENTS.md content
    agents_text = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert "ZERO REPO EXPLORATION" in agents_text
    assert "TREAT EVERY QUERY AS AN ENGINEERING DEBUGGING TASK" in agents_text
    assert "SANDBOX MATH & LOGIC VERIFICATION" in agents_text
    assert "NO FABRICATED BASELINES" in agents_text
    assert "EFFICIENCY CAP" in agents_text

    # Verify query.txt
    assert (workspace / "query.txt").read_text(encoding="utf-8") == "How to diagnose agricultural wilt?"

    # Verify vault_context.md
    assert "### PLAYBOOK: test" in (workspace / "vault_context.md").read_text(encoding="utf-8")

    # Verify search_cli.py is executable
    search_cli = workspace / "search_cli.py"
    assert search_cli.exists()
    assert (search_cli.stat().st_mode & 0o111) != 0


def test_build_codex_prompt() -> None:
    """Verify prompt embeds the exact required system directive and query."""
    prompt = _build_codex_prompt("Sample query", "Sample chunks")
    assert "### SYSTEM DIRECTIVE: PLAYBOOK-GUIDED FORENSIC REASONING & GROUNDING" in prompt
    assert "1. PLAYBOOK-DRIVEN SEARCH PLANNING:" in prompt
    assert "2. INLINE PROS, CONS & FALSIFICATION AUDIT (NO EXTERNAL CRITIC):" in prompt
    assert "<vault_playbooks>" in prompt
    assert "Sample chunks" in prompt
    assert "</vault_playbooks>" in prompt
    assert "### USER QUERY:" in prompt
    assert "Sample query" in prompt


def test_engine_result_dataclass() -> None:
    res = EngineResult(
        output="Test output",
        model="gpt-6-astra",
        wall_clock_seconds=12.5,
        output_chars=11,
        output_words=2,
    )
    assert res.output == "Test output"
    assert res.model == "gpt-6-astra"
    assert res.wall_clock_seconds == 12.5
    assert res.return_code == 0

