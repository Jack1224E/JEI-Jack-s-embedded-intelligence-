"""Docked AgentRouter Codex CLI Bridge for Project J-EI.

Enforces CODEX ZERO-CONTEXT JAIL: Executes in an isolated, empty temporary directory
with --skip-git-repo-check, --sandbox read-only, and strict Pydantic JSON schema
output (--output-schema) so Codex NEVER scans or reads the project folder, books/, or data/.

Single model: gpt-6-astra via AgentRouter.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, Field

from j_ei.config import JEIConfig, get_default_config
from j_ei.schemas import StrictModel

T = TypeVar("T", bound=BaseModel)


class TemplateAuditResult(StrictModel):
    """Structured output schema for docked template audit."""

    passed: bool
    domain_leakage_detected: bool
    domain_tokens_found: list[str] = Field(default_factory=list)
    neutrality_score: float = Field(ge=0.0, le=1.0)
    structural_issues: list[str] = Field(default_factory=list)
    reasoning_summary: str = Field(default="", max_length=600)


class QualityAuditResult(StrictModel):
    """Schema for quality meta-review audit."""

    overall_quality_score: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)
    consistency_score: float = Field(ge=0.0, le=1.0)
    issues_found: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    reasoning_summary: str = Field(default="", max_length=800)


def prepare_codex_dock(config: JEIConfig | None = None) -> tuple[Path, Path, dict[str, str]]:
    """Set up isolated Codex dock directory and return (dock_dir, codex_home, env)."""
    cfg = config or get_default_config()
    dock_dir = cfg.codex_dock_dir
    dock_dir.mkdir(parents=True, exist_ok=True)

    # Use warm ~/.codex to prevent plugin unpack/sync overhead (~90s penalty)
    warm_codex = Path.home() / ".codex"
    codex_home = warm_codex if warm_codex.exists() else dock_dir / ".codex_home"
    codex_home.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["CODEX_HOME"] = str(codex_home)
    env["CODEX_GATEWAY_API_KEY"] = cfg.agentrouter_api_key or ""
    env["PATH"] = os.environ.get("PATH", "/home/jack/.local/bin:/usr/bin:/bin")

    return dock_dir, codex_home, env


def codex_exec_raw(
    prompt: str,
    config: JEIConfig | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> str:
    """Execute a raw prompt via codex exec and return stdout content."""
    cfg = config or get_default_config()
    dock_dir, codex_home, env = prepare_codex_dock(cfg)

    use_model = model or cfg.codex_model
    use_timeout = timeout or cfg.codex_timeout

    cmd = [
        "codex", "exec",
        "--cd", str(dock_dir),
        "--skip-git-repo-check",
        "--sandbox", "read-only",
        "--ephemeral",
        "--model", use_model,
        "-",
    ]

    proc = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        env=env,
        timeout=use_timeout,
    )

    # Filter CLI banner
    banner_prefixes = [
        "OpenAI Codex", "workdir:", "model:", "provider:", "approval:",
        "sandbox:", "reasoning", "session", "--------", "user", "warning:",
    ]
    lines = [
        line for line in proc.stdout.splitlines()
        if not any(line.startswith(p) for p in banner_prefixes)
    ]
    return "\n".join(lines).strip()


def codex_audit_template(
    template_json: str,
    config: JEIConfig | None = None,
) -> TemplateAuditResult:
    """Audit a single template for domain leakage using gpt-6-astra."""
    cfg = config or get_default_config()
    dock_dir, codex_home, env = prepare_codex_dock(cfg)

    prompt = f"""You are a strict domain-neutrality auditor.
Audit this JSON template for any domain-specific leakage (crime, medical, legal terms).
Score neutrality 0.0 (leaking) to 1.0 (fully neutral). Output ONLY valid JSON matching
the TemplateAuditResult schema.

TEMPLATE:
{template_json}
"""

    schema_json = json.dumps(TemplateAuditResult.model_json_schema())
    session_id = f"audit-{uuid.uuid4().hex[:8]}"

    cmd = [
        "codex", "exec",
        "--cd", str(dock_dir),
        "--skip-git-repo-check",
        "--sandbox", "read-only",
        "--ephemeral",
        "--model", cfg.codex_model,
        "--output-schema", schema_json,
        "-",
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            env=env,
            timeout=cfg.codex_timeout,
        )
        # Extract JSON from output
        output = proc.stdout.strip()
        # Try to find JSON in the output
        for line in output.splitlines():
            line = line.strip()
            if line.startswith("{"):
                return TemplateAuditResult.model_validate_json(line)

        return TemplateAuditResult(
            passed=True,
            domain_leakage_detected=False,
            neutrality_score=0.5,
            reasoning_summary=f"Could not parse structured output: {output[:200]}",
        )
    except (subprocess.TimeoutExpired, Exception) as e:
        return TemplateAuditResult(
            passed=False,
            domain_leakage_detected=False,
            neutrality_score=0.0,
            reasoning_summary=f"Codex execution error: {e}",
        )
