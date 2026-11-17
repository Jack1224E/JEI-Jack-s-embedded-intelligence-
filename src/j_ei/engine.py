"""Project J-EI Single-Shot Engine: GPT-6-Astra via Codex Gateway + SearXNG.

Executes each query inside an isolated dummy workspace with exactly 4 files:
  AGENTS.md, vault_context.md, query.txt, search_cli.py

Astra uses Codex's native shell harness to run search_cli.py and python3 -c
for math verification — no multi-turn regex loop needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

from j_ei.config import JEIConfig, get_default_config
from j_ei.retriever import (
    load_vault_playbooks,
    playbook_to_context_chunk,
    retrieve_top_k,
)
from j_ei.search import ddg_fallback_search, searxng_search

# ---------------------------------------------------------------------------
# AGENTS.md content — strict operational lockdown for gpt-6-astra
# ---------------------------------------------------------------------------

AGENTS_MD = textwrap.dedent("""\
    # AGENTS.md — Project J-EI Operational Rules

    ## ZERO REPO EXPLORATION
    You are operating in an isolated single-task workspace. Do NOT run `ls`, `find`,
    `tree`, `pwd`, `git`, or inspect parent/sibling directories (`..`). Only read
    `vault_context.md` and `query.txt` in the current working directory.

    ## TREAT EVERY QUERY AS AN ENGINEERING DEBUGGING TASK
    Even if `query.txt` is non-code (e.g., agronomy, physics, operations,
    investigations), apply formal systems engineering discipline:

    1. **Read `vault_context.md` first** to extract hidden failure modes, domain
       mechanics, and 3-4 high-signal investigative angles.

    2. **Execute 2 to 4 surgical web searches** via:
       ```
       python3 search_cli.py "<specific query>"
       ```
       to verify empirical thresholds, constants, and failure post-mortems.
       Do NOT search generic surface words — formulate precise, targeted queries
       derived from the playbook mechanisms.

    3. **SANDBOX MATH & LOGIC VERIFICATION:** Before writing your final response,
       run a quick `python3 -c "..."` command in your shell tool to verify every
       formula, sign convention, unit conversion, and numerical example so there
       are zero arithmetic or dimensional errors.

    4. **INLINE PROS, CONS & FALSIFICATION AUDIT:** Weigh all competing hypotheses
       symmetrically against conservation laws and missing telemetry (flagging
       Goodhart's Law / false nominals). For every proposed diagnostic or
       intervention, explicitly weigh pros, cons, operational costs, and backfire
       risks.

    5. **NO FABRICATED BASELINES:** Never invent sensor readings or sample numbers
       not present in `query.txt` or `search_cli.py` output. Mark missing baselines
       as `[DATA REQUIRED]` and gate irreversible actions behind data acquisition.

    ## EFFICIENCY CAP
    Keep total shell tool calls to ≤ 5 per run. Write the final deliverable
    directly to stdout as your last assistant message.
""")

# ---------------------------------------------------------------------------
# search_cli.py content — standalone CLI helper for SearXNG + DDG
# ---------------------------------------------------------------------------

SEARCH_CLI_PY = textwrap.dedent("""\
    #!/usr/bin/env python3
    \"\"\"Standalone search CLI for Project J-EI.
    Usage: python3 search_cli.py "<query>"
    Returns JSON array of {source, title, url, body} search snippets.
    \"\"\"
    import json
    import re
    import sys
    import urllib.parse
    import urllib.request

    SEARXNG_URL = "{searxng_url}"
    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    def ddg_lite_search(query: str, max_results: int = 4) -> list[dict]:
        snippets = []
        try:
            clean_q = re.sub(r"[^\\w\\s\\-\\.]", " ", query).strip()
            data = urllib.parse.urlencode({"q": clean_q}).encode("utf-8")
            req = urllib.request.Request("https://lite.duckduckgo.com/lite/", data=data, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=4) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            from lxml.html import document_fromstring
            tree = document_fromstring(html)
            rows = tree.xpath("//table[last()]//tr")
            curr = None
            for tr in rows:
                links = tr.xpath('.//a[contains(@class, "result-link")]')
                snips = tr.xpath('.//td[contains(@class, "result-snippet")]')
                if links:
                    href = links[0].get("href", "")
                    if "uddg=" in href:
                        qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        href = qs.get("uddg", [href])[0]
                    curr = {"source": "DuckDuckGo", "title": links[0].text_content().strip(), "url": href, "body": ""}
                elif snips and curr:
                    curr["body"] = snips[0].text_content().strip()
                    snippets.append(curr)
                    curr = None
                    if len(snippets) >= max_results:
                        break
        except Exception:
            pass
        return snippets

    def europepmc_search(query: str, max_results: int = 3) -> list[dict]:
        snippets = []
        try:
            clean_q = re.sub(r"[^\\w\\s\\-\\.]", " ", query).strip()
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode({
                "query": clean_q, "format": "json", "pageSize": str(max_results), "resultType": "core"
            })
            req = urllib.request.Request(url, headers={"User-Agent": "Project-J-EI/2.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("resultList", {}).get("result", []):
                    title = item.get("title", "").rstrip(".")
                    abstract = re.sub(r"<[^>]+>", " ", item.get("abstractText", "")).strip()
                    sid = item.get("source", "MED")
                    eid = item.get("id", "")
                    snippets.append({
                        "source": "EuropePMC",
                        "title": title,
                        "url": f"https://europepmc.org/article/{sid}/{eid}" if eid else "https://europepmc.org",
                        "body": abstract[:350] if abstract else item.get("journalTitle", ""),
                    })
        except Exception:
            pass
        return snippets

    if __name__ == "__main__":
        if len(sys.argv) < 2:
            print("Usage: python3 search_cli.py \\"<query>\\"", file=sys.stderr)
            sys.exit(1)
        query = " ".join(sys.argv[1:])
        results = ddg_lite_search(query)
        if not results:
            results = europepmc_search(query)
        print(json.dumps(results, indent=2))
""")

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    ### SYSTEM DIRECTIVE: PLAYBOOK-GUIDED FORENSIC REASONING & GROUNDING

    You are the core intelligence engine for Project J-EI. Before answering the user's request, execute this internal protocol:

    1. PLAYBOOK-DRIVEN SEARCH PLANNING:
    Analyze the retrieved domain playbooks in `<vault_playbooks>` alongside live empirical search evidence in `<live_grounding_search_results>` below:
    <vault_playbooks>
    {retrieved_playbook_chunks}
    </vault_playbooks>

    <live_grounding_search_results>
    {search_grounding_chunks}
    </live_grounding_search_results>

    Use these playbooks and empirical search findings to identify non-obvious failure modes, exact domain mechanics, physical constants, and empirical thresholds (e.g. osmotic potential, leaching fractions, sensor calibration curves).

    2. INLINE PROS, CONS & FALSIFICATION AUDIT (NO EXTERNAL CRITIC):
    - Weigh all competing hypotheses symmetrically before jumping to conclusions, testing each against physical conservation laws and missing telemetry (flagging any false nominals or Goodhart's Law traps).
    - For every diagnostic step or intervention you propose, explicitly weigh its pros, cons, operational costs, and conditions where it backfires.
    - Check all formulas, signs, and units for strict mathematical consistency.
    - Never invent or hallucinate field sensor readings or baseline numbers not present in the prompt or search results; mark missing baselines as `[DATA REQUIRED]` and prioritize data acquisition over irreversible actions.

    ### USER QUERY:
    {user_query}
""")


# ---------------------------------------------------------------------------
# Engine result
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    """Result from a single J-EI engine run."""
    output: str
    model: str
    wall_clock_seconds: float
    output_chars: int
    output_words: int
    raw_stderr: str = ""
    return_code: int = 0


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

def _provision_workspace(
    user_query: str,
    playbook_chunks: str,
    config: JEIConfig,
) -> Path:
    """Create an isolated dummy workspace with exactly 4 files."""
    base = config.codex_dock_dir
    base.mkdir(parents=True, exist_ok=True)

    workspace = Path(tempfile.mkdtemp(dir=base, prefix="jei_run_"))

    # 1. AGENTS.md
    (workspace / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")

    # 2. vault_context.md
    (workspace / "vault_context.md").write_text(playbook_chunks, encoding="utf-8")

    # 3. query.txt
    (workspace / "query.txt").write_text(user_query, encoding="utf-8")

    # 4. search_cli.py (with SearXNG URL baked in)
    cli_content = SEARCH_CLI_PY.replace("{searxng_url}", config.searxng_url)
    (workspace / "search_cli.py").write_text(cli_content, encoding="utf-8")
    os.chmod(workspace / "search_cli.py", 0o755)

    return workspace


def _build_codex_prompt(user_query: str, playbook_chunks: str, search_chunks: str = "") -> str:
    """Build the full prompt injected into codex exec."""
    grounding = search_chunks.strip() or "No empirical search snippets returned."
    return SYSTEM_PROMPT_TEMPLATE.format(
        retrieved_playbook_chunks=playbook_chunks,
        search_grounding_chunks=grounding,
        user_query=user_query,
    )


def run_jei_query(
    user_query: str,
    config: JEIConfig | None = None,
    alpha_path: Path | None = None,
    top_k: int = 4,
    output_file: Path | None = None,
    cleanup_workspace: bool = True,
) -> EngineResult:
    """Execute a single J-EI query end-to-end.

    1. Retrieve top-k playbook chunks via BAAI/bge-small-en-v1.5.
    2. Retrieve live web grounding snippets via SearXNG.
    3. Provision isolated dummy workspace.
    4. Call codex exec in read-only sandbox.
    5. Capture output, print verification info, save to file.
    """
    cfg = config or get_default_config()
    t_start = time.time()

    # --- Step A: Retrieve playbook chunks ---
    t_retrieval = time.time()
    top_playbooks = retrieve_top_k(
        query=user_query,
        alpha_path=alpha_path,
        k=top_k,
        config=cfg,
    )
    playbook_chunks = "\n\n".join(
        playbook_to_context_chunk(t) for t, _ in top_playbooks
    )
    retrieval_time = time.time() - t_retrieval
    print(f"[J-EI] Step A: Retrieved {len(top_playbooks)} playbook chunks in {retrieval_time:.2f}s", flush=True)
    for t, score in top_playbooks:
        print(f"  - {t.title} (score={score:.3f})", flush=True)

    # --- Step B: Multi-Source Web & Scientific Literature Grounding ---
    t_search = time.time()
    # Extract domain search terms from user query
    clean_words = [
        w.strip(".,;:?!()[]\"'")
        for w in user_query.split()
        if len(w) > 3 and w.lower() not in {
            "this", "that", "with", "from", "into", "across", "about",
            "what", "when", "where", "which", "how", "does", "have",
            "provide", "end-to-end", "framework", "system", "severe", "unexplained"
        }
    ]
    search_query = " ".join(clean_words[:6])
    from j_ei.search import web_search
    raw_snippets = web_search(search_query, config=cfg)

    search_chunks = "\n\n".join(
        f"[{s.get('source', 'Web')}: {s.get('title', 'Snippet')}] ({s.get('url', '')})\n{s.get('body', '')}"
        for s in raw_snippets
    )
    search_time = time.time() - t_search
    print(f"[J-EI] Step B: Retrieved {len(raw_snippets)} live search snippets in {search_time:.2f}s (query: {search_query!r})", flush=True)

    # --- Step C: Provision isolated workspace ---
    workspace = _provision_workspace(user_query, playbook_chunks, cfg)
    print(f"[J-EI] Step C: Provisioned isolated workspace at {workspace}", flush=True)
    print(f"  Files: {sorted(f.name for f in workspace.iterdir())}", flush=True)

    # --- Step D: codex exec single-shot in read-only sandbox ---
    codex_prompt = _build_codex_prompt(user_query, playbook_chunks, search_chunks)

    # Use warm ~/.codex to prevent plugin unpack/sync overhead (~90s penalty)
    warm_codex = Path.home() / ".codex"
    env = dict(os.environ)
    if warm_codex.exists():
        env["CODEX_HOME"] = str(warm_codex)
    env["CODEX_GATEWAY_API_KEY"] = cfg.agentrouter_api_key or ""
    env["PATH"] = os.environ.get("PATH", "/home/jack/.local/bin:/usr/bin:/bin")

    cmd = [
        "codex", "exec",
        "--cd", str(workspace),
        "--skip-git-repo-check",
        "--sandbox", "read-only",
        "--ephemeral",
        "--model", cfg.codex_model,
        "-c", 'model_reasoning_effort="low"',
        "-"
    ]

    print(f"[J-EI] Step D: Launching codex exec --model {cfg.codex_model} (timeout={cfg.codex_timeout}s)...", flush=True)
    t_codex = time.time()

    try:
        proc = subprocess.run(
            cmd,
            input=codex_prompt,
            text=True,
            capture_output=True,
            env=env,
            cwd=str(workspace),
            timeout=cfg.codex_timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[J-EI] ERROR: codex exec timed out after {cfg.codex_timeout}s", flush=True)
        return EngineResult(
            output="ERROR: Codex execution timed out.",
            model=cfg.codex_model,
            wall_clock_seconds=time.time() - t_start,
            output_chars=0,
            output_words=0,
            return_code=-1,
        )

    codex_time = time.time() - t_codex
    total_time = time.time() - t_start

    # Filter out Codex CLI banner lines from stdout
    raw_lines = proc.stdout.splitlines()
    banner_prefixes = [
        "OpenAI Codex", "workdir:", "model:", "provider:", "approval:",
        "sandbox:", "reasoning", "session", "--------", "user", "warning:",
    ]
    content_lines = [
        line for line in raw_lines
        if not any(line.startswith(p) for p in banner_prefixes)
    ]
    output_text = "\n".join(content_lines).strip()

    result = EngineResult(
        output=output_text,
        model=cfg.codex_model,
        wall_clock_seconds=total_time,
        output_chars=len(output_text),
        output_words=len(output_text.split()),
        raw_stderr=proc.stderr or "",
        return_code=proc.returncode,
    )

    # --- Verification output ---
    print(f"\n{'='*72}", flush=True)
    print(f"[J-EI] ✅ EXECUTION COMPLETE", flush=True)
    print(f"  Model:              {result.model}", flush=True)
    print(f"  Return Code:        {result.return_code}", flush=True)
    print(f"  Retrieval Time:     {retrieval_time:.2f}s", flush=True)
    print(f"  Codex Exec Time:    {codex_time:.2f}s", flush=True)
    print(f"  Total Wall-Clock:   {total_time:.2f}s", flush=True)
    print(f"  Output Size:        {result.output_chars} chars, {result.output_words} words", flush=True)
    print(f"{'='*72}", flush=True)

    # Save to file
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(output_text, encoding="utf-8")
        print(f"[J-EI] Saved output to {output_file}", flush=True)

    # Cleanup
    if cleanup_workspace:
        shutil.rmtree(workspace, ignore_errors=True)

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m j_ei.engine '<user query>'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = run_jei_query(
        user_query=query,
        output_file=Path("docs/jei_astra_output.md"),
    )

    print(f"\n--- DELIVERABLE ({result.output_chars} chars) ---")
    print(result.output)

