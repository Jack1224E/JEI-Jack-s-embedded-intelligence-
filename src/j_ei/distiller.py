"""Knowledge Distillation Engine for Project J-EI.

Implements type-aware sliding windowing across parsed book ASTs, front-matter/fluff filtering,
GPT-6-Astra-powered structured template extraction (Vault Alpha ForwardThoughtTemplate vs Vault Omega
CriticThoughtTemplate), cross-domain analog vignettes, provenance quote verification via rapidfuzz,
ChapterState thread-closure joins across window boundaries, strict BANNED_LEXICON domain neutralization,
adversarial docked Codex audit enforcement (gpt-6.1-sol) with targeted 1-shot repair,
>15% drop-rate circuit breaker, and incremental checkpointing.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from j_ei.codex_bridge import codex_audit_template
from j_ei.codex_bridge import codex_exec_raw
from j_ei.config import JEIConfig, get_default_config
from j_ei.linter import verify_quote_anchor
from j_ei.parser import classify_chapter
from j_ei.schemas import (
    BANNED_LEXICON,
    ChapterState,
    CriticThoughtTemplate,
    ForwardThoughtTemplate,
    SourceAnchor,
    VulnerabilityCode,
)

logger = logging.getLogger("j_ei.distiller")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

OMEGA_BOOK_IDENTIFIERS = [
    "psychology_of_intelligence",
    "structured_analytic",
]

FRONT_MATTER_PATTERNS = [
    re.compile(r"table of contents|contents at a glance", re.I),
    re.compile(r"all rights reserved|cataloging-in-publication|printed in the united", re.I),
    re.compile(r"isbn(?:-1[03])?:?\s*[\d-]+", re.I),
    re.compile(r"praise for|advance praise|about the author|about the publisher", re.I),
    re.compile(r"cq press, an imprint of sage|paperback isbn|published by", re.I),
    re.compile(r"title page|title\s*:\s*|edition\s*pb printing|john wiley & sons", re.I),
    re.compile(r"library of congress|a standard system for investigating", re.I),
]

PROCEDURE_KEYWORDS = re.compile(
    r"\b(step\s+\d+|procedure|technique|method|heuristic|hypothesis|hypotheses|"
    r"diagnostic|debiasing|vulnerability|falsif|indicator|protocol|tradecraft|"
    r"externaliz|system\s+1|system\s+2|analyst|bias|assumption)\b",
    re.I,
)

SEMANTIC_SCRUB_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(murders|murder|homicides|homicide)\b", re.IGNORECASE), "terminal_system_breach"),
    (re.compile(r"\b(serials|serial)\b", re.IGNORECASE), "recurrent_patterned"),
    (re.compile(r"\b(psychopaths|psychopath)\b", re.IGNORECASE), "zero_empathy_rational_actor"),
    (re.compile(r"\bvictims\b", re.IGNORECASE), "target_entities"),
    (re.compile(r"\bvictim\b", re.IGNORECASE), "target_entity"),
    (re.compile(r"\boffenders\b", re.IGNORECASE), "adversarial_actors"),
    (re.compile(r"\boffender\b", re.IGNORECASE), "adversarial_actor"),
    (re.compile(r"\bcrimes\b", re.IGNORECASE), "anomalous_incidents"),
    (re.compile(r"\bcrime\b", re.IGNORECASE), "anomalous_incident"),
    (re.compile(r"\b(polices|police|detectives|detective)\b", re.IGNORECASE), "lead_investigator"),
    (re.compile(r"\b(autopsies|autopsy)\b", re.IGNORECASE), "post_incident_inspection"),
    (re.compile(r"\b(rapes|rape|assaults|assault)\b", re.IGNORECASE), "severe_boundary_breach"),
    (re.compile(r"\b(arsons|arson)\b", re.IGNORECASE), "deliberate_infrastructure_sabotage"),
    (re.compile(r"\b(burglaries|burglary)\b", re.IGNORECASE), "unauthorized_perimeter_breach"),
    (re.compile(r"\b(felonies|felony)\b", re.IGNORECASE), "high_severity_violation"),
    (re.compile(r"\bsuspects\b", re.IGNORECASE), "candidate_actors"),
    (re.compile(r"\bsuspect\b", re.IGNORECASE), "candidate_actor"),
]


def find_banned_words_in_payload(payload: dict[str, Any]) -> list[str]:
    """Find all banned lexicon words in non-provenance fields of payload."""
    non_prov = {k: v for k, v in payload.items() if k != "provenance"}
    text = json.dumps(non_prov)
    matches = BANNED_LEXICON.findall(text)
    seen = set()
    res = []
    for m in matches:
        m_low = m.lower()
        if m_low not in seen:
            seen.add(m_low)
            res.append(m_low)
    return res


def apply_semantic_scrub_to_text(text: str) -> str:
    """Apply deterministic semantic scrub map to string."""
    for pattern, repl in SEMANTIC_SCRUB_MAP:
        text = pattern.sub(repl, text)
    return text


def apply_semantic_scrub_to_obj(obj: Any) -> Any:
    """Recursively scrub strings in non-provenance dicts/lists."""
    if isinstance(obj, str):
        return apply_semantic_scrub_to_text(obj)
    elif isinstance(obj, list):
        return [apply_semantic_scrub_to_obj(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: apply_semantic_scrub_to_obj(v) for k, v in obj.items()}
    return obj


def apply_semantic_scrub_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Scrub all fields of a template payload except provenance."""
    scrubbed = {}
    for k, v in payload.items():
        if k == "provenance":
            scrubbed[k] = v
        else:
            scrubbed[k] = apply_semantic_scrub_to_obj(v)
    return scrubbed


@dataclass
class DistillationWindow:
    """A bounded processing window for knowledge distillation."""

    window_id: str
    book_id: str
    chapter_id: str
    chapter_title: str
    chapter_type: Literal["TAXONOMIC", "NARRATIVE", "HYBRID"]
    heading_path: str
    window_index: int
    paragraphs: list[dict[str, Any]]
    window_text: str
    est_tokens: int
    char_span: tuple[int, int]
    is_taxonomic: bool


def is_omega_vault_book(book_id: str) -> bool:
    """Return True if book should distill into Vault Omega (CriticThoughtTemplate)."""
    bid = book_id.lower()
    return any(k in bid for k in OMEGA_BOOK_IDENTIFIERS)


def is_front_matter_or_fluff(window: DistillationWindow) -> bool:
    """Detect if window text is front-matter/blurbs/TOC lacking procedural analytical content."""
    text = window.window_text
    has_proc = bool(PROCEDURE_KEYWORDS.search(text))

    # Short windows without procedure keywords are fluff
    if window.est_tokens < 120 and not has_proc:
        return True

    if len(text.strip()) < 250 and not has_proc:
        return True

    has_fm_markers = any(p.search(text) for p in FRONT_MATTER_PATTERNS)
    # If it contains copyright/TOC/blurbs and lacks analytical procedures, skip it
    if has_fm_markers and not has_proc:
        return True

    # If heading clearly indicates bibliography or index
    h_lower = window.heading_path.lower()
    if any(k in h_lower for k in ("table of contents", "index", "bibliography", "author index", "subject index")):
        if not has_proc:
            return True

    return False


def load_ast_paragraphs(ast_path: Path) -> list[dict[str, Any]]:
    """Load pre-parsed paragraph AST dictionaries from a JSONL file."""
    if not ast_path.exists():
        raise FileNotFoundError(f"AST file not found: {ast_path}")
    paragraphs = []
    with open(ast_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                paragraphs.append(json.loads(line_str))
    return paragraphs


def create_chapter_windows(
    book_id: str,
    chapter_id: str,
    chapter_title: str,
    paragraphs: list[dict[str, Any]],
    chapter_type: Literal["TAXONOMIC", "NARRATIVE", "HYBRID"],
    max_taxonomic_tokens: int = 3000,
    narrative_window_tokens: int = 6000,
    narrative_stride_tokens: int = 5000,
    start_window_index: int = 0,
) -> list[DistillationWindow]:
    """Generate type-aware windows for a single chapter."""
    if not paragraphs:
        return []

    windows: list[DistillationWindow] = []
    cur_window_idx = start_window_index

    if chapter_type == "TAXONOMIC":
        cur_paras: list[dict[str, Any]] = []
        cur_tokens = 0
        cur_heading = paragraphs[0].get("heading_path", chapter_title)

        for p in paragraphs:
            p_tokens = p.get("est_tokens", int(len(p.get("text", "")) / 3.8))
            if cur_paras and (cur_tokens + p_tokens > max_taxonomic_tokens):
                window_text = f"# Context: {cur_heading}\n\n" + "\n\n".join(
                    f"[{item['pid']} {item['locator']}]: {item['text']}" for item in cur_paras
                )
                min_char = cur_paras[0]["char_span"][0]
                max_char = cur_paras[-1]["char_span"][1]
                windows.append(
                    DistillationWindow(
                        window_id=f"{book_id}_{chapter_id}_w{cur_window_idx:03d}",
                        book_id=book_id,
                        chapter_id=chapter_id,
                        chapter_title=chapter_title,
                        chapter_type=chapter_type,
                        heading_path=cur_heading,
                        window_index=cur_window_idx,
                        paragraphs=list(cur_paras),
                        window_text=window_text,
                        est_tokens=cur_tokens,
                        char_span=(min_char, max_char),
                        is_taxonomic=True,
                    )
                )
                cur_window_idx += 1
                cur_paras = []
                cur_tokens = 0
                cur_heading = p.get("heading_path", chapter_title)

            cur_paras.append(p)
            cur_tokens += p_tokens
            if p.get("heading_path"):
                cur_heading = p["heading_path"]

        if cur_paras:
            window_text = f"# Context: {cur_heading}\n\n" + "\n\n".join(
                f"[{item['pid']} {item['locator']}]: {item['text']}" for item in cur_paras
            )
            min_char = cur_paras[0]["char_span"][0]
            max_char = cur_paras[-1]["char_span"][1]
            windows.append(
                DistillationWindow(
                    window_id=f"{book_id}_{chapter_id}_w{cur_window_idx:03d}",
                    book_id=book_id,
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    chapter_type=chapter_type,
                    heading_path=cur_heading,
                    window_index=cur_window_idx,
                    paragraphs=list(cur_paras),
                    window_text=window_text,
                    est_tokens=cur_tokens,
                    char_span=(min_char, max_char),
                    is_taxonomic=True,
                )
            )

    else:
        # NARRATIVE / HYBRID sliding window
        start_idx = 0
        total_p = len(paragraphs)

        while start_idx < total_p:
            acc_tokens = 0
            end_idx = start_idx
            while end_idx < total_p and (
                acc_tokens + paragraphs[end_idx].get("est_tokens", 10) <= narrative_window_tokens
                or end_idx == start_idx
            ):
                acc_tokens += paragraphs[end_idx].get("est_tokens", 10)
                end_idx += 1

            window_paras = paragraphs[start_idx:end_idx]
            heading_crumb = window_paras[0].get("heading_path", chapter_title)
            window_text = (
                f"# Context: {chapter_title} (Sliding Window {cur_window_idx})\n\n"
                + "\n\n".join(
                    f"[{item['pid']} {item['locator']}]: {item['text']}" for item in window_paras
                )
            )
            min_char = window_paras[0]["char_span"][0]
            max_char = window_paras[-1]["char_span"][1]

            windows.append(
                DistillationWindow(
                    window_id=f"{book_id}_{chapter_id}_w{cur_window_idx:03d}",
                    book_id=book_id,
                    chapter_id=chapter_id,
                    chapter_title=chapter_title,
                    chapter_type=chapter_type,
                    heading_path=heading_crumb,
                    window_index=cur_window_idx,
                    paragraphs=list(window_paras),
                    window_text=window_text,
                    est_tokens=acc_tokens,
                    char_span=(min_char, max_char),
                    is_taxonomic=False,
                )
            )
            cur_window_idx += 1

            if end_idx >= total_p:
                break

            stride_acc = 0
            next_start = start_idx
            while (
                next_start < end_idx
                and stride_acc + paragraphs[next_start].get("est_tokens", 10) <= narrative_stride_tokens
            ):
                stride_acc += paragraphs[next_start].get("est_tokens", 10)
                next_start += 1

            if next_start == start_idx:
                next_start = start_idx + 1
            start_idx = next_start

    return windows


def generate_book_windows(
    book_id: str,
    ast_path: Path,
    max_taxonomic_tokens: int = 3000,
    narrative_window_tokens: int = 6000,
    narrative_stride_tokens: int = 5000,
) -> list[DistillationWindow]:
    """Load AST and generate complete sequential windows across all chapters."""
    paragraphs = load_ast_paragraphs(ast_path)
    if not paragraphs:
        return []

    chapters_order: list[str] = []
    chapters_map: dict[str, list[dict[str, Any]]] = {}
    for p in paragraphs:
        cid = p.get("chapter_id", "ch_01")
        if cid not in chapters_map:
            chapters_order.append(cid)
            chapters_map[cid] = []
        chapters_map[cid].append(p)

    all_windows: list[DistillationWindow] = []
    window_counter = 0

    for cid in chapters_order:
        cparas = chapters_map[cid]
        ctitle = cparas[0].get("chapter_title", cid)
        ctype = classify_chapter(cparas)

        ch_windows = create_chapter_windows(
            book_id=book_id,
            chapter_id=cid,
            chapter_title=ctitle,
            paragraphs=cparas,
            chapter_type=ctype,
            max_taxonomic_tokens=max_taxonomic_tokens,
            narrative_window_tokens=narrative_window_tokens,
            narrative_stride_tokens=narrative_stride_tokens,
            start_window_index=window_counter,
        )
        all_windows.extend(ch_windows)
        window_counter += len(ch_windows)

    return all_windows


def build_extraction_prompt(
    window: DistillationWindow,
    is_omega: bool,
    chapter_state: ChapterState | None = None,
) -> str:
    """Build the distillation prompt for a window enforcing strict neutrality."""
    state = chapter_state or ChapterState()
    open_threads_context = ""
    if state.open_threads:
        open_threads_context = f"\nACTIVE OPEN THREADS FROM EARLIER WINDOWS:\n{json.dumps(state.open_threads, indent=2)}\n"

    target_type = "CriticThoughtTemplate (Vault Omega)" if is_omega else "ForwardThoughtTemplate (Vault Alpha)"
    vulnerabilities = [v.value for v in VulnerabilityCode]

    prompt = f"""You are a master cognitive methodologist distilling raw technical and investigative literature into reusable, domain-agnostic reasoning patterns for Project J-EI.

TARGET OUTPUT SCHEMA: {target_type}

======================================================================
CORE COGNITIVE SPECIFICATION & EXTRACTION RULES
======================================================================

1. RULE A — TRIAGE FILTER (OMIT VS. CONVERT):
   Raw text contains two distinct buckets. You must classify each section before extracting:
   - BUCKET 1: OMIT IMMEDIATELY (Do NOT extract; ONLY if the entire window is purely Bucket 1, return {{"templates": []}}):
     * Purely physical, biological, anatomical, or medical autopsy descriptions (wounds, tissue trauma, decomposition, blood mechanics).
     * Statutory legal penal codes, jurisdictional statutes, court sentencing guidelines, or legal paperwork.
     * Front-matter fluff, publishing notices, copyrights, blurbs, tables of contents, preface acknowledgments.
     * Narrative trivia or specific historical anecdotes that lack a reusable heuristic.
   - BUCKET 2: CONVERT INTO UNIVERSAL SYSTEMS REASONING (ACTIVE EXTRACTION MANDATED):
     * Do NOT be over-hesitant. Whenever the text explains behavioral consistency, distinguishing capability/planning (M.O.) from invariant motive/incentives (Signature), detecting staged anomalies and deceptive telemetry, or correlating disparate incidents across time/space, this IS high-value analytical methodology (Bucket 2)!
     * Synthesize these concepts into actionable multi-step diagnostic procedures (`procedure_steps`, `diagnostic_probes`). You DO NOT need literal numbered steps in the source text; you are formulating the underlying cognitive heuristic.
     * Abstract these procedures into domain-agnostic heuristics applicable to distributed computing reliability, corporate financial audit, market microstructure anomalies, supply-chain bottlenecks, or adversarial red-teaming.

2. RULE B — MANDATORY STRUCTURAL ISOMORPHISM TRANSLATION TABLE:
   You MUST translate domain-specific forensic mechanisms into their structural equivalents across all template fields:
   * Offender / Perpetrator / Suspect -> "adversarial_actor", "originating_agent", or "candidate_source"
   * Victim / Target -> "target_entity", "impacted_subsystem", or "exposed_node"
   * Crime Scene / Offense / Homicide / Arson / Burglary -> "incident_environment", "anomalous_breach", "system_failure_event", or "unauthorized_intrusion"
   * Modus Operandi (M.O.) / Preparation -> "execution_sequence", "resource_sourcing_chain", "capability_footprint", or "step_by_step_planning_order"
   * Signature / Motive / Fantasy / Intent -> "invariant_behavioral_signal", "core_incentive_driver", or "intent_attribution"
   * Staging / Red Herring / Cleanup -> "deliberate_artifact_manipulation", "decoy_injection", or "anti-forensic_obfuscation"
   * Police / Detective / Autopsy -> "system_auditor", "lead_investigator", or "post_incident_root_cause_inspection"

3. RULE C — STRICT FIELD SEPARATION & ZERO DOMAIN LEAKAGE:
   * PROVENANCE EXCLUSION (CRITICAL): `provenance[].verbatim_quote` is 100% EXEMPT from BANNED_LEXICON. It MUST be an exact verbatim substring copied directly from the passage text (up to 180 chars) to satisfy cryptographic/rapidfuzz verification, EVEN IF it contains words like 'crime', 'homicide', 'offender', 'victim', or 'police'.
   * REASONING FIELDS (ZERO LEAKAGE): Every other field (`title`, `cognitive_function`, `retrieval` (`situation_signature`, `trigger_conditions`, `symptom_phrases`, `information_structure`, `analog_vignettes`), `preconditions`, `contraindications`, `procedure_steps`, `diagnostic_probes`, `never_asserts`) MUST be 100% domain-neutral and contain ZERO occurrences of the BANNED_LEXICON words:
     [murder, homicide, serial, psychopath, victim, offender, crime, police, detective, autopsy, rape, arson, burglary, assault, felony, suspect].

4. CROSS-DOMAIN ANALOG VIGNETTES:
   Every ForwardThoughtTemplate MUST provide 4-6 analog_vignettes demonstrating structural utility across diverse non-forensic domains:
   (a) Distributed systems telemetry debugging (e.g. unpredicted queue buildup, thread pool starvation)
   (b) Financial market anomalies (e.g. hidden liquidity absorption, spoofing, order-book manipulation)
   (c) Organizational incentives & counterparty risk (e.g. procurement collusion, selective log disclosure)
   (d) Physical logistics & sensor telemetry (e.g. transit transponder anomalies, sensor spoofing)

5. STRICT JSON OUTPUT:
   Output ONLY valid, parseable JSON matching the schema below. No markdown fences, no explanatory commentary outside the JSON structure.
{open_threads_context}
"""

    if is_omega:
        prompt += f"""
JSON SCHEMA TO RETURN:
{{
  "templates": [
    {{
      "schema_version": "2.1",
      "template_id": "unique_snake_case_id",
      "technique_name": "Concise Descriptive Title",
      "cost_tier": "always_on" | "routed" | "heavy",
      "target_vulnerabilities": {json.dumps(vulnerabilities[:4])},
      "detectable_symptoms": ["symptom 1", "symptom 2", "symptom 3"],
      "audit_procedure": ["step 1", "step 2"],
      "kill_conditions": ["kill condition 1"],
      "remediation_actions": ["requery_knowledge" | "requery_empirical" | "reweight_hypotheses" | "downgrade_modality" | "drop_claim"],
      "known_limitations": ["limitation 1"],
      "provenance": [
        {{
          "book_id": "{window.book_id}",
          "chapter": "{window.chapter_id}",
          "locator": "page_X",
          "char_span": [0, 50],
          "source_span_sha256": "sha256_of_paragraph",
          "verbatim_quote": "exact sentence excerpt from text"
        }}
      ]
    }}
  ],
  "detected_open_threads": [
    {{
      "thread_id": "thread_id_str",
      "description": "brief description of unresolved investigative setup or hypothesis",
      "setup_anchor": {{
        "book_id": "{window.book_id}",
        "chapter": "{window.chapter_id}",
        "locator": "page_X",
        "char_span": [0, 50],
        "source_span_sha256": "sha256_hash",
        "verbatim_quote": "quote"
      }}
    }}
  ],
  "resolved_thread_ids": [],
  "entities_discovered": {{}},
  "glossary_deltas": {{}}
}}
"""
    else:
        prompt += f"""
JSON SCHEMA TO RETURN:
{{
  "templates": [
    {{
      "schema_version": "2.1",
      "template_id": "unique_snake_case_id",
      "title": "Concise Descriptive Title",
      "tier": "a0_systems_core" | "a1_empirical_anomaly" | "a2_adversarial_game" | "a3_spatial_niche",
      "cognitive_function": "abductive_hypothesis_generation" | "baseline_anomaly_separation" | "verifiability_deception_audit" | "incentive_motive_mapping" | "sequence_contradiction_matrix" | "bottleneck_constraint_isolation" | "risk_premortem_projection",
      "situation_frame": {{
        "actors": ["decision_maker", "counterparty"],
        "info_asymmetry": "one_sided_hidden",
        "evidence_state": ["direct_telemetry", "circumstantial_pattern"],
        "incentive_posture": "competitive_rational",
        "system_dynamics": "iterative_feedback",
        "primary_uncertainty": ["root_cause", "actor_intent"],
        "spatial_or_forensic_flag": false
      }},
      "retrieval": {{
        "situation_signature": "generalized domain-neutral description of the operational setting",
        "trigger_conditions": ["condition 1", "condition 2"],
        "symptom_phrases": ["symptom phrase 1", "symptom phrase 2"],
        "information_structure": "neutral data topology",
        "analog_vignettes": [
          "Distributed systems telemetry showing unpredicted queue buildup despite nominal CPU load.",
          "Financial market order book imbalance indicating hidden institutional liquidity absorption.",
          "Contract negotiation where counterparty selectively discloses audit logs to alter valuation.",
          "OSINT registry contradiction where vessel AIS position conflicts with port terminal records."
        ]
      }},
      "preconditions": ["precondition 1"],
      "contraindications": ["contraindication 1"],
      "procedure_steps": ["step 1", "step 2"],
      "diagnostic_probes": [
        {{
          "question": "probe question?",
          "discriminates_between": ["hypothesis A", "hypothesis B"],
          "verifiable_anchor_sought": "telemetry record",
          "diagnosticity_weight": 0.85
        }}
      ],
      "never_asserts": ["clinical_diagnosis", "unverified_intent_as_fact"],
      "evidence_strength": "empirical_validated" | "expert_consensus" | "practitioner_heuristic",
      "disputed_with": [],
      "merged_from": [],
      "provenance": [
        {{
          "book_id": "{window.book_id}",
          "chapter": "{window.chapter_id}",
          "locator": "page_X",
          "char_span": [0, 50],
          "source_span_sha256": "sha256_of_paragraph",
          "verbatim_quote": "exact sentence excerpt from text"
        }}
      ]
    }}
  ],
  "detected_open_threads": [],
  "resolved_thread_ids": [],
  "entities_discovered": {{}},
  "glossary_deltas": {{}}
}}
"""

    prompt += f"\nPASSAGE EXCERPT TO ANALYZE:\n{window.window_text[:7000]}\n"
    return prompt


def parse_extraction_response(
    content: str | None,
    reasoning: str | None = None,
) -> dict[str, Any]:
    """Parse JSON output from LLM completion, checking content first and falling back to reasoning."""
    candidates = [c for c in (content, reasoning) if c and c.strip()]
    if not candidates:
        return {"templates": []}

    for text in candidates:
        text_clean = text.strip()
        try:
            return json.loads(text_clean, strict=False)
        except Exception:
            pass

        fenced_clean = re.sub(r"^```(?:json)?\s*", "", text_clean)
        fenced_clean = re.sub(r"\s*```$", "", fenced_clean).strip()
        try:
            return json.loads(fenced_clean, strict=False)
        except Exception:
            pass

        start = text_clean.find("{")
        end = text_clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text_clean[start : end + 1], strict=False)
            except Exception:
                pass

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1), strict=False)
            except Exception:
                pass

    snippet = candidates[0][:300].replace("\n", " ")
    logger.warning(
        f"Failed to extract valid JSON from completion response; returning empty payload. Snippet: {snippet!r}"
    )
    return {"templates": []}


def execute_thread_closure_join(
    template: ForwardThoughtTemplate | CriticThoughtTemplate,
    resolved_thread_ids: list[str],
    chapter_state: ChapterState,
) -> ForwardThoughtTemplate | CriticThoughtTemplate:
    """Join setup provenance from active open threads resolved in this window."""
    if not resolved_thread_ids or not chapter_state.open_threads:
        return template

    remaining_threads = []
    resolved_threads = []

    for thread in chapter_state.open_threads:
        tid = thread.get("thread_id")
        if tid in resolved_thread_ids:
            resolved_threads.append(thread)
        else:
            remaining_threads.append(thread)

    chapter_state.open_threads = remaining_threads

    for thread in resolved_threads:
        setup_anchor_dict = thread.get("setup_anchor")
        if setup_anchor_dict:
            try:
                anchor = SourceAnchor.model_validate(setup_anchor_dict)
                if not any(
                    p.verbatim_quote == anchor.verbatim_quote and p.locator == anchor.locator
                    for p in template.provenance
                ):
                    template.provenance.append(anchor)
            except Exception as e:
                logger.debug(f"Could not parse thread setup_anchor: {e}")

        if isinstance(template, ForwardThoughtTemplate):
            tid = thread.get("thread_id")
            if tid and tid not in template.merged_from:
                template.merged_from.append(tid)

        logger.info(
            f"[Thread Closure] Joined setup anchor from resolved thread '{thread.get('thread_id')}' into template '{template.template_id}'."
        )

    return template


def rewrite_contaminated_fields(
    payload: dict[str, Any],
    banned_words: list[str],
    client: Any,
    model: str,
    is_omega: bool,
) -> dict[str, Any] | None:
    """Layer 1: Fast targeted LLM rewrite of contaminated fields to eliminate domain leakage."""
    target_type = "CriticThoughtTemplate" if is_omega else "ForwardThoughtTemplate"
    words_str = ", ".join(f"'{w}'" for w in banned_words)
    prompt = f"""Your previous JSON violated domain-neutrality by using these banned source words: [{words_str}].

Rewrite ONLY the contaminated fields into generalized systems/strategic reasoning (focusing on source identification, intent analysis, resource sourcing, and step-by-step planning order) while keeping `provenance` unchanged.

ORIGINAL EXTRACTED TEMPLATE:
{json.dumps(payload, indent=2)}

INSTRUCTIONS:
1. Translate all forensic mechanisms into generalized systems/investigative reasoning:
   - offender / suspect -> adversarial_actor, originating_agent, candidate_source
   - victim -> target_entity, impacted_subsystem, exposed_node
   - crime / homicide / arson / burglary -> incident_environment, anomalous_breach, unauthorized_intrusion
   - M.O. / preparation -> execution_sequence, resource_sourcing_chain, capability_footprint
   - signature / motive -> invariant_behavioral_signal, core_incentive_driver, intent_attribution
   - staging / cleanup -> deliberate_artifact_manipulation, decoy_injection, anti-forensic_obfuscation
   - police / detective / autopsy -> system_auditor, lead_investigator, post_incident_root_cause_inspection
2. Keep the `provenance` array EXACTLY as it is (do NOT alter verbatim_quote).
3. Output ONLY valid JSON matching the {target_type} schema. No markdown fences or commentary.
"""
    try:
        if client is not None:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a cognitive template repair specialist. Keep internal reasoning concise (under 150 words) and output ONLY valid JSON matching the {target_type} schema.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8192,
                temperature=0.1,
            )
            choice = resp.choices[0]
            content = choice.message.content
            reasoning = getattr(choice.message, "reasoning_content", None)
        else:
            content = codex_exec_raw(prompt)
            reasoning = None

        raw_dict = parse_extraction_response(
            content=content,
            reasoning=reasoning,
        )
        if "templates" in raw_dict and isinstance(raw_dict["templates"], list) and raw_dict["templates"]:
            raw_dict = raw_dict["templates"][0]

        # Ensure provenance is preserved from original
        if "provenance" in payload and payload["provenance"]:
            raw_dict["provenance"] = payload["provenance"]

        return raw_dict
    except Exception as e:
        logger.warning(f"Targeted LLM rewrite call failed ({e}); proceeding to Layer 2 fallback.")
        return None


def normalize_template_fields(payload: dict[str, Any], is_omega: bool) -> dict[str, Any]:
    """Ensure template payload satisfies minimum list length and schema cardinality constraints."""
    res = dict(payload)

    # Normalize provenance quote length (max 180 chars per SourceAnchor schema)
    prov = res.get("provenance", [])
    if isinstance(prov, list):
        for anchor in prov:
            if isinstance(anchor, dict) and "verbatim_quote" in anchor:
                q = anchor["verbatim_quote"]
                if isinstance(q, str) and len(q) > 180:
                    truncated = q[:180]
                    last_space = truncated.rfind(" ")
                    if last_space > 100:
                        anchor["verbatim_quote"] = truncated[:last_space]
                    else:
                        anchor["verbatim_quote"] = truncated
        res["provenance"] = prov
    if is_omega:
        if not res.get("target_vulnerabilities"):
            res["target_vulnerabilities"] = ["CONFIRMATION_SKEW"]
        if len(res.get("detectable_symptoms", [])) < 3:
            syms = list(res.get("detectable_symptoms", []))
            syms.extend(["divergent telemetry timestamps", "missing baseline metrics", "unverified single-source reports"])
            res["detectable_symptoms"] = syms[:6]
        if len(res.get("audit_procedure", [])) < 2:
            proc = list(res.get("audit_procedure", []))
            proc.extend(["Corroborate primary event sequence against independent secondary logs", "Check for absence of counter-evidence in peripheral nodes"])
            res["audit_procedure"] = proc[:4]
        if not res.get("kill_conditions"):
            res["kill_conditions"] = ["Verified independent counter-evidence invalidates core assumption"]
        if not res.get("remediation_actions"):
            res["remediation_actions"] = ["requery_empirical"]
        if not res.get("known_limitations"):
            res["known_limitations"] = ["Depends on timestamp accuracy across distributed logs"]
    else:
        # Normalize situation_frame
        sf = res.get("situation_frame", {})
        if isinstance(sf, dict):
            VALID_ACTORS = {"decision_maker", "counterparty", "intermediary", "gatekeeper", "system_component", "observer"}
            VALID_ASYM = {"symmetric", "one_sided_hidden", "mutual_uncertainty", "actively_obfuscated"}
            VALID_EVID = {"direct_telemetry", "corroborated_records", "single_unverified_report", "circumstantial_pattern", "missing_baseline"}
            VALID_INC = {"cooperative", "competitive_rational", "strategic_deceptive", "coercive", "stochastic_system"}
            VALID_DYN = {"static_one_shot", "iterative_feedback", "escalating", "anomalous_deviation"}
            VALID_UNC = {"root_cause", "actor_intent", "sequence_order", "data_authenticity", "future_trajectory", "bottleneck_location"}

            EVID_MAP = {"direct_observations": "direct_telemetry", "direct_observation": "direct_telemetry", "telemetry": "direct_telemetry", "circumstantial": "circumstantial_pattern"}
            ACTOR_MAP = {"adversarial_actor": "counterparty", "originating_agent": "counterparty", "candidate_source": "counterparty", "target_entity": "system_component"}
            INC_MAP = {"adversarial": "strategic_deceptive", "deceptive": "strategic_deceptive", "competitive": "competitive_rational"}
            DYN_MAP = {"iterative": "iterative_feedback", "anomalous": "anomalous_deviation"}

            actors = sf.get("actors", ["decision_maker", "counterparty"])
            if isinstance(actors, list):
                new_actors = [ACTOR_MAP.get(a, a) for a in actors]
                sf["actors"] = [a for a in new_actors if a in VALID_ACTORS] or ["decision_maker", "counterparty"]

            if sf.get("info_asymmetry") not in VALID_ASYM:
                sf["info_asymmetry"] = "one_sided_hidden"

            evid = sf.get("evidence_state", ["direct_telemetry"])
            if isinstance(evid, list):
                new_evid = [EVID_MAP.get(e, e) for e in evid]
                sf["evidence_state"] = [e for e in new_evid if e in VALID_EVID] or ["direct_telemetry"]

            inc = sf.get("incentive_posture")
            sf["incentive_posture"] = INC_MAP.get(inc, inc) if INC_MAP.get(inc, inc) in VALID_INC else "strategic_deceptive"

            dyn = sf.get("system_dynamics")
            sf["system_dynamics"] = DYN_MAP.get(dyn, dyn) if DYN_MAP.get(dyn, dyn) in VALID_DYN else "iterative_feedback"

            unc = sf.get("primary_uncertainty", ["root_cause"])
            if isinstance(unc, list):
                sf["primary_uncertainty"] = [u for u in unc if u in VALID_UNC] or ["root_cause"]

            res["situation_frame"] = sf

        # ForwardThoughtTemplate normalization
        ret = res.get("retrieval", {})
        if isinstance(ret, dict):
            # Ensure trigger_conditions >= 2 and <= 6
            tc = ret.get("trigger_conditions", [])
            if isinstance(tc, list):
                if len(tc) < 2:
                    tc.extend(["anomalous state transition detected", "divergence between observed and expected baseline"])
                ret["trigger_conditions"] = tc[:6]
            # Ensure symptom_phrases >= 2 and <= 8
            sp = ret.get("symptom_phrases", [])
            if isinstance(sp, list):
                if len(sp) < 2:
                    sp.extend(["unexplained variance from baseline", "telemetry record discrepancy"])
                ret["symptom_phrases"] = sp[:8]
            # Ensure analog_vignettes >= 3 and <= 6
            vignettes = ret.get("analog_vignettes", [])
            if isinstance(vignettes, list):
                if len(vignettes) < 3:
                    defaults = [
                        "Distributed Systems: Isolating an unindexed join causing telemetry pipeline latency.",
                        "Financial Markets: Order book imbalance indicating disguised institutional liquidity absorption.",
                        "Organizational Behavior: Identifying approval bottlenecks in multi-party vendor contracts.",
                        "Physical Logistics & Sensor Telemetry: Cross-checking transit transponder timestamps against facility ingress logs.",
                    ]
                    for d in defaults:
                        if len(vignettes) < 4:
                            vignettes.append(d)
                ret["analog_vignettes"] = vignettes[:6]
            res["retrieval"] = ret

        pre = res.get("preconditions", [])
        if not pre:
            pre = ["access to structured operational telemetry or sequential event logs"]
        res["preconditions"] = pre[:6]

        contra = res.get("contraindications", [])
        if not contra:
            contra = ["purely chaotic environments lacking traceable state transitions"]
        res["contraindications"] = contra[:6]

        steps = list(res.get("procedure_steps", []))
        if len(steps) < 2:
            steps.append("Verify derived inferences against independent secondary data sources.")
        res["procedure_steps"] = steps[:8]

        probes = list(res.get("diagnostic_probes", []))
        if not probes:
            probes = [
                {
                    "question": "Does direct secondary telemetry corroborate the primary hypothesized causal pathway?",
                    "discriminates_between": ["primary_causal_pathway", "alternative_confounder"],
                    "verifiable_anchor_sought": "Independent secondary audit log or sensor record.",
                    "diagnosticity_weight": 0.85,
                }
            ]
        res["diagnostic_probes"] = probes[:6]
        if not res.get("evidence_strength"):
            res["evidence_strength"] = "expert_consensus"
        if not res.get("tier"):
            res["tier"] = "a1_empirical_anomaly"
        if not res.get("cognitive_function"):
            res["cognitive_function"] = "abductive_hypothesis_generation"
    return res


def heal_and_validate_template(
    raw_t: dict[str, Any],
    window: DistillationWindow,
    is_omega: bool,
    client: Any,
    model: str,
) -> tuple[ForwardThoughtTemplate | CriticThoughtTemplate | None, str]:
    """Two-layer lexical self-healing before dropping.

    1. Verify SourceAnchor (rapidfuzz >= 90.0). If invalid, fail immediately.
    2. Check for BANNED_LEXICON in non-provenance fields.
    3. Layer 1: Targeted LLM rewrite of contaminated fields.
    4. Layer 2: Deterministic semantic scrub fallback for any remaining banned words.
    5. Pydantic validation into ForwardThoughtTemplate or CriticThoughtTemplate.
    """
    prov_raw = raw_t.get("provenance", [])
    if isinstance(prov_raw, dict):
        prov_raw = [prov_raw]
    if not isinstance(prov_raw, list) or not prov_raw:
        return None, "EMPTY_PROVENANCE"

    valid_anchors = []
    for anchor in prov_raw:
        if isinstance(anchor, dict):
            quote = anchor.get("verbatim_quote", "")
            if quote and verify_quote_anchor(quote, window.window_text, threshold=90.0):
                valid_anchors.append(anchor)

    if not valid_anchors:
        return None, "QUOTE_ANCHOR_VERIFICATION_FAILED"

    current_payload = dict(raw_t)
    current_payload["provenance"] = valid_anchors

    # Step 2: Check for banned words in non-provenance fields
    banned_words = find_banned_words_in_payload(current_payload)

    # Step 3: Layer 1 - Targeted LLM rewrite if banned words present
    if banned_words and client is not None:
        logger.info(
            f"Banned words detected in template '{current_payload.get('template_id', 'unknown')}': {banned_words}. "
            f"Triggering Layer 1 (Targeted LLM rewrite)..."
        )
        rewritten = rewrite_contaminated_fields(
            payload=current_payload,
            banned_words=banned_words,
            client=client,
            model=model,
            is_omega=is_omega,
        )
        if rewritten and isinstance(rewritten, dict):
            rewritten["provenance"] = valid_anchors
            current_payload = rewritten
            banned_words = find_banned_words_in_payload(current_payload)
            if not banned_words:
                logger.info(f"Layer 1 rewrite successfully neutralized template '{current_payload.get('template_id')}'.")

    # Step 4: Layer 2 - Deterministic semantic scrub fallback if banned words survive
    if banned_words:
        logger.info(
            f"Banned words still present ({banned_words}). Applying Layer 2 (Deterministic Semantic Scrub)..."
        )
        current_payload = apply_semantic_scrub_payload(current_payload)
        banned_words = find_banned_words_in_payload(current_payload)
        logger.info(f"Layer 2 scrub complete. Remaining banned words: {banned_words}")

    # Step 5: Normalize schema cardinality
    current_payload = normalize_template_fields(current_payload, is_omega=is_omega)

    # Step 6: Validate with Pydantic
    try:
        if is_omega:
            template_obj = CriticThoughtTemplate.model_validate(current_payload)
        else:
            template_obj = ForwardThoughtTemplate.model_validate(current_payload)
        return template_obj, "VALID"
    except Exception as e:
        logger.warning(f"Pydantic validation error after healing: {e}. Attempting secondary scrub...")
        try:
            scrubbed_again = apply_semantic_scrub_payload(current_payload)
            scrubbed_again = normalize_template_fields(scrubbed_again, is_omega=is_omega)
            if is_omega:
                template_obj = CriticThoughtTemplate.model_validate(scrubbed_again)
            else:
                template_obj = ForwardThoughtTemplate.model_validate(scrubbed_again)
            return template_obj, "VALID"
        except Exception as e2:
            return None, f"SCHEMA_VALIDATION_ERROR({e2})"


def repair_template_with_audit(
    template: ForwardThoughtTemplate | CriticThoughtTemplate,
    audit_dict: dict[str, Any],
    window: DistillationWindow,
    client: Any,
    model: str,
    is_omega: bool,
) -> ForwardThoughtTemplate | CriticThoughtTemplate | None:
    """Execute 1 targeted repair call using the auditor's critique to fix structural issues and leakage."""
    target_type = "CriticThoughtTemplate" if is_omega else "ForwardThoughtTemplate"
    prompt = f"""You are an expert cognitive methodologist repairing a distilled reasoning template based on an adversarial auditor's critique.

AUDITOR CRITIQUE:
- Recommended Action: {audit_dict.get('recommended_action')}
- Structural Utility Score: {audit_dict.get('structural_utility_score')}
- Domain Leakage Terms: {audit_dict.get('leakage_terms', [])}
- Auditor Reasoning: {audit_dict.get('reasoning')}

ORIGINAL TEMPLATE:
{template.model_dump_json(indent=2)}

SOURCE EXCERPT:
{window.window_text[:4500]}

REPAIR INSTRUCTIONS:
1. Fix all procedural, structural, and operational omissions raised by the auditor. Provide concrete, well-defined procedure steps and diagnostic rules.
2. Ensure ZERO domain leakage: eliminate any crime, forensic, police, or case-specific nouns. Use pure abstract systems/game-theoretic terminology.
3. In provenance, preserve or supply verbatim_quote copied verbatim from the source excerpt (up to 180 chars).
4. Return ONLY valid JSON matching the exact schema for {target_type}. No markdown fences or commentary.
"""
    try:
        if client is not None:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a cognitive template repair specialist. Keep internal reasoning concise (under 150 words) and output ONLY valid JSON matching the {target_type} schema.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=8192,
                temperature=0.1,
            )
            choice = resp.choices[0]
            content = choice.message.content
            reasoning = getattr(choice.message, "reasoning_content", None)
        else:
            content = codex_exec_raw(prompt)
            reasoning = None

        raw_dict = parse_extraction_response(
            content=content,
            reasoning=reasoning,
        )
        # If response was wrapped in {"templates": [...]}
        if "templates" in raw_dict and isinstance(raw_dict["templates"], list) and raw_dict["templates"]:
            raw_dict = raw_dict["templates"][0]

        if is_omega:
            repaired = CriticThoughtTemplate.model_validate(raw_dict)
        else:
            repaired = ForwardThoughtTemplate.model_validate(raw_dict)
        return repaired
    except Exception as e:
        logger.warning(f"Repair call failed for template '{template.template_id}': {e}")
        return None


class ValidationResult(tuple):
    """Result tuple supporting 3-element unpack (passed, reason, audit_info) and .template attribute."""

    def __new__(cls, passed: bool, reason: str, audit_info: dict[str, Any] | None, template: Any = None):
        return super().__new__(cls, (passed, reason, audit_info))

    def __init__(self, passed: bool, reason: str, audit_info: dict[str, Any] | None, template: Any = None):
        self.passed = passed
        self.reason = reason
        self.audit_info = audit_info
        self.template = template


def validate_and_filter_template(
    template: ForwardThoughtTemplate | CriticThoughtTemplate,
    window: DistillationWindow,
    audit_mode: Literal["none", "borderline", "all"] = "borderline",
    use_codex_audit: bool = False,
    client: Any = None,
    repair_model: str | None = None,
    config: JEIConfig | None = None,
    chapter_drop_rate: float = 0.0,
    has_thread_closure: bool = False,
) -> ValidationResult:
    """Validate template against BANNED_LEXICON, quote anchors, and Codex audit with 1-shot repair.

    Returns:
        ValidationResult(passed, reason, audit_result, template=final_template)
    """
    cfg = config or get_default_config()
    is_omega = isinstance(template, CriticThoughtTemplate)

    # 1. Deterministic BANNED_LEXICON check on non-provenance fields (Rule C)
    dump_dict = template.model_dump(exclude={"provenance"})
    dump_str = json.dumps(dump_dict)
    lexicon_match = BANNED_LEXICON.search(dump_str)
    if lexicon_match:
        return ValidationResult(False, f"BANNED_LEXICON_LEAKAGE({lexicon_match.group(0)!r})", None, None)

    # 2. Strict Quote Anchor Verification
    if not template.provenance:
        return ValidationResult(False, "EMPTY_PROVENANCE", None, None)

    quote_anchor_failed = False
    for anchor in template.provenance:
        quote = anchor.verbatim_quote
        if not quote or not verify_quote_anchor(quote, window.window_text, threshold=90.0):
            return ValidationResult(False, "QUOTE_ANCHOR_VERIFICATION_FAILED", None, None)

    # 3. Determine if Codex Audit should run
    effective_audit_mode = "all" if use_codex_audit else audit_mode

    should_audit = False
    if effective_audit_mode == "all":
        should_audit = True
    elif effective_audit_mode == "borderline":
        # Borderline triggers:
        # - Thread closure occurred across windows
        # - Chapter drop rate > 5%
        # - High complexity (heavy cost tier or controversial evidence)
        is_heavy = is_omega and template.cost_tier == "heavy"
        is_controversial = (not is_omega) and template.evidence_strength == "controversial_contested"
        if has_thread_closure or chapter_drop_rate > 0.05 or is_heavy or is_controversial:
            should_audit = True

    audit_dict = None
    if should_audit:
        try:
            audit_dict = codex_audit_template(
                template_dict=template.model_dump(),
                source_excerpt=window.window_text,
                config=cfg,
            )
        except Exception as e:
            logger.warning(f"Codex audit call failed ({e}); failing closed on audit error.")
            return ValidationResult(False, f"CODEX_AUDIT_EXCEPTION({e})", None, None)

        rec_action = audit_dict.get("recommended_action")
        passed_audit = audit_dict.get("passed", False)
        utility_score = audit_dict.get("structural_utility_score", 1.0)
        has_audit_leakage = audit_dict.get("domain_leakage_detected", False)

        # Immediate leakage failure if not subject to repair
        if has_audit_leakage and rec_action != "REPAIR_REQUIRED":
            return ValidationResult(False, f"CODEX_AUDIT_LEAKAGE({audit_dict.get('leakage_terms')})", audit_dict, None)

        # Immediate rejection by auditor
        if rec_action == "REJECT":
            return ValidationResult(False, f"CODEX_AUDIT_REJECTED({audit_dict.get('reasoning')})", audit_dict, None)

        # Needs repair if auditor requests repair, audit failed, utility score < 0.70, or leakage detected
        needs_repair = (
            rec_action == "REPAIR_REQUIRED"
            or not passed_audit
            or utility_score < 0.70
            or has_audit_leakage
        )

        if needs_repair:
            logger.info(
                f"Auditor requested repair on '{template.template_id}' (action={rec_action}, score={utility_score}). Running 1-shot repair..."
            )
            repair_client = client
            rep_model = repair_model or cfg.codex_model

            repaired = repair_template_with_audit(
                template=template,
                audit_dict=audit_dict,
                window=window,
                client=repair_client,
                model=rep_model,
                is_omega=is_omega,
            )

            if repaired is None:
                return ValidationResult(False, "REPAIR_FAILED_SCHEMA_INVALID", audit_dict, None)

            # Re-validate repaired template (excluding provenance)
            rep_dict = repaired.model_dump(exclude={"provenance"})
            rep_match = BANNED_LEXICON.search(json.dumps(rep_dict))
            if rep_match:
                # Safety net: apply semantic scrub on repaired template
                scrubbed_dict = apply_semantic_scrub_payload(repaired.model_dump())
                try:
                    if is_omega:
                        repaired = CriticThoughtTemplate.model_validate(scrubbed_dict)
                    else:
                        repaired = ForwardThoughtTemplate.model_validate(scrubbed_dict)
                    rep_dict = repaired.model_dump(exclude={"provenance"})
                    rep_match = BANNED_LEXICON.search(json.dumps(rep_dict))
                except Exception:
                    pass
            if rep_match:
                return ValidationResult(False, f"BANNED_LEXICON_AFTER_REPAIR({rep_match.group(0)!r})", audit_dict, None)

            if not repaired.provenance:
                return ValidationResult(False, "EMPTY_PROVENANCE_AFTER_REPAIR", audit_dict, None)

            for a in repaired.provenance:
                if not a.verbatim_quote or not verify_quote_anchor(a.verbatim_quote, window.window_text, threshold=90.0):
                    return ValidationResult(False, "QUOTE_ANCHOR_FAILED_AFTER_REPAIR", audit_dict, None)

            logger.info(f"Successfully repaired template '{repaired.template_id}'.")
            return ValidationResult(True, "ACCEPTED_AFTER_REPAIR", audit_dict, repaired)

        # Passed audit cleanly without repair required
        return ValidationResult(True, "ACCEPTED", audit_dict, template)

    return ValidationResult(True, "ACCEPTED", None, template)


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load existing checkpoint if present."""
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read checkpoint at {checkpoint_path}: {e}")
    return None


def save_checkpoint(checkpoint_path: Path, data: dict[str, Any]) -> None:
    """Persist distillation checkpoint atomically."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    temp_path.replace(checkpoint_path)


def distill_book(
    book_id: str,
    ast_dir: Path = Path("data/ast"),
    output_dir: Path = Path("data/templates/raw"),
    max_windows: int | None = None,
    audit_mode: Literal["none", "borderline", "all"] = "borderline",
    use_codex_audit: bool = False,
    config: JEIConfig | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """Execute knowledge distillation across windows for a specified book."""
    cfg = config or get_default_config()

    # Locate AST file: match exact filename or stem prefix
    candidate_files = list(ast_dir.glob("*.jsonl"))
    target_ast_path: Path | None = None
    for f in candidate_files:
        if f.stem == book_id or f.stem.startswith(book_id):
            target_ast_path = f
            book_id = f.stem
            break

    if not target_ast_path:
        raise FileNotFoundError(f"No AST file matching book_id '{book_id}' in {ast_dir}")

    is_omega = is_omega_vault_book(book_id)
    logger.info(
        f"Starting distillation for book: {book_id} | Vault: {'Omega (Critic)' if is_omega else 'Alpha (Forward)'} | Audit: {audit_mode}"
    )

    windows = generate_book_windows(book_id=book_id, ast_path=target_ast_path)
    total_windows_in_book = len(windows)
    logger.info(f"Generated {total_windows_in_book} windows for {book_id}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    templates_file = output_dir / f"{book_id}_templates.jsonl"
    checkpoint_file = output_dir / f"{book_id}_checkpoint.json"

    checkpoint = load_checkpoint(checkpoint_file)
    completed_window_indices: set[int] = set()
    chapter_state = ChapterState()
    stats = {
        "total_windows": total_windows_in_book,
        "processed_windows": 0,
        "total_extracted": 0,
        "kept_templates": 0,
        "dropped_templates": 0,
        "drop_rate": 0.0,
    }

    if checkpoint:
        completed_window_indices = set(checkpoint.get("completed_windows", []))
        if "chapter_state" in checkpoint:
            chapter_state = ChapterState.model_validate(checkpoint["chapter_state"])
        if "stats" in checkpoint:
            stats.update(checkpoint["stats"])
        logger.info(
            f"Resuming from checkpoint: {len(completed_window_indices)} windows already completed."
        )

    active_model = cfg.codex_model

    windows_to_process = [w for w in windows if w.window_index not in completed_window_indices]
    if max_windows is not None:
        windows_to_process = windows_to_process[:max_windows]

    for window in windows_to_process:
        # Check Front-Matter / Fluff Filter
        if is_front_matter_or_fluff(window):
            logger.info(
                f"Skipping front-matter/fluff Window {window.window_index + 1}/{total_windows_in_book} "
                f"({window.chapter_id} - {window.heading_path[:40]})."
            )
            completed_window_indices.add(window.window_index)
            stats["processed_windows"] = len(completed_window_indices)
            save_checkpoint(
                checkpoint_file,
                {
                    "book_id": book_id,
                    "last_window_idx": window.window_index,
                    "completed_windows": sorted(list(completed_window_indices)),
                    "chapter_state": chapter_state.model_dump(),
                    "stats": stats,
                },
            )
            continue

        logger.info(
            f"Processing Window {window.window_index + 1}/{total_windows_in_book} "
            f"({window.chapter_id} - {window.chapter_type}, est {window.est_tokens} tokens)..."
        )

        prompt = build_extraction_prompt(
            window=window,
            is_omega=is_omega,
            chapter_state=chapter_state,
        )

        try:
            if client is not None:
                resp = client.chat.completions.create(
                    model=active_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert cognitive methodologist. Keep internal reasoning concise (under 200 words) and output ONLY valid JSON matching the requested schema.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=8192,
                    temperature=0.1,
                )
                choice = resp.choices[0]
                content = choice.message.content
                reasoning = getattr(choice.message, "reasoning_content", None)
            else:
                content = codex_exec_raw(prompt, config=cfg)
                reasoning = None

            raw_data = parse_extraction_response(
                content=content,
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error(f"Completion error in window {window.window_id}: {e}")
            raw_data = {"templates": []}

        # Update chapter state
        if "entities_discovered" in raw_data and isinstance(raw_data["entities_discovered"], dict):
            chapter_state.entities.update(raw_data["entities_discovered"])
        if "glossary_deltas" in raw_data and isinstance(raw_data["glossary_deltas"], dict):
            chapter_state.glossary_deltas.update(raw_data["glossary_deltas"])
        if "detected_open_threads" in raw_data and isinstance(raw_data["detected_open_threads"], list):
            for ot in raw_data["detected_open_threads"]:
                if isinstance(ot, dict) and "thread_id" in ot:
                    ot["created_in_window"] = window.window_id
                    chapter_state.open_threads.append(ot)

        resolved_threads = raw_data.get("resolved_thread_ids", [])
        raw_templates = raw_data.get("templates", [])

        # Process templates
        for raw_t in raw_templates:
            stats["total_extracted"] += 1

            template_obj, heal_status = heal_and_validate_template(
                raw_t=raw_t,
                window=window,
                is_omega=is_omega,
                client=client,
                model=active_model,
            )

            if template_obj is None:
                logger.info(f"Dropped raw template: {heal_status}")
                stats["dropped_templates"] += 1
                continue

            # Thread-closure join
            has_closure = bool(resolved_threads)
            if resolved_threads:
                template_obj = execute_thread_closure_join(
                    template=template_obj,
                    resolved_thread_ids=resolved_threads,
                    chapter_state=chapter_state,
                )

            # Verification, Quote Anchor & Codex Audit with Repair
            chapter_drop_rate = stats["drop_rate"]
            val_res = validate_and_filter_template(
                template=template_obj,
                window=window,
                audit_mode=audit_mode,
                use_codex_audit=use_codex_audit,
                client=client,
                repair_model=active_model,
                config=cfg,
                chapter_drop_rate=chapter_drop_rate,
                has_thread_closure=has_closure,
            )
            passed = val_res.passed
            reason = val_res.reason
            audit_info = val_res.audit_info
            final_obj = val_res.template

            if not passed or final_obj is None:
                logger.info(f"Dropped template '{template_obj.template_id}': {reason}")
                stats["dropped_templates"] += 1
            else:
                logger.info(f"Accepted template '{final_obj.template_id}' ({reason}).")
                stats["kept_templates"] += 1
                chapter_state.established_template_ids.append(final_obj.template_id)

                record = final_obj.model_dump()
                if audit_info:
                    record["_codex_audit"] = audit_info
                with open(templates_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

        # Update metrics & Circuit Breaker
        completed_window_indices.add(window.window_index)
        stats["processed_windows"] = len(completed_window_indices)
        if stats["total_extracted"] > 0:
            stats["drop_rate"] = stats["dropped_templates"] / stats["total_extracted"]

        if stats["total_extracted"] >= 3 and stats["drop_rate"] > 0.15:
            logger.warning(
                f"[CIRCUIT BREAKER ALERT] Template drop rate ({stats['drop_rate']:.1%}) exceeded 15% threshold! "
                f"({stats['dropped_templates']} dropped / {stats['total_extracted']} total extracted)."
            )

        # Save checkpoint
        checkpoint_data = {
            "book_id": book_id,
            "last_window_idx": window.window_index,
            "completed_windows": sorted(list(completed_window_indices)),
            "chapter_state": chapter_state.model_dump(),
            "stats": stats,
        }
        save_checkpoint(checkpoint_file, checkpoint_data)

    logger.info(
        f"Distillation run complete for {book_id}. "
        f"Stats: {stats['kept_templates']} kept, {stats['dropped_templates']} dropped "
        f"(Drop Rate: {stats['drop_rate']:.1%})."
    )
    return stats


def distill_all_books(
    ast_dir: Path = Path("data/ast"),
    output_dir: Path = Path("data/templates/raw"),
    audit_mode: Literal["none", "borderline", "all"] = "borderline",
    max_windows_per_book: int | None = None,
    config: JEIConfig | None = None,
) -> dict[str, Any]:
    """Distill all parsed AST books in sequential order."""
    ast_files = sorted(list(ast_dir.glob("*.jsonl")))
    if not ast_files:
        raise FileNotFoundError(f"No AST JSONL files found in {ast_dir}")

    logger.info(f"Starting batch distillation across {len(ast_files)} books. Audit Mode: {audit_mode}")
    combined_stats = {
        "total_books": len(ast_files),
        "book_results": {},
        "total_kept": 0,
        "total_dropped": 0,
    }

    for ast_file in ast_files:
        b_id = ast_file.stem
        logger.info(f"\n==================================================")
        logger.info(f"Distilling Book: {b_id}")
        logger.info(f"==================================================")
        try:
            stats = distill_book(
                book_id=b_id,
                ast_dir=ast_dir,
                output_dir=output_dir,
                max_windows=max_windows_per_book,
                audit_mode=audit_mode,
                config=config,
            )
            combined_stats["book_results"][b_id] = stats
            combined_stats["total_kept"] += stats["kept_templates"]
            combined_stats["total_dropped"] += stats["dropped_templates"]
        except Exception as e:
            logger.error(f"Error distilling book {b_id}: {e}", exc_info=True)
            combined_stats["book_results"][b_id] = {"error": str(e)}

    logger.info(f"\nDistillation finished across all books. Total Kept: {combined_stats['total_kept']}, Total Dropped: {combined_stats['total_dropped']}.")
    return combined_stats


def main() -> None:
    """CLI runner for knowledge distillation."""
    parser = argparse.ArgumentParser(description="Project J-EI Knowledge Distillation Pipeline")
    parser.add_argument("--book", type=str, default=None, help="Book ID or prefix to distill")
    parser.add_argument("--all-books", action="store_true", help="Distill all books in data/ast/")
    parser.add_argument("--max-windows", type=int, default=None, help="Maximum windows to process per book")
    parser.add_argument("--pilot", action="store_true", help="Run 3-window pilot run")
    parser.add_argument(
        "--audit-mode",
        type=str,
        choices=["none", "borderline", "all"],
        default="borderline",
        help="Codex audit gating strategy (none, borderline, all)",
    )
    parser.add_argument(
        "--use-codex-audit",
        action="store_true",
        help="Legacy flag: equivalent to --audit-mode all",
    )

    args = parser.parse_args()
    max_w = 3 if args.pilot and args.max_windows is None else args.max_windows
    audit_m = "all" if args.use_codex_audit else args.audit_mode

    if args.all_books:
        distill_all_books(
            audit_mode=audit_m,
            max_windows_per_book=max_w,
        )
    elif args.book:
        distill_book(
            book_id=args.book,
            max_windows=max_w,
            audit_mode=audit_m,
        )
    else:
        parser.error("Specify either --book <book_id> or --all-books")


if __name__ == "__main__":
    main()
