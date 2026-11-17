"""Comprehensive pytest suite for Step 2B (Knowledge Distillation Engine & Docked Codex Audit)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from j_ei.distiller import (
    DistillationWindow,
    build_extraction_prompt,
    create_chapter_windows,
    distill_book,
    execute_thread_closure_join,
    heal_and_validate_template,
    is_front_matter_or_fluff,
    is_omega_vault_book,
    parse_extraction_response,
    validate_and_filter_template,
)
from j_ei.schemas import (
    BANNED_LEXICON,
    ChapterState,
    CriticThoughtTemplate,
    ForwardThoughtTemplate,
    RetrievalViews,
    SituationFrame,
    SourceAnchor,
)


@pytest.fixture
def sample_paragraphs():
    """Generate 25 synthetic paragraphs of 500 tokens each."""
    paras = []
    for i in range(25):
        paras.append(
            {
                "pid": f"[P{i+1:04d}]",
                "book_id": "test_book",
                "chapter_id": "ch_01",
                "chapter_title": "System Architecture",
                "heading_path": "System Architecture > Subsystem A",
                "locator": f"page_{i+1}",
                "char_span": [i * 1000, (i + 1) * 1000],
                "source_span_sha256": f"hash_{i:04d}",
                "est_tokens": 500,
                "text": f"This is synthetic paragraph {i+1} describing algorithmic state synchronization and telemetry anomaly bounds.",
            }
        )
    return paras


def test_is_omega_vault_book():
    """Verify Heuer books route to Vault Omega and others to Vault Alpha."""
    assert is_omega_vault_book("psychology_of_intelligence_analysis_richards_j_heuer_jr") is True
    assert is_omega_vault_book("structured_analytic_techniques_for_intelligence_analysis") is True
    assert is_omega_vault_book("CrimeClassificationManual") is False
    assert is_omega_vault_book("mindhunter") is False


def test_window_generator_taxonomic(sample_paragraphs):
    """Verify taxonomic windowing chunks up to 3,000 tokens with 0% overlap and prepends breadcrumbs."""
    windows = create_chapter_windows(
        book_id="test_book",
        chapter_id="ch_01",
        chapter_title="System Architecture",
        paragraphs=sample_paragraphs[:12],  # 12 * 500 = 6,000 tokens
        chapter_type="TAXONOMIC",
        max_taxonomic_tokens=3000,
    )

    # 6,000 tokens with 3,000 tokens per chunk = 2 windows
    assert len(windows) == 2
    assert windows[0].is_taxonomic is True
    assert windows[0].est_tokens == 3000
    assert windows[1].est_tokens == 3000

    # Verify heading breadcrumb prepended
    assert windows[0].window_text.startswith("# Context: System Architecture > Subsystem A")
    # Verify no paragraph overlap between windows
    pids_w0 = {p["pid"] for p in windows[0].paragraphs}
    pids_w1 = {p["pid"] for p in windows[1].paragraphs}
    assert pids_w0.isdisjoint(pids_w1)


def test_window_generator_narrative_hybrid(sample_paragraphs):
    """Verify narrative/hybrid sliding window produces 6,000-token windows with 1,000-token overlap."""
    windows = create_chapter_windows(
        book_id="test_book",
        chapter_id="ch_01",
        chapter_title="Investigative Narrative",
        paragraphs=sample_paragraphs,  # 25 * 500 = 12,500 tokens
        chapter_type="NARRATIVE",
        narrative_window_tokens=6000,
        narrative_stride_tokens=5000,
    )

    assert len(windows) >= 3
    # Window 0: paras 0..11 (12 paras * 500 = 6000 tokens)
    assert windows[0].est_tokens == 6000
    assert len(windows[0].paragraphs) == 12

    # Window 1: stride is 5,000 tokens (10 paras), so starts at para 10
    # Paras 10..21 (12 paras * 500 = 6000 tokens)
    assert windows[1].est_tokens == 6000
    assert windows[1].paragraphs[0]["pid"] == sample_paragraphs[10]["pid"]

    # Overlap between Window 0 and Window 1 should be paras 10 and 11 = 2 paras * 500 = 1,000 tokens
    pids_w0 = {p["pid"] for p in windows[0].paragraphs}
    pids_w1 = {p["pid"] for p in windows[1].paragraphs}
    overlap_pids = pids_w0.intersection(pids_w1)
    assert len(overlap_pids) == 2


def test_chapter_state_serialization():
    """Verify ChapterState serializes and deserializes accurately."""
    state = ChapterState(
        entities={"component_x": "gateway node"},
        open_threads=[{"thread_id": "t1", "description": "unverified anomaly"}],
        established_template_ids=["temp_01"],
        glossary_deltas={"latency_spike": "unexpected 10x deviation"},
    )
    data = state.model_dump()
    loaded = ChapterState.model_validate(data)
    assert loaded.entities["component_x"] == "gateway node"
    assert len(loaded.open_threads) == 1
    assert loaded.established_template_ids == ["temp_01"]


def test_thread_closure_join():
    """Verify that resolving an open thread joins setup provenance across window boundaries."""
    state = ChapterState(
        open_threads=[
            {
                "thread_id": "thread_anomaly_setup",
                "description": "Initial setup marker for anomalous latency pattern",
                "setup_anchor": {
                    "book_id": "test_book",
                    "chapter": "ch_01",
                    "locator": "page_2",
                    "char_span": [100, 200],
                    "source_span_sha256": "setup_hash_123",
                    "verbatim_quote": "Latency telemetry deviated beyond standard bounds.",
                },
            }
        ]
    )

    current_anchor = SourceAnchor(
        book_id="test_book",
        chapter="ch_01",
        locator="page_12",
        char_span=[1000, 1100],
        source_span_sha256="res_hash_456",
        verbatim_quote="Resolution was traced to circular network buffer saturation.",
    )

    template = CriticThoughtTemplate(
        template_id="buffer_saturation_audit",
        technique_name="Buffer Saturation Audit",
        cost_tier="routed",
        target_vulnerabilities=["MODEL_PRIOR_UNSUPPORTED_FACT"],
        detectable_symptoms=["throughput stall", "buffer growth", "telemetry gap"],
        audit_procedure=["inspect buffer limits", "trace packet queues"],
        kill_conditions=["direct telemetry unavailable"],
        remediation_actions=["requery_empirical"],
        known_limitations=["requires access to queue metrics"],
        provenance=[current_anchor],
    )

    updated_template = execute_thread_closure_join(
        template=template,
        resolved_thread_ids=["thread_anomaly_setup"],
        chapter_state=state,
    )

    # Provenance should now contain 2 anchors: current anchor + setup anchor
    assert len(updated_template.provenance) == 2
    quotes = [a.verbatim_quote for a in updated_template.provenance]
    assert "Latency telemetry deviated beyond standard bounds." in quotes
    assert "Resolution was traced to circular network buffer saturation." in quotes
    # The resolved thread must be removed from open_threads
    assert len(state.open_threads) == 0


def test_banned_lexicon_rejection():
    """Verify template with domain leakage (e.g. murder, police, homicide) is rejected."""
    window = DistillationWindow(
        window_id="w1",
        book_id="b1",
        chapter_id="c1",
        chapter_title="T",
        chapter_type="TAXONOMIC",
        heading_path="H",
        window_index=0,
        paragraphs=[],
        window_text="Some text mentioning anomaly.",
        est_tokens=100,
        char_span=(0, 100),
        is_taxonomic=True,
    )

    # Template with domain leakage in detectable_symptoms
    template = CriticThoughtTemplate(
        template_id="leaky_template",
        technique_name="Forensic Investigation",
        cost_tier="routed",
        target_vulnerabilities=["CONFIRMATION_SKEW"],
        detectable_symptoms=["police report contradiction", "witness uncertainty", "evidence tampering"],
        audit_procedure=["review records", "interview party"],
        kill_conditions=["no records"],
        remediation_actions=["requery_knowledge"],
        known_limitations=["limit 1"],
        provenance=[
            SourceAnchor(
                book_id="b1",
                chapter="c1",
                locator="p1",
                char_span=(0, 10),
                source_span_sha256="hash",
                verbatim_quote="Some text mentioning anomaly.",
            )
        ],
    )

    passed, reason, _ = validate_and_filter_template(template, window)
    assert passed is False
    assert "BANNED_LEXICON_LEAKAGE" in reason


def test_quote_anchor_verification():
    """Verify rapidfuzz quote anchor validation rejects hallucinated quotes."""
    window_text = (
        "Algorithmic verification requires strictly independent diagnostic metrics. "
        "Analysts must never rely on single unverified telemetry streams."
    )
    window = DistillationWindow(
        window_id="w1",
        book_id="b1",
        chapter_id="c1",
        chapter_title="T",
        chapter_type="TAXONOMIC",
        heading_path="H",
        window_index=0,
        paragraphs=[],
        window_text=window_text,
        est_tokens=50,
        char_span=(0, 100),
        is_taxonomic=True,
    )

    # Valid template matching text
    valid_template = CriticThoughtTemplate(
        template_id="valid_crit",
        technique_name="Independent Metric Verification",
        cost_tier="routed",
        target_vulnerabilities=["UNVERIFIED_SOURCE_DEPENDENCE"],
        detectable_symptoms=["single feed", "missing baseline", "discrepant timestamps"],
        audit_procedure=["cross-check telemetry", "isolate data source"],
        kill_conditions=["no alternate telemetry"],
        remediation_actions=["requery_empirical"],
        known_limitations=["telemetry lag"],
        provenance=[
            SourceAnchor(
                book_id="b1",
                chapter="c1",
                locator="p1",
                char_span=(0, 10),
                source_span_sha256="hash",
                verbatim_quote="Algorithmic verification requires strictly independent diagnostic metrics.",
            )
        ],
    )

    passed, reason, _ = validate_and_filter_template(valid_template, window)
    assert passed is True
    assert reason == "ACCEPTED"

    # Hallucinated quote
    invalid_template = valid_template.model_copy(deep=True)
    invalid_template.provenance[0].verbatim_quote = "Completely hallucinated quote that does not appear anywhere."
    passed_inv, reason_inv, _ = validate_and_filter_template(invalid_template, window)
    assert passed_inv is False
    assert "QUOTE_ANCHOR_VERIFICATION_FAILED" in reason_inv


def test_parse_extraction_response():
    """Verify JSON parsing handles clean JSON, markdown fenced blocks, and fallback."""
    # 1. Clean JSON
    res1 = parse_extraction_response('{"templates": [{"id": 1}]}')
    assert len(res1["templates"]) == 1

    # 2. Markdown fenced JSON
    res2 = parse_extraction_response('```json\n{"templates": [{"id": 2}]}\n```')
    assert len(res2["templates"]) == 1
    assert res2["templates"][0]["id"] == 2

    # 3. Content None but reasoning has markdown JSON
    res3 = parse_extraction_response(None, 'Thinking process... ```json\n{"templates": [{"id": 3}]}\n```')
    assert len(res3["templates"]) == 1
    assert res3["templates"][0]["id"] == 3


@patch("j_ei.distiller.codex_audit_template")
def test_codex_audit_integration(mock_codex):
    """Verify docked Codex CLI template audit integration."""
    window = DistillationWindow(
        window_id="w1",
        book_id="b1",
        chapter_id="c1",
        chapter_title="T",
        chapter_type="TAXONOMIC",
        heading_path="H",
        window_index=0,
        paragraphs=[],
        window_text="Algorithmic verification requires strictly independent diagnostic metrics.",
        est_tokens=50,
        char_span=(0, 100),
        is_taxonomic=True,
    )

    template = CriticThoughtTemplate(
        template_id="test_audit",
        technique_name="Independent Metric Verification",
        cost_tier="routed",
        target_vulnerabilities=["UNVERIFIED_SOURCE_DEPENDENCE"],
        detectable_symptoms=["single feed", "missing baseline", "discrepant timestamps"],
        audit_procedure=["cross-check telemetry", "isolate data source"],
        kill_conditions=["no alternate telemetry"],
        remediation_actions=["requery_empirical"],
        known_limitations=["telemetry lag"],
        provenance=[
            SourceAnchor(
                book_id="b1",
                chapter="c1",
                locator="p1",
                char_span=(0, 10),
                source_span_sha256="hash",
                verbatim_quote="Algorithmic verification requires strictly independent diagnostic metrics.",
            )
        ],
    )

    # 1. Codex accepts template
    mock_codex.return_value = {
        "passed": True,
        "domain_leakage_detected": False,
        "leakage_terms": [],
        "structural_utility_score": 0.95,
        "reasoning": "High structural utility and zero domain leakage.",
        "recommended_action": "ACCEPT",
    }
    passed, reason, audit_info = validate_and_filter_template(
        template=template,
        window=window,
        use_codex_audit=True,
    )
    assert passed is True
    assert reason == "ACCEPTED"
    assert audit_info["structural_utility_score"] == 0.95

    # 2. Codex rejects template due to leakage
    mock_codex.return_value = {
        "passed": False,
        "domain_leakage_detected": True,
        "leakage_terms": ["suspect"],
        "structural_utility_score": 0.3,
        "reasoning": "Detected domain leakage.",
        "recommended_action": "REJECT",
    }
    passed_bad, reason_bad, _ = validate_and_filter_template(
        template=template,
        window=window,
        use_codex_audit=True,
    )
    assert passed_bad is False
    assert "CODEX_AUDIT_LEAKAGE" in reason_bad


def test_distill_book_circuit_breaker_and_checkpoint(tmp_path):
    """Verify distillation loop, checkpoint saving/resuming, and circuit breaker warning."""
    ast_dir = tmp_path / "ast"
    ast_dir.mkdir()
    ast_file = ast_dir / "structured_analytic_mock.jsonl"

    paragraphs = [
        {
            "pid": f"[P{i:04d}]",
            "book_id": "structured_analytic_mock",
            "chapter_id": "ch_01",
            "chapter_title": "Heuer Analytic Methods",
            "heading_path": "Heuer Analytic Methods > Core",
            "locator": f"page_{i+1}",
            "char_span": [i * 100, (i + 1) * 100],
            "source_span_sha256": "abc123hash",
            "est_tokens": 100,
            "text": f"Paragraph {i+1} discussing cognitive bias and competing hypotheses analysis.",
        }
        for i in range(5)
    ]
    with open(ast_file, "w", encoding="utf-8") as f:
        for p in paragraphs:
            f.write(json.dumps(p) + "\n")

    out_dir = tmp_path / "templates_raw"

    # Mock client response returning 1 valid and 2 invalid templates
    mock_client = MagicMock()

    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(
        {
            "templates": [
                {
                    "schema_version": "2.1",
                    "template_id": "ach_technique",
                    "technique_name": "Analysis of Competing Hypotheses",
                    "cost_tier": "routed",
                    "target_vulnerabilities": ["CONFIRMATION_SKEW"],
                    "detectable_symptoms": ["early closure", "ignoring non-confirming data", "single option focus"],
                    "audit_procedure": ["list hypotheses", "evaluate diagnosticity"],
                    "kill_conditions": ["fewer than two hypotheses"],
                    "remediation_actions": ["reweight_hypotheses"],
                    "known_limitations": ["depends on initial set"],
                    "provenance": [
                        {
                            "book_id": "structured_analytic_mock",
                            "chapter": "ch_01",
                            "locator": "page_1",
                            "char_span": [0, 50],
                            "source_span_sha256": "abc123hash",
                            "verbatim_quote": "Paragraph 1 discussing cognitive bias and competing hypotheses analysis.",
                        }
                    ],
                },
                {
                    # Invalid quote anchor (will be dropped)
                    "schema_version": "2.1",
                    "template_id": "fake_anchor_technique",
                    "technique_name": "Fake Anchor Technique",
                    "cost_tier": "routed",
                    "target_vulnerabilities": ["CONFIRMATION_SKEW"],
                    "detectable_symptoms": ["early closure", "ignoring non-confirming data", "single option focus"],
                    "audit_procedure": ["list hypotheses", "evaluate diagnosticity"],
                    "kill_conditions": ["fewer than two hypotheses"],
                    "remediation_actions": ["reweight_hypotheses"],
                    "known_limitations": ["depends on initial set"],
                    "provenance": [
                        {
                            "book_id": "structured_analytic_mock",
                            "chapter": "ch_01",
                            "locator": "page_1",
                            "char_span": [0, 50],
                            "source_span_sha256": "abc123hash",
                            "verbatim_quote": "This quote is nowhere in paragraph 1 text.",
                        }
                    ],
                },
                {
                    # Banned lexicon (will be dropped)
                    "schema_version": "2.1",
                    "template_id": "banned_technique",
                    "technique_name": "Crime Analysis",
                    "cost_tier": "routed",
                    "target_vulnerabilities": ["CONFIRMATION_SKEW"],
                    "detectable_symptoms": ["homicide investigation", "police suspicion", "suspect interrogation"],
                    "audit_procedure": ["list hypotheses", "evaluate diagnosticity"],
                    "kill_conditions": ["fewer than two hypotheses"],
                    "remediation_actions": ["reweight_hypotheses"],
                    "known_limitations": ["depends on initial set"],
                    "provenance": [
                        {
                            "book_id": "structured_analytic_mock",
                            "chapter": "ch_01",
                            "locator": "page_1",
                            "char_span": [0, 50],
                            "source_span_sha256": "abc123hash",
                            "verbatim_quote": "This second quote is also completely absent from paragraph 1 text.",
                        }
                    ],
                },
            ],
            "detected_open_threads": [],
            "resolved_thread_ids": [],
            "entities_discovered": {},
            "glossary_deltas": {},
        }
    )
    mock_resp.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_resp

    stats = distill_book(
        book_id="structured_analytic_mock",
        ast_dir=ast_dir,
        output_dir=out_dir,
        max_windows=1,
        client=mock_client,
    )

    assert stats["processed_windows"] == 1
    assert stats["total_extracted"] == 3
    assert stats["kept_templates"] == 1
    assert stats["dropped_templates"] == 2
    # Drop rate is 2/3 = 66.7% > 15%
    assert stats["drop_rate"] > 0.15

    # Verify templates file written
    templates_file = out_dir / "structured_analytic_mock_templates.jsonl"
    assert templates_file.exists()
    with open(templates_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 1
        t_json = json.loads(lines[0])
        assert t_json["template_id"] == "ach_technique"

    # Verify checkpoint file written
    ckpt_file = out_dir / "structured_analytic_mock_checkpoint.json"
    assert ckpt_file.exists()
    with open(ckpt_file, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
        assert ckpt["book_id"] == "structured_analytic_mock"
        assert ckpt["completed_windows"] == [0]


def test_self_healing_banned_lexicon_in_forward_template():
    """Verify raw LLM template with forensic words in RetrievalViews is healed and passes with 0 drops."""
    window_text = (
        "In complex investigations, the planning sequence reveals underlying actor capabilities. "
        "Analyzing resource sourcing and preparation time isolates genuine operational signatures."
    )
    window = DistillationWindow(
        window_id="test_window_001",
        book_id="crimeclassificationmanual",
        chapter_id="ch_02",
        chapter_title="Homicide Investigation Heuristics",
        chapter_type="TAXONOMIC",
        heading_path="Homicide > Modus Operandi",
        window_index=0,
        paragraphs=[],
        window_text=window_text,
        est_tokens=80,
        char_span=(0, len(window_text)),
        is_taxonomic=True,
    )

    # Raw LLM response containing banned words in title, procedure, and RetrievalViews
    raw_template = {
        "schema_version": "2.1",
        "template_id": "offender_planning_sequence",
        "title": "Offender Planning Sequence and Crime Staging",
        "tier": "a1_empirical_anomaly",
        "cognitive_function": "sequence_contradiction_matrix",
        "situation_frame": {
            "actors": ["decision_maker", "counterparty"],
            "info_asymmetry": "one_sided_hidden",
            "evidence_state": ["direct_telemetry", "circumstantial_pattern"],
            "incentive_posture": "strategic_deceptive",
            "system_dynamics": "iterative_feedback",
            "primary_uncertainty": ["root_cause", "actor_intent"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "An offender selects a vulnerable victim and stages the homicide scene.",
            "trigger_conditions": ["crime scene staging detected", "victim vulnerability exploitation"],
            "symptom_phrases": ["offender preparation signature", "homicide evidence manipulation"],
            "information_structure": "Temporal sequence of resource sourcing and offense execution.",
            "analog_vignettes": [
                "Distributed systems telemetry showing unpredicted queue buildup despite nominal CPU load.",
                "Financial market order book imbalance indicating hidden institutional liquidity absorption.",
                "Contract negotiation where counterparty selectively discloses audit logs to alter valuation.",
                "Physical logistics transponder anomaly where vessel AIS position conflicts with terminal records.",
            ],
        },
        "preconditions": ["access to sequential telemetry logs"],
        "contraindications": ["unverified single-source reports"],
        "procedure_steps": [
            "Reconstruct step-by-step preparation order of the suspect.",
            "Identify anomalies between resource sourcing and actual execution.",
        ],
        "diagnostic_probes": [
            {
                "question": "Does the preparation timeline match observed telemetry?",
                "discriminates_between": ["deliberate_staging", "opportunistic_action"],
                "verifiable_anchor_sought": "Physical telemetry or timestamped record.",
                "diagnosticity_weight": 0.85,
            }
        ],
        "never_asserts": ["clinical_diagnosis", "unverified_intent_as_fact"],
        "evidence_strength": "expert_consensus",
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            {
                "book_id": "crimeclassificationmanual",
                "chapter": "ch_02",
                "locator": "page_25",
                "char_span": [0, 50],
                "source_span_sha256": "sha256_mock",
                "verbatim_quote": "In complex investigations, the planning sequence reveals underlying actor capabilities.",
            }
        ],
    }

    # Heal and validate
    healed, status = heal_and_validate_template(
        raw_t=raw_template,
        window=window,
        is_omega=False,
        client=None,  # Layer 2 deterministic fallback handles test offline
        model="omni",
    )

    assert status == "VALID"
    assert healed is not None
    assert isinstance(healed, ForwardThoughtTemplate)

    # Verify that all banned words were eliminated from reasoning fields
    dump_dict = healed.model_dump(exclude={"provenance"})
    dump_str = json.dumps(dump_dict)
    assert BANNED_LEXICON.search(dump_str) is None

    # Verify provenance quote anchor remained verbatim and intact
    assert (
        healed.provenance[0].verbatim_quote
        == "In complex investigations, the planning sequence reveals underlying actor capabilities."
    )

    # Verify validate_and_filter_template accepts it with 0 drops
    val_res = validate_and_filter_template(healed, window, audit_mode="none")
    assert val_res.passed is True
    assert val_res.reason == "ACCEPTED"


def test_non_procedural_fluff_window_returns_empty():
    """Verify non-procedural fluff windows safely return [] without triggering circuit-breaker drops."""
    window = DistillationWindow(
        window_id="w_fluff",
        book_id="crimeclassificationmanual",
        chapter_id="ch_01",
        chapter_title="Front Matter",
        chapter_type="TAXONOMIC",
        heading_path="Title Page > Editorial Board",
        window_index=0,
        paragraphs=[],
        window_text="# Crime Classification Manual\nA Standard System for Investigating and Classifying Violent Crimes\nSecond Edition\nJohn E. Douglas, Ann W. Burgess",
        est_tokens=65,
        char_span=(0, 150),
        is_taxonomic=True,
    )

    # 1. Front-matter filter correctly flags it as fluff
    assert is_front_matter_or_fluff(window) is True

    # 2. Extraction prompt instructs empty templates return for fluff
    prompt = build_extraction_prompt(window, is_omega=False)
    assert "RULE A — TRIAGE FILTER" in prompt
    assert "OMIT IMMEDIATELY" in prompt

