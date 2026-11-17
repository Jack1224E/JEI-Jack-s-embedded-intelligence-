"""Test suite for Step 1 of Project J-EI.

Tests NIM configuration loader, MRL vector normalization, Pydantic v2 schemas,
domain-neutrality regex filtering, epistemic linter, and evidence hashing.
"""

from __future__ import annotations

import math
from datetime import date
from unittest.mock import MagicMock
import numpy as np
import pytest
from pydantic import ValidationError

from j_ei.config import (
    JEIConfig,
    embed_texts,
    normalize_mrl_vector,
    resolve_agentrouter_api_key,
)
from j_ei.linter import (
    compute_evidence_hash,
    lint_claim_graph,
    verify_quote_anchor,
)
from j_ei.schemas import (
    BANNED_LEXICON,
    ChapterState,
    ClaimNode,
    CriticThoughtTemplate,
    DiagnosticProbe,
    EvidenceCard,
    EvidenceStrength,
    ForwardThoughtTemplate,
    RetrievalViews,
    SituationFrame,
    SourceAnchor,
    VaultTier,
    VulnerabilityCode,
)


# =====================================================================
# 1. Configuration & Key Resolution Tests
# =====================================================================

def test_jei_config_defaults() -> None:
    cfg = JEIConfig()
    assert cfg.codex_model in ("gpt-6-astra", "gpt-6.1-sol", "gpt-5.6-sol")
    assert cfg.agentrouter_base_url == "https://agentrouter.org/v1"
    assert cfg.embed_model_name == "BAAI/bge-small-en-v1.5"
    assert cfg.embed_dim == 384
    assert cfg.codex_timeout in (120, 180)
    assert cfg.searxng_url == "http://localhost:8888"


def test_key_resolution_agentrouter() -> None:
    mock_env = {
        "AGENTROUTER_API_KEY": "mock-agentrouter-key",
    }
    assert resolve_agentrouter_api_key(mock_env) == "mock-agentrouter-key"

    mock_env2 = {
        "CODEX_GATEWAY_API_KEY": "Bearer mock-gateway-key",
    }
    assert resolve_agentrouter_api_key(mock_env2) == "mock-gateway-key"


# =====================================================================
# 2. Vector Normalization & Embedding Wrapper Tests
# =====================================================================

def test_normalize_mrl_vector_l2_norm() -> None:
    raw_vec = [float(i + 1) for i in range(512)]
    normalized = normalize_mrl_vector(raw_vec, target_dim=384)

    assert len(normalized) == 384
    l2_norm = math.sqrt(sum(x * x for x in normalized))
    assert pytest.approx(l2_norm, abs=1e-6) == 1.0


def test_normalize_mrl_vector_zero_norm_raises() -> None:
    zero_vec = [0.0] * 384
    with pytest.raises(ValueError, match="Zero-norm or invalid vector"):
        normalize_mrl_vector(zero_vec, target_dim=384)


def test_normalize_mrl_vector_dimension_too_short_raises() -> None:
    short_vec = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="smaller than requested target_dim"):
        normalize_mrl_vector(short_vec, target_dim=384)


def test_embed_texts_empty_and_invalid_type() -> None:
    assert embed_texts([], input_type="query") == []
    with pytest.raises(ValueError, match="input_type must be 'passage' or 'query'"):
        embed_texts(["test"], input_type="invalid")  # type: ignore[arg-type]


# =====================================================================
# 3. Pydantic v2 Schemas & Domain-Neutrality Regex Tests
# =====================================================================

def test_clean_retrieval_views_passes() -> None:
    views = RetrievalViews(
        situation_signature="Anomalous network bottleneck under escalating load feedback.",
        trigger_conditions=["latency p99 spike", "packet drop rate exceeds 5%"],
        symptom_phrases=["intermittent timeout", "queue saturation"],
        information_structure="Directed acyclic graph with asymmetric telemetry feeds.",
        analog_vignettes=[
            "Distributed cache thrashing under concurrent write bursts.",
            "Semiconductor assembly pipeline delayed by wafer shipment halt.",
            "Financial clearinghouse settlement backlog due to clock drift.",
        ],
    )
    assert len(views.trigger_conditions) == 2
    assert len(views.analog_vignettes) == 3


@pytest.mark.parametrize(
    "banned_word",
    [
        "murder",
        "homicide",
        "serial",
        "psychopath",
        "victim",
        "offender",
        "crime",
        "police",
        "detective",
        "autopsy",
        "rape",
        "arson",
        "burglary",
        "assault",
        "felony",
        "suspect",
    ],
)
def test_retrieval_views_banned_lexicon_rejection(banned_word: str) -> None:
    clean_kwargs = {
        "situation_signature": "Anomalous network bottleneck under escalating load feedback.",
        "trigger_conditions": ["latency p99 spike", "packet drop rate exceeds 5%"],
        "symptom_phrases": ["intermittent timeout", "queue saturation"],
        "information_structure": "Directed acyclic graph with asymmetric telemetry feeds.",
        "analog_vignettes": [
            "Distributed cache thrashing under concurrent write bursts.",
            "Semiconductor assembly pipeline delayed by wafer shipment halt.",
            "Financial clearinghouse settlement backlog due to clock drift.",
        ],
    }

    # Test signature contamination
    with pytest.raises(ValidationError, match="Domain-specific leakage detected"):
        dirty = dict(clean_kwargs)
        dirty["situation_signature"] = f"A {banned_word} was observed in telemetry"
        RetrievalViews(**dirty)

    # Test trigger conditions contamination
    with pytest.raises(ValidationError, match="Domain-specific leakage detected"):
        dirty = dict(clean_kwargs)
        dirty["trigger_conditions"] = ["normal check", f"identifying {banned_word} pattern"]
        RetrievalViews(**dirty)

    # Test information structure contamination
    with pytest.raises(ValidationError, match="Domain-specific leakage detected"):
        dirty = dict(clean_kwargs)
        dirty["information_structure"] = f"Structure containing {banned_word} logs."
        RetrievalViews(**dirty)


def test_strict_models_forbid_extra_fields() -> None:
    anchor_data = {
        "book_id": "sat-2019",
        "chapter": "ch04",
        "locator": "p.88",
        "char_span": (100, 250),
        "source_span_sha256": "abc123sha",
        "verbatim_quote": "Key assumptions must be stated explicitly.",
        "unauthorized_field": "disallowed",
    }
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SourceAnchor(**anchor_data)


def test_forward_thought_template_instantiation() -> None:
    anchor = SourceAnchor(
        book_id="sat-2019",
        chapter="ch04",
        locator="p.88",
        char_span=(100, 250),
        source_span_sha256="abc123sha",
        verbatim_quote="Key assumptions must be stated explicitly.",
    )
    frame = SituationFrame(
        actors=["decision_maker", "counterparty"],
        info_asymmetry="one_sided_hidden",
        evidence_state=["direct_telemetry", "corroborated_records"],
        incentive_posture="strategic_deceptive",
        system_dynamics="escalating",
        primary_uncertainty=["root_cause", "actor_intent"],
        spatial_or_forensic_flag=False,
    )
    views = RetrievalViews(
        situation_signature="Asymmetric incentive divergence across competing subsystems.",
        trigger_conditions=["divergent telemetry reports", "unexplained state deviation"],
        symptom_phrases=["unverified status claims", "contradictory metrics"],
        information_structure="Bi-level game tree with hidden node payoffs.",
        analog_vignettes=[
            "Microservice cluster reporting conflicting health states.",
            "Supply contract renegotiation under unannounced production constraints.",
            "Aviation fleet maintenance schedule drift against flight log entries.",
        ],
    )
    probe = DiagnosticProbe(
        question="Does telemetry correlate with secondary independent sensor timestamps?",
        discriminates_between=("H1_systemic_clock_drift", "H2_intentional_spoofing"),
        verifiable_anchor_sought="Hardware NTP log hashes",
        diagnosticity_weight=0.92,
    )

    template = ForwardThoughtTemplate(
        schema_version="2.1",
        template_id="FT-SYS-001",
        title="Divergent Telemetry Decomposition",
        tier=VaultTier.A0_SYSTEMS_CORE,
        cognitive_function="baseline_anomaly_separation",
        situation_frame=frame,
        retrieval=views,
        preconditions=["Direct logging telemetry available"],
        contraindications=["Total absence of clock synchronization data"],
        procedure_steps=[
            "Extract independent secondary sensor logs",
            "Align timestamps along monotonic hardware clock",
        ],
        diagnostic_probes=[probe],
        evidence_strength=EvidenceStrength.EMPIRICAL_VALIDATED,
        provenance=[anchor],
    )
    assert template.template_id == "FT-SYS-001"
    assert template.tier == VaultTier.A0_SYSTEMS_CORE


def test_critic_thought_template_instantiation() -> None:
    anchor = SourceAnchor(
        book_id="heuer-psych-1999",
        chapter="ch08",
        locator="p.95",
        char_span=(50, 180),
        source_span_sha256="sha888",
        verbatim_quote="Evaluating alternative explanations simultaneously reduces bias.",
    )
    critic = CriticThoughtTemplate(
        schema_version="2.1",
        template_id="CT-ACH-001",
        technique_name="Analysis of Competing Hypotheses",
        cost_tier="always_on",
        target_vulnerabilities=[VulnerabilityCode.SINGLE_HYPOTHESIS_LOCKIN],
        detectable_symptoms=[
            "Single narrative dominates argumentation",
            "Absence of explicit rival propositions",
            "Unquestioned acceptance of early telemetry",
        ],
        audit_procedure=[
            "List all mutually exclusive rival explanations",
            "Evaluate evidence diagnosticity across each hypothesis matrix",
        ],
        kill_conditions=["Rival hypothesis explains identical evidence with fewer assumptions"],
        remediation_actions=["reweight_hypotheses", "downgrade_modality"],
        known_limitations=["Requires explicit generation of non-obvious alternatives"],
        provenance=[anchor],
    )
    assert critic.technique_name == "Analysis of Competing Hypotheses"


# =====================================================================
# 4. Quote Verification & Evidence Hashing Tests
# =====================================================================

def test_verify_quote_anchor() -> None:
    source_span = (
        "In complex investigations, the analyst must ensure that all key assumptions "
        "are stated explicitly and subjected to systematic challenge before reaching conclusions."
    )
    verbatim_quote = "all key assumptions are stated explicitly and subjected to systematic challenge"
    assert verify_quote_anchor(verbatim_quote, source_span, threshold=90.0) is True

    # Whitespace and case variations
    variant_quote = "  ALL key   assumptions ARE stated explicitly   and subjected to systematic challenge\n"
    assert verify_quote_anchor(variant_quote, source_span, threshold=90.0) is True

    # Fabricated / completely irrelevant quote
    fabricated_quote = "The quick brown fox jumps over the lazy dog."
    assert verify_quote_anchor(fabricated_quote, source_span, threshold=90.0) is False

    # Empty inputs
    assert verify_quote_anchor("", source_span) is False
    assert verify_quote_anchor(verbatim_quote, "") is False


def test_compute_evidence_hash_order_invariance() -> None:
    card1 = EvidenceCard(
        evidence_id="E-001",
        source_tier="PRIMARY_SPEC",
        citation_locator="RFC 7540 sec 5.1",
        pinpoint_excerpt="Stream states: idle, reserved, open, half-closed, closed.",
        verified_in_source=True,
        diagnosticity=0.85,
        as_of_date=date(2026, 1, 1),
    )
    card2 = EvidenceCard(
        evidence_id="E-002",
        source_tier="OPERATIONAL_SOP",
        citation_locator="Runbook sec 3.2",
        pinpoint_excerpt="Restart proxy when stream state count exceeds 10,000.",
        verified_in_source=True,
        diagnosticity=0.60,
        as_of_date=date(2026, 2, 1),
    )

    hash_order_a = compute_evidence_hash([card1, card2])
    hash_order_b = compute_evidence_hash([card2, card1])
    assert hash_order_a == hash_order_b

    # Modifying excerpt changes hash
    card2_mod = card2.model_copy(update={"pinpoint_excerpt": "Restart proxy when stream count > 5,000."})
    hash_order_c = compute_evidence_hash([card1, card2_mod])
    assert hash_order_a != hash_order_c


# =====================================================================
# 5. Deterministic Epistemic Linter Tests
# =====================================================================

def test_linter_pruning_low_diagnosticity_noise() -> None:
    card_high = EvidenceCard(
        evidence_id="E-HIGH",
        source_tier="PRIMARY_SPEC",
        citation_locator="Spec A",
        pinpoint_excerpt="High diagnosticity spec excerpt",
        verified_in_source=True,
        diagnosticity=0.80,
        as_of_date=date(2026, 1, 1),
    )
    card_noise = EvidenceCard(
        evidence_id="E-NOISE",
        source_tier="OPERATIONAL_SOP",
        citation_locator="Log B",
        pinpoint_excerpt="Generic log line that applies to all hypotheses",
        verified_in_source=True,
        diagnosticity=0.10,  # Below 0.15 threshold
        as_of_date=date(2026, 1, 1),
    )

    claim = ClaimNode(
        claim_id="C-001",
        proposition="Subsystem A timed out.",
        claim_type="OBSERVED_FACT",
        modality="PLAUSIBLE",
        warrant_link="verified_documentation",
        supporting_evidence_ids=["E-HIGH", "E-NOISE"],
        hypothesis_id="H1",
    )
    claim_h2 = ClaimNode(
        claim_id="C-002",
        proposition="Subsystem B caused backpressure.",
        claim_type="CAUSAL_INFERENCE",
        modality="PLAUSIBLE",
        warrant_link="corroborated_inference",
        supporting_evidence_ids=["E-HIGH"],
        hypothesis_id="H2",
    )

    res = lint_claim_graph([claim, claim_h2], [card_high, card_noise])
    assert "E-NOISE" in res["pruned_evidence_ids"]
    assert "E-HIGH" in res["active_evidence_ids"]
    assert "E-NOISE" not in res["active_evidence_ids"]


def test_linter_model_prior_unsupported_fact_trigger() -> None:
    # 1. OBSERVED_FACT relying solely on MODEL_PRIOR
    card_prior = EvidenceCard(
        evidence_id="E-PRIOR",
        source_tier="MODEL_PRIOR",
        citation_locator="LLM internal prior",
        pinpoint_excerpt="Model pre-trained association",
        verified_in_source=True,
        diagnosticity=0.50,
        as_of_date=date(2026, 1, 1),
    )
    claim_prior = ClaimNode(
        claim_id="C-001",
        proposition="Port 443 is blocked by corporate firewall.",
        claim_type="OBSERVED_FACT",
        modality="HIGH_PROBABILITY",
        warrant_link="single_report",
        supporting_evidence_ids=["E-PRIOR"],
        hypothesis_id="H1",
    )
    claim_h2 = ClaimNode(
        claim_id="C-002",
        proposition="DNS resolution failed.",
        claim_type="DOMAIN_RULE",
        modality="PLAUSIBLE",
        warrant_link="verified_documentation",
        supporting_evidence_ids=["E-PRIOR"],
        hypothesis_id="H2",
    )

    res = lint_claim_graph([claim_prior, claim_h2], [card_prior])
    assert VulnerabilityCode.MODEL_PRIOR_UNSUPPORTED_FACT.value in res["fatal_flags"]

    # 2. DOMAIN_RULE with verified_in_source == False
    card_unverified = EvidenceCard(
        evidence_id="E-UNVERIFIED",
        source_tier="PRIMARY_SPEC",
        citation_locator="Spec sec 9",
        pinpoint_excerpt="Unverified excerpt",
        verified_in_source=False,
        diagnosticity=0.70,
        as_of_date=date(2026, 1, 1),
    )
    claim_unverified = ClaimNode(
        claim_id="C-003",
        proposition="Timeout must reset connection.",
        claim_type="DOMAIN_RULE",
        modality="HIGH_PROBABILITY",
        warrant_link="verified_documentation",
        supporting_evidence_ids=["E-UNVERIFIED"],
        hypothesis_id="H1",
    )
    res2 = lint_claim_graph([claim_unverified, claim_h2], [card_unverified])
    assert VulnerabilityCode.MODEL_PRIOR_UNSUPPORTED_FACT.value in res2["fatal_flags"]


def test_linter_single_hypothesis_lockin_trigger() -> None:
    card = EvidenceCard(
        evidence_id="E-001",
        source_tier="PRIMARY_SPEC",
        citation_locator="Spec",
        pinpoint_excerpt="Excerpt",
        verified_in_source=True,
        diagnosticity=0.60,
        as_of_date=date(2026, 1, 1),
    )
    # Only H1 claims
    claims_h1_only = [
        ClaimNode(
            claim_id=f"C-{i}",
            proposition=f"Proposition {i}",
            claim_type="OBSERVED_FACT",
            modality="PLAUSIBLE",
            warrant_link="verified_documentation",
            supporting_evidence_ids=["E-001"],
            hypothesis_id="H1",
        )
        for i in range(4)
    ]
    res1 = lint_claim_graph(claims_h1_only, [card])
    assert VulnerabilityCode.SINGLE_HYPOTHESIS_LOCKIN.value in res1["major_flags"]

    # 9 claims for H1, 1 claim for H2 (90% > 80%)
    claims_skewed = list(claims_h1_only) + [
        ClaimNode(
            claim_id=f"C-{i}",
            proposition=f"Proposition {i}",
            claim_type="OBSERVED_FACT",
            modality="PLAUSIBLE",
            warrant_link="verified_documentation",
            supporting_evidence_ids=["E-001"],
            hypothesis_id="H1",
        )
        for i in range(4, 9)
    ] + [
        ClaimNode(
            claim_id="C-H2",
            proposition="Rival claim",
            claim_type="OBSERVED_FACT",
            modality="PLAUSIBLE",
            warrant_link="verified_documentation",
            supporting_evidence_ids=["E-001"],
            hypothesis_id="H2",
        )
    ]
    res2 = lint_claim_graph(claims_skewed, [card])
    assert VulnerabilityCode.SINGLE_HYPOTHESIS_LOCKIN.value in res2["major_flags"]


def test_linter_unverified_source_dependence_trigger() -> None:
    card1 = EvidenceCard(
        evidence_id="E-001",
        source_tier="PRIMARY_SPEC",
        citation_locator="Spec",
        pinpoint_excerpt="Spec text",
        verified_in_source=True,
        diagnosticity=0.70,
        as_of_date=date(2026, 1, 1),
    )
    card2 = EvidenceCard(
        evidence_id="E-002",
        source_tier="PRIMARY_SPEC",
        citation_locator="Spec 2",
        pinpoint_excerpt="Spec text 2",
        verified_in_source=True,
        diagnosticity=0.70,
        as_of_date=date(2026, 1, 1),
    )

    # 2 load-bearing claims with >= 2 supporting evidence items, both relying on single_report warrants (100% > 30%)
    c1 = ClaimNode(
        claim_id="C-001",
        proposition="Load bearing premise 1",
        claim_type="DOMAIN_RULE",
        modality="PLAUSIBLE",
        warrant_link="single_report",
        supporting_evidence_ids=["E-001", "E-002"],
        hypothesis_id="H1",
    )
    c2 = ClaimNode(
        claim_id="C-002",
        proposition="Load bearing premise 2",
        claim_type="DOMAIN_RULE",
        modality="PLAUSIBLE",
        warrant_link="single_report",
        supporting_evidence_ids=["E-001", "E-002"],
        hypothesis_id="H2",
    )

    res = lint_claim_graph([c1, c2], [card1, card2])
    assert VulnerabilityCode.UNVERIFIED_SOURCE_DEPENDENCE.value in res["major_flags"]


def test_linter_inferential_leap_trigger() -> None:
    card = EvidenceCard(
        evidence_id="E-001",
        source_tier="PRIMARY_SPEC",
        citation_locator="Spec",
        pinpoint_excerpt="Spec text",
        verified_in_source=True,
        diagnosticity=0.70,
        as_of_date=date(2026, 1, 1),
    )
    # 2 CAUSAL_INFERENCE claims with 1 verified EvidenceCard -> ratio 2.0 > 0.60
    c1 = ClaimNode(
        claim_id="C-001",
        proposition="Inference 1",
        claim_type="CAUSAL_INFERENCE",
        modality="PLAUSIBLE",
        warrant_link="corroborated_inference",
        supporting_evidence_ids=["E-001"],
        hypothesis_id="H1",
    )
    c2 = ClaimNode(
        claim_id="C-002",
        proposition="Inference 2",
        claim_type="CAUSAL_INFERENCE",
        modality="PLAUSIBLE",
        warrant_link="corroborated_inference",
        supporting_evidence_ids=["E-001"],
        hypothesis_id="H2",
    )
    res = lint_claim_graph([c1, c2], [card])
    assert VulnerabilityCode.INFERENTIAL_LEAP.value in res["major_flags"]


def test_linter_confidence_miscalibration_trigger() -> None:
    card1 = EvidenceCard(
        evidence_id="E-001",
        source_tier="PRIMARY_SPEC",
        citation_locator="Spec",
        pinpoint_excerpt="Spec text",
        verified_in_source=True,
        diagnosticity=0.70,
        as_of_date=date(2026, 1, 1),
    )
    # ESTABLISHED claim with only 1 supporting card (< 2)
    c1 = ClaimNode(
        claim_id="C-001",
        proposition="This is definitively established.",
        claim_type="OBSERVED_FACT",
        modality="ESTABLISHED",
        warrant_link="direct_measurement",
        supporting_evidence_ids=["E-001"],
        hypothesis_id="H1",
    )
    c2 = ClaimNode(
        claim_id="C-002",
        proposition="Rival possibility.",
        claim_type="OBSERVED_FACT",
        modality="PLAUSIBLE",
        warrant_link="direct_measurement",
        supporting_evidence_ids=["E-001"],
        hypothesis_id="H2",
    )

    res = lint_claim_graph([c1, c2], [card1])
    assert VulnerabilityCode.CONFIDENCE_MISCALIBRATION.value in res["major_flags"]


def test_linter_clean_balanced_graph() -> None:
    card1 = EvidenceCard(
        evidence_id="E-001",
        source_tier="PRIMARY_SPEC",
        citation_locator="Spec A",
        pinpoint_excerpt="Verified spec A excerpt",
        verified_in_source=True,
        diagnosticity=0.85,
        as_of_date=date(2026, 1, 1),
    )
    card2 = EvidenceCard(
        evidence_id="E-002",
        source_tier="EMPIRICAL_BENCHMARK",
        citation_locator="Benchmark B",
        pinpoint_excerpt="Verified benchmark B excerpt",
        verified_in_source=True,
        diagnosticity=0.80,
        as_of_date=date(2026, 1, 1),
    )

    c1 = ClaimNode(
        claim_id="C-001",
        proposition="Subsystem A experienced high latency.",
        claim_type="OBSERVED_FACT",
        modality="ESTABLISHED",
        warrant_link="direct_measurement",
        supporting_evidence_ids=["E-001", "E-002"],
        hypothesis_id="H1",
    )
    c2 = ClaimNode(
        claim_id="C-002",
        proposition="Subsystem B queue limit was reached.",
        claim_type="OBSERVED_FACT",
        modality="HIGH_PROBABILITY",
        warrant_link="verified_documentation",
        supporting_evidence_ids=["E-002"],
        hypothesis_id="H2",
    )

    res = lint_claim_graph([c1, c2], [card1, card2])
    assert res["fatal_flags"] == []
    assert res["major_flags"] == []
    assert len(res["active_evidence_ids"]) == 2
    assert len(res["pruned_evidence_ids"]) == 0

