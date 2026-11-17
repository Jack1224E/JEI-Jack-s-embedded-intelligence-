"""Pytest suite for Step 3: Tier A0 Seed, Deduplication, and Preserved-Dissent Invariant."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from j_ei.deduplicator import (
    canonical_strings_alpha,
    canonical_strings_omega,
    cluster_templates,
    deduplicate_vault,
    max_quote_jaccard,
    merge_alpha_templates,
    merge_omega_templates,
    quote_jaccard_similarity,
)
from j_ei.distiller import (
    DistillationWindow,
    is_front_matter_or_fluff,
    repair_template_with_audit,
    validate_and_filter_template,
)
from j_ei.schemas import (
    BANNED_LEXICON,
    CriticThoughtTemplate,
    DiagnosticProbe,
    EvidenceStrength,
    ForwardThoughtTemplate,
    RetrievalViews,
    SituationFrame,
    SourceAnchor,
    VaultTier,
    VulnerabilityCode,
)
from j_ei.seed_a0 import load_and_validate_tier_a0_seed


def _sample_anchor(quote: str, book_id: str = "test_book") -> SourceAnchor:
    sha = hashlib.sha256(quote.encode("utf-8")).hexdigest()
    return SourceAnchor(
        book_id=book_id,
        chapter="ch_01",
        locator="p10",
        char_span=(0, len(quote)),
        source_span_sha256=sha,
        verbatim_quote=quote,
    )


def _sample_forward_template(
    template_id: str,
    title: str = "Sample Test Strategy",
    quote: str = "First principles dictate that fundamental constraints govern system throughput.",
) -> ForwardThoughtTemplate:
    return ForwardThoughtTemplate(
        schema_version="2.1",
        template_id=template_id,
        title=title,
        tier=VaultTier.A0_SYSTEMS_CORE,
        cognitive_function="bottleneck_constraint_isolation",
        situation_frame=SituationFrame(
            actors=["decision_maker", "system_component"],
            info_asymmetry="symmetric",
            evidence_state=["direct_telemetry"],
            incentive_posture="cooperative",
            system_dynamics="static_one_shot",
            primary_uncertainty=["root_cause", "bottleneck_location"],
            spatial_or_forensic_flag=False,
        ),
        retrieval=RetrievalViews(
            situation_signature="System latency spikes due to an unindexed bottleneck.",
            trigger_conditions=["queue accumulation at processing node", "upstream throughput throttling"],
            symptom_phrases=["unexplained latency spike", "thread pool exhaustion"],
            information_structure="Pipeline DAG annotated with node latency.",
            analog_vignettes=[
                "Distributed Systems: Isolating an unindexed join in a Kafka stream.",
                "Financial Markets: Isolating clearinghouse settlement batch windows.",
                "Organizational Behavior: Identifying legal approval queues in product launches.",
                "Physical Logistics & Sensor Telemetry: Identifying customs document checkpoints in logistics.",
            ],
        ),
        preconditions=["access to node-level telemetry"],
        contraindications=["fully decentralized non-deterministic networks"],
        procedure_steps=[
            "Identify the single rate-limiting constraint in the pipeline.",
            "Subordinate all peripheral processes to the constraint tempo.",
        ],
        diagnostic_probes=[
            DiagnosticProbe(
                question="Does increasing capacity at this node scale total system throughput?",
                discriminates_between=("true_bottleneck", "local_subsystem"),
                verifiable_anchor_sought="Throughput metric comparison before and after capacity addition.",
                diagnosticity_weight=0.9,
            )
        ],
        evidence_strength=EvidenceStrength.EMPIRICAL_VALIDATED,
        provenance=[_sample_anchor(quote)],
    )


def _sample_critic_template(
    template_id: str,
    technique_name: str = "Analysis of Competing Hypotheses",
    quote: str = "Evaluating hypotheses by disproving evidence rather than seeking confirmation.",
) -> CriticThoughtTemplate:
    return CriticThoughtTemplate(
        schema_version="2.1",
        template_id=template_id,
        technique_name=technique_name,
        cost_tier="routed",
        target_vulnerabilities=[VulnerabilityCode.CONFIRMATION_SKEW, VulnerabilityCode.SINGLE_HYPOTHESIS_LOCKIN],
        detectable_symptoms=[
            "analyst searches only for corroborating evidence",
            "competing hypotheses dismissed without formal evaluation",
            "overconfidence in preliminary narrative",
        ],
        audit_procedure=[
            "Construct a complete matrix of mutually exclusive hypotheses against all observed evidence.",
            "Focus analysis strictly on diagnostic evidence that disproves specific candidate hypotheses.",
        ],
        kill_conditions=["hypothesis matrix fails to include obvious alternative explanations"],
        remediation_actions=["reweight_hypotheses", "requery_operational"],
        known_limitations=["high cognitive overhead in real-time operational triage"],
        provenance=[_sample_anchor(quote, book_id="psychology_of_intelligence")],
    )


def test_tier_a0_seed_schema_and_banned_lexicon() -> None:
    """Verify all 25 Tier A0 seed playbooks adhere to schema and pass BANNED_LEXICON."""
    templates = load_and_validate_tier_a0_seed()
    assert len(templates) == 25

    template_ids = set()
    for t in templates:
        assert t.tier == VaultTier.A0_SYSTEMS_CORE
        assert t.schema_version == "2.1"
        assert t.template_id not in template_ids, f"Duplicate ID: {t.template_id}"
        template_ids.add(t.template_id)

        # Ensure 4-6 analog vignettes
        vignettes = t.retrieval.analog_vignettes
        assert 4 <= len(vignettes) <= 6

        # Ensure provenance anchor matches sha256
        for p in t.provenance:
            calc_sha = hashlib.sha256(p.verbatim_quote.encode("utf-8")).hexdigest()
            assert p.source_span_sha256 == calc_sha

        # Ensure zero BANNED_LEXICON matches
        serialized = t.model_dump_json()
        match = BANNED_LEXICON.search(serialized)
        assert match is None, f"Leakage {match.group(0)!r} found in {t.template_id}"


def test_quote_jaccard_similarity() -> None:
    """Test verbatim quote Jaccard similarity computation."""
    q1 = "A system can run only as fast as its slowest process."
    q2 = "A system can run only as fast as its slowest process."
    assert quote_jaccard_similarity(q1, q2) == 1.0

    q3 = "Completely different text with zero common words."
    assert quote_jaccard_similarity(q1, q3) == 0.0

    q4 = "A pipeline can run only as fast as its slowest process."
    sim = quote_jaccard_similarity(q1, q4)
    assert 0.70 < sim < 1.0


def test_canonical_strings() -> None:
    """Test emission of Situation and Procedure views for Alpha and Omega."""
    alpha = _sample_forward_template("test_alpha")
    s_alpha, p_alpha = canonical_strings_alpha(alpha)
    assert "System latency spikes" in s_alpha
    assert "Procedures:" in p_alpha
    assert "Probe 1:" in p_alpha

    omega = _sample_critic_template("test_omega")
    s_omega, p_omega = canonical_strings_omega(omega)
    assert "Analysis of Competing Hypotheses" in s_omega
    assert "Audit Procedure:" in p_omega


def test_auto_merge_shortcut() -> None:
    """Verify auto-merge occurs when sim >= 0.93 and quote_jaccard >= 0.50."""
    quote = "A system runs only as fast as its bottleneck constraint."
    t1 = _sample_forward_template("template_primary", quote=quote)
    t2 = _sample_forward_template("template_secondary", quote=quote)

    # Mock similarity matrix with 0.95 similarity between t1 and t2
    sim_mat = np.array([[1.0, 0.95], [0.95, 1.0]], dtype=np.float32)

    with patch("j_ei.deduplicator.compute_similarity_matrix", return_value=sim_mat):
        masters, metrics = deduplicate_vault(
            templates=[t1, t2],
            is_omega=False,
            auto_merge_sim_threshold=0.93,
            auto_merge_quote_threshold=0.50,
        )

    assert len(masters) == 1
    assert metrics["auto_merged_count"] == 1
    master = masters[0]
    assert master.template_id == "template_primary"
    assert "template_secondary" in master.merged_from


def test_preserved_dissent_contradictory_invariant() -> None:
    """CRITICAL INVARIANT: CONTRADICTORY templates must NEVER be merged!

    Both must remain in the master list, and their IDs cross-linked in disputed_with.
    """
    t1 = _sample_forward_template("model_centralization_thesis")
    t2 = _sample_forward_template("model_decentralization_antithesis")

    # High borderline similarity (0.88), but differing quotes (q_sim < 0.50)
    sim_mat = np.array([[1.0, 0.88], [0.88, 1.0]], dtype=np.float32)

    with patch("j_ei.deduplicator.compute_similarity_matrix", return_value=sim_mat):
        with patch("j_ei.deduplicator.adjudicate_pair_relationship", return_value="CONTRADICTORY"):
            masters, metrics = deduplicate_vault(
                templates=[t1, t2],
                is_omega=False,
                auto_merge_sim_threshold=0.93,
                borderline_sim_threshold=0.82,
            )

    # Assert both templates survive!
    assert len(masters) == 2
    assert metrics["contradictory_preserved_count"] == 1
    assert metrics["auto_merged_count"] == 0
    assert metrics["llm_merged_count"] == 0

    master_ids = {m.template_id for m in masters}
    assert master_ids == {"model_centralization_thesis", "model_decentralization_antithesis"}

    # Assert mutual cross-linking in disputed_with
    m1 = next(m for m in masters if m.template_id == "model_centralization_thesis")
    m2 = next(m for m in masters if m.template_id == "model_decentralization_antithesis")
    assert "model_decentralization_antithesis" in m1.disputed_with
    assert "model_centralization_thesis" in m2.disputed_with


def test_cluster_templates_grouping() -> None:
    """Test AgglomerativeClustering groups similar items and caps cluster sizes."""
    # Create 4 items: items 0,1 are close (sim=0.9), items 2,3 are close (sim=0.9), but 0/1 are distant from 2/3 (sim=0.1)
    sim_mat = np.array(
        [
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.9],
            [0.1, 0.1, 0.9, 1.0],
        ],
        dtype=np.float32,
    )

    clusters = cluster_templates(sim_mat, distance_threshold=0.18, max_cluster_size=10)
    assert len(clusters) == 2
    cluster_sets = [set(c) for c in clusters]
    assert {0, 1} in cluster_sets
    assert {2, 3} in cluster_sets


def test_repair_required_enforcement_in_distiller() -> None:
    """Verify that when codex audit returns REPAIR_REQUIRED, repair loop is invoked and drops if still failing."""
    quote = "First principles dictate that fundamental constraints govern system throughput."
    t = _sample_forward_template("test_repair_template", quote=quote)

    window = DistillationWindow(
        window_id="w_test",
        book_id="test_book",
        chapter_id="ch_01",
        chapter_title="Test Chapter",
        chapter_type="TAXONOMIC",
        heading_path="Test Heading",
        window_index=0,
        paragraphs=[],
        window_text=f"Introductory text. {quote} Trailing text.",
        est_tokens=50,
        char_span=(0, 100),
        is_taxonomic=True,
    )

    audit_fail = {
        "passed": False,
        "structural_utility_score": 0.55,
        "leakage_terms": ["suspect"],
        "reasoning": "Contains leakage term suspect and low structural utility.",
        "recommended_action": "REPAIR_REQUIRED",
    }

    # When repair still leaves leakage, validate_and_filter_template must drop it
    with patch("j_ei.distiller.codex_audit_template", return_value=audit_fail):
        with patch("j_ei.distiller.repair_template_with_audit", return_value=None):
            res = validate_and_filter_template(
                template=t,
                window=window,
                audit_mode="all",
            )
            assert res.passed is False
            assert "REPAIR_FAILED" in res.reason or "CODEX_AUDIT" in res.reason


def test_front_matter_fluff_filter() -> None:
    """Verify front matter and copyright fluff are skipped."""
    window = DistillationWindow(
        window_id="w_0",
        book_id="book_1",
        chapter_id="ch_01",
        chapter_title="Front Matter",
        chapter_type="TAXONOMIC",
        heading_path="Table of Contents",
        window_index=0,
        paragraphs=[],
        window_text="Table of Contents\nAll rights reserved. Printed in the United States. ISBN 978-0-123456-78-9.",
        est_tokens=25,
        char_span=(0, 100),
        is_taxonomic=True,
    )
    assert is_front_matter_or_fluff(window) is True
