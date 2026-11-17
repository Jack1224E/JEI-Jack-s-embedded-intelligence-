"""Deterministic Epistemic Linter for Project J-EI.

Implements sub-15ms deterministic graph verification, Bayesian diagnosticity pruning,
quote anchor verification, and deduplication hashing without LLM calls.
"""

from __future__ import annotations

import hashlib
from typing import Any
from rapidfuzz import fuzz

from j_ei.schemas import ClaimNode, EvidenceCard, VulnerabilityCode


def verify_quote_anchor(quote: str, source_span: str, threshold: float = 90.0) -> bool:
    """Verify that quote verbatim or near-verbatim exists within source_span.
    
    Collapses whitespace, lowercases both strings, and calculates rapidfuzz partial ratio.
    """
    quote_clean = " ".join(quote.split()).lower()
    source_clean = " ".join(source_span.split()).lower()

    if not quote_clean or not source_clean:
        return False

    ratio = fuzz.partial_ratio(quote_clean, source_clean)
    return bool(ratio >= threshold)


def compute_evidence_hash(cards: list[EvidenceCard]) -> str:
    """Compute deterministic SHA-256 digest over sorted (citation_locator, pinpoint_excerpt) tuples.
    
    Ensures order-invariant deduplication hashing to detect cyclic retrieval loops.
    """
    tuples = sorted((card.citation_locator, card.pinpoint_excerpt) for card in cards)
    canonical_repr = "\n".join(f"{loc}::{excerpt}" for loc, excerpt in tuples)
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()


def lint_claim_graph(
    claims: list[ClaimNode],
    evidence: list[EvidenceCard],
) -> dict[str, Any]:
    """Execute deterministic <15ms epistemic scan on Claim-Evidence Graph.
    
    Returns:
        {
            "fatal_flags": list[str],
            "major_flags": list[str],
            "pruned_evidence_ids": list[str],
            "active_evidence_ids": list[str],
        }
    """
    fatal_flags: list[str] = []
    major_flags: list[str] = []

    # 1. Prune low-diagnosticity noise (< 0.15)
    pruned_evidence_ids: list[str] = []
    active_evidence_ids: list[str] = []
    active_evidence_map: dict[str, EvidenceCard] = {}

    for card in evidence:
        if card.diagnosticity < 0.15:
            pruned_evidence_ids.append(card.evidence_id)
        else:
            active_evidence_ids.append(card.evidence_id)
            active_evidence_map[card.evidence_id] = card

    # 2. Check MODEL_PRIOR_UNSUPPORTED_FACT (FATAL)
    # Any DOMAIN_RULE or OBSERVED_FACT claim relying solely on MODEL_PRIOR,
    # having verified_in_source == False, or having zero active supporting cards.
    for claim in claims:
        if claim.claim_type in ("DOMAIN_RULE", "OBSERVED_FACT"):
            active_supporting = [
                active_evidence_map[eid]
                for eid in claim.supporting_evidence_ids
                if eid in active_evidence_map
            ]

            # Zero active supporting cards (either none provided or all pruned)
            if not active_supporting:
                if VulnerabilityCode.MODEL_PRIOR_UNSUPPORTED_FACT.value not in fatal_flags:
                    fatal_flags.append(VulnerabilityCode.MODEL_PRIOR_UNSUPPORTED_FACT.value)
                break

            # Relies solely on MODEL_PRIOR
            if all(c.source_tier == "MODEL_PRIOR" for c in active_supporting):
                if VulnerabilityCode.MODEL_PRIOR_UNSUPPORTED_FACT.value not in fatal_flags:
                    fatal_flags.append(VulnerabilityCode.MODEL_PRIOR_UNSUPPORTED_FACT.value)
                break

            # Has unverified active supporting evidence
            if any(not c.verified_in_source for c in active_supporting):
                if VulnerabilityCode.MODEL_PRIOR_UNSUPPORTED_FACT.value not in fatal_flags:
                    fatal_flags.append(VulnerabilityCode.MODEL_PRIOR_UNSUPPORTED_FACT.value)
                break

    # 3. Check SINGLE_HYPOTHESIS_LOCKIN (MAJOR)
    # < 2 distinct hypothesis_id values exist or one holds > 80% of claims.
    if not claims:
        major_flags.append(VulnerabilityCode.SINGLE_HYPOTHESIS_LOCKIN.value)
    else:
        hyp_counts: dict[str, int] = {}
        for c in claims:
            hyp_counts[c.hypothesis_id] = hyp_counts.get(c.hypothesis_id, 0) + 1

        if len(hyp_counts) < 2:
            major_flags.append(VulnerabilityCode.SINGLE_HYPOTHESIS_LOCKIN.value)
        else:
            max_share = max(hyp_counts.values()) / len(claims)
            if max_share > 0.80:
                major_flags.append(VulnerabilityCode.SINGLE_HYPOTHESIS_LOCKIN.value)

    # 4. Check UNVERIFIED_SOURCE_DEPENDENCE (MAJOR)
    # > 30% of load-bearing claims rely on single_report warrants.
    # Load-bearing = >= 2 supporting evidence items OR proposition shared across multiple hypotheses.
    if claims:
        prop_to_hyps: dict[str, set[str]] = {}
        for c in claims:
            prop_key = c.proposition.strip().lower()
            prop_to_hyps.setdefault(prop_key, set()).add(c.hypothesis_id)

        load_bearing_claims = [
            c
            for c in claims
            if len(c.supporting_evidence_ids) >= 2
            or len(prop_to_hyps[c.proposition.strip().lower()]) > 1
        ]

        if load_bearing_claims:
            single_report_count = sum(
                1 for c in load_bearing_claims if c.warrant_link == "single_report"
            )
            if (single_report_count / len(load_bearing_claims)) > 0.30:
                major_flags.append(VulnerabilityCode.UNVERIFIED_SOURCE_DEPENDENCE.value)

    # 5. Check INFERENTIAL_LEAP (MAJOR)
    # CAUSAL_INFERENCE claims / verified active EvidenceCard count > 0.60.
    causal_count = sum(1 for c in claims if c.claim_type == "CAUSAL_INFERENCE")
    verified_active_evidence_count = sum(
        1 for c in active_evidence_map.values() if c.verified_in_source
    )

    if causal_count > 0:
        if verified_active_evidence_count == 0:
            major_flags.append(VulnerabilityCode.INFERENTIAL_LEAP.value)
        elif (causal_count / verified_active_evidence_count) > 0.60:
            major_flags.append(VulnerabilityCode.INFERENTIAL_LEAP.value)

    # 6. Check CONFIDENCE_MISCALIBRATION (MAJOR)
    # Any ESTABLISHED modality claim (or certainty language) has < 2 distinct verified active supporting cards.
    for c in claims:
        if c.modality == "ESTABLISHED" or c.asserts_certainty_language:
            distinct_verified = {
                eid
                for eid in c.supporting_evidence_ids
                if eid in active_evidence_map and active_evidence_map[eid].verified_in_source
            }
            if len(distinct_verified) < 2:
                if VulnerabilityCode.CONFIDENCE_MISCALIBRATION.value not in major_flags:
                    major_flags.append(VulnerabilityCode.CONFIDENCE_MISCALIBRATION.value)
                break

    return {
        "fatal_flags": fatal_flags,
        "major_flags": major_flags,
        "pruned_evidence_ids": pruned_evidence_ids,
        "active_evidence_ids": active_evidence_ids,
    }

