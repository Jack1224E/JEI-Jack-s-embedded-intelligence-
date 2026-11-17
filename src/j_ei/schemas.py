"""Domain-agnostic Pydantic v2 schemas for Project J-EI.

Defines schemas for Vault Alpha (Forward Cognitive Templates), Vault Omega (Adversarial Critic),
and the inter-stage Probabilistic Claim-Evidence Graph.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

BANNED_LEXICON = re.compile(
    r"\b(murder|homicide|serial|psychopath|victim|offender|crime|police|"
    r"detective|autopsy|rape|arson|burglary|assault|felony|suspect)\b",
    re.IGNORECASE,
)


class StrictModel(BaseModel):
    """Base model enforcing strict configuration: no extra fields and whitespace stripping."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class VaultTier(StrEnum):
    A0_SYSTEMS_CORE = "a0_systems_core"
    A1_EMPIRICAL_ANOMALY = "a1_empirical_anomaly"
    A2_ADVERSARIAL_GAME = "a2_adversarial_game"
    A3_SPATIAL_NICHE = "a3_spatial_niche"


class EvidenceStrength(StrEnum):
    EMPIRICAL_VALIDATED = "empirical_validated"
    EXPERT_CONSENSUS = "expert_consensus"
    PRACTITIONER_HEURISTIC = "practitioner_heuristic"
    CONTROVERSIAL_CONTESTED = "controversial_contested"


class SourceAnchor(StrictModel):
    book_id: str
    chapter: str
    locator: str
    char_span: tuple[int, int]
    source_span_sha256: str
    verbatim_quote: str = Field(max_length=180)


class ChapterState(StrictModel):
    """Rolling state ledger for 6,000-token sliding window book distillation."""

    entities: dict[str, str] = Field(default_factory=dict)
    open_threads: list[dict] = Field(default_factory=list)
    established_template_ids: list[str] = Field(default_factory=list)
    glossary_deltas: dict[str, str] = Field(default_factory=dict)


class SituationFrame(StrictModel):
    actors: list[
        Literal[
            "decision_maker",
            "counterparty",
            "intermediary",
            "gatekeeper",
            "system_component",
            "observer",
        ]
    ]
    info_asymmetry: Literal[
        "symmetric",
        "one_sided_hidden",
        "mutual_uncertainty",
        "actively_obfuscated",
    ]
    evidence_state: list[
        Literal[
            "direct_telemetry",
            "corroborated_records",
            "single_unverified_report",
            "circumstantial_pattern",
            "missing_baseline",
        ]
    ]
    incentive_posture: Literal[
        "cooperative",
        "competitive_rational",
        "strategic_deceptive",
        "coercive",
        "stochastic_system",
    ]
    system_dynamics: Literal[
        "static_one_shot",
        "iterative_feedback",
        "escalating",
        "anomalous_deviation",
    ]
    primary_uncertainty: list[
        Literal[
            "root_cause",
            "actor_intent",
            "sequence_order",
            "data_authenticity",
            "future_trajectory",
            "bottleneck_location",
        ]
    ]
    spatial_or_forensic_flag: bool = False


class DiagnosticProbe(StrictModel):
    question: str = Field(max_length=220)
    discriminates_between: tuple[str, str]
    verifiable_anchor_sought: str
    diagnosticity_weight: float = Field(ge=0.0, le=1.0)


class RetrievalViews(StrictModel):
    situation_signature: str = Field(max_length=600)
    trigger_conditions: list[str] = Field(min_length=2, max_length=6)
    symptom_phrases: list[str] = Field(min_length=2, max_length=8)
    information_structure: str = Field(max_length=300)
    analog_vignettes: list[str] = Field(min_length=3, max_length=6)

    @field_validator(
        "situation_signature",
        "trigger_conditions",
        "symptom_phrases",
        "information_structure",
        "analog_vignettes",
        mode="after",
    )
    @classmethod
    def enforce_domain_neutrality(cls, v: str | list[str]) -> str | list[str]:
        text = " ".join(v) if isinstance(v, list) else str(v)
        match = BANNED_LEXICON.search(text)
        if match:
            raise ValueError(
                f"Domain-specific leakage detected ({match.group(0)!r}) in retrieval view: {text}"
            )
        return v


class ForwardThoughtTemplate(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    template_id: str
    title: str = Field(max_length=120)
    tier: VaultTier
    cognitive_function: Literal[
        "abductive_hypothesis_generation",
        "baseline_anomaly_separation",
        "verifiability_deception_audit",
        "incentive_motive_mapping",
        "sequence_contradiction_matrix",
        "bottleneck_constraint_isolation",
        "risk_premortem_projection",
    ]
    situation_frame: SituationFrame
    retrieval: RetrievalViews
    preconditions: list[str] = Field(min_length=1, max_length=6)
    contraindications: list[str] = Field(min_length=1, max_length=6)
    procedure_steps: list[str] = Field(min_length=2, max_length=8)
    diagnostic_probes: list[DiagnosticProbe] = Field(min_length=1, max_length=6)
    never_asserts: list[str] = Field(
        default_factory=lambda: ["clinical_diagnosis", "unverified_intent_as_fact"]
    )
    evidence_strength: EvidenceStrength
    disputed_with: list[str] = Field(default_factory=list)
    merged_from: list[str] = Field(default_factory=list)
    provenance: list[SourceAnchor] = Field(min_length=1)


class VulnerabilityCode(StrEnum):
    SINGLE_HYPOTHESIS_LOCKIN = "SINGLE_HYPOTHESIS_LOCKIN"
    CONFIRMATION_SKEW = "CONFIRMATION_SKEW"
    UNVERIFIED_SOURCE_DEPENDENCE = "UNVERIFIED_SOURCE_DEPENDENCE"
    INFERENTIAL_LEAP = "INFERENTIAL_LEAP"
    UNTESTED_LOAD_BEARING_ASSUMPTION = "UNTESTED_LOAD_BEARING_ASSUMPTION"
    CONFIDENCE_MISCALIBRATION = "CONFIDENCE_MISCALIBRATION"
    MIRROR_IMAGING = "MIRROR_IMAGING"
    ABSENCE_OF_EVIDENCE_FALLACY = "ABSENCE_OF_EVIDENCE_FALLACY"
    LOW_DIAGNOSTICITY_NOISE = "LOW_DIAGNOSTICITY_NOISE"
    UNRESOLVED_SOURCE_CONFLICT = "UNRESOLVED_SOURCE_CONFLICT"
    MODEL_PRIOR_UNSUPPORTED_FACT = "MODEL_PRIOR_UNSUPPORTED_FACT"


class CriticThoughtTemplate(StrictModel):
    schema_version: Literal["2.1"] = "2.1"
    template_id: str
    technique_name: str = Field(max_length=120)
    cost_tier: Literal["always_on", "routed", "heavy"] = "routed"
    target_vulnerabilities: list[VulnerabilityCode] = Field(min_length=1)
    detectable_symptoms: list[str] = Field(min_length=3, max_length=8)
    audit_procedure: list[str] = Field(min_length=2, max_length=8)
    kill_conditions: list[str] = Field(min_length=1, max_length=6)
    remediation_actions: list[
        Literal[
            "requery_knowledge",
            "requery_empirical",
            "requery_operational",
            "reweight_hypotheses",
            "downgrade_modality",
            "drop_claim",
        ]
    ]
    known_limitations: list[str] = Field(min_length=1, max_length=5)
    provenance: list[SourceAnchor] = Field(min_length=1)


class EvidenceCard(StrictModel):
    evidence_id: str
    evidence_hash: str = ""
    source_tier: Literal[
        "PRIMARY_SPEC",
        "EMPIRICAL_BENCHMARK",
        "PEER_REVIEWED",
        "OPERATIONAL_SOP",
        "USER_PROVIDED",
        "MODEL_PRIOR",
    ]
    citation_locator: str
    pinpoint_excerpt: str = Field(max_length=240)
    verified_in_source: bool
    completeness_checked: bool = False
    diagnosticity: float = Field(ge=0.0, le=1.0)
    as_of_date: date


class ClaimNode(StrictModel):
    claim_id: str
    proposition: str = Field(max_length=220)
    claim_type: Literal[
        "OBSERVED_FACT",
        "DOMAIN_RULE",
        "CAUSAL_INFERENCE",
        "OPERATIONAL_CONSTRAINT",
        "ASSUMPTION",
    ]
    modality: Literal["ESTABLISHED", "HIGH_PROBABILITY", "PLAUSIBLE", "SPECULATIVE"]
    asserts_certainty_language: bool = False
    warrant_link: Literal[
        "direct_measurement",
        "verified_documentation",
        "corroborated_inference",
        "single_report",
        "unverified_leap",
    ]
    supporting_evidence_ids: list[str]
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    hypothesis_id: str

