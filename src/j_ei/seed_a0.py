"""Tier A0 Systems Core Seed Knowledge Playbooks for Project J-EI.

Defines 25 foundational first-principles cognitive templates spanning core analytical,
epistemological, cybernetic, and systems frameworks missing from empirical forensic literature.
All templates adhere strictly to the ForwardThoughtTemplate Pydantic v2 schema, VaultTier.A0_SYSTEMS_CORE,
contain 4-6 cross-domain analog vignettes, and strictly pass the BANNED_LEXICON deterministic linter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from j_ei.codex_bridge import codex_audit_template
from j_ei.schemas import (
    BANNED_LEXICON,
    DiagnosticProbe,
    EvidenceStrength,
    ForwardThoughtTemplate,
    RetrievalViews,
    SituationFrame,
    SourceAnchor,
    VaultTier,
)

logger = logging.getLogger("j_ei.seed_a0")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _make_anchor(quote: str, chapter: str, locator: str) -> SourceAnchor:
    """Helper to generate a valid SourceAnchor for Tier A0 seed playbooks."""
    sha = hashlib.sha256(quote.encode("utf-8")).hexdigest()
    return SourceAnchor(
        book_id="tier_a0_seed",
        chapter=chapter,
        locator=locator,
        char_span=(0, len(quote)),
        source_span_sha256=sha,
        verbatim_quote=quote,
    )


RAW_SEED_DEFINITIONS: list[dict] = [
    {
        "template_id": "a0_first_principles_decomposition",
        "title": "First-Principles Foundational Decomposition",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "bottleneck_constraint_isolation",
        "situation_frame": {
            "actors": ["decision_maker", "system_component"],
            "info_asymmetry": "symmetric",
            "evidence_state": ["direct_telemetry", "corroborated_records"],
            "incentive_posture": "cooperative",
            "system_dynamics": "static_one_shot",
            "primary_uncertainty": ["root_cause", "bottleneck_location"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "System analysis is obfuscated by accumulated historical conventions, analogies, or inherited operational assumptions that mask fundamental physical, logical, or economic boundaries.",
            "trigger_conditions": [
                "inherited design assumptions create cost or performance plateaus",
                "conventional heuristics produce diminishing returns on architectural improvements",
            ],
            "symptom_phrases": [
                "we have always built it this way",
                "industry standard heuristic failure",
                "unexplained cost-efficiency ceiling",
                "premature optimization of legacy abstractions",
            ],
            "information_structure": "Hierarchical component tree mapping high-level system functions down to fundamental irreducible invariants.",
            "analog_vignettes": [
                "Distributed Systems: Breaking down distributed database write latency into raw NVMe persistence, PCIe transfer limits, and Paxos consensus quorum correctness constraints rather than accepting cluster-wide defaults.",
                "Financial Markets: Recomputing market-making inventory risk models from first-principles order-book queue dynamics, adverse selection probabilities, and matching engine throughput rather than relying on Black-Scholes approximations.",
                "Game Theory & Mechanism Design: Deriving dominant strategy incentive compatibility and Nash equilibria directly from agent utility functions and payoff matrices rather than assuming voluntary cooperative compliance.",
                "Organizational Behavior: Deconstructing enterprise procurement cycles into raw legal review duration, compliance sign-offs, and administrative idle time rather than accepting standard multi-quarter vendor onboarding schedules.",
                "Physical Logistics & Sensor Telemetry: Tracing cargo vessel supply chain routes by calculating physical fuel burn, deadweight tonnage displacement, and maritime canal chokepoints rather than relying on declared cargo manifests.",
            ],
        },
        "preconditions": [
            "access to direct physical, logical, or structural telemetry",
            "mandate to challenge legacy institutional rules and conventions",
        ],
        "contraindications": [
            "emergency operational triage where rapid heuristic execution outweighs architectural redesign",
        ],
        "procedure_steps": [
            "Identify and catalogue every inherited premise, industry analogy, and operational assumption governing current system design.",
            "Distinguish verified physical and mathematical constraints from unvalidated operational assumptions.",
            "Strip all secondary and tertiary layers until reaching irreducible foundational constraints such as thermodynamics, arithmetic, or raw computational capacity.",
            "Reconstruct the system architecture from these foundational truths upward, evaluating optimal combinations without historical precedent.",
        ],
        "diagnostic_probes": [
            {
                "question": "Is this design boundary dictated by fundamental physical limits or merely inherited procedural convenience?",
                "discriminates_between": ["physical_invariant_boundary", "inherited_convention_artifact"],
                "verifiable_anchor_sought": "Empirical telemetry measuring theoretical hardware or mathematical limits versus observed production parameters.",
                "diagnosticity_weight": 0.95,
            }
        ],
        "never_asserts": ["unsupported_historical_analogy", "conventional_wisdom_as_fact"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="In every discipline, knowledge emerges from understanding fundamental elements and principles rather than superficial analogies.",
                chapter="ch_01_foundations",
                locator="section_epistemology_p01",
            )
        ],
    },
    {
        "template_id": "a0_theory_of_constraints",
        "title": "Theory of Constraints Bottleneck Isolation",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "bottleneck_constraint_isolation",
        "situation_frame": {
            "actors": ["decision_maker", "system_component", "gatekeeper"],
            "info_asymmetry": "one_sided_hidden",
            "evidence_state": ["direct_telemetry", "missing_baseline"],
            "incentive_posture": "cooperative",
            "system_dynamics": "iterative_feedback",
            "primary_uncertainty": ["bottleneck_location", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "End-to-end throughput is throttled by a single rate-limiting stage whose capacity limits total output, while upstream and downstream optimizations produce local efficiency gains without system-wide improvement.",
            "trigger_conditions": [
                "upstream inventory or work-in-progress queue accumulation before a single stage",
                "downstream stages starvation despite high resource utilization across adjacent nodes",
            ],
            "symptom_phrases": [
                "growing backlog at specific pipeline stage",
                "local optimizations yield zero end-to-end throughput gain",
                "subsystem overutilization without delivery impact",
            ],
            "information_structure": "Directed acyclic pipeline graph annotated with stage capacities, queue depths, and processing latency.",
            "analog_vignettes": [
                "Distributed Systems: In an ETL pipeline, isolating an unindexed database join causing kafka consumer lag rather than prematurely scaling upstream ingest pods.",
                "Financial Markets: Isolating clearinghouse settlement window bottlenecks throttling trade velocity despite high-frequency exchange execution speeds.",
                "Organizational Behavior: Identifying executive legal review queues as the single throughput gatekeeper for multi-team software feature launches.",
                "Physical Logistics & Sensor Telemetry: Isolating customs document verification clearance rates as the true rate-limiting factor in an observed maritime logistics cluster.",
            ],
        },
        "preconditions": [
            "complete sequential flow visibility across all stages",
            "quantifiable throughput and queue metrics",
        ],
        "contraindications": [
            "unstable non-deterministic pipelines where the constraint shifts dynamically from transaction to transaction",
        ],
        "procedure_steps": [
            "Map the sequential workflow and identify the single constraint node exhibiting maximum queue accumulation and continuous utilization.",
            "Exploit the constraint by ensuring it operates at 100 percent productive capacity with zero idle time or wasted cycles.",
            "Subordinate all upstream and downstream processes to pace their work strictly to the constraint processing rate.",
            "Elevate the constraint through capacity addition, structural re-engineering, or architectural offloading.",
        ],
        "diagnostic_probes": [
            {
                "question": "Does adding capacity to this specific node increase end-to-end system throughput proportionally?",
                "discriminates_between": ["true_system_bottleneck", "non_constraining_local_subsystem"],
                "verifiable_anchor_sought": "Differential throughput telemetry before and after temporary capacity injection at the target candidate node.",
                "diagnosticity_weight": 0.92,
            }
        ],
        "never_asserts": ["local_efficiency_equals_global_throughput", "unsubordinated_capacity_addition"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="A system can run only as fast as its slowest process, the constraint, which determines the throughput of the entire chain.",
                chapter="ch_02_constraints",
                locator="section_throughput_p14",
            )
        ],
    },
    {
        "template_id": "a0_bayesian_base_rate_calibration",
        "title": "Bayesian Updating & Base-Rate Calibration",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "baseline_anomaly_separation",
        "situation_frame": {
            "actors": ["decision_maker", "observer"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["single_unverified_report", "missing_baseline", "corroborated_records"],
            "incentive_posture": "stochastic_system",
            "system_dynamics": "anomalous_deviation",
            "primary_uncertainty": ["data_authenticity", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "A dramatic, low-probability signal or incident report arrives, threatening to distort decision-making through base-rate neglect or cognitive availability bias.",
            "trigger_conditions": [
                "high-salience anomalous event reported with low prior historical occurrence",
                "decision makers demanding extreme corrective measures based on an isolated observation",
            ],
            "symptom_phrases": [
                "base rate neglect",
                "overreacting to rare signal",
                "false alarm cascade",
                "uncalibrated probability estimation",
            ],
            "information_structure": "Prior probability distribution paired with empirical likelihood ratios and posterior updates.",
            "analog_vignettes": [
                "Distributed Systems: Calibrating automated microservice restart triggers upon receiving a rare cosmic ray memory bit-flip signal versus common transient network dropouts.",
                "Financial Markets: Adjusting credit default swap spreads following a single debt covenant dispute by establishing historical sovereign default base-rates across economic cycles.",
                "Organizational Behavior: Evaluating allegations of corporate industrial espionage by calculating base rates of disgruntled employee leaks versus innocent clerical filing errors.",
                "Physical Logistics & Sensor Telemetry: Assessing satellite sensor thermal anomalies as missile launch indicators versus regular industrial agricultural stubble burning base rates.",
            ],
        },
        "preconditions": [
            "access to historical reference class data for prior probability calibration",
            "quantified true-positive and false-positive rates",
        ],
        "contraindications": [
            "black swan events where reference classes are mathematically non-existent or ill-defined",
        ],
        "procedure_steps": [
            "Establish the objective empirical base rate of the target phenomenon within an appropriate reference class before examining new evidence.",
            "Quantify the diagnostic specificity and sensitivity (likelihood ratio) of the newly observed evidence.",
            "Calculate the posterior probability using Bayes rule, updating the baseline proportionately without emotional weighting.",
        ],
        "diagnostic_probes": [
            {
                "question": "What is the historical frequency of this phenomenon in the broader population when no specific signal is detected?",
                "discriminates_between": ["unbiased_reference_class_prior", "salience_driven_intuitive_bias"],
                "verifiable_anchor_sought": "Historical population frequency table establishing baseline occurrence rates over multiple observation periods.",
                "diagnosticity_weight": 0.90,
            }
        ],
        "never_asserts": ["unconditional_posterior_from_isolated_cue", "base_rate_omission"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="The probability of any hypothesis is proportional to the product of its prior probability and the likelihood of the observed evidence.",
                chapter="ch_03_probability",
                locator="section_bayes_theorem_p03",
            )
        ],
    },
    {
        "template_id": "a0_fermi_order_of_magnitude_estimation",
        "title": "Fermi Order-of-Magnitude Bounds Estimation",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "baseline_anomaly_separation",
        "situation_frame": {
            "actors": ["decision_maker", "observer"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["missing_baseline", "circumstantial_pattern"],
            "incentive_posture": "stochastic_system",
            "system_dynamics": "static_one_shot",
            "primary_uncertainty": ["bottleneck_location", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Direct measurement is impossible or prohibitively expensive, yet bounding the plausible order of magnitude is essential to eliminate wildly unrealistic hypotheses.",
            "trigger_conditions": [
                "zero direct measurement data available for a critical decision variable",
                "competing claims diverge by several orders of magnitude without grounding",
            ],
            "symptom_phrases": [
                "back of napkin bounding",
                "order of magnitude sanity check",
                "dimensional analysis",
                "ballpark reality check",
            ],
            "information_structure": "Dimensional decomposition equation factoring a target parameter into independent bounded sub-estimates.",
            "analog_vignettes": [
                "Distributed Systems: Estimating peak cross-datacenter replication bandwidth needs by decomposing daily active users, write operations per session, and payload sizes.",
                "Financial Markets: Bounding total addressable market size for autonomous air-freight transport by decomposing total ton-miles, current diesel costs, and airspace regulatory capacity.",
                "Organizational Behavior: Estimating shadow software spend across a 10,000-person enterprise by bounding corporate credit card usage, software license seats, and engineering churn.",
                "Physical Logistics & Sensor Telemetry: Estimating clandestine oil refinery output from commercial satellite tank volume dimensions, crude delivery pipeline diameters, and thermal dissipation signatures.",
            ],
        },
        "preconditions": [
            "clear dimensional units defining the target quantity",
            "plausible conservative bounds for intermediate sub-variables",
        ],
        "contraindications": [
            "scenarios requiring high-precision decimal tolerances where 10x error margins lead to immediate catastrophic failure",
        ],
        "procedure_steps": [
            "Formulate the unknown quantity as a dimensional product or quotient of fundamental sub-variables.",
            "Assign conservative upper and lower geometric bounds to each independent sub-variable based on verifiable physical or structural constraints.",
            "Multiply the component bounds to establish logarithmic orders of magnitude, testing the robustness of conclusions against extreme boundary assumptions.",
        ],
        "diagnostic_probes": [
            {
                "question": "Does the proposed claim violate upper physical or economic capacity limits by more than one order of magnitude?",
                "discriminates_between": ["physically_plausible_claim", "dimensionally_impossible_fabrication"],
                "verifiable_anchor_sought": "Calculated geometric mean boundary versus declared operational claims.",
                "diagnosticity_weight": 0.88,
            }
        ],
        "never_asserts": ["unbounded_estimation_certainty", "pseudo_precise_point_estimates"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="By breaking a large problem into small, independent geometric estimates, individual errors cancel out to produce a reliable order of magnitude.",
                chapter="ch_04_estimation",
                locator="section_fermi_bounds_p09",
            )
        ],
    },
    {
        "template_id": "a0_goodharts_law_metric_gaming",
        "title": "Goodhart's Law & Metric Gaming Detection",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "verifiability_deception_audit",
        "situation_frame": {
            "actors": ["decision_maker", "counterparty", "gatekeeper"],
            "info_asymmetry": "one_sided_hidden",
            "evidence_state": ["corroborated_records", "circumstantial_pattern"],
            "incentive_posture": "strategic_deceptive",
            "system_dynamics": "escalating",
            "primary_uncertainty": ["data_authenticity", "actor_intent"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "A statistical proxy metric chosen for performance monitoring or compensation becomes a direct target, collapsing its diagnostic correlation with underlying operational health.",
            "trigger_conditions": [
                "target metric shows sustained dramatic improvement while overall system health or customer satisfaction degrades",
                "actors exploit structural loopholes to satisfy scorecard targets without delivering intended value",
            ],
            "symptom_phrases": [
                "metric gaming",
                "proxy metric optimization collapse",
                "perverse incentive distortion",
                "hitting the target but missing the point",
            ],
            "information_structure": "Bivariate time-series comparing the target proxy metric trajectory against independent holistic health indicators.",
            "analog_vignettes": [
                "Distributed Systems: Engineering teams artificially boosting uptime metrics by suppressing error logging or routing critical failures to unmonitored background queues.",
                "Financial Markets: Corporate treasuries executing end-of-quarter repo trades to window-dress leverage ratios while underlying balance sheet risk remains elevated.",
                "Organizational Behavior: Customer support agents closing complex customer tickets prematurely without resolution to meet strict first-response-time SLAs.",
                "Physical Logistics & Sensor Telemetry: State transport authorities meeting road safety quotas by reclassifying high-incident thoroughfares as private access roads to hide collision statistics.",
            ],
        },
        "preconditions": [
            "performance metric coupled to external reward or penalty",
            "availability of independent un-incentivized reference data",
        ],
        "contraindications": [
            "purely objective physical measurements where human agency and behavioral gaming cannot alter the outcome",
        ],
        "procedure_steps": [
            "Map the operational mechanism connecting the target metric to actor compensation, promotion, or regulatory compliance.",
            "Identify friction-free gaming vectors where an actor can inflate the metric without creating the underlying real-world value.",
            "Cross-reference the reported metric with independent, un-incentivized secondary indicators to audit divergence.",
        ],
        "diagnostic_probes": [
            {
                "question": "Has the correlation between the monitored metric and real-world system health broken down over the observation period?",
                "discriminates_between": ["authentic_performance_gain", "strategic_metric_gaming_artifact"],
                "verifiable_anchor_sought": "Longitudinal divergence plot comparing targeted KPI performance against un-incentivized audit benchmarks.",
                "diagnosticity_weight": 0.94,
            }
        ],
        "never_asserts": ["metric_integrity_under_high_stakes", "uncorrupted_target_surrogate"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="When a measure becomes a target, it ceases to be a good measure because actors optimize the measure rather than the objective.",
                chapter="ch_05_incentives",
                locator="section_goodhart_law_p21",
            )
        ],
    },
    {
        "template_id": "a0_inversion_analytical_protocol",
        "title": "Inversion Analytical Failure Protocol",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "risk_premortem_projection",
        "situation_frame": {
            "actors": ["decision_maker", "counterparty"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["circumstantial_pattern", "corroborated_records"],
            "incentive_posture": "competitive_rational",
            "system_dynamics": "iterative_feedback",
            "primary_uncertainty": ["future_trajectory", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Forward planning has stalled due to combinatorial complexity; inverting the objective to determine how to guarantee catastrophe illuminates critical structural vulnerabilities.",
            "trigger_conditions": [
                "teams fixated on ideal success paths while ignoring obvious catastrophic blind spots",
                "complex strategic initiative suffering from analysis paralysis",
            ],
            "symptom_phrases": [
                "invert always invert",
                "how to guarantee failure",
                "catastrophic failure pathway",
                "negative capability analysis",
            ],
            "information_structure": "Inverted causal graph mapping deliberate failure triggers to necessary defensive prerequisites.",
            "analog_vignettes": [
                "Distributed Systems: Inverting microservice reliability by designing the exact sequence of packet delays and database locks needed to trigger complete cascade blackouts.",
                "Financial Markets: Inverting portfolio construction by identifying the precise macroeconomic shocks that would guarantee fund liquidation, then hedging those vectors.",
                "Organizational Behavior: Inverting corporate retention strategy by asking what policies would guarantee immediate resignation of top engineering talent.",
                "Physical Logistics & Sensor Telemetry: Inverting operational security analysis by auditing how an adversary would easily pinpoint covert logistics networks via public shipping registries.",
            ],
        },
        "preconditions": [
            "clearly defined strategic objective or mission",
            "willingness to confront worst-case organizational failure modes",
        ],
        "contraindications": [
            "trivial linear tasks where risk is strictly negligible and forward planning suffices",
        ],
        "procedure_steps": [
            "State the positive objective and invert it: define the exact state of total failure, bankruptcy, or system collapse.",
            "Brainstorm every mechanism, vulnerability, and decision path that would directly produce or accelerate this inverted state.",
            "Establish active institutional guardrails and architectural tripwires to preempt each identified failure vector.",
        ],
        "diagnostic_probes": [
            {
                "question": "What sequence of decisions would make failure virtually certain, and are any of those decisions currently being executed?",
                "discriminates_between": ["active_unmitigated_failure_vector", "benign_operational_noise"],
                "verifiable_anchor_sought": "Audit of current active operational procedures matching the catalogued failure pathways.",
                "diagnosticity_weight": 0.91,
            }
        ],
        "never_asserts": ["exhaustive_risk_immunity", "inevitable_success_without_negative_guardrails"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="Invert, always invert: turn a problem upside down and consider what guarantees failure to reveal the path to success.",
                chapter="ch_06_inversion",
                locator="section_jacobi_rule_p02",
            )
        ],
    },
    {
        "template_id": "a0_chestertons_fence_prerequisite_audit",
        "title": "Chesterton's Fence Structural Prerequisite Audit",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "verifiability_deception_audit",
        "situation_frame": {
            "actors": ["decision_maker", "system_component", "gatekeeper"],
            "info_asymmetry": "one_sided_hidden",
            "evidence_state": ["missing_baseline", "corroborated_records"],
            "incentive_posture": "cooperative",
            "system_dynamics": "static_one_shot",
            "primary_uncertainty": ["root_cause", "future_trajectory"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "An institutional rule, legacy code block, or procedural friction is proposed for demolition because its current utility is not immediately obvious to reformers.",
            "trigger_conditions": [
                "proposal to dismantle a legacy protocol or policy labelled as obsolete without documentation of its origin",
                "new leadership attempting rapid deregulation or code refactoring without understanding historical failures",
            ],
            "symptom_phrases": [
                "why is this fence here",
                "unintended consequence of removal",
                "legacy constraint purpose",
                "reform without historical understanding",
            ],
            "information_structure": "Historical dependency matrix connecting legacy constraints to the latent hazards they were erected to mitigate.",
            "analog_vignettes": [
                "Distributed Systems: Developers removing an aparentemente redundant rate-limiter middleware, only to cause an upstream third-party webhook partner to ban their IP range during batch spikes.",
                "Financial Markets: Deregulating margin collateral restrictions on overnight repo markets without recalling the liquidity freeze of prior credit contractions.",
                "Organizational Behavior: Eliminating multi-signature approval gates on corporate disbursements, leading to unauthorized vendor fraud within thirty days.",
                "Physical Logistics & Sensor Telemetry: Investigating why an obscure border crossing checkpoint exists in a remote valley, revealing unmapped seasonal smuggling corridors.",
            ],
        },
        "preconditions": [
            "existence of an established constraint, rule, or architectural barrier",
            "initiative proposing the immediate removal or bypass of said barrier",
        ],
        "contraindications": [
            "situations where the original rationale is fully documented, the underlying hazard is demonstrably extinct, and migration paths are verified",
        ],
        "procedure_steps": [
            "Halt removal of the constraint until its original installation purpose and historical failure context are definitively identified.",
            "Reconstruct the historical operational crisis or structural condition that prompted the creation of the rule.",
            "Evaluate whether the underlying hazard has truly vanished or whether the fence is the only reason the hazard remains dormant.",
        ],
        "diagnostic_probes": [
            {
                "question": "Can the proponents of removal articulate the exact failure mode this rule was originally constructed to prevent?",
                "discriminates_between": ["rigorous_prerequisite_comprehension", "naive_optimistic_demolition"],
                "verifiable_anchor_sought": "Archival documentation or historical post-mortem logs detailing the incident that established the protocol.",
                "diagnosticity_weight": 0.93,
            }
        ],
        "never_asserts": ["uselessness_derived_from_silence", "safe_removal_without_origin_audit"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="Do not remove a fence until you know why it was put up in the first place, lest you unleash the hazard it was built to contain.",
                chapter="ch_07_prerequisites",
                locator="section_chesterton_fence_p11",
            )
        ],
    },
    {
        "template_id": "a0_second_order_effects_feedback_loops",
        "title": "Second-Order Thinking & Dynamic Feedback Loops",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "risk_premortem_projection",
        "situation_frame": {
            "actors": ["decision_maker", "counterparty", "system_component"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["circumstantial_pattern", "corroborated_records"],
            "incentive_posture": "competitive_rational",
            "system_dynamics": "iterative_feedback",
            "primary_uncertainty": ["future_trajectory", "actor_intent"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "An immediate intervention solves an acute first-order problem but triggers delayed, higher-order systemic feedback that exacerbates the original condition.",
            "trigger_conditions": [
                "decision makers celebrating immediate first-order gains while ignoring actor behavioral adaptations",
                "policy interventions generating paradoxical downstream systemic side-effects",
            ],
            "symptom_phrases": [
                "and then what",
                "unintended secondary consequences",
                "feedback loop cascade",
                "cobra effect distortion",
            ],
            "information_structure": "Causal loop diagram tracing balancing and reinforcing feedback pathways over multiple time horizons.",
            "analog_vignettes": [
                "Distributed Systems: Implementing aggressive client retry logic during service degradation, creating a thundering herd that completely prevents backend database recovery.",
                "Financial Markets: Central banks subsidizing corporate credit spreads, which encourages excessive corporate leverage and increases systemic default fragility.",
                "Organizational Behavior: Rewarding developers by raw line-of-code counts, causing codebase bloat, increased refactoring latency, and higher bug densities.",
                "Physical Logistics & Sensor Telemetry: Tracking how international trade sanctions on raw mineral exports spurred rapid domestic illicit processing syndicates in adjacent jurisdictions.",
            ],
        },
        "preconditions": [
            "system comprised of adaptive agents who modify behavior in response to external rules",
            "temporal delay between intervention and equilibrium response",
        ],
        "contraindications": [
            "closed, non-adaptive mechanical systems with zero feedback loops or autonomous agents",
        ],
        "procedure_steps": [
            "Identify the proposed first-order action and its immediate direct intended outcome.",
            "Ask 'And then what?'—projecting how counterparties, competitors, and system components will rationally adapt their incentives.",
            "Trace tertiary feedback loops to determine whether the systemic equilibrium stabilizes or destabilizes the original environment.",
        ],
        "diagnostic_probes": [
            {
                "question": "How will adaptive counterparties modify their operational behavior in response to this new rule over a twelve-month horizon?",
                "discriminates_between": ["systemic_feedback_equilibrium", "naive_linear_projection"],
                "verifiable_anchor_sought": "Behavioral response simulations or historical precedent logs under similar incentive shifts.",
                "diagnosticity_weight": 0.89,
            }
        ],
        "never_asserts": ["static_counterparty_behavior", "linear_extrapolation_of_first_order_gains"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="First-order thinking considers only immediate results, while second-order thinking asks what happens next and how agents adapt.",
                chapter="ch_08_systems_thinking",
                locator="section_feedback_loops_p05",
            )
        ],
    },
    {
        "template_id": "a0_ooda_loop_rapid_synthesis",
        "title": "OODA Loop Rapid Iterative Synthesis",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "sequence_contradiction_matrix",
        "situation_frame": {
            "actors": ["decision_maker", "counterparty"],
            "info_asymmetry": "actively_obfuscated",
            "evidence_state": ["direct_telemetry", "circumstantial_pattern"],
            "incentive_posture": "strategic_deceptive",
            "system_dynamics": "escalating",
            "primary_uncertainty": ["future_trajectory", "sequence_order"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "In a fast-moving, contested environment, compressing the cycle time between observation, orientation, decision, and action allows an agent to out-tempo and disorient competitors.",
            "trigger_conditions": [
                "adversary operating with shorter decision latency, rendering strategic plans obsolete upon execution",
                "dynamic environments where situational awareness decays faster than formal committee review cycles",
            ],
            "symptom_phrases": [
                "decision cycle tempo",
                "operating inside adversary loop",
                "orientation mismatch",
                "rapid iterative synthesis",
            ],
            "information_structure": "Cyclical phase state machine tracking cycle latency across Observe, Orient, Decide, and Act stages.",
            "analog_vignettes": [
                "Distributed Systems: An automated incident response bot cycling health probes, diagnosing split-brain states, and isolating partitions within seconds before human operators assemble.",
                "Financial Markets: High-frequency market-making algorithms reacting to breaking economic data by cancelling stale quotes inside microsecond horizons before institutional flow arrives.",
                "Organizational Behavior: A nimble startup deploying daily customer-driven product experiments to out-iterate a bureaucratic market incumbent paralyzed by quarterly review cycles.",
                "Physical Logistics & Sensor Telemetry: Rapidly updating satellite and transponder telemetry across contested maritime straits to expose deceptive vessel maneuvers before regional authorities issue notices.",
            ],
        },
        "preconditions": [
            "low-latency telemetry pipeline for continuous real-time observation",
            "decentralized authority allowing immediate autonomous execution without multistage signoffs",
        ],
        "contraindications": [
            "deep capital-allocation decisions where multi-year execution horizons cannot be accelerated by tactical cycle tempo",
        ],
        "procedure_steps": [
            "Observe: Gather raw unfiltered telemetry from multiple sensors, detecting environmental shifts without cognitive filtering.",
            "Orient: Synthesize observations against prior mental models, genetics, heritage, and newly processed insights to destroy obsolete dogmas.",
            "Decide: Formulate hypotheses and select the optimal operational vector rapidly under probabilistic uncertainty.",
            "Act: Execute decisively, observing subsequent environmental feedback to trigger the next iterative loop.",
        ],
        "diagnostic_probes": [
            {
                "question": "Is the cycle time from sensor observation to physical operational action faster than the environmental rate of change?",
                "discriminates_between": ["decisional_tempo_advantage", "reactive_paralysis_lag"],
                "verifiable_anchor_sought": "End-to-end telemetry timestamp logs measuring delay between event occurrence and corrective deployment.",
                "diagnosticity_weight": 0.91,
            }
        ],
        "never_asserts": ["static_operational_superiority", "untested_orientation_validity"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="To survive and win in conflict, an entity must operate at a faster tempo than its adversary through continuous observation and orientation.",
                chapter="ch_09_tempo",
                locator="section_boyd_cycle_p17",
            )
        ],
    },
    {
        "template_id": "a0_cynefin_domain_classification",
        "title": "Cynefin Complex Domain Sense-Making",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "baseline_anomaly_separation",
        "situation_frame": {
            "actors": ["decision_maker", "system_component"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["circumstantial_pattern", "missing_baseline"],
            "incentive_posture": "stochastic_system",
            "system_dynamics": "anomalous_deviation",
            "primary_uncertainty": ["root_cause", "bottleneck_location"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Management misapplies linear best practices to a complex emergent problem, causing governance paralysis and severe intervention backfires.",
            "trigger_conditions": [
                "teams demanding predictable multi-year roadmaps for novel, non-linear exploratory challenges",
                "failure of standardized operational templates in novel volatile domains",
            ],
            "symptom_phrases": [
                "complex vs complicated",
                "emergent order probe sense respond",
                "domain mismatch error",
                "inappropriate best practice application",
            ],
            "information_structure": "Four-quadrant ontological matrix classifying system state into Clear, Complicated, Complex, or Chaotic.",
            "analog_vignettes": [
                "Distributed Systems: Recognizing that an unprecedented multi-cloud cascading outage is in the Chaotic domain requiring immediate stabilization actions before root-cause analysis.",
                "Financial Markets: Treating sudden algorithmic market flash crashes as complex emergent phenomena rather than linear balance sheet accounting errors.",
                "Organizational Behavior: Distinguishing between Complicated tasks like payroll compliance and Complex challenges like engineering cultural motivation.",
                "Physical Logistics & Sensor Telemetry: Classifying geopolitical proxy faction alignments in a fractured state as complex emergent dynamics rather than fixed hierarchical command chains.",
            ],
        },
        "preconditions": [
            "openness to recognize that cause-and-effect relationships may only be coherent in retrospect",
            "capacity to run safe-to-fail exploratory probes",
        ],
        "contraindications": [
            "routine manufacturing or regulatory reporting where standard operating procedures are strictly optimal",
        ],
        "procedure_steps": [
            "Diagnose the system ontology: Clear (best practice), Complicated (expert analysis), Complex (emergent probes), or Chaotic (immediate triage).",
            "If Complex, avoid premature categorical frameworks; instead execute small, safe-to-fail probes to stimulate environmental feedback.",
            "Sense the resulting patterns from probes and amplify successful behaviors while dampening destructive vectors.",
        ],
        "diagnostic_probes": [
            {
                "question": "Are cause-and-effect links repeatable and predictable in advance, or only understandable in retrospect?",
                "discriminates_between": ["complicated_predictable_system", "complex_emergent_system"],
                "verifiable_anchor_sought": "Historical variance analysis of identical intervention outcomes across multiple deployment instances.",
                "diagnosticity_weight": 0.90,
            }
        ],
        "never_asserts": ["universal_best_practice_applicability", "deterministic_outcomes_in_complex_domains"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="In a complex domain, cause and effect are only clear in retrospect; leadership must probe, sense, and respond rather than impose rigid plans.",
                chapter="ch_10_sensemaking",
                locator="section_cynefin_domains_p08",
            )
        ],
    },
    {
        "template_id": "a0_occams_razor_complexity_penalization",
        "title": "Occam's Razor & Complexity Penalization",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "abductive_hypothesis_generation",
        "situation_frame": {
            "actors": ["decision_maker", "observer"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["circumstantial_pattern", "single_unverified_report"],
            "incentive_posture": "stochastic_system",
            "system_dynamics": "static_one_shot",
            "primary_uncertainty": ["root_cause", "data_authenticity"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "An intricate, multi-layered conspiracy or Byzantine architectural hypothesis is proposed to explain an anomaly that is fully accounted for by simple human clerical error or physical noise.",
            "trigger_conditions": [
                "elaborate speculative theories requiring dozens of unverified auxiliary assumptions",
                "analysts multiplying entities unnecessarily to preserve a disproven initial model",
            ],
            "symptom_phrases": [
                "entities should not be multiplied",
                "conspiracy over clerical error",
                "epicycles on hypotheses",
                "parsimonious explanation penalty",
            ],
            "information_structure": "Comparative hypothesis ledger penalizing each explanation by the count and unlikelihood of its unverified auxiliary premises.",
            "analog_vignettes": [
                "Distributed Systems: Explaining sudden packet loss by a state-sponsored zero-day BGP hijack versus an unannounced telco fiber cut along a railway line.",
                "Financial Markets: Attributing a sudden stock price drop to an intricate syndicate short-and-distort conspiracy rather than an institutional index fund mechanical rebalance.",
                "Organizational Behavior: Explaining a delayed board presentation as deliberate political sabotage rather than a junior analyst misplacing spreadsheet attachments.",
                "Physical Logistics & Sensor Telemetry: Auditing claims of deepfake satellite manipulation versus common orbital sensor scanline artifacts and sun glare reflections.",
            ],
        },
        "preconditions": [
            "multiple competing hypotheses that explain the observed empirical evidence equally well",
            "ability to quantify or enumerate the number of independent assumptions required by each model",
        ],
        "contraindications": [
            "demonstrated adversarial environments where deception deliberately mimics simple benign explanations",
        ],
        "procedure_steps": [
            "Enumerate all plausible hypotheses capable of explaining the observed empirical anomalies.",
            "Count and evaluate the independent, unverified assumptions required by each competing hypothesis.",
            "Select the hypothesis that minimizes extraneous assumptions while fully satisfying empirical observations.",
        ],
        "diagnostic_probes": [
            {
                "question": "How many independent unverified assumptions does this hypothesis require compared to the simplest alternative?",
                "discriminates_between": ["parsimonious_optimal_hypothesis", "overfitted_speculative_construct"],
                "verifiable_anchor_sought": "Explicit count of unevidenced load-bearing propositions in the claim tree.",
                "diagnosticity_weight": 0.87,
            }
        ],
        "never_asserts": ["elaborate_conspiracy_without_direct_evidence", "immunity_to_parsimony_penalties"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="Entities should not be multiplied beyond necessity; the explanation requiring the fewest assumptions is generally preferred.",
                chapter="ch_11_parsimony",
                locator="section_occams_rule_p04",
            )
        ],
    },
    {
        "template_id": "a0_hanlons_razor_incentive_audit",
        "title": "Hanlon's Razor & Incentive Misalignment Audit",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "incentive_motive_mapping",
        "situation_frame": {
            "actors": ["decision_maker", "counterparty", "intermediary"],
            "info_asymmetry": "one_sided_hidden",
            "evidence_state": ["circumstantial_pattern", "corroborated_records"],
            "incentive_posture": "strategic_deceptive",
            "system_dynamics": "iterative_feedback",
            "primary_uncertainty": ["actor_intent", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "An operational failure or broken contract is immediately attributed to malicious strategic sabotage, when structural incompetence, miscommunication, or conflicting incentives explain the outcome.",
            "trigger_conditions": [
                "counterparty delivers substandard results, triggering intense accusations of bad faith",
                "organizational silos engaging in destructive conflict based on presumed malice",
            ],
            "symptom_phrases": [
                "never attribute to malice",
                "incompetence over sabotage",
                "misaligned operational incentives",
                "bureaucratic neglect mistaken for hostility",
            ],
            "information_structure": "Matrix contrasting malicious intent models against structural competence, training, and incentive bottlenecks.",
            "analog_vignettes": [
                "Distributed Systems: Accusing a partner service of deliberately dropping API calls to stifle competition when their engineers were struggling with misconfigured ingress proxy timeouts.",
                "Financial Markets: Assuming a prime broker liquidated client collateral maliciously when automated back-office margin calculation batch jobs suffered corrupted database indexes.",
                "Organizational Behavior: Believing IT support denied hardware upgrades out of personal animus when corporate procurement had frozen departmental capital expenditure ceilings.",
                "Physical Logistics & Sensor Telemetry: Interpreting diplomatic customs transit delays as geopolitical retaliation rather than local port administrative personnel strikes.",
            ],
        },
        "preconditions": [
            "observed harmful outcome generated by an external party or internal team",
            "uncertainty regarding the underlying psychological intent or mechanical driver",
        ],
        "contraindications": [
            "verified adversarial environments where clear operational intelligence establishes documented malicious intent",
        ],
        "procedure_steps": [
            "Document the exact operational damage and trace the proximate human or software actions that caused it.",
            "Audit the counterparty operational training, tooling limitations, and bureaucratic incentives for obvious points of failure.",
            "Evaluate whether structural incompetence, exhaustion, or incentive misalignment fully explains the failure before alleging bad faith.",
        ],
        "diagnostic_probes": [
            {
                "question": "Did the responsible party derive a concrete, rational strategic advantage from this failure, or did they suffer administrative embarrassment?",
                "discriminates_between": ["malicious_strategic_sabotage", "structural_incompetence_or_neglect"],
                "verifiable_anchor_sought": "Audit of counterparty tangible benefits versus incurred operational costs following the incident.",
                "diagnosticity_weight": 0.88,
            }
        ],
        "never_asserts": ["unverified_malevolent_motive", "presumed_malice_without_substantiation"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="Never attribute to malice that which is adequately explained by carelessness, incompetence, or structural misunderstanding.",
                chapter="ch_12_intent",
                locator="section_hanlon_razor_p12",
            )
        ],
    },
    {
        "template_id": "a0_pareto_80_20_leverage_isolation",
        "title": "Pareto 80/20 High-Leverage Isolation",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "bottleneck_constraint_isolation",
        "situation_frame": {
            "actors": ["decision_maker", "system_component"],
            "info_asymmetry": "symmetric",
            "evidence_state": ["direct_telemetry", "corroborated_records"],
            "incentive_posture": "cooperative",
            "system_dynamics": "static_one_shot",
            "primary_uncertainty": ["bottleneck_location", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Resources and engineering efforts are scattered uniformly across dozens of minor issues, diluting impact while a tiny minority of vital factors drives the vast majority of outcomes.",
            "trigger_conditions": [
                "teams attempting to resolve 100 backlog items with equal priority",
                "flat resource allocation across highly skewed power-law distributions",
            ],
            "symptom_phrases": [
                "vital few trivial many",
                "power law distribution",
                "80 20 leverage point",
                "unequal cause effect distribution",
            ],
            "information_structure": "Rank-ordered cumulative distribution histogram isolating the top 20 percent percentile driving 80 percent of variance.",
            "analog_vignettes": [
                "Distributed Systems: Discovering that 85 percent of database query latency is generated by exactly 3 unindexed queries out of 200 total endpoints.",
                "Financial Markets: Finding that 80 percent of trading desk risk volatility is driven by two illiquid interest rate swap tranches.",
                "Organizational Behavior: Identifying that 80 percent of customer churn complaints originate from one specific buggy onboarding screen.",
                "Physical Logistics & Sensor Telemetry: Concentrating maritime monitoring on 15 key chokepoint straits that handle over 75 percent of global illicit transshipment tonnage.",
            ],
        },
        "preconditions": [
            "measurable outcome metric tied to discrete causal categories",
            "sufficient sample size to construct a cumulative Pareto distribution",
        ],
        "contraindications": [
            "strictly Gaussian domains where variance is uniformly distributed with zero heavy tails",
        ],
        "procedure_steps": [
            "Quantify and rank-order all contributing inputs, bugs, or risks by their total empirical impact.",
            "Plot the cumulative impact curve to locate the inflection point separating the vital few from the trivial many.",
            "Reallocate 80 percent of available operational and cognitive capacity strictly to the top 20 percent leverage drivers.",
        ],
        "diagnostic_probes": [
            {
                "question": "What percentage of the total problem or cost is eliminated if we only resolve the top three discrete drivers?",
                "discriminates_between": ["power_law_distribution", "uniform_gaussian_distribution"],
                "verifiable_anchor_sought": "Cumulative frequency distribution chart showing cumulative impact percentage across inputs.",
                "diagnosticity_weight": 0.92,
            }
        ],
        "never_asserts": ["uniform_importance_across_tasks", "linear_resource_dilution"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="In any series of elements to be controlled, a selected small fraction, in terms of numbers, always accounts for any large fraction in terms of effect.",
                chapter="ch_13_distributions",
                locator="section_pareto_principle_p06",
            )
        ],
    },
    {
        "template_id": "a0_premortem_failure_decomposition",
        "title": "Pre-Mortem Failure Mode Decomposition",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "risk_premortem_projection",
        "situation_frame": {
            "actors": ["decision_maker", "counterparty", "observer"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["circumstantial_pattern", "corroborated_records"],
            "incentive_posture": "cooperative",
            "system_dynamics": "iterative_feedback",
            "primary_uncertainty": ["future_trajectory", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Overconfidence, groupthink, and social pressure prevent team members from raising vital doubts prior to a major system launch, setting the stage for unmitigated disaster.",
            "trigger_conditions": [
                "unanimous leadership consensus on a high-stakes, unvalidated project rollout",
                "subtle reservations suppressed due to fear of appearing uncooperative",
            ],
            "symptom_phrases": [
                "prospective hindsight",
                "assume the project failed completely",
                "debiasing groupthink",
                "pre-launch failure narrative",
            ],
            "information_structure": "Narrative retrospective timeline generated under the axiomatic premise that total catastrophic failure has already occurred.",
            "analog_vignettes": [
                "Distributed Systems: Assembling the engineering team before a datacenter migration and asking: 'It is six months later and we lost customer data; what happened?'",
                "Financial Markets: Forcing an investment committee to write the post-mortem obituary of a multi-billion dollar acquisition before signing the definitive merger agreement.",
                "Organizational Behavior: Asking sales leadership before a major product pivot: 'We missed quarterly revenue by 70 percent; where did our assumptions snap?'",
                "Physical Logistics & Sensor Telemetry: Evaluating an operational field mission plan by assuming the undercover communications channel was compromised from day one.",
            ],
        },
        "preconditions": [
            "initiative at the pre-launch or final design approval stage",
            "psychological safety established to voice candid structural criticisms without penalty",
        ],
        "contraindications": [
            "operations already actively unfolding where retrospective simulation comes too late for architectural changes",
        ],
        "procedure_steps": [
            "Gather key stakeholders and declare: 'Imagine we are five years in the future, and this initiative has failed catastrophically. Write a comprehensive history of how it collapsed.'",
            "Collect every individual failure mechanism and categorize them into systemic clusters without attribution.",
            "Restructure the launch plan to build explicit preventative controls and tripwire monitors around the top failure vectors.",
        ],
        "diagnostic_probes": [
            {
                "question": "What critical failure mode emerged during the retrospective simulation that had never been documented in standard risk registers?",
                "discriminates_between": ["unveiled_latent_vulnerability", "standard_boiler_plate_risk_catalog"],
                "verifiable_anchor_sought": "Comparison of pre-mortem identified failure modes against formal legacy risk logs.",
                "diagnosticity_weight": 0.94,
            }
        ],
        "never_asserts": ["unflinching_consensus_as_proof_of_safety", "suppression_of_dissent"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="By assuming a project has already failed, prospective hindsight legitimizes doubts and reveals hidden catastrophic failure modes.",
                chapter="ch_14_debiasing",
                locator="section_premortem_klein_p19",
            )
        ],
    },
    {
        "template_id": "a0_red_teaming_adversarial_simulation",
        "title": "Red Teaming & Adversarial Counterfactual Simulation",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "risk_premortem_projection",
        "situation_frame": {
            "actors": ["decision_maker", "counterparty", "gatekeeper"],
            "info_asymmetry": "actively_obfuscated",
            "evidence_state": ["circumstantial_pattern", "direct_telemetry"],
            "incentive_posture": "strategic_deceptive",
            "system_dynamics": "escalating",
            "primary_uncertainty": ["future_trajectory", "actor_intent"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Defensive postures are evaluated only against polite, compliant, or historical threats, leaving the system completely exposed to an intelligent, opportunistic adversary.",
            "trigger_conditions": [
                "security or business strategy validated only through static checklist compliance",
                "absence of active adversarial pressure testing of institutional defenses",
            ],
            "symptom_phrases": [
                "adversarial simulation",
                "red team stress testing",
                "finding the blind spot",
                "thinking like the adversary",
            ],
            "information_structure": "Adversarial exploit graph matching target defensive blind spots with hostile operational maneuvers.",
            "analog_vignettes": [
                "Distributed Systems: Commissioning an independent team to perform penetration testing and data exfiltration through forgotten staging environment DNS tunnels.",
                "Financial Markets: Simulating how predatory algorithmic trading firms would exploit market-order slippage during index rebalancing windows.",
                "Organizational Behavior: Running an internal red team exercise simulating how a hostile activist shareholder would orchestrate a boardroom takeover.",
                "Physical Logistics & Sensor Telemetry: Stress-testing public disclosures by trying to reverse-engineer sensitive corporate facility locations using public job postings and satellite imagery.",
            ],
        },
        "preconditions": [
            "independence and empowerment of the red team to bypass organizational taboos",
            "commitment by system leadership to remediate uncovered vulnerabilities",
        ],
        "contraindications": [
            "environments where simulated stress introduces unacceptable live operational safety hazards without containment",
        ],
        "procedure_steps": [
            "Grant an independent red team the explicit charter, objective, and freedom to exploit system weaknesses from an adversarial perspective.",
            "Execute the attack simulation without notifying operational defenders in advance, logging response latencies and blind spots.",
            "Conduct a structured joint debriefing to convert identified exploitable seams into resilient structural mitigations.",
        ],
        "diagnostic_probes": [
            {
                "question": "Did the system defense rely on the adversary playing by established institutional rules or courtesy?",
                "discriminates_between": ["robust_adversarial_resilience", "fragile_compliance_theater"],
                "verifiable_anchor_sought": "Red team exercise log demonstrating unhindered penetration through compliant perimeters.",
                "diagnosticity_weight": 0.93,
            }
        ],
        "never_asserts": ["checklist_compliance_as_true_security", "passive_perimeter_impenetrability"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="To understand your own vulnerabilities, you must adopt the perspective and ruthlessness of an intelligent adversary.",
                chapter="ch_15_adversarial",
                locator="section_red_teaming_p03",
            )
        ],
    },
    {
        "template_id": "a0_opportunity_cost_tradeoff_matrix",
        "title": "Opportunity Cost & Foregone Alternative Matrix",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "bottleneck_constraint_isolation",
        "situation_frame": {
            "actors": ["decision_maker", "system_component"],
            "info_asymmetry": "symmetric",
            "evidence_state": ["corroborated_records", "missing_baseline"],
            "incentive_posture": "cooperative",
            "system_dynamics": "static_one_shot",
            "primary_uncertainty": ["bottleneck_location", "future_trajectory"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Capital or engineering time is committed to a marginally profitable or cosmetic project because the far greater returns of the next-best foregone alternative are completely omitted from accounting.",
            "trigger_conditions": [
                "evaluating an investment purely on whether it produces a positive return rather than optimal return",
                "ignoring the scarcity of executive attention and engineering bandwidth",
            ],
            "symptom_phrases": [
                "what are we giving up",
                "unseen foregone alternative",
                "opportunity cost omission",
                "accounting profit vs economic profit",
            ],
            "information_structure": "Comparative matrix contrasting nominal project ROI against the expected return of the best foregone alternative.",
            "analog_vignettes": [
                "Distributed Systems: Allocating senior infrastructure architects to rewrite internal tooling for marginal speedups while core customer database reliability issues remain unaddressed.",
                "Financial Markets: Holding low-yielding cash equivalents to avoid mark-to-market volatility while inflation compounds a negative real return.",
                "Organizational Behavior: Committing engineering teams to build a bespoke CRM system in-house instead of adopting commercial off-the-shelf software and focusing on proprietary algorithms.",
                "Physical Logistics & Sensor Telemetry: Spending analyst manpower manually translating foreign shipping manifests rather than building automated parsers to unlock wider regional datasets.",
            ],
        },
        "preconditions": [
            "finite resource pool (capital, cognitive focus, or developer hours)",
            "clear enumeration of alternative allocation candidates",
        ],
        "contraindications": [
            "scenarios where surplus uncommitted capacity exists with zero trade-offs or friction",
        ],
        "procedure_steps": [
            "Explicitly define the next-best alternative use for every dollar, hour, and engineer committed to the proposed initiative.",
            "Calculate economic return by subtracting the value of the foregone alternative from the projected outcome of the chosen path.",
            "Reject projects that show positive accounting returns but negative economic profit after accounting for opportunity cost.",
        ],
        "diagnostic_probes": [
            {
                "question": "What is the single most valuable project that will be starved of resources if this initiative is approved?",
                "discriminates_between": ["optimal_economic_allocation", "suboptimal_resource_squandering"],
                "verifiable_anchor_sought": "Quantified expected value comparison between proposed initiative and the top foregone alternative.",
                "diagnosticity_weight": 0.89,
            }
        ],
        "never_asserts": ["zero_cost_resource_utilization", "isolated_roi_without_alternative_benchmarks"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="The true cost of anything is the value of the next-best alternative that must be foregone to obtain it.",
                chapter="ch_16_economics",
                locator="section_opportunity_cost_p22",
            )
        ],
    },
    {
        "template_id": "a0_survivorship_bias_denominator_reconstruction",
        "title": "Survivorship Bias & Denominator Reconstruction",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "baseline_anomaly_separation",
        "situation_frame": {
            "actors": ["decision_maker", "observer"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["circumstantial_pattern", "missing_baseline"],
            "incentive_posture": "stochastic_system",
            "system_dynamics": "static_one_shot",
            "primary_uncertainty": ["root_cause", "data_authenticity"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Analyses evaluate only successful or surviving entities that made it past an unobserved filtering process, mistakenly attributing success to traits that were equally prevalent among failed entities.",
            "trigger_conditions": [
                "benchmarking performance solely against industry winners while ignoring the graveyard of bankrupt firms",
                "deriving architectural guidelines from survivors of a catastrophic outage who were merely lucky",
            ],
            "symptom_phrases": [
                "where are the missing bullet holes",
                "silent graveyard of failures",
                "unobserved denominator",
                "survivor filter distortion",
            ],
            "information_structure": "Two-by-two contingency table restoring the unobserved cell containing failed entities that possessed the target attribute.",
            "analog_vignettes": [
                "Distributed Systems: Studying server instances that did not crash during a power spike to find hardening traits, ignoring that they were simply on an unrelated circuit breaker.",
                "Financial Markets: Evaluating hedge fund manager alpha by analyzing surviving funds, masking the reality that underperforming funds were simply liquidated and erased from database indices.",
                "Organizational Behavior: Emulating the aggressive habits of successful technology founders while overlooking that thousands of bankrupt founders shared identical habits.",
                "Physical Logistics & Sensor Telemetry: Estimating border patrol intercept rates based strictly on captured contraband shipments, completely missing shipments that bypassed checkpoints undetected.",
            ],
        },
        "preconditions": [
            "dataset generated by an historical process containing selective attrition or survival gates",
            "ability to estimate or model the unobserved population of non-survivors",
        ],
        "contraindications": [
            "controlled experimental trials where the full initial cohort is completely tracked without attrition",
        ],
        "procedure_steps": [
            "Identify the historical selection filter that separated surviving specimens from non-survivors.",
            "Reconstruct the unobserved denominator: estimate the volume and characteristics of entities that entered the funnel and failed.",
            "Recalculate attribute correlations across both survivors and non-survivors to test whether the trait actually predicts survival.",
        ],
        "diagnostic_probes": [
            {
                "question": "Are we analyzing the armor where returning aircraft were hit, or where the planes that were lost suffered damage?",
                "discriminates_between": ["unbiased_full_cohort_analysis", "distorted_survivor_selection_bias"],
                "verifiable_anchor_sought": "Historical census data reconstructing the lost or dropped cohort size and traits.",
                "diagnosticity_weight": 0.95,
            }
        ],
        "never_asserts": ["correlation_equals_causation_in_survivor_cohorts", "ignoring_unobserved_attrition"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="The armor belongs where the bullet holes are not, for the aircraft hit in those vital spots did not return to be counted.",
                chapter="ch_17_selection_bias",
                locator="section_wald_survivorship_p07",
            )
        ],
    },
    {
        "template_id": "a0_comparative_advantage_resource_allocation",
        "title": "Comparative Advantage & Allocation Optimization",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "bottleneck_constraint_isolation",
        "situation_frame": {
            "actors": ["decision_maker", "system_component"],
            "info_asymmetry": "symmetric",
            "evidence_state": ["direct_telemetry", "corroborated_records"],
            "incentive_posture": "cooperative",
            "system_dynamics": "static_one_shot",
            "primary_uncertainty": ["bottleneck_location", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "An actor or team insists on performing tasks they are absolute masters at, despite possessing a far greater comparative advantage elsewhere, bottlenecking overall output.",
            "trigger_conditions": [
                "senior engineers spending hours on basic server provisioning because they do it faster than juniors",
                "failure to delegate tasks based on relative opportunity cost differences across units",
            ],
            "symptom_phrases": [
                "comparative vs absolute advantage",
                "misallocated specialization",
                "opportunity cost of execution",
                "relative efficiency divergence",
            ],
            "information_structure": "Matrix comparing relative opportunity cost ratios for each task across participating operational nodes.",
            "analog_vignettes": [
                "Distributed Systems: Offloading cryptographic hashing to specialized hardware ASICs not because CPUs cannot compute hashes, but to free CPU cycles for complex routing logic.",
                "Financial Markets: A top portfolio manager delegating macroeconomic data scraping to junior analysts even if the manager could scrape faster, freeing hours for trade generation.",
                "Organizational Behavior: Having chief architects write production code when their architectural guidance across twenty teams yields 100x greater systemic value.",
                "Physical Logistics & Sensor Telemetry: Assigning raw translation parsing to machine models so human domain linguists can concentrate exclusively on subtle cultural dialect nuances.",
            ],
        },
        "preconditions": [
            "multiple agents or subsystems with heterogeneous productivity across multiple tasks",
            "fungible workflow that permits modular task reassignment",
        ],
        "contraindications": [
            "isolated single-agent environments where delegation or parallel execution is impossible",
        ],
        "procedure_steps": [
            "Quantify the productivity of each agent across all key operational tasks.",
            "Calculate the opportunity cost of having each agent perform task A versus task B.",
            "Assign each task strictly to the agent with the lowest opportunity cost, regardless of absolute proficiency.",
        ],
        "diagnostic_probes": [
            {
                "question": "Is this senior node performing this task because of absolute competence, or because their relative opportunity cost is lowest?",
                "discriminates_between": ["comparative_advantage_assignment", "naive_absolute_advantage_hoarding"],
                "verifiable_anchor_sought": "Ratio analysis of foregone high-value production hours spent on low-leverage execution.",
                "diagnosticity_weight": 0.88,
            }
        ],
        "never_asserts": ["absolute_skill_as_sole_allocation_criterion", "efficient_monopolization_of_subtasks"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="Total production is maximized when each participant specializes in activities where their relative opportunity cost is lowest.",
                chapter="ch_18_allocation",
                locator="section_ricardo_comparative_p15",
            )
        ],
    },
    {
        "template_id": "a0_ashbys_law_requisite_variety",
        "title": "Ashby's Law of Requisite Variety",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "baseline_anomaly_separation",
        "situation_frame": {
            "actors": ["decision_maker", "system_component"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["direct_telemetry", "circumstantial_pattern"],
            "incentive_posture": "stochastic_system",
            "system_dynamics": "iterative_feedback",
            "primary_uncertainty": ["future_trajectory", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "A control system fails because its internal repertoire of responses is far narrower than the state space of disturbances generated by the external environment.",
            "trigger_conditions": [
                "automated regulatory compliance or firewall systems bypassed by novel environmental states",
                "management attempting to control a complex, volatile market with rigid, three-state decision rules",
            ],
            "symptom_phrases": [
                "only variety can absorb variety",
                "control system state deficit",
                "inadequate response repertoire",
                "cybernetic control mismatch",
            ],
            "information_structure": "State-space dimension table comparing environmental disturbance entropy against controller response capacity.",
            "analog_vignettes": [
                "Distributed Systems: An auto-scaling algorithm possessing only CPU threshold rules failing to absorb complex cache contention and network socket exhaustion spikes.",
                "Financial Markets: A risk management engine utilizing single-factor VaR failing to contain systemic liquidity cascades across multiple uncorrelated assets.",
                "Organizational Behavior: A centralized corporate board attempting to micromanage decentralized regional branches operating under divergent local legal systems.",
                "Physical Logistics & Sensor Telemetry: Tracking an adversary dynamic communications network using fixed keyword filters that are rendered useless by rapid slang evolution.",
            ],
        },
        "preconditions": [
            "identifiable boundary separating controller from the external environment",
            "ability to enumerate or bound environmental disturbance dimensions",
        ],
        "contraindications": [
            "highly constrained, deterministic closed environments where disturbance states are strictly fixed and known",
        ],
        "procedure_steps": [
            "Map the dimensionality and variance of disturbances generated by the external operating environment.",
            "Audit the internal controller state space and verify whether it possesses sufficient distinct counter-responses.",
            "Expand controller internal variety through modular adaptation, or attenuate external variety through filtering boundaries.",
        ],
        "diagnostic_probes": [
            {
                "question": "Does the control mechanism possess a distinct, calibrated countermeasure for each credible environmental disturbance state?",
                "discriminates_between": ["requisite_variety_sufficiency", "controller_state_space_collapse"],
                "verifiable_anchor_sought": "Ratio of controller response states to distinct observed external disturbance modes.",
                "diagnosticity_weight": 0.91,
            }
        ],
        "never_asserts": ["external_complexity_suppressed_by_simple_rules", "unbounded_controller_dominance"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="Only variety can destroy variety; a control system must possess at least as much variety as the disturbances it seeks to regulate.",
                chapter="ch_19_cybernetics",
                locator="section_ashby_law_p09",
            )
        ],
    },
    {
        "template_id": "a0_lindy_effect_robustness_testing",
        "title": "Lindy Effect & Robustness Horizon Testing",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "baseline_anomaly_separation",
        "situation_frame": {
            "actors": ["decision_maker", "observer"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["corroborated_records", "circumstantial_pattern"],
            "incentive_posture": "stochastic_system",
            "system_dynamics": "static_one_shot",
            "primary_uncertainty": ["future_trajectory", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Teams abandon battle-tested protocols in favor of shiny, unvalidated novelties that lack empirical survival proof across multi-decade stressors.",
            "trigger_conditions": [
                "replacing proven relational database engines with hype-driven ephemeral databases lacking fault tolerance history",
                "conflating technological novelty with structural durability",
            ],
            "symptom_phrases": [
                "future life expectancy proportional to age",
                "time as the ultimate stressor",
                "neomania vulnerability",
                "robustness proven by survival",
            ],
            "information_structure": "Survival curve mapping technology age against projected future operational longevity.",
            "analog_vignettes": [
                "Distributed Systems: Choosing PostgreSQL and C libraries for core transactional pipelines over brand-new framework releases that lack ten-year stability track records.",
                "Financial Markets: Assessing the long-term survival probability of sovereign debt instruments versus speculative digital assets lacking economic cycle seasoning.",
                "Organizational Behavior: Preserving double-entry bookkeeping principles established five centuries ago over opaque modern financial engineering constructs.",
                "Physical Logistics & Sensor Telemetry: Relying on physical maritime triangulation methods to verify vessel positions when electronic GPS signals are spoofed.",
            ],
        },
        "preconditions": [
            "non-perishable domain (ideas, software architectures, algorithms, institutions)",
            "historical survival data spanning multiple technological or economic cycles",
        ],
        "contraindications": [
            "perishable physical organisms where biological aging accelerates mortality over time",
        ],
        "procedure_steps": [
            "Audit the historical age of competing technologies, frameworks, or institutional rules under consideration.",
            "Evaluate whether the candidate has survived diverse macro shocks, economic recessions, and technological paradigm shifts.",
            "Default to Lindy-proven components for mission-critical core architectures, isolating unvalidated novelties in non-critical edges.",
        ],
        "diagnostic_probes": [
            {
                "question": "Has this technology or protocol survived continuous operational stress for as long as we expect our new architecture to last?",
                "discriminates_between": ["lindy_robust_foundational_asset", "ephemeral_hype_driven_novelty"],
                "verifiable_anchor_sought": "Historical production uptime and adoption logs spanning multiple market or compute cycles.",
                "diagnosticity_weight": 0.89,
            }
        ],
        "never_asserts": ["novelty_equals_superiority", "linear_obsolescence_of_non_perishables"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="For the non-perishable, every year of survival adds to its expected future longevity, proving its robustness against unforeseen shocks.",
                chapter="ch_20_robustness",
                locator="section_lindy_effect_p13",
            )
        ],
    },
    {
        "template_id": "a0_signal_detection_decision_boundary",
        "title": "Signal Detection & Decision Boundary Calibration",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "verifiability_deception_audit",
        "situation_frame": {
            "actors": ["decision_maker", "observer", "gatekeeper"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["direct_telemetry", "circumstantial_pattern"],
            "incentive_posture": "stochastic_system",
            "system_dynamics": "anomalous_deviation",
            "primary_uncertainty": ["data_authenticity", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "A monitoring system is overwhelmed by alert fatigue or catastrophic false negatives because the decision threshold was set arbitrarily without calculating asymmetric payoff matrices.",
            "trigger_conditions": [
                "security operations center ignoring alerts due to 99 percent false-positive alert fatigue",
                "critical equipment failures slipping through because thresholds were tuned to minimize false alarms",
            ],
            "symptom_phrases": [
                "receiver operating characteristic",
                "asymmetric payoff matrix",
                "tuning the detection threshold",
                "d-prime vs criterion bias",
            ],
            "information_structure": "ROC curve plotting hit rate versus false alarm rate parameterized by decision threshold criterion.",
            "analog_vignettes": [
                "Distributed Systems: Tuning automated microservice failover alert thresholds so transient network blips do not cause split-brain thrashing.",
                "Financial Markets: Calibrating anti-fraud transaction blocking thresholds to balance customer checkout abandonment against unauthorized chargeback liability.",
                "Organizational Behavior: Calibrating internal audit investigation triggers between employee whistleblowing reports and administrative distraction costs.",
                "Physical Logistics & Sensor Telemetry: Calibrating radar anomaly detection thresholds between flock of birds noise and low-altitude drone intrusions.",
            ],
        },
        "preconditions": [
            "noisy background distribution with overlapping signal distribution",
            "quantifiable costs and payoffs for hits, false alarms, misses, and correct rejections",
        ],
        "contraindications": [
            "noiseless deterministic telemetry where signal and background never overlap",
        ],
        "procedure_steps": [
            "Plot the empirical distribution of signal-plus-noise against pure noise to determine separation distance.",
            "Construct a payoff matrix assigning explicit economic values to false alarms, misses, hits, and correct rejections.",
            "Set the operational decision threshold criterion to maximize net expected value rather than minimizing raw alarm counts.",
        ],
        "diagnostic_probes": [
            {
                "question": "What is the concrete cost ratio between a false alarm and a missed critical detection in this operating domain?",
                "discriminates_between": ["economically_calibrated_criterion", "arbitrary_uncalibrated_threshold"],
                "verifiable_anchor_sought": "Explicit cost-benefit payoff matrix applied to receiver operating characteristic data.",
                "diagnosticity_weight": 0.93,
            }
        ],
        "never_asserts": ["zero_false_positive_and_zero_miss_feasibility", "arbitrary_threshold_neutrality"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="Decision criteria must balance the cost of false alarms against the catastrophe of misses, calibrating thresholds to objective payoff asymmetries.",
                chapter="ch_21_detection",
                locator="section_signal_theory_p04",
            )
        ],
    },
    {
        "template_id": "a0_triangulation_multi_channel_audit",
        "title": "Triangulation & Multi-Channel Orthogonal Audit",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "verifiability_deception_audit",
        "situation_frame": {
            "actors": ["decision_maker", "observer", "counterparty"],
            "info_asymmetry": "actively_obfuscated",
            "evidence_state": ["single_unverified_report", "corroborated_records", "circumstantial_pattern"],
            "incentive_posture": "strategic_deceptive",
            "system_dynamics": "anomalous_deviation",
            "primary_uncertainty": ["data_authenticity", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "An apparent confirmation of facts is completely hollow because multiple corroborating reports trace back to a single shared circular source or correlated sensor vulnerability.",
            "trigger_conditions": [
                "apparent multi-source corroboration that shares an uninspected single point of origin",
                "high-stakes claim supported only by co-dependent intelligence channels",
            ],
            "symptom_phrases": [
                "circular reporting echo",
                "orthogonal source triangulation",
                "independent sensor confirmation",
                "correlated error mode",
            ],
            "information_structure": "Provenance directed graph tracing multiple surface reports back to their root origin nodes to expose circular loops.",
            "analog_vignettes": [
                "Distributed Systems: Verifying datacenter power loss through independent external ping telemetry, ambient thermal sensor spikes, and grid substation status rather than relying on a single network switch reporting port down.",
                "Financial Markets: Confirming corporate revenue authenticity by cross-referencing sales tax filings, ocean freight bills of lading, and bank cash deposits rather than accepting vendor invoices alone.",
                "Organizational Behavior: Auditing workplace grievance claims by interviewing witnesses across different reporting hierarchies to avoid coordinated collusion.",
                "Physical Logistics & Sensor Telemetry: Confirming an unacknowledged airstrike by pairing synthetic aperture radar imagery, infrared heat blooms, and local social media geolocated timestamps.",
            ],
        },
        "preconditions": [
            "access to at least two fundamentally independent, orthogonal information channels",
            "ability to trace evidentiary provenance back to initial capture sensors",
        ],
        "contraindications": [
            "situations where only a single proprietary physical sensor exists in the universe for that phenomenon",
        ],
        "procedure_steps": [
            "Map the lineage of each corroborating report to verify that sources do not share a common root ancestor.",
            "Identify orthogonal observation modes subject to entirely different physical, operational, or legal error mechanisms.",
            "Synthesize data across orthogonal channels, downgrading certainty if correlation is confined to a single collection medium.",
        ],
        "diagnostic_probes": [
            {
                "question": "Do these three corroborating sources originate from independent collection physics, or do they share an uninspected root feed?",
                "discriminates_between": ["genuine_orthogonal_triangulation", "circular_information_echo_chamber"],
                "verifiable_anchor_sought": "Graph provenance trace confirming zero shared intermediate nodes across source pipelines.",
                "diagnosticity_weight": 0.95,
            }
        ],
        "never_asserts": ["corroboration_via_circular_echo", "treating_correlated_sensors_as_independent"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="True confirmation requires independent orthogonal measurement channels; three echoes of a single report provide zero incremental validity.",
                chapter="ch_22_triangulation",
                locator="section_orthogonal_proof_p18",
            )
        ],
    },
    {
        "template_id": "a0_root_cause_ishikawa_synthesis",
        "title": "Root Cause Ishikawa Synthesis (5 Whys)",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "abductive_hypothesis_generation",
        "situation_frame": {
            "actors": ["decision_maker", "system_component"],
            "info_asymmetry": "symmetric",
            "evidence_state": ["direct_telemetry", "corroborated_records"],
            "incentive_posture": "cooperative",
            "system_dynamics": "iterative_feedback",
            "primary_uncertainty": ["root_cause", "bottleneck_location"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Remediation stops prematurely at the proximate human or mechanical trigger, allowing the underlying systemic, procedural, and cultural root cause to generate recurring outages.",
            "trigger_conditions": [
                "fixing recurring production bugs by merely patching local symptoms",
                "blaming operator human error without auditing user interface design or systemic incentives",
            ],
            "symptom_phrases": [
                "stopping at proximate cause",
                "five whys drilldown",
                "ishikawa fishbone diagram",
                "systemic root vulnerability",
            ],
            "information_structure": "Fishbone causal hierarchy decomposing failure modes across Machinery, Methods, Materials, Measurements, and Milieu.",
            "analog_vignettes": [
                "Distributed Systems: Drilling down from a database crash through out-of-memory errors, rogue cron jobs, missing container cgroup limits, to an engineering culture lacking memory budget reviews.",
                "Financial Markets: Tracing an errant algorithmic trade through an unvalidated config file to a lack of pre-deployment validation scripts in the CI/CD pipeline.",
                "Organizational Behavior: Investigating accounting compliance breaches beyond the rogue employee to uncover aggressive quarterly executive bonus pressures.",
                "Physical Logistics & Sensor Telemetry: Tracing repeated maritime navigational collisions past individual helm error to systematic crew sleep deprivation policies on budget container lines.",
            ],
        },
        "preconditions": [
            "incident event log with verified chronological timeline",
            "blameless operational culture focused on systemic discovery rather than individual scapegoating",
        ],
        "contraindications": [
            "true non-deterministic stochastic fluctuations with zero repeatable underlying structural mechanism",
        ],
        "procedure_steps": [
            "Document the immediate observable failure event and its proximate technical trigger.",
            "Iterate downward through at least five levels of 'Why?', questioning why the preceding safeguard or control failed to prevent the condition.",
            "Map findings onto an Ishikawa diagram across structural categories to formulate systemic mitigations that render recurrence physically impossible.",
        ],
        "diagnostic_probes": [
            {
                "question": "Does our proposed remediation prevent this entire class of failure from recurring, or does it merely prevent this exact script from running?",
                "discriminates_between": ["systemic_root_cause_remediation", "superficial_symptom_patching"],
                "verifiable_anchor_sought": "Architectural architectural changes that eliminate the failure condition by design.",
                "diagnosticity_weight": 0.92,
            }
        ],
        "never_asserts": ["human_error_as_root_cause", "superficial_symptom_suppression"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="To see root causes clearly, one must ask why repeatedly until the systemic flaw in design, training, or process is laid bare.",
                chapter="ch_23_causality",
                locator="section_ishikawa_five_whys_p10",
            )
        ],
    },
    {
        "template_id": "a0_regret_minimization_trajectory_audit",
        "title": "Regret Minimization Long-Horizon Trajectory",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "risk_premortem_projection",
        "situation_frame": {
            "actors": ["decision_maker", "observer"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["missing_baseline", "circumstantial_pattern"],
            "incentive_posture": "cooperative",
            "system_dynamics": "static_one_shot",
            "primary_uncertainty": ["future_trajectory", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "Decision makers are paralyzed by short-term embarrassment, transitory market noise, or immediate career friction, blinding them to enormous long-horizon trajectory divergence.",
            "trigger_conditions": [
                "hesitating on transformative strategic investments due to fear of immediate quarterly earnings volatility",
                "conflating short-term social friction with long-term structural regret",
            ],
            "symptom_phrases": [
                "looking back from age eighty",
                "minimizing terminal regret",
                "long horizon trajectory divergence",
                "transitory noise vs permanent impact",
            ],
            "information_structure": "Decision tree evaluated at asymptotic terminal horizons rather than immediate tactical periods.",
            "analog_vignettes": [
                "Distributed Systems: Migrating away from a proprietary cloud vendor database despite short-term migration pain to prevent permanent multi-decade vendor lock-in.",
                "Financial Markets: Allocating research capital to foundational clean energy generation technologies despite multi-year capital expenditure drawdowns.",
                "Organizational Behavior: An executive leaving a safe bureaucratic corporation to launch an innovative frontier venture, minimizing regret of never trying.",
                "Physical Logistics & Sensor Telemetry: Publishing critical audit findings regarding supply chain abuse despite threats of short-term retaliatory litigation.",
            ],
        },
        "preconditions": [
            "strategic choice point with asymmetric long-term irreversibility",
            "freedom to decouple evaluation from immediate short-term quarterly incentives",
        ],
        "contraindications": [
            "existential short-term insolvency crises where failing to survive the present month renders long-term horizons moot",
        ],
        "procedure_steps": [
            "Project the decision maker perspective forward to the end of life or the conclusion of the multi-decade institutional horizon.",
            "Evaluate each candidate decision vector specifically by the magnitude of permanent regret generated by acts of omission.",
            "Select the trajectory that minimizes terminal regret, accepting short-term volatility and social friction as necessary friction.",
        ],
        "diagnostic_probes": [
            {
                "question": "Looking back across a thirty-year horizon, will we regret the temporary embarrassment of failure or the permanent regret of never attempting this path?",
                "discriminates_between": ["terminal_regret_minimization", "short_term_loss_aversion_paralysis"],
                "verifiable_anchor_sought": "Documented decision matrix ranking options by terminal regret scores over multi-decade windows.",
                "diagnosticity_weight": 0.88,
            }
        ],
        "never_asserts": ["tactical_noise_over_terminal_trajectory", "short_term_comfort_as_strategic_safety"],
        "evidence_strength": EvidenceStrength.EXPERT_CONSENSUS,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="I wanted to project myself forward to age eighty and look back on my life, minimizing the number of regrets I would carry.",
                chapter="ch_24_horizons",
                locator="section_regret_framework_p02",
            )
        ],
    },
    {
        "template_id": "a0_antifragile_convexity_profiling",
        "title": "Antifragility & Convex Optionality Profiling",
        "tier": VaultTier.A0_SYSTEMS_CORE,
        "cognitive_function": "risk_premortem_projection",
        "situation_frame": {
            "actors": ["decision_maker", "system_component"],
            "info_asymmetry": "mutual_uncertainty",
            "evidence_state": ["circumstantial_pattern", "direct_telemetry"],
            "incentive_posture": "stochastic_system",
            "system_dynamics": "anomalous_deviation",
            "primary_uncertainty": ["future_trajectory", "root_cause"],
            "spatial_or_forensic_flag": False,
        },
        "retrieval": {
            "situation_signature": "A system is optimized for hyper-efficient performance in a predictable environment but shatters catastrophically upon exposure to volatility, while antifragile architectures thrive and gain from disorder.",
            "trigger_conditions": [
                "supply chains stripped of all inventory redundancy to maximize short-term accounting margins",
                "structures with concave payoff profiles where downside is unbounded and upside is capped",
            ],
            "symptom_phrases": [
                "convex payoff profile",
                "benefiting from disorder",
                "hyper efficiency fragility",
                "barbell optionality strategy",
            ],
            "information_structure": "Payoff convexity curve showing positive asymmetry where gains from volatility strictly exceed potential losses.",
            "analog_vignettes": [
                "Distributed Systems: Implementing chaos engineering to continuously inject network latency and kill servers, causing autonomous self-healing microservices to grow more resilient over time.",
                "Financial Markets: Structuring a barbell investment portfolio consisting of 90 percent ultra-safe treasury bills and 10 percent convex asymmetric out-of-the-money options.",
                "Organizational Behavior: Empowering independent engineering pods to test decentralized product prototypes, capturing unlimited upside from hits while capping failure costs to small sprints.",
                "Physical Logistics & Sensor Telemetry: Utilizing decentralized distributed volunteer satellite imagery review networks that accelerate analysis throughput as regional crises escalate.",
            ],
        },
        "preconditions": [
            "exposure to volatile, uncertain, or disordered external environments",
            "structural capacity to bound downside risk while leaving upside unbound",
        ],
        "contraindications": [
            "subsystems with non-negotiable zero-tolerance safety limits (e.g., nuclear reactor cooling cores) where all volatility is catastrophic",
        ],
        "procedure_steps": [
            "Audit the system payoff profile to identify hidden concave vulnerabilities where volatility produces non-linear losses.",
            "Construct a barbell architecture: aggressively protect against catastrophic downside risk while creating small, decentralized convex options.",
            "Expose the system to controlled, low-stakes stressors to stimulate adaptive strengthening without risking systemic ruin.",
        ],
        "diagnostic_probes": [
            {
                "question": "Does this architecture gain more from positive volatility than it loses from negative volatility of equal magnitude?",
                "discriminates_between": ["convex_antifragile_structure", "concave_fragile_optimization"],
                "verifiable_anchor_sought": "Empirical payoff curvature plot demonstrating positive second-derivative response under stress.",
                "diagnosticity_weight": 0.94,
            }
        ],
        "never_asserts": ["hyper_efficiency_without_fragility", "volatility_immunity_without_optionality"],
        "evidence_strength": EvidenceStrength.EMPIRICAL_VALIDATED,
        "disputed_with": [],
        "merged_from": [],
        "provenance": [
            _make_anchor(
                quote="Some things benefit from shocks; they thrive and grow when exposed to volatility, randomness, disorder, and stressors.",
                chapter="ch_25_antifragility",
                locator="section_taleb_convexity_p01",
            )
        ],
    },
]


def load_and_validate_tier_a0_seed() -> list[ForwardThoughtTemplate]:
    """Instantiate and validate all 25 Tier A0 seed playbooks against Pydantic schemas and BANNED_LEXICON."""
    templates: list[ForwardThoughtTemplate] = []

    for raw in RAW_SEED_DEFINITIONS:
        # 1. Pydantic schema validation
        template = ForwardThoughtTemplate.model_validate(raw)

        # 2. BANNED_LEXICON validation across all serialized text
        serialized_json = template.model_dump_json()
        match = BANNED_LEXICON.search(serialized_json)
        if match:
            raise ValueError(
                f"BANNED_LEXICON leakage ({match.group(0)!r}) detected in Tier A0 seed playbook {template.template_id}!"
            )

        # 3. Vignette diversity check (must have 4-6 vignettes)
        vignettes = template.retrieval.analog_vignettes
        if len(vignettes) < 4:
            raise ValueError(
                f"Playbook {template.template_id} has only {len(vignettes)} analog vignettes (expected >= 4)."
            )

        templates.append(template)

    return templates


def audit_seed_templates_with_codex(
    templates: list[ForwardThoughtTemplate],
    max_audit: int = 1,
) -> None:
    """Audit a sample or all seed templates via docked Codex CLI (gpt-6.1-sol)."""
    to_audit = templates[:max_audit] if max_audit > 0 else templates
    logger.info("Auditing %d Tier A0 seed templates with docked Codex (gpt-6.1-sol)...", len(to_audit))

    for idx, t in enumerate(to_audit, 1):
        logger.info("[%d/%d] Auditing template: %s", idx, len(to_audit), t.template_id)
        audit_res = codex_audit_template(
            template_dict=t.model_dump(),
            source_excerpt=f"Tier A0 seed epistemological framework: {t.title}. Provenance quote: {t.provenance[0].verbatim_quote}",
        )

        passed = bool(audit_res.get("passed", False))
        score = float(audit_res.get("structural_utility_score", 0.0))
        action = audit_res.get("recommended_action", "UNKNOWN")
        reasoning = audit_res.get("reasoning", "")
        logger.info(
            "Codex Audit Verdict for %s: passed=%s, score=%.2f, action=%s, reasoning=%s",
            t.template_id,
            passed,
            score,
            action,
            reasoning[:120],
        )

        if not passed or score < 0.75:
            raise RuntimeError(
                f"Tier A0 seed template {t.template_id} failed Codex audit! Score: {score}, Action: {action}, Reasoning: {reasoning}"
            )


def write_seed_templates(
    templates: list[ForwardThoughtTemplate],
    output_path: Path,
) -> None:
    """Serialize validated Tier A0 templates to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for t in templates:
            f.write(t.model_dump_json() + "\n")
    logger.info("Successfully wrote %d Tier A0 seed templates to %s", len(templates), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tier A0 Systems Core Seed Knowledge Playbooks Generator")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/templates/raw/tier_a0_seed_templates.jsonl"),
        help="Target output JSONL path",
    )
    parser.add_argument(
        "--audit-mode",
        choices=["none", "sample", "all"],
        default="none",
        help="Docked Codex audit mode (default: none; sample audits first 1, all audits all 25)",
    )
    parser.add_argument(
        "--audit-sample-size",
        type=int,
        default=1,
        help="Number of templates to audit in sample mode",
    )
    args = parser.parse_args()

    logger.info("Instantiating and validating 25 Tier A0 seed playbooks...")
    templates = load_and_validate_tier_a0_seed()
    logger.info("Successfully validated %d Tier A0 seed playbooks with zero banned terms.", len(templates))

    if args.audit_mode == "sample":
        audit_seed_templates_with_codex(templates, max_audit=args.audit_sample_size)
    elif args.audit_mode == "all":
        audit_seed_templates_with_codex(templates, max_audit=len(templates))
    else:
        logger.info("Codex audit skipped by mode '%s'.", args.audit_mode)

    write_seed_templates(templates, args.output)


if __name__ == "__main__":
    main()
