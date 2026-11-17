"""Preserved-Dissent Deduplicator and Master Playbook Builder for Project J-EI.

Processes Vault Alpha (ForwardThoughtTemplate) and Vault Omega (CriticThoughtTemplate) strictly
separately. Uses dual canonical views (Situation S and Procedure P), 384-dim L2-normalized
embeddings (BAAI/bge-small-en-v1.5), combined cosine similarity, and verbatim quote
Jaccard indexing.

Key Invariants:
1. Auto-Merge: Pairs with sim >= 0.93 and quote_jaccard >= 0.50 merge automatically without LLM.
2. Agglomerative Clustering: Clusters borderline pairs (distance_threshold=0.18, max_cluster_size=10).
3. Borderline Adjudication: LLM classifies candidate pairs into DUPLICATE | SUBSUMES | COMPLEMENTARY | CONTRADICTORY | DISTINCT.
4. Preserved-Dissent Invariant: CONTRADICTORY templates are NEVER merged; both are preserved
   as distinct Master Playbooks with mutually cross-referenced IDs in `disputed_with`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from j_ei.codex_bridge import codex_exec_raw
from j_ei.config import JEIConfig, embed_texts, get_default_config
from j_ei.schemas import (
    CriticThoughtTemplate,
    DiagnosticProbe,
    ForwardThoughtTemplate,
    SourceAnchor,
)

logger = logging.getLogger("j_ei.deduplicator")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


AdjudicationVerdict = Literal["DUPLICATE", "SUBSUMES", "COMPLEMENTARY", "CONTRADICTORY", "DISTINCT"]


def canonical_strings_alpha(template: ForwardThoughtTemplate) -> tuple[str, str]:
    """Emit dual canonical views (Situation S, Procedure P) for a ForwardThoughtTemplate."""
    triggers = " | ".join(template.retrieval.trigger_conditions)
    symptoms = " | ".join(template.retrieval.symptom_phrases)
    view_s = (
        f"Title: {template.title}\n"
        f"Cognitive Function: {template.cognitive_function}\n"
        f"Signature: {template.retrieval.situation_signature}\n"
        f"Triggers: {triggers}\n"
        f"Symptoms: {symptoms}\n"
        f"Structure: {template.retrieval.information_structure}"
    )

    steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(template.procedure_steps))
    probes = "\n".join(
        f"Probe {i+1}: {p.question} (discriminates: {p.discriminates_between[0]} vs {p.discriminates_between[1]})"
        for i, p in enumerate(template.diagnostic_probes)
    )
    pre = " | ".join(template.preconditions)
    contra = " | ".join(template.contraindications)
    view_p = (
        f"Procedures:\n{steps}\n"
        f"Diagnostic Probes:\n{probes}\n"
        f"Preconditions: {pre}\n"
        f"Contraindications: {contra}"
    )

    return view_s, view_p


def canonical_strings_omega(template: CriticThoughtTemplate) -> tuple[str, str]:
    """Emit dual canonical views (Situation S, Procedure P) for a CriticThoughtTemplate."""
    vulns = " | ".join(template.target_vulnerabilities)
    symptoms = " | ".join(template.detectable_symptoms)
    view_s = (
        f"Technique: {template.technique_name}\n"
        f"Target Vulnerabilities: {vulns}\n"
        f"Detectable Symptoms: {symptoms}"
    )

    proc = "\n".join(f"{i+1}. {step}" for i, step in enumerate(template.audit_procedure))
    kills = " | ".join(template.kill_conditions)
    remed = " | ".join(template.remediation_actions)
    limits = " | ".join(template.known_limitations)
    view_p = (
        f"Audit Procedure:\n{proc}\n"
        f"Kill Conditions: {kills}\n"
        f"Remediation: {remed}\n"
        f"Limitations: {limits}"
    )

    return view_s, view_p


def quote_jaccard_similarity(q1: str, q2: str) -> float:
    """Compute token Jaccard similarity between two verbatim quotes."""
    t1 = set(re.findall(r"\w+", q1.lower()))
    t2 = set(re.findall(r"\w+", q2.lower()))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def max_quote_jaccard(
    anchors_a: list[SourceAnchor],
    anchors_b: list[SourceAnchor],
) -> float:
    """Compute maximum pairwise verbatim quote Jaccard similarity between two anchor sets."""
    if not anchors_a or not anchors_b:
        return 0.0
    max_sim = 0.0
    for a in anchors_a:
        for b in anchors_b:
            sim = quote_jaccard_similarity(a.verbatim_quote, b.verbatim_quote)
            if sim > max_sim:
                max_sim = sim
    return max_sim


def compute_similarity_matrix(
    templates: list[ForwardThoughtTemplate | CriticThoughtTemplate],
    is_omega: bool = False,
    config: JEIConfig | None = None,
) -> np.ndarray:
    """Compute pairwise similarity matrix: 0.5 * cos(S) + 0.5 * cos(P).

    Uses MRL 768-dim L2-normalized embeddings via embed_texts.
    """
    n = len(templates)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if n == 1:
        return np.ones((1, 1), dtype=np.float32)

    cfg = config or get_default_config()

    s_views: list[str] = []
    p_views: list[str] = []

    for t in templates:
        if is_omega:
            s, p = canonical_strings_omega(t)  # type: ignore[arg-type]
        else:
            s, p = canonical_strings_alpha(t)  # type: ignore[arg-type]
        s_views.append(s)
        p_views.append(p)

    logger.info("Embedding %d Situation views (input_type='passage', dim=768)...", n)
    s_embeddings = embed_texts(s_views, input_type="passage", target_dim=768, config=cfg)
    logger.info("Embedding %d Procedure views (input_type='passage', dim=768)...", n)
    p_embeddings = embed_texts(p_views, input_type="passage", target_dim=768, config=cfg)

    s_mat = np.array(s_embeddings, dtype=np.float32)
    p_mat = np.array(p_embeddings, dtype=np.float32)

    # Since embed_texts vectors are already L2-normalized, dot product equals cosine similarity
    s_cos = np.clip(s_mat @ s_mat.T, -1.0, 1.0)
    p_cos = np.clip(p_mat @ p_mat.T, -1.0, 1.0)

    combined = 0.5 * s_cos + 0.5 * p_cos
    np.fill_diagonal(combined, 1.0)
    return np.clip(combined, 0.0, 1.0)


def cluster_templates(
    sim_matrix: np.ndarray,
    distance_threshold: float = 0.18,
    max_cluster_size: int = 10,
) -> list[list[int]]:
    """Cluster templates using AgglomerativeClustering on 1 - sim_matrix.

    Partitions clusters exceeding max_cluster_size into sub-clusters.
    """
    n = sim_matrix.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    dist_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)
    # Ensure exact symmetric distance matrix with 0 diagonal
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = clustering.fit_predict(dist_matrix)

    clusters_dict: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        clusters_dict.setdefault(lbl, []).append(idx)

    final_clusters: list[list[int]] = []
    for c_indices in clusters_dict.values():
        if len(c_indices) <= max_cluster_size:
            final_clusters.append(c_indices)
        else:
            # Sub-partition large clusters into chunks of at most max_cluster_size
            for chunk_start in range(0, len(c_indices), max_cluster_size):
                final_clusters.append(c_indices[chunk_start : chunk_start + max_cluster_size])

    return final_clusters


def adjudicate_pair_relationship(
    template_a: ForwardThoughtTemplate | CriticThoughtTemplate,
    template_b: ForwardThoughtTemplate | CriticThoughtTemplate,
    is_omega: bool = False,
    config: JEIConfig | None = None,
) -> AdjudicationVerdict:
    """Call LLM to classify relationship: DUPLICATE | SUBSUMES | COMPLEMENTARY | CONTRADICTORY | DISTINCT."""
    cfg = config or get_default_config()

    if is_omega:
        sa, pa = canonical_strings_omega(template_a)  # type: ignore[arg-type]
        sb, pb = canonical_strings_omega(template_b)  # type: ignore[arg-type]
        role_desc = "Vault Omega Adversarial Critic Techniques"
    else:
        sa, pa = canonical_strings_alpha(template_a)  # type: ignore[arg-type]
        sb, pb = canonical_strings_omega(template_b) if is_omega else canonical_strings_alpha(template_b)  # type: ignore[arg-type]
        role_desc = "Vault Alpha Forward Cognitive Playbooks"

    prompt = f"""You are an expert cognitive epistemologist adjudicating two candidate reasoning templates in {role_desc}.

Analyze Template A and Template B carefully:

--- TEMPLATE A ({template_a.template_id}) ---
{sa}
{pa}

--- TEMPLATE B ({template_b.template_id}) ---
{sb}
{pb}

Classify their semantic and operational relationship into EXACTLY ONE category:
1. DUPLICATE: They express the exact same cognitive procedure, diagnostic questions, and triggers under slightly different phrasing.
2. SUBSUMES: One template is a broader, higher-level generalization that completely contains the analytical utility of the other.
3. COMPLEMENTARY: They address related phenomena or sequential stages, but describe distinct procedures that should both be retained.
4. CONTRADICTORY: They represent directly conflicting epistemic models, mutually exclusive diagnostic advice, or disputed analytical axioms (e.g. one asserts presence of cue X proves Y, while the other asserts X disproves Y).
5. DISTINCT: They describe entirely different cognitive operations or problem classes with minimal overlap.

Respond in strict JSON format:
{{"verdict": "DUPLICATE" | "SUBSUMES" | "COMPLEMENTARY" | "CONTRADICTORY" | "DISTINCT", "reasoning": "1-2 concise sentences"}}
"""

    try:
        content = codex_exec_raw(prompt, config=cfg)
        # Match JSON in response
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            verdict = parsed.get("verdict", "").strip().upper()
            if verdict in ("DUPLICATE", "SUBSUMES", "COMPLEMENTARY", "CONTRADICTORY", "DISTINCT"):
                logger.debug(
                    "Adjudication (%s vs %s) -> %s: %s",
                    template_a.template_id,
                    template_b.template_id,
                    verdict,
                    parsed.get("reasoning", ""),
                )
                return verdict  # type: ignore[return-value]
    except Exception as exc:
        logger.warning(
            "Adjudication LLM call failed for %s vs %s: %s. Defaulting to DISTINCT.",
            template_a.template_id,
            template_b.template_id,
            exc,
        )

    return "DISTINCT"


def merge_alpha_templates(
    master: ForwardThoughtTemplate,
    secondary: ForwardThoughtTemplate,
) -> ForwardThoughtTemplate:
    """Merge secondary ForwardThoughtTemplate into master, preserving all schema invariants."""
    data = master.model_dump()

    # 1. Update merged_from
    merged_set = set(data.get("merged_from", []))
    merged_set.add(secondary.template_id)
    merged_set.update(secondary.merged_from)
    merged_set.discard(master.template_id)
    data["merged_from"] = sorted(merged_set)

    # 2. Unite provenance anchors (deduplicate by source_span_sha256)
    existing_anchors = {a["source_span_sha256"]: a for a in data["provenance"]}
    for a in secondary.provenance:
        a_dict = a.model_dump()
        if a.source_span_sha256 not in existing_anchors:
            existing_anchors[a.source_span_sha256] = a_dict
    data["provenance"] = list(existing_anchors.values())

    # 3. Combine analog vignettes (deduplicating, max 6)
    vignettes = list(data["retrieval"]["analog_vignettes"])
    for v in secondary.retrieval.analog_vignettes:
        if v not in vignettes and len(vignettes) < 6:
            vignettes.append(v)
    data["retrieval"]["analog_vignettes"] = vignettes

    # 4. Combine trigger conditions (deduplicating, max 6)
    triggers = list(data["retrieval"]["trigger_conditions"])
    for tc in secondary.retrieval.trigger_conditions:
        if tc not in triggers and len(triggers) < 6:
            triggers.append(tc)
    data["retrieval"]["trigger_conditions"] = triggers

    # 5. Combine symptom phrases (deduplicating, max 8)
    symptoms = list(data["retrieval"]["symptom_phrases"])
    for s in secondary.retrieval.symptom_phrases:
        if s not in symptoms and len(symptoms) < 8:
            symptoms.append(s)
    data["retrieval"]["symptom_phrases"] = symptoms

    # 6. Combine diagnostic probes (deduplicating by question text, max 6)
    existing_probes = {p["question"].strip().lower() for p in data["diagnostic_probes"]}
    for p in secondary.diagnostic_probes:
        p_dict = p.model_dump()
        q_norm = p.question.strip().lower()
        if q_norm not in existing_probes and len(data["diagnostic_probes"]) < 6:
            data["diagnostic_probes"].append(p_dict)
            existing_probes.add(q_norm)

    # 7. Combine preconditions & contraindications (max 6 each)
    pre = list(data["preconditions"])
    for p in secondary.preconditions:
        if p not in pre and len(pre) < 6:
            pre.append(p)
    data["preconditions"] = pre

    contra = list(data["contraindications"])
    for c in secondary.contraindications:
        if c not in contra and len(contra) < 6:
            contra.append(c)
    data["contraindications"] = contra

    # 8. Unite disputed_with
    disputes = set(data.get("disputed_with", []))
    disputes.update(secondary.disputed_with)
    disputes.discard(master.template_id)
    data["disputed_with"] = sorted(disputes)

    return ForwardThoughtTemplate.model_validate(data)


def merge_omega_templates(
    master: CriticThoughtTemplate,
    secondary: CriticThoughtTemplate,
) -> CriticThoughtTemplate:
    """Merge secondary CriticThoughtTemplate into master, preserving all schema invariants."""
    data = master.model_dump()

    # 1. Unite provenance anchors (deduplicate by source_span_sha256)
    existing_anchors = {a["source_span_sha256"]: a for a in data["provenance"]}
    for a in secondary.provenance:
        a_dict = a.model_dump()
        if a.source_span_sha256 not in existing_anchors:
            existing_anchors[a.source_span_sha256] = a_dict
    data["provenance"] = list(existing_anchors.values())

    # 2. Unite target vulnerabilities
    vulns = list(data["target_vulnerabilities"])
    for v in secondary.target_vulnerabilities:
        if v not in vulns:
            vulns.append(v)
    data["target_vulnerabilities"] = vulns

    # 3. Unite detectable symptoms (max 8)
    symptoms = list(data["detectable_symptoms"])
    for s in secondary.detectable_symptoms:
        if s not in symptoms and len(symptoms) < 8:
            symptoms.append(s)
    data["detectable_symptoms"] = symptoms

    # 4. Unite kill conditions (max 6)
    kills = list(data["kill_conditions"])
    for k in secondary.kill_conditions:
        if k not in kills and len(kills) < 6:
            kills.append(k)
    data["kill_conditions"] = kills

    # 5. Unite remediation actions
    remed = list(data["remediation_actions"])
    for r in secondary.remediation_actions:
        if r not in remed:
            remed.append(r)
    data["remediation_actions"] = remed

    # 6. Unite known limitations (max 5)
    limits = list(data["known_limitations"])
    for l in secondary.known_limitations:
        if l not in limits and len(limits) < 5:
            limits.append(l)
    data["known_limitations"] = limits

    return CriticThoughtTemplate.model_validate(data)


def deduplicate_vault(
    templates: list[ForwardThoughtTemplate | CriticThoughtTemplate],
    is_omega: bool = False,
    auto_merge_sim_threshold: float = 0.93,
    auto_merge_quote_threshold: float = 0.50,
    borderline_sim_threshold: float = 0.82,
    config: JEIConfig | None = None,
) -> tuple[list[ForwardThoughtTemplate | CriticThoughtTemplate], dict[str, Any]]:
    """Deduplicate a single vault library with preserved-dissent guarantees.

    Returns (master_templates, metrics_dict).
    """
    n = len(templates)
    metrics: dict[str, Any] = {
        "initial_count": n,
        "auto_merged_count": 0,
        "llm_adjudicated_count": 0,
        "llm_merged_count": 0,
        "contradictory_preserved_count": 0,
        "final_count": 0,
    }

    if n <= 1:
        metrics["final_count"] = n
        return templates, metrics

    logger.info("Computing pairwise similarity matrix for %d templates...", n)
    sim_matrix = compute_similarity_matrix(templates, is_omega=is_omega, config=config)

    logger.info("Clustering templates with AgglomerativeClustering (threshold=0.18)...")
    clusters = cluster_templates(sim_matrix, distance_threshold=0.18, max_cluster_size=10)
    logger.info("Identified %d candidate clusters across %d templates.", len(clusters), n)

    # Active templates pool indexed by original position
    # Each entry is either a template object or None (if merged into another)
    pool: list[ForwardThoughtTemplate | CriticThoughtTemplate | None] = list(templates)

    for cluster_idx, c_indices in enumerate(clusters, 1):
        if len(c_indices) <= 1:
            continue

        logger.info(
            "Evaluating cluster %d/%d (%d templates): %s",
            cluster_idx,
            len(clusters),
            len(c_indices),
            [templates[i].template_id for i in c_indices],
        )

        # Compare pairs within the cluster
        for i_pos in range(len(c_indices)):
            for j_pos in range(i_pos + 1, len(c_indices)):
                idx_a = c_indices[i_pos]
                idx_b = c_indices[j_pos]

                template_a = pool[idx_a]
                template_b = pool[idx_b]

                if template_a is None or template_b is None:
                    continue

                sim = float(sim_matrix[idx_a, idx_b])
                q_sim = max_quote_jaccard(template_a.provenance, template_b.provenance)

                # Path 1: High Similarity + High Quote Jaccard -> Auto-Merge Shortcut
                if sim >= auto_merge_sim_threshold and q_sim >= auto_merge_quote_threshold:
                    logger.info(
                        "Auto-merging '%s' and '%s' (sim=%.3f, quote_jaccard=%.3f)",
                        template_a.template_id,
                        template_b.template_id,
                        sim,
                        q_sim,
                    )
                    if is_omega:
                        pool[idx_a] = merge_omega_templates(template_a, template_b)  # type: ignore[arg-type]
                    else:
                        pool[idx_a] = merge_alpha_templates(template_a, template_b)  # type: ignore[arg-type]
                    pool[idx_b] = None
                    metrics["auto_merged_count"] += 1
                    continue

                # Path 2: Borderline Similarity -> LLM Adjudication
                if sim >= borderline_sim_threshold:
                    metrics["llm_adjudicated_count"] += 1
                    verdict = adjudicate_pair_relationship(
                        template_a,
                        template_b,
                        is_omega=is_omega,
                        config=config,
                    )
                    logger.info(
                        "Adjudication verdict for '%s' vs '%s' (sim=%.3f): %s",
                        template_a.template_id,
                        template_b.template_id,
                        sim,
                        verdict,
                    )

                    if verdict in ("DUPLICATE", "SUBSUMES"):
                        if is_omega:
                            pool[idx_a] = merge_omega_templates(template_a, template_b)  # type: ignore[arg-type]
                        else:
                            pool[idx_a] = merge_alpha_templates(template_a, template_b)  # type: ignore[arg-type]
                        pool[idx_b] = None
                        metrics["llm_merged_count"] += 1

                    elif verdict == "CONTRADICTORY":
                        # CRITICAL PRESERVED-DISSENT INVARIANT:
                        # NEVER MERGE CONTRADICTORY TEMPLATES!
                        # Cross-link their IDs in disputed_with and keep both intact!
                        if not is_omega:
                            # ForwardThoughtTemplate supports disputed_with
                            t_a = pool[idx_a]
                            t_b = pool[idx_b]
                            if t_a is not None and t_b is not None:
                                d_a = set(t_a.disputed_with)
                                d_a.add(t_b.template_id)
                                pool[idx_a] = t_a.model_copy(update={"disputed_with": sorted(d_a)})

                                d_b = set(t_b.disputed_with)
                                d_b.add(t_a.template_id)
                                pool[idx_b] = t_b.model_copy(update={"disputed_with": sorted(d_b)})

                        logger.info(
                            "Preserving contradictory dissent between '%s' and '%s' (cross-linked in disputed_with).",
                            template_a.template_id,
                            template_b.template_id,
                        )
                        metrics["contradictory_preserved_count"] += 1

                    # For COMPLEMENTARY or DISTINCT: do nothing, both remain separate in pool

    final_masters = [t for t in pool if t is not None]
    metrics["final_count"] = len(final_masters)
    logger.info(
        "Vault deduplication complete. Initial: %d -> Final: %d (Auto-merged: %d, LLM-merged: %d, Contradictory preserved: %d)",
        metrics["initial_count"],
        metrics["final_count"],
        metrics["auto_merged_count"],
        metrics["llm_merged_count"],
        metrics["contradictory_preserved_count"],
    )
    return final_masters, metrics


def load_raw_templates(raw_dir: Path) -> tuple[list[ForwardThoughtTemplate], list[CriticThoughtTemplate]]:
    """Scan raw templates directory and partition into Vault Alpha and Vault Omega lists."""
    alpha_templates: list[ForwardThoughtTemplate] = []
    omega_templates: list[CriticThoughtTemplate] = []

    jsonl_files = sorted(raw_dir.glob("*templates.jsonl"))
    logger.info("Found %d template JSONL files in %s", len(jsonl_files), raw_dir)

    for fpath in jsonl_files:
        with fpath.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    payload.pop("_codex_audit", None)
                    payload.pop("codex_audit", None)
                    # Discriminate schema by key fields
                    if "target_vulnerabilities" in payload or "audit_procedure" in payload:
                        omega_templates.append(CriticThoughtTemplate.model_validate(payload))
                    else:
                        alpha_templates.append(ForwardThoughtTemplate.model_validate(payload))
                except Exception as exc:
                    logger.warning("Failed to parse template at %s:%d: %s", fpath.name, line_no, exc)

    logger.info("Loaded %d Vault Alpha templates and %d Vault Omega templates.", len(alpha_templates), len(omega_templates))
    return alpha_templates, omega_templates


def deduplicate_all(
    raw_dir: Path,
    output_dir: Path,
    config: JEIConfig | None = None,
) -> dict[str, Any]:
    """Run full deduplication across Vault Alpha and Vault Omega, emitting master playbooks."""
    alpha_templates, omega_templates = load_raw_templates(raw_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}

    # 1. Process Vault Alpha
    if alpha_templates:
        logger.info("=== Deduplicating Vault Alpha (%d templates) ===", len(alpha_templates))
        masters_alpha, metrics_alpha = deduplicate_vault(
            templates=alpha_templates,  # type: ignore[arg-type]
            is_omega=False,
            config=config,
        )
        alpha_out = output_dir / "vault_alpha_masters.jsonl"
        with alpha_out.open("w", encoding="utf-8") as f:
            for t in masters_alpha:
                f.write(t.model_dump_json() + "\n")
        logger.info("Saved %d Vault Alpha Master Playbooks to %s", len(masters_alpha), alpha_out)
        summary["vault_alpha"] = metrics_alpha
    else:
        logger.info("No Vault Alpha templates found in %s", raw_dir)
        summary["vault_alpha"] = {"initial_count": 0, "final_count": 0}

    # 2. Process Vault Omega
    if omega_templates:
        logger.info("=== Deduplicating Vault Omega (%d templates) ===", len(omega_templates))
        masters_omega, metrics_omega = deduplicate_vault(
            templates=omega_templates,  # type: ignore[arg-type]
            is_omega=True,
            config=config,
        )
        omega_out = output_dir / "vault_omega_masters.jsonl"
        with omega_out.open("w", encoding="utf-8") as f:
            for t in masters_omega:
                f.write(t.model_dump_json() + "\n")
        logger.info("Saved %d Vault Omega Master Playbooks to %s", len(masters_omega), omega_out)
        summary["vault_omega"] = metrics_omega
    else:
        logger.info("No Vault Omega templates found in %s", raw_dir)
        summary["vault_omega"] = {"initial_count": 0, "final_count": 0}

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="J-EI Preserved-Dissent Deduplicator & Master Playbook Builder")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/templates/raw"),
        help="Path to raw templates directory containing JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/templates/masters"),
        help="Path to masters output directory",
    )
    args = parser.parse_args()

    logger.info("Starting J-EI deduplication pipeline...")
    summary = deduplicate_all(raw_dir=args.input_dir, output_dir=args.output_dir)
    print("\n--- Deduplication Summary ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

