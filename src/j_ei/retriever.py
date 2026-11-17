"""Vault Alpha playbook retrieval via local BAAI/bge-small-en-v1.5 embeddings.

Loads the 26 Vault Alpha ForwardThoughtTemplates and retrieves the top-k
most relevant playbooks for a given user query using cosine similarity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from j_ei.config import JEIConfig, embed_texts, get_default_config
from j_ei.schemas import ForwardThoughtTemplate


def load_vault_playbooks(
    alpha_path: Path | None = None,
) -> list[ForwardThoughtTemplate]:
    """Load all Vault Alpha ForwardThoughtTemplates from JSONL file."""
    if alpha_path is None:
        alpha_path = Path("data/templates/masters/vault_alpha_masters.jsonl")

    templates: list[ForwardThoughtTemplate] = []
    with open(alpha_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                templates.append(ForwardThoughtTemplate.model_validate_json(line))
    return templates


def playbook_to_text(t: ForwardThoughtTemplate) -> str:
    """Convert a playbook template to a searchable text representation."""
    parts = [
        f"{t.title} ({t.cognitive_function})",
        f"Signature: {t.retrieval.situation_signature}",
        f"Triggers: {', '.join(t.retrieval.trigger_conditions[:3])}",
        f"Symptoms: {', '.join(t.retrieval.symptom_phrases[:3])}",
    ]
    return ". ".join(parts)


def playbook_to_context_chunk(t: ForwardThoughtTemplate) -> str:
    """Convert a playbook template to a rich context chunk for injection."""
    probes = "\n".join(
        f"  - Q: {p.question} (discriminates: {p.discriminates_between[0]} vs {p.discriminates_between[1]}, weight={p.diagnosticity_weight})"
        for p in t.diagnostic_probes[:3]
    )
    return f"""### PLAYBOOK: {t.title}
- **Cognitive Function:** {t.cognitive_function}
- **Tier:** {t.tier.value}
- **Situation Signature:** {t.retrieval.situation_signature}
- **Trigger Conditions:** {', '.join(t.retrieval.trigger_conditions)}
- **Symptom Phrases:** {', '.join(t.retrieval.symptom_phrases)}
- **Procedure Steps:**
  {chr(10).join(f'  {i+1}. {s}' for i, s in enumerate(t.procedure_steps))}
- **Diagnostic Probes:**
{probes}
- **Preconditions:** {', '.join(t.preconditions)}
- **Contraindications:** {', '.join(t.contraindications)}
"""


def retrieve_top_k(
    query: str,
    playbooks: list[ForwardThoughtTemplate] | None = None,
    alpha_path: Path | None = None,
    k: int = 4,
    config: JEIConfig | None = None,
) -> list[tuple[ForwardThoughtTemplate, float]]:
    """Retrieve top-k playbooks by cosine similarity to the query.

    Returns list of (template, similarity_score) tuples sorted by descending score.
    """
    cfg = config or get_default_config()

    if playbooks is None:
        playbooks = load_vault_playbooks(alpha_path)

    if not playbooks:
        return []

    # Embed query
    q_vecs = embed_texts([query], input_type="query", config=cfg)
    q_vec = np.asarray(q_vecs[0], dtype=np.float32)

    # Embed all playbook texts
    pb_texts = [playbook_to_text(t) for t in playbooks]
    pb_vecs = embed_texts(pb_texts, input_type="passage", config=cfg)

    # Compute cosine similarities (vectors are already L2-normed)
    sims: list[float] = []
    for pv in pb_vecs:
        d_vec = np.asarray(pv, dtype=np.float32)
        sims.append(float(np.dot(q_vec, d_vec)))

    # Rank and return top-k
    ranked = np.argsort(sims)[::-1][:k]
    return [(playbooks[idx], sims[idx]) for idx in ranked]

