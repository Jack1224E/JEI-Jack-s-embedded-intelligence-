<div align="center">

# ⚡ Project J-EI
### Jack's Embedded Intelligence

**Turbocharging Frontier AI Reasoning by Injecting Systems Engineering Playbooks & Live Grounding RAG**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/Tests-69%20Passing-brightgreen.svg)]()
[![Embeddings: BGE-small](https://img.shields.io/badge/Embeddings-BAAI%2Fbge--small--en-orange.svg)]()
[![Engine: Codex Gateway](https://img.shields.io/badge/Gateway-AgentRouter%20Codex-purple.svg)]()

<p align="center">
  <a href="#-the-big-idea-in-plain-english">The Big Idea</a> •
  <a href="#-how-it-works-the-deep-dive">Technical Deep Dive</a> •
  <a href="#-benchmarks--case-studies">Real-World Case Studies</a> •
  <a href="#-getting-started">Quickstart</a>
</p>

---

</div>

## 💡 The Big Idea (In Plain English)

### Why do the smartest AI models still fail at tough real-world problems?

Even the most advanced Large Language Models often stumble when debugging complex, mission-critical systems—whether it is a catastrophic crop failure on a 5,000-acre smart farm or numerical divergence in custom neural network code.

**Why?** Because an LLM out of the box is like a brilliant savant trapped in a room with no books and no internet. It remembers billions of words, but when pushed into intricate engineering corners, it relies on pattern guesswork. It confuses averages with ground truth and hallucinates missing data.

### How Project J-EI Turbocharges the AI

**Project J-EI gives the AI an elite cognitive operating system:**

1. 🧠 **Cognitive Playbook RAG:** Instead of dumping random text into context, J-EI searches a curated vault of battle-tested systems engineering frameworks (First-Principles Decomposition, Inversion Analysis, Bayesian Updating, Ishikawa Root Cause). The AI is handed the exact playbook for *how an elite investigator thinks* before it writes a single word.
2. 🌐 **Real-Time Scientific Grounding:** The AI doesn't guess empirical numbers. J-EI automatically queries live scientific engines (DuckDuckGo Lite, EuropePMC, and PubMed) to verify physical constants, peer-reviewed thresholds, and domain manuals on the fly.
3. 🛡️ **Anti-Hallucination Discipline:** If real-world telemetry or field baselines are missing, the AI is mathematically forbidden from inventing them—it must isolate the gap and flag `[DATA REQUIRED]`.

> **The Result:** The model shifts from *generating believable-sounding text* to *executing rigorous systems forensics*.

---

## 🏗️ Architecture: How It Works

```
                                USER QUERY
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│     Cognitive Playbook RAG    │               │    Multi-Source Grounding     │
│   (BAAI/bge-small-en-v1.5)    │               │    (Live Web & Literature)    │
│                               │               │                               │
│ • 100% Local 384-dim Vectors  │               │ • DuckDuckGo Lite (No Bot-Ban)│
│ • Vault Alpha Mental Models   │               │ • EuropePMC Agronomy Research │
│ • Falsification Protocols     │               │ • NCBI PubMed / PMC E-Utils   │
└───────────────┬───────────────┘               └───────────────┬───────────────┘
                │                                               │
                └───────────────────────┬───────────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │      Codex Gateway Engine     │
                        │      (Single-Shot Docker)     │
                        │       Model: gpt-5.6-sol      │
                        │                               │
                        │ • Read-Only Sandboxed Exec    │
                        │ • Inversion & Falsification   │
                        │ • Strict [DATA REQUIRED] Tag  │
                        └───────────────┬───────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │   Verified Forensic Output    │
                        │  (Formulas, Root Causes, Math)│
                        └───────────────────────────────┘
```

---

## 🔬 Technical Deep Dive

> [!IMPORTANT]
> **Engine Runtime Scope & Model Prerequisites:**  
> In its current release, Project J-EI is hardcoded to orchestrate single-shot reasoning through the **OpenAI Codex CLI (`codex exec`)** powered by **ChatGPT / OpenAI-compatible reasoning models via AgentRouter API keys** (defaulting to `gpt-5.6-sol`). It operates via isolated, read-only sandboxed task workspaces rather than generic multi-turn chat loops. To run this project, an active Codex CLI installation and an AgentRouter or ChatGPT-compatible gateway key are strictly required.

### 1. Zero-Cost, 100% Local Playbook Retrieval
- **Vector Model:** `BAAI/bge-small-en-v1.5` via `sentence-transformers`.
- **Dimensions:** 384-dimensional normalized dense vectors.
- **Speed:** Instant in-memory cosine similarity across the 26 Vault Alpha master playbooks in `< 1.5s` warm.
- **Zero Cloud Leakage:** No proprietary playbook chunks or investigative heuristics ever hit commercial embedding APIs.

### 2. Multi-Source Resilient Grounding Engine
Instead of relying on fragile web scrapers that get tripped by Cloudflare and CAPTCHAs, J-EI routes searches across specialized protocols:
- **DuckDuckGo Lite Endpoint:** Direct server-rendered HTML parser for agricultural guides, manufacturer datasheets, and physical thresholds without JavaScript penalties or bot blocks.
- **EuropePMC REST API:** Direct unauthenticated scientific literature gateway returning abstracts on crop biology, rootstock genetics, and soil chemistry.
- **NCBI Entrez E-Utilities:** Direct querying of peer-reviewed biological literature.
- **Wikipedia API:** Conceptual taxonomy definitions.

### 3. AgentRouter Codex Gateway Execution
- Dispatches prompts via Codex CLI (`codex exec`) in an ephemeral `--sandbox read-only` execution dock.
- Runs on active high-reasoning models (`gpt-5.6-sol`) with guaranteed zero quota stalls.
- Enforces strict mathematical consistency (dimensional checks, sign conventions, conservation of mass/flow).

---

## 📊 Real-World Case Studies

### 🚜 Case 1: Arid Valley Precision Farming Crop Wilt Crisis
* **Challenge:** 4 agricultural blocks (almonds, citrus, wheat, legumes) suffering sudden mid-season wilt despite automated drip telemetry showing nominal N-P-K and normal water flow.
* **J-EI Output:** 
  - Proved aggregate telemetry masked localized emitter clogging and osmotic shock.
  - Retrieved exact empirical Maas-Hoffman salinity thresholds ($1.5\text{ dS/m}$ for almonds, $1.4\text{–}1.7\text{ dS/m}$ for citrus).
  - Derived total soil water potential ($\Psi_{\text{soil}} = \Psi_m + \Psi_s$) and leaching requirement fractions ($LF$) to rescue orchards without destroying diagnostic evidence.

### ⚡ Case 2: Neural Networks from Scratch Mastery
* **Challenge:** Formulate a zero-framework master progression to master deep learning implementations from first principles.
* **J-EI Output:**
  - Derived scalar reverse-mode autodiff with strict gradient accumulation invariants (`+=`).
  - Implemented numerical gradient verification ($g_{\text{num}} = \frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}$) targeting relative error $< 10^{-7}$.
  - Provided exact tensor algebra equations ($dW = X^\top dZ$, $dX = dZ W^\top$) and a full debugging hierarchy.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.11+
- [Codex CLI](https://github.com/openai/codex) installed (`npm install -g @openai/codex`)

### 2. Installation

```bash
git clone https://github.com/Jack1224E/JEI-Jack-s-embedded-intelligence-.git
cd JEI-Jack-s-embedded-intelligence-

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### 3. Configuration

Copy the example environment file and configure your AgentRouter key:

```bash
cp .env.example .env
```

Edit `.env`:
```env
agentrouter=sk-your-agentrouter-api-key
CODEX_MODEL=gpt-5.6-sol
```

### 4. Run the Verification Test Suite

Verify all 69 unit tests pass:

```bash
pytest tests/
```

### 5. Execute an Investigative Query

```python
from pathlib import Path
from j_ei.config import get_default_config
from j_ei.engine import run_jei_query

cfg = get_default_config()

result = run_jei_query(
    user_query="Your complex engineering or forensic problem here",
    config=cfg,
    top_k=3,
    output_file=Path("output.md")
)

print(result.output)
```

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.
