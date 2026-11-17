# Project J-EI (Jack's Embedded Intelligence)

Project J-EI is a domain-agnostic, forensic systems reasoning and investigative intelligence engine. It combines local semantic playbook retrieval, live empirical web and literature grounding, and single-shot execution via the AgentRouter Codex Gateway.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |              User Query               |
                      +-------------------+-------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
       +----------------------------+            +----------------------------+
       |   Local BGE Embeddings     |            |    Multi-Source Grounding  |
       |  (BAAI/bge-small-en-v1.5)  |            |   (DuckDuckGo Lite,        |
       |  Retrieves top-k playbooks |            |    EuropePMC, PubMed,      |
       |  from Vault Alpha Masters  |            |    Wikipedia API)          |
       +-------------+--------------+            +-------------+--------------+
                     |                                         |
                     +--------------------+--------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         Codex Gateway Bridge          |
                      |       (Single-Shot Execution)         |
                      |          Model: gpt-5.6-sol           |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Verified Forensic Output         |
                      |  (Formulas, Pros/Cons, Grounded Data) |
                      +---------------------------------------+
```

1. **Local Semantic Playbook Retrieval:**
   - 100% local embedding via `BAAI/bge-small-en-v1.5` (sentence-transformers).
   - Generates 384-dimensional normalized vectors to retrieve specialized investigative protocols.
2. **Multi-Source Grounding Search:**
   - **DuckDuckGo Lite:** Direct server-rendered HTML search for real-world manuals and thresholds (bypassing CAPTCHAs).
   - **Europe PMC & PubMed:** Peer-reviewed open-access scientific literature across agronomy, biology, and chemistry.
   - **Wikipedia API:** High-level domain definitions and terminology.
3. **Single-Shot Systems Reasoning:**
   - Routes inference through Codex CLI (`codex exec`) via AgentRouter.
   - Enforces strict anti-hallucination discipline: missing field parameters are labeled `[DATA REQUIRED]`.

---

## Getting Started

### 1. Installation

Clone the repository and set up a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Configuration

Create your `.env` file from the provided template:

```bash
cp .env.example .env
```

Add your AgentRouter API key to `.env`:
```bash
agentrouter=your_agentrouter_api_key_here
```

### 3. Running Unit Tests

Verify the complete test suite:

```bash
pytest tests/
```

### 4. Running a Query

Execute an end-to-end investigative query:

```python
from pathlib import Path
from j_ei.config import get_default_config
from j_ei.engine import run_jei_query

cfg = get_default_config()
res = run_jei_query(
    user_query="Your engineering or investigative question here",
    config=cfg,
    output_file=Path("output.md")
)
print(res.output)
```

---

## License

MIT License.
