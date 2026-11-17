import os
import shutil
import tempfile
import zipfile
from pathlib import Path

staging = Path(tempfile.mkdtemp(prefix="jei_zip_update_"))
with zipfile.ZipFile("jei_clean_migration.zip", "r") as zf:
    zf.extractall(staging)

# Copy updated search.py
shutil.copy("src/j_ei/search.py", staging / "search.py")

# Update manual 02
manual_02 = staging / "manuals" / "02_SEARXNG_AND_WEB_GROUNDING.md"
manual_02.write_text("""# Manual 02: Multi-Source Web & Scientific Literature Grounding

Project J-EI utilizes a resilient, multi-source search grounding architecture designed to prevent hallucinations on physical constants, empirical thresholds, and failure post-mortems without fragile external daemons or scraping blocks.

---

## 1. Grounding Search Engines

1. **DuckDuckGo Lite (Primary Web Search):**
   - End-point: `https://lite.duckduckgo.com/lite/`
   - Pure HTML parser via standard `urllib` + `lxml.html`.
   - **Zero CAPTCHA / Bot Blocks:** Unlike standard scrapers or SearXNG, DDG Lite serves plain server-rendered HTML without triggering Cloudflare / bot-detection challenges.
   - Extracts real grower guides, industry standards (e.g. `almonds.org`), manufacturer technical manuals, and empirical thresholds.

2. **EuropePMC REST API (Open Botanical & Agronomic Science Literature):**
   - End-point: `https://www.ebi.ac.uk/europepmc/webservices/rest/search`
   - Free, unauthenticated REST API returning peer-reviewed journal papers with abstracts.
   - Covers crop physiology, soil salinity, rootstocks (e.g., GF677), irrigation dynamics, and osmotic shock.

3. **NCBI PubMed / PMC E-Utilities (Biological & Pathology Research):**
   - End-point: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`
   - Direct biological research query via NCBI Entrez `esearch` + `esummary`.

4. **Wikipedia API (Domain Terminology & Definitions):**
   - High-level encyclopedia disambiguation for specialized concepts.

5. **SearXNG (Optional Local Aggregator):**
   - If a local SearXNG daemon is online at `http://localhost:8888`, it is queried; otherwise, the multi-source DDG Lite + EuropePMC pipeline engages automatically.

---

## 2. Python Usage Example

```python
from search import web_search

# Multi-source live search query
results = web_search("almond salinity threshold dS/m")
for r in results:
    print(f"[{r['source']}] {r['title']} -> {r['url']}")
    print(f"  {r['body'][:120]}...")
```

---

## 3. Why Local SearXNG Tripped CAPTCHA

When running SearXNG on localhost without rotating proxy pools, upstream DuckDuckGo and Brave detect repeated requests from a single IP and trigger `SearxEngineCaptchaException` / 429 Too Many Requests. The new direct multi-source architecture bypasses this entirely.
""", encoding="utf-8")

# Re-zip
if os.path.exists("jei_clean_migration.zip"):
    os.remove("jei_clean_migration.zip")

with zipfile.ZipFile("jei_clean_migration.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for f in staging.rglob("*"):
        if f.is_file():
            zf.write(f, f.relative_to(staging))

shutil.rmtree(staging)
print("jei_clean_migration.zip successfully updated.")

