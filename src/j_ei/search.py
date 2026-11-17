"""Multi-source web & scientific literature search module for Project J-EI.

Provides unified web and scientific search across:
1. DuckDuckGo (via lightweight resilient DDG Lite HTML parser — zero bot-detection / no CAPTCHAs)
2. EuropePMC REST API (peer-reviewed agronomic, botanical, and life-science papers)
3. NCBI PubMed / PMC E-Utilities (peer-reviewed biological and plant science literature)
4. Wikipedia Search API (domain terminology & definitions)
5. SearXNG JSON API (optional local aggregator fallback)
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any

from j_ei.config import JEIConfig, get_default_config

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# 1. DuckDuckGo Lite Search (Primary Web Search — No CAPTCHA / Bot Blocks)
# ---------------------------------------------------------------------------

def duckduckgo_lite_search(
    query: str,
    max_results: int = 4,
    timeout: float = 4.0,
) -> list[dict[str, str]]:
    """Search DuckDuckGo Lite HTML endpoint.
    
    Returns structured list of {title, url, body}.
    Does not trigger bot challenges or require JS/playwright.
    """
    clean_query = re.sub(r"[^\w\s\-\.]", " ", query).strip()
    if not clean_query:
        return []

    snippets: list[dict[str, str]] = []
    try:
        url = "https://lite.duckduckgo.com/lite/"
        data = urllib.parse.urlencode({"q": clean_query}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Parse via lxml.html
        from lxml.html import document_fromstring

        tree = document_fromstring(html)
        rows = tree.xpath("//table[last()]//tr")
        current: dict[str, str] | None = None

        for tr in rows:
            links = tr.xpath('.//a[contains(@class, "result-link")]')
            snips = tr.xpath('.//td[contains(@class, "result-snippet")]')

            if links:
                raw_href = links[0].get("href", "")
                # Decode DuckDuckGo redirection link if needed
                if "uddg=" in raw_href:
                    parsed_qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_href).query)
                    href = parsed_qs.get("uddg", [raw_href])[0]
                else:
                    href = raw_href

                current = {
                    "source": "DuckDuckGo",
                    "title": links[0].text_content().strip(),
                    "url": href,
                    "body": "",
                }
            elif snips and current:
                current["body"] = snips[0].text_content().strip()
                snippets.append(current)
                current = None
                if len(snippets) >= max_results:
                    break

    except Exception:
        pass

    return snippets


# ---------------------------------------------------------------------------
# 2. EuropePMC REST API (Open Agricultural & Botanical Science Literature)
# ---------------------------------------------------------------------------

def europepmc_search(
    query: str,
    max_results: int = 3,
    timeout: float = 4.0,
) -> list[dict[str, str]]:
    """Search Europe PMC REST API for peer-reviewed papers (agronomy/plant science).
    
    Free, unauthenticated, returns full paper titles and abstracts.
    """
    clean_query = re.sub(r"[^\w\s\-\.]", " ", query).strip()
    if not clean_query:
        return []

    snippets: list[dict[str, str]] = []
    try:
        url_epmc = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode({
            "query": clean_query,
            "format": "json",
            "pageSize": str(max_results),
            "resultType": "core",
        })
        req = urllib.request.Request(
            url_epmc,
            headers={"User-Agent": "Project-J-EI/2.0 (AgronomyResearch)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("resultList", {}).get("result", [])
            for item in results:
                title = item.get("title", "").rstrip(".")
                abstract = item.get("abstractText", "")
                # Clean html tags from abstract
                abstract_clean = re.sub(r"<[^>]+>", " ", abstract).strip()
                source_id = item.get("source", "MED")
                ext_id = item.get("id", "")
                url = f"https://europepmc.org/article/{source_id}/{ext_id}" if ext_id else "https://europepmc.org"
                body = abstract_clean[:400] if abstract_clean else f"Journal: {item.get('journalTitle', 'Unknown')}"

                snippets.append({
                    "source": "EuropePMC",
                    "title": title,
                    "url": url,
                    "body": body,
                })
    except Exception:
        pass

    return snippets


# ---------------------------------------------------------------------------
# 3. NCBI PubMed / PMC E-Utilities (Peer-Reviewed Biological Literature)
# ---------------------------------------------------------------------------

def pubmed_search(
    query: str,
    max_results: int = 3,
    timeout: float = 4.0,
) -> list[dict[str, str]]:
    """Search NCBI PMC / PubMed E-utilities for biological/crop science papers."""
    clean_query = re.sub(r"[^\w\s\-\.]", " ", query).strip()
    if not clean_query:
        return []

    snippets: list[dict[str, str]] = []
    try:
        # Step 1: ESearch
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode({
            "db": "pmc",
            "term": clean_query,
            "retmode": "json",
            "retmax": str(max_results),
        })
        req = urllib.request.Request(esearch_url, headers={"User-Agent": "Project-J-EI/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        id_list = data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return snippets

        # Step 2: ESummary
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode({
            "db": "pmc",
            "id": ",".join(id_list),
            "retmode": "json",
        })
        req2 = urllib.request.Request(summary_url, headers={"User-Agent": "Project-J-EI/2.0"})
        with urllib.request.urlopen(req2, timeout=timeout) as resp:
            sum_data = json.loads(resp.read().decode("utf-8"))

        for uid in id_list:
            item = sum_data.get("result", {}).get(uid, {})
            title = item.get("title", "").rstrip(".")
            journal = item.get("source", "")
            pubdate = item.get("pubdate", "")
            snippets.append({
                "source": "PubMed/PMC",
                "title": title,
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{uid}/",
                "body": f"Published in: {journal} ({pubdate})",
            })
    except Exception:
        pass

    return snippets


# ---------------------------------------------------------------------------
# 4. Wikipedia Search API (Domain Definitions & Terminology)
# ---------------------------------------------------------------------------

def wikipedia_search(
    query: str,
    max_results: int = 2,
    timeout: float = 3.0,
) -> list[dict[str, str]]:
    """Search Wikipedia API for conceptual definitions."""
    clean_query = re.sub(r"[^\w\s\-\.]", " ", query).strip()
    if not clean_query:
        return []

    snippets: list[dict[str, str]] = []
    try:
        url_wiki = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query",
            "list": "search",
            "srsearch": clean_query,
            "format": "json",
            "srlimit": str(max_results),
        })
        req = urllib.request.Request(
            url_wiki,
            headers={"User-Agent": "Project-J-EI/2.0 (Research)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("query", {}).get("search", []):
                clean_snippet = re.sub(r"<[^>]+>", " ", item.get("snippet", "")).strip()
                snippets.append({
                    "source": "Wikipedia",
                    "title": item["title"],
                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(item['title'])}",
                    "body": clean_snippet,
                })
    except Exception:
        pass

    return snippets


# ---------------------------------------------------------------------------
# 5. SearXNG Search (Optional Local Server Fallback)
# ---------------------------------------------------------------------------

def searxng_search(
    query: str,
    searxng_url: str = "http://localhost:8888",
    max_results: int = 5,
    timeout: float = 2.0,
) -> list[dict[str, str]]:
    """Search via SearXNG JSON API if locally running."""
    snippets: list[dict[str, str]] = []
    try:
        url = f"{searxng_url.rstrip('/')}/search?" + urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "categories": "general,science",
        })
        req = urllib.request.Request(url, headers={"User-Agent": "Project-J-EI/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("results", [])[:max_results]:
                snippets.append({
                    "source": "SearXNG",
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "body": item.get("content", "")[:400],
                })
    except Exception:
        pass
    return snippets


# ---------------------------------------------------------------------------
# 6. Unified Multi-Source Grounding Search
# ---------------------------------------------------------------------------

def ddg_fallback_search(
    query: str,
    timeout: float = 4.0,
) -> list[dict[str, str]]:
    """Multi-source search: DuckDuckGo Lite + EuropePMC + Wikipedia.
    
    Preserves exact backward-compatible signature for tests and engine calls.
    """
    snippets: list[dict[str, str]] = []

    # 1. Primary: DuckDuckGo Lite
    ddg_hits = duckduckgo_lite_search(query, max_results=3, timeout=timeout)
    snippets.extend(ddg_hits)

    # 2. Scientific: Europe PMC (especially powerful for agronomy, biology, chemistry)
    epmc_hits = europepmc_search(query, max_results=2, timeout=timeout)
    snippets.extend(epmc_hits)

    # 3. Terminology: Wikipedia
    wiki_hits = wikipedia_search(query, max_results=2, timeout=timeout)
    snippets.extend(wiki_hits)

    return snippets


def web_search(
    query: str,
    config: JEIConfig | None = None,
) -> list[dict[str, str]]:
    """Unified search: tries SearXNG if online, otherwise uses multi-source DDG+EuropePMC+Wiki."""
    cfg = config or get_default_config()

    # If SearXNG is running, use it
    results = searxng_search(query, searxng_url=cfg.searxng_url)
    if results:
        return results

    # Otherwise fallback to multi-source DDG Lite + EuropePMC + Wikipedia
    return ddg_fallback_search(query)
