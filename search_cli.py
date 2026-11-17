#!/usr/bin/env python3
"""Standalone search CLI for Project J-EI.
Usage: python3 search_cli.py "<query>"
Returns JSON array of {title, url, body} search snippets.
"""
import json
import sys
import urllib.parse
import urllib.request

SEARXNG_URL = "http://localhost:8888"

def searxng_search(query: str) -> list[dict]:
    snippets = []
    try:
        url = SEARXNG_URL.rstrip("/") + "/search?" + urllib.parse.urlencode({
            "q": query, "format": "json", "categories": "general,science",
        })
        req = urllib.request.Request(url, headers={"User-Agent": "J-EI/2.0"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("results", [])[:5]:
                snippets.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "body": item.get("content", "")[:400],
                })
    except Exception:
        pass
    return snippets

def ddg_fallback(query: str) -> list[dict]:
    snippets = []
    try:
        url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
            "q": query, "format": "json", "no_html": 1, "skip_disambig": 1,
        })
        req = urllib.request.Request(url, headers={"User-Agent": "J-EI/2.0"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            abstract = data.get("AbstractText", "")
            if abstract:
                snippets.append({
                    "title": data.get("Heading", query),
                    "url": data.get("AbstractURL", ""),
                    "body": abstract[:400],
                })
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    snippets.append({
                        "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                        "url": topic.get("FirstURL", ""),
                        "body": topic["Text"][:300],
                    })
    except Exception:
        pass
    try:
        url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": 3,
        })
        req = urllib.request.Request(url, headers={"User-Agent": "J-EI/2.0 (Research)"})
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            for item in data.get("query", {}).get("search", []):
                clean = item["snippet"].replace('<span class="searchmatch">', "").replace("</span>", "")
                snippets.append({
                    "title": item["title"],
                    "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(item['title'])}",
                    "body": clean,
                })
    except Exception:
        pass
    return snippets

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 search_cli.py \"<query>\"", file=sys.stderr)
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    results = searxng_search(query)
    if not results:
        results = ddg_fallback(query)
    print(json.dumps(results, indent=2))
