"""Test suite for Step 2A: Pre-Flight OCR Triage, Layout-Aware AST Parser, and Chapter Classifier."""

from __future__ import annotations

import hashlib
from pathlib import Path
from ebooklib import epub
import pymupdf
import pytest

from j_ei.parser import (
    classify_chapter,
    clean_paragraph_text,
    parse_epub,
    parse_native_pdf,
    slugify,
    triage_pdf,
)


@pytest.fixture
def sample_native_pdf(tmp_path: Path) -> Path:
    """Create a synthetic 2-page native text PDF with headers and steps."""
    pdf_path = tmp_path / "sample_native.pdf"
    doc = pymupdf.open()

    # Page 1: Taxonomic section
    p1 = doc.new_page()
    text_p1 = (
        "# Chapter 1: Forensic Triage Architecture\n\n"
        "This foundational chapter specifies the deterministic triage procedure for all incoming streams.\n\n"
        "1. Identify the focal anomaly and isolate boundary telemetry.\n"
        "2. Formulate at least two mutually exclusive rival hypotheses.\n"
        "3. Audit evidence diagnosticity against each hypothesis matrix.\n"
        "4. Calculate information-theoretic entropy delta before dispatching workers.\n"
    )
    p1.insert_text((50, 50), text_p1)

    # Page 2: Narrative case study
    p2 = doc.new_page()
    text_p2 = (
        "## Chapter 2: Incident Retrospective\n\n"
        '"We did not expect the network failover to cascade into distributed storage," stated the commander.\n\n'
        '"The secondary link was supposed to buffer the incoming write bursts," answered the site lead.\n\n'
        '"Why was the heartbeat monitor completely silent?" asked the director. "It should have alerted us immediately."\n\n'
        '"The alert threshold was suppressed by the maintenance window flag," replied the operator.\n'
    )
    p2.insert_text((50, 50), text_p2)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_scanned_pdf(tmp_path: Path) -> Path:
    """Create a synthetic 3-page low-text PDF simulating a scanned document needing OCR."""
    pdf_path = tmp_path / "sample_scanned.pdf"
    doc = pymupdf.open()
    for i in range(3):
        p = doc.new_page()
        p.insert_text((50, 50), f"p. {i + 1}")  # < 100 characters per page
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_epub(tmp_path: Path) -> Path:
    """Create a synthetic EPUB with taxonomic and narrative chapters."""
    epub_path = tmp_path / "sample_book.epub"
    book = epub.EpubBook()
    book.set_identifier("sample-001")
    book.set_title("Cognitive Systems Analysis")
    book.set_language("en")

    # Chapter 1 (Taxonomic)
    c1 = epub.EpubHtml(title="Systems Triage", file_name="ch1.xhtml", lang="en")
    c1.content = (
        "<h1>Systems Triage Guidelines</h1>"
        "<p>This methodology document defines the structured diagnostics for root-cause isolation.</p>"
        "<p>1. Formulate rival explanation branches across all active entities.</p>"
        "<p>2. Quantify diagnostic weight for each observable symptom.</p>"
        "<p>3. Prune low-diagnosticity noise observations prior to synthesis.</p>"
        "<p>4. Execute verification checks across all independent witness records.</p>"
    )
    book.add_item(c1)

    # Chapter 2 (Narrative)
    c2 = epub.EpubHtml(title="Case Study Alpha", file_name="ch2.xhtml", lang="en")
    c2.content = (
        "<h1>Case Study Alpha Analysis</h1>"
        '<p>"What caused the sudden divergence in cluster telemetry?" asked the investigator.</p>'
        '<p>"The replication thread encountered an unexpected deadlock," explained the engineer.</p>'
        '<p>"Did you verify the transaction logs before restarting?" asked the security lead.</p>'
        '<p>"No, the automated watchdog restarted the process immediately," admitted the operator.</p>'
    )
    book.add_item(c2)

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", c1, c2]
    epub.write_epub(str(epub_path), book)
    return epub_path


# =====================================================================
# Unit Tests
# =====================================================================

def test_slugify() -> None:
    assert slugify("CrimeClassificationManual.pdf") == "crimeclassificationmanual"
    assert slugify("Inside The Criminal Mind PDF.pdf") == "inside_the_criminal_mind_pdf"
    assert slugify("The Gift of Fear (Gavin de Becker).epub") == "the_gift_of_fear_gavin_de_becker"


def test_clean_paragraph_text() -> None:
    raw = "Inves- \n tigation of the sys- \n temic anomaly with <mark>highlight</mark>."
    cleaned = clean_paragraph_text(raw)
    assert "Investigation" in cleaned
    assert "systemic" in cleaned
    assert "<mark>" not in cleaned
    assert "\n" not in cleaned


def test_triage_pdf_native_vs_scanned(
    sample_native_pdf: Path,
    sample_scanned_pdf: Path,
) -> None:
    native_report = triage_pdf(sample_native_pdf)
    assert native_report["ocr_status"] == "NATIVE_TEXT"
    assert native_report["total_pages"] == 2
    assert native_report["text_ratio"] >= 0.70
    assert native_report["low_text_pages"] == []

    scanned_report = triage_pdf(sample_scanned_pdf)
    assert scanned_report["ocr_status"] == "NEEDS_VISION_OCR"
    assert scanned_report["total_pages"] == 3
    assert scanned_report["text_ratio"] == 0.0
    assert scanned_report["low_text_pages"] == [1, 2, 3]


def test_parse_native_pdf_ast_schema(sample_native_pdf: Path) -> None:
    paragraphs = parse_native_pdf(sample_native_pdf, book_id="sample_native")
    assert len(paragraphs) >= 2

    for idx, p in enumerate(paragraphs):
        expected_pid = f"[P{idx + 1:04d}]"
        assert p["pid"] == expected_pid
        assert p["book_id"] == "sample_native"
        assert p["chapter_id"].startswith("ch_")
        assert len(p["chapter_title"]) > 0
        assert len(p["heading_path"]) > 0
        assert p["locator"].startswith("page_")
        assert len(p["char_span"]) == 2
        assert p["char_span"][0] < p["char_span"][1]

        # Verify SHA-256 integrity
        expected_sha = hashlib.sha256(p["text"].encode("utf-8")).hexdigest()
        assert p["source_span_sha256"] == expected_sha
        assert p["est_tokens"] == int(len(p["text"]) / 3.8)


def test_parse_epub_ast_schema(sample_epub: Path) -> None:
    paragraphs = parse_epub(sample_epub, book_id="sample_epub")
    assert len(paragraphs) >= 2

    for idx, p in enumerate(paragraphs):
        expected_pid = f"[P{idx + 1:04d}]"
        assert p["pid"] == expected_pid
        assert p["book_id"] == "sample_epub"
        assert p["chapter_id"].startswith("ch_")
        assert p["locator"].startswith("spine_")

        expected_sha = hashlib.sha256(p["text"].encode("utf-8")).hexdigest()
        assert p["source_span_sha256"] == expected_sha


def test_classify_chapter_taxonomic_vs_narrative() -> None:
    taxonomic_paras = [
        {
            "text": (
                "1. Isolate anomalous telemetry packet headers.\n"
                "2. Construct pairwise discrimination matrix.\n"
                "3. Verify hardware time synchronization counters.\n"
                "4. Check boundary conditions against specification."
            ),
            "est_tokens": 80,
        },
        {
            "text": "Step 5: Apply Bayesian likelihood updates to rival hypothesis branches.",
            "est_tokens": 40,
        },
    ]
    assert classify_chapter(taxonomic_paras) == "TAXONOMIC"

    narrative_paras = [
        {
            "text": (
                '"We arrived at the datacenter at midnight," recalled the engineer. '
                '"The cooling units had failed completely," he said.'
            ),
            "est_tokens": 60,
        },
        {
            "text": (
                '"Did you check the secondary generator?" asked the director. '
                '"Yes, but the fuel valve was shut," replied the supervisor.'
            ),
            "est_tokens": 60,
        },
        {
            "text": (
                '"Why was nobody notified?" shouted the vice president. '
                '"The alarm network was down as well," whispered the technician.'
            ),
            "est_tokens": 60,
        },
    ]
    assert classify_chapter(narrative_paras) == "NARRATIVE"

    hybrid_paras = [
        {
            "text": (
                'In our analysis of the incident, we asked: "What caused the initial trigger?"\n'
                '"The memory leak caused catastrophic exhaustion," stated the architect.\n'
                "The remediation procedure requires three sequential steps:\n"
                "1. Database schema migration executed without index lock.\n"
                "2. Client connections accumulated in backpressure queue.\n"
                "3. Service workers restarted under graceful drain mode."
            ),
            "est_tokens": 120,
        }
    ]
    assert classify_chapter(hybrid_paras) == "HYBRID"
