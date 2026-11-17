"""Pass P0: Pre-Flight OCR Triage, Layout-Aware AST Parser, and Chapter Classifier.

Processes PDF and EPUB books into a unified paragraph-indexed AST format (JSONL),
evaluates OCR necessity, and classifies chapters into TAXONOMIC, NARRATIVE, or HYBRID.
Zero external API calls (100% local processing).
"""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Literal

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub
import pymupdf
import pymupdf4llm

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ChapterType = Literal["TAXONOMIC", "NARRATIVE", "HYBRID"]

STEP_PATTERN = re.compile(r"^\s*(\d+\.|Step\s+\d+|[-*])", re.IGNORECASE | re.MULTILINE)
HEADING_PATTERN = re.compile(r"^\s*#{1,6}\s+", re.MULTILINE)


def slugify(text: str) -> str:
    """Produce filesystem-safe and identifier-safe slug from filename."""
    stem = Path(text).stem if ("." in text) else text
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    return cleaned or "document"


def triage_pdf(pdf_path: Path) -> dict[str, Any]:
    """Inspect text yield per page and determine if PDF needs Vision OCR.
    
    If text_pages / max(total_pages, 1) < 0.70, status is 'NEEDS_VISION_OCR',
    otherwise 'NATIVE_TEXT'.
    """
    path = Path(pdf_path)
    book_id = slugify(path.stem)
    doc = pymupdf.open(str(path))
    total_pages = len(doc)
    text_pages = 0
    low_text_pages: list[int] = []

    for i in range(total_pages):
        page = doc[i]
        page_text = page.get_text("text").strip()
        page_num = i + 1
        if len(page_text) > 100:
            text_pages += 1
        else:
            low_text_pages.append(page_num)

    doc.close()
    ratio = text_pages / max(total_pages, 1)
    status = "NATIVE_TEXT" if ratio >= 0.70 else "NEEDS_VISION_OCR"

    return {
        "book_id": book_id,
        "format": "pdf",
        "file_path": str(path),
        "total_pages": total_pages,
        "text_pages": text_pages,
        "text_ratio": round(ratio, 4),
        "ocr_status": status,
        "low_text_pages": low_text_pages,
    }


def clean_paragraph_text(raw_text: str) -> str:
    """Clean paragraph text: repair line-break hyphens and collapse internal whitespace."""
    # Repair hyphenated word breaks (e.g., 'inves- \n tigation' -> 'investigation')
    repaired = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", raw_text)
    # Strip HTML tags inserted by converters
    repaired = re.sub(r"<[^>]+>", "", repaired)
    # Collapse whitespace / internal newlines
    lines = [line.strip() for line in repaired.splitlines() if line.strip()]
    return " ".join(lines)


def classify_chapter(paragraphs: list[dict[str, Any]]) -> ChapterType:
    """Classify chapter type based on heading/step density vs dialogue density."""
    if not paragraphs:
        return "HYBRID"

    total_tokens = sum(p.get("est_tokens", 0) for p in paragraphs)
    if total_tokens <= 0:
        total_tokens = max(sum(len(p.get("text", "")) for p in paragraphs) / 3.8, 1.0)
    if total_tokens < 30:
        return "HYBRID"

    scale = 1000.0 / total_tokens

    step_count = sum(len(STEP_PATTERN.findall(p.get("text", ""))) for p in paragraphs)
    heading_count = sum(len(HEADING_PATTERN.findall(p.get("text", ""))) for p in paragraphs)
    quote_count = sum(
        p.get("text", "").count('"')
        + p.get("text", "").count("“")
        + p.get("text", "").count("”")
        for p in paragraphs
    )

    step_density = step_count * scale
    heading_density = heading_count * scale
    quote_density = quote_count * scale
    structure_density = step_density + heading_density

    # High step/heading density with low dialogue -> TAXONOMIC
    if structure_density >= 2.0 and quote_density < 3.0:
        return "TAXONOMIC"
    # High dialogue/story arc with low step density -> NARRATIVE
    elif quote_density >= 3.5 and step_density < 1.0:
        return "NARRATIVE"
    else:
        return "HYBRID"


def parse_native_pdf(pdf_path: Path, book_id: str | None = None) -> list[dict[str, Any]]:
    """Extract AST paragraphs from native text PDF using PyMuPDF and Markdown segmentation."""
    path = Path(pdf_path)
    b_id = book_id or slugify(path.stem)
    doc = pymupdf.open(str(path))
    total_pages = len(doc)

    # Disable heavy AI layout model / OCR to run fast deterministic extraction
    pymupdf4llm.use_layout(False)
    chunks = pymupdf4llm.to_markdown(
        doc,
        page_chunks=True,
        margins=0,
        ignore_images=True,
    )
    doc.close()

    # Pre-scan for Markdown chapter headings (# or ##)
    heading_candidates: list[tuple[int, int, str]] = []  # (chunk_idx, level, title)
    for c_idx, chunk in enumerate(chunks):
        for line in chunk.get("text", "").splitlines():
            line_str = line.strip()
            m = re.match(r"^(#{1,2})\s+(.+)$", line_str)
            if m:
                level = len(m.group(1))
                title = clean_paragraph_text(m.group(2))
                if len(title) >= 3:
                    heading_candidates.append((c_idx, level, title))

    # Determine chapter mode: heading-based if >= 2 headings, else fixed 25-page blocks
    use_fixed_page_blocks = len(heading_candidates) < 2

    paragraphs: list[dict[str, Any]] = []
    pid_counter = 1
    char_offset = 0

    chapter_idx = 1
    current_chapter_id = "ch_01"
    current_chapter_title = (
        f"Pages 1-{min(25, total_pages)}" if use_fixed_page_blocks else "Introduction"
    )
    heading_stack: list[str] = [current_chapter_title]

    for c_idx, chunk in enumerate(chunks):
        page_num = chunk.get("metadata", {}).get("page_number", c_idx + 1)
        raw_text = chunk.get("text", "")

        # Fixed 25-page block mode
        if use_fixed_page_blocks:
            target_ch_idx = (page_num - 1) // 25 + 1
            if target_ch_idx != chapter_idx:
                chapter_idx = target_ch_idx
                current_chapter_id = f"ch_{chapter_idx:02d}"
                start_p = (chapter_idx - 1) * 25 + 1
                end_p = min(chapter_idx * 25, total_pages)
                current_chapter_title = f"Pages {start_p}-{end_p}"
                heading_stack = [current_chapter_title]

        # Split page text into block elements
        raw_blocks = re.split(r"\n\s*\n", raw_text)
        for block in raw_blocks:
            cleaned = clean_paragraph_text(block)
            if len(cleaned) < 5:
                continue

            # Check if block is a heading
            h_match = re.match(r"^(#{1,6})\s+(.+)$", cleaned)
            if h_match:
                h_level = len(h_match.group(1))
                h_title = clean_paragraph_text(h_match.group(2))

                # If in heading mode and level is # or ##, evaluate chapter transition
                if not use_fixed_page_blocks and h_level <= 2:
                    if paragraphs:
                        chapter_idx += 1
                        current_chapter_id = f"ch_{chapter_idx:02d}"
                    current_chapter_title = h_title

                # Update heading stack
                if h_level == 1:
                    heading_stack = [h_title]
                elif h_level == 2:
                    heading_stack = heading_stack[:1] + [h_title]
                else:
                    heading_stack = heading_stack[: h_level - 1] + [h_title]

            heading_path = " > ".join(heading_stack) if heading_stack else current_chapter_title
            sha256 = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            est_tokens = int(len(cleaned) / 3.8)
            end_offset = char_offset + len(cleaned)

            p_entry = {
                "pid": f"[P{pid_counter:04d}]",
                "book_id": b_id,
                "chapter_id": current_chapter_id,
                "chapter_title": current_chapter_title,
                "heading_path": heading_path,
                "locator": f"page_{page_num}",
                "char_span": [char_offset, end_offset],
                "source_span_sha256": sha256,
                "est_tokens": est_tokens,
                "text": cleaned,
            }
            paragraphs.append(p_entry)
            char_offset = end_offset + 1
            pid_counter += 1

    return paragraphs


def parse_epub(epub_path: Path, book_id: str | None = None) -> list[dict[str, Any]]:
    """Extract AST paragraphs from EPUB using ebooklib and BeautifulSoup."""
    path = Path(epub_path)
    b_id = book_id or slugify(path.stem)
    book = epub.read_epub(str(path), options={"ignore_ncx": True})
    spine_ids = [item[0] for item in book.spine]

    paragraphs: list[dict[str, Any]] = []
    pid_counter = 1
    char_offset = 0
    chapter_idx = 0

    for idx, item_id in enumerate(spine_ids):
        item = book.get_item_with_id(item_id)
        if not item or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            soup = BeautifulSoup(item.get_content(), "lxml")
        full_text = soup.get_text(separator=" ", strip=True)

        # Skip empty/cover/copyright front-matter items (< 150 chars)
        if len(full_text) < 150:
            continue

        chapter_idx += 1
        current_chapter_id = f"ch_{chapter_idx:02d}"

        # Extract title from first heading if present
        first_h = soup.find(["h1", "h2", "h3"])
        current_chapter_title = (
            clean_paragraph_text(first_h.get_text())
            if first_h
            else f"Section {chapter_idx}"
        )
        heading_stack: list[str] = [current_chapter_title]

        # Extract paragraph and heading blocks
        blocks = soup.find_all(
            ["p", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "li"]
        )
        if not blocks:
            raw_blocks = [full_text]
        else:
            raw_blocks = [b.get_text(separator=" ", strip=True) for b in blocks]

        for block in raw_blocks:
            cleaned = clean_paragraph_text(block)
            if len(cleaned) < 10:
                continue

            heading_path = " > ".join(heading_stack)
            sha256 = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            est_tokens = int(len(cleaned) / 3.8)
            end_offset = char_offset + len(cleaned)

            p_entry = {
                "pid": f"[P{pid_counter:04d}]",
                "book_id": b_id,
                "chapter_id": current_chapter_id,
                "chapter_title": current_chapter_title,
                "heading_path": heading_path,
                "locator": f"spine_{idx}",
                "char_span": [char_offset, end_offset],
                "source_span_sha256": sha256,
                "est_tokens": est_tokens,
                "text": cleaned,
            }
            paragraphs.append(p_entry)
            char_offset = end_offset + 1
            pid_counter += 1

    return paragraphs


def process_books_directory(
    books_dir: Path = Path("books"),
    output_dir: Path = Path("data/ast"),
) -> dict[str, Any]:
    """Scan books_dir, process all PDF/EPUB books, write JSONL ASTs, and produce triage report."""
    books_path = Path(books_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_items: list[dict[str, Any]] = []

    files = sorted(list(books_path.glob("*.pdf")) + list(books_path.glob("*.epub")))
    for file_path in files:
        book_id = slugify(file_path.stem)
        fmt = file_path.suffix.lstrip(".").lower()

        if fmt == "pdf":
            triage = triage_pdf(file_path)
            ocr_status = triage["ocr_status"]
            total_pages_or_spines = triage["total_pages"]

            if ocr_status == "NATIVE_TEXT":
                paragraphs = parse_native_pdf(file_path, book_id=book_id)
            else:
                paragraphs = []
        elif fmt == "epub":
            ocr_status = "NATIVE_TEXT"
            paragraphs = parse_epub(file_path, book_id=book_id)
            # Count total spine items
            book_epub = epub.read_epub(str(file_path), options={"ignore_ncx": True})
            total_pages_or_spines = len(book_epub.spine)
        else:
            continue

        # Classify chapters
        chapters_dict: dict[str, list[dict[str, Any]]] = {}
        for p in paragraphs:
            chapters_dict.setdefault(p["chapter_id"], []).append(p)

        breakdown = {"TAXONOMIC": 0, "NARRATIVE": 0, "HYBRID": 0}
        for ch_id, ch_paras in chapters_dict.items():
            ch_type = classify_chapter(ch_paras)
            breakdown[ch_type] += 1

        total_est_tokens = sum(p.get("est_tokens", 0) for p in paragraphs)

        # Write AST JSONL if paragraphs were extracted
        if paragraphs:
            ast_file = out_path / f"{book_id}.jsonl"
            with open(ast_file, "w", encoding="utf-8") as f:
                for p in paragraphs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

        item = {
            "book_id": book_id,
            "filename": file_path.name,
            "format": fmt,
            "ocr_status": ocr_status,
            "total_pages_or_spines": total_pages_or_spines,
            "paragraph_count": len(paragraphs),
            "chapter_count": len(chapters_dict),
            "total_est_tokens": total_est_tokens,
            "chapter_breakdown": breakdown,
        }
        report_items.append(item)

    report_path = out_path / "triage_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_items, f, indent=2)

    return {"books": report_items, "report_path": str(report_path)}


def print_triage_table(report_items: list[dict[str, Any]]) -> None:
    """Print clean formatted triage summary table."""
    header = (
        f"{'Book ID':<35} | {'Fmt':<4} | {'OCR Status':<15} | "
        f"{'Pages/Sp':<8} | {'Chaps':<5} | {'Tokens':<9} | {'Tax / Nar / Hyb'}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print(" PROJECT J-EI — PASS P0 PRE-FLIGHT TRIAGE & AST REPORT")
    print(separator)
    print(header)
    print(separator)

    total_tokens = 0
    total_paras = 0

    for item in report_items:
        b_id = item["book_id"][:35]
        fmt = item["format"].upper()
        status = item["ocr_status"]
        pages = item["total_pages_or_spines"]
        chaps = item["chapter_count"]
        tokens = item["total_est_tokens"]
        bd = item["chapter_breakdown"]
        bd_str = f"{bd['TAXONOMIC']} / {bd['NARRATIVE']} / {bd['HYBRID']}"

        total_tokens += tokens
        total_paras += item["paragraph_count"]

        print(
            f"{b_id:<35} | {fmt:<4} | {status:<15} | "
            f"{pages:<8} | {chaps:<5} | {tokens:<9,d} | {bd_str}"
        )

    print(separator)
    print(f"Total Books: {len(report_items)} | Total Paragraphs: {total_paras:,d} | Total Tokens: {total_tokens:,d}")
    print(separator + "\n")


def main() -> None:
    """CLI entrypoint for running Pass P0 AST parser and OCR triage."""
    result = process_books_directory()
    print_triage_table(result["books"])


if __name__ == "__main__":
    main()
