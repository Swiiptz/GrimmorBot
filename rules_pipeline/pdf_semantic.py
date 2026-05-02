from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from rules_pipeline.models import DocumentSection, ParsedDocument, TocChapter

CHAPTER_RE = re.compile(r"^Chapitre\s+[IVXLC]+\s*:\s*.+$", re.IGNORECASE)
PAGE_COUNTER_RE = re.compile(r"^\d+\s*/\s*1\.\d+$")
MAX_SECTION_CHARS = 7000
OVERLAP_CHARS = 500

HEADING_HINTS = {
    "introduction",
    "systeme",
    "système",
    "score de test",
    "focus",
    "succes exceptionnel",
    "succès exceptionnel",
    "faq",
    "connaissances communes",
    "connaissances secretes",
    "connaissances secrètes",
}


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"\s+", " ", normalized.lower())
    return normalized.strip(" :\t-")


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _normalize(text))
    return slug.strip("-") or "section"


def _clean_page_text(text: str) -> str:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        norm = _normalize(line)
        if norm in {
            "federation camarilla francaise",
            "document confidentiel - diffusion interdite en dehors de la federation sans authorisation du bureau",
            "sommaire",
        }:
            continue
        if line == "[sommaire]" or PAGE_COUNTER_RE.match(line):
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _parse_toc(reader: PdfReader) -> list[TocChapter]:
    toc_lines: list[str] = []
    for page_num in (2, 3):
        text = _clean_page_text(reader.pages[page_num - 1].extract_text() or "")
        toc_lines.extend(line.strip() for line in text.splitlines() if line.strip())

    chapters: list[TocChapter] = []
    current: TocChapter | None = None
    for line in toc_lines:
        if line.upper() == "SOMMAIRE":
            continue
        if CHAPTER_RE.match(line) or line in {"Annexes", "FAQ (Questions Frequemment Posees)", "FAQ (Questions Fréquemment Posées)"}:
            current = TocChapter(title=line)
            chapters.append(current)
            continue
        if current is not None:
            current.subsections.append(line)
    return chapters or [TocChapter(title="Document")]


def _find_chapter_boundaries(reader: PdfReader, chapters: list[TocChapter]) -> None:
    first_hits: dict[str, int] = {}
    candidates = {
        chapter.title: {_normalize(chapter.title), *(_normalize(s) for s in chapter.subsections[:8])}
        for chapter in chapters
    }

    for page_num in range(4, len(reader.pages) + 1):
        text = _clean_page_text(reader.pages[page_num - 1].extract_text() or "")
        page_lines = {_normalize(line) for line in text.splitlines() if line.strip()}
        for chapter in chapters:
            if chapter.title in first_hits:
                continue
            if candidates[chapter.title] & page_lines:
                first_hits[chapter.title] = page_num

    last_start = 4
    for index, chapter in enumerate(chapters):
        chapter.start_page = first_hits.get(chapter.title, 4 if index == 0 else last_start)
        last_start = chapter.start_page

    for index, chapter in enumerate(chapters):
        next_start = chapters[index + 1].start_page if index + 1 < len(chapters) else len(reader.pages) + 1
        chapter.end_page = max(chapter.start_page, next_start - 1)


def _looks_like_heading(line: str) -> bool:
    if not line or len(line) > 90:
        return False
    norm = _normalize(line)
    if norm in HEADING_HINTS or norm.startswith("focus "):
        return True
    if line.startswith("•"):
        return True
    if norm.startswith(("conditions prealables", "prerequis")):
        return True
    if ":" in line or line.endswith((".", "!", "?", ";")):
        return False
    words = line.split()
    if not 1 <= len(words) <= 8:
        return False
    lower_words = sum(1 for word in words if word[:1].islower())
    return lower_words <= max(1, len(words) // 3)


def _split_large(section: DocumentSection) -> list[DocumentSection]:
    if len(section.text) <= MAX_SECTION_CHARS:
        return [section]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
    results: list[DocumentSection] = []
    current: list[str] = []
    part = 1
    for paragraph in paragraphs:
        candidate = "\n\n".join([*current, paragraph])
        if current and len(candidate) > MAX_SECTION_CHARS:
            text = "\n\n".join(current).strip()
            results.append(
                DocumentSection(
                    section_id=f"{section.section_id}-part-{part}",
                    chapter=section.chapter,
                    subsection=section.subsection,
                    heading=section.heading,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    order=section.order * 100 + part,
                    text=text,
                    headings_path=section.headings_path,
                )
            )
            tail = text[-OVERLAP_CHARS:].strip()
            current = [tail, paragraph] if tail else [paragraph]
            part += 1
        else:
            current.append(paragraph)
    if current:
        results.append(
            DocumentSection(
                section_id=f"{section.section_id}-part-{part}",
                chapter=section.chapter,
                subsection=section.subsection,
                heading=section.heading,
                page_start=section.page_start,
                page_end=section.page_end,
                order=section.order * 100 + part,
                text="\n\n".join(current).strip(),
                headings_path=section.headings_path,
            )
        )
    return results


def _build_chapter_sections(reader: PdfReader, chapter: TocChapter, order_offset: int) -> list[DocumentSection]:
    sections: list[DocumentSection] = []
    subsection_lookup = {_normalize(value): value for value in chapter.subsections}
    current_subsection = ""
    current_heading = chapter.title
    start_page = chapter.start_page
    end_page = chapter.start_page
    buffer: list[str] = []
    local_order = 0

    def flush() -> None:
        nonlocal buffer, local_order, start_page
        text = "\n\n".join(part.strip() for part in buffer if part.strip()).strip()
        if not text:
            buffer = []
            start_page = end_page
            return
        local_order += 1
        section_id = f"{order_offset + local_order:04d}-{_slugify(current_heading)}"
        section = DocumentSection(
            section_id=section_id,
            chapter=chapter.title,
            subsection=current_subsection,
            heading=current_heading,
            page_start=start_page,
            page_end=end_page,
            order=order_offset + local_order,
            text=text,
            headings_path=[chapter.title, current_subsection, current_heading],
        )
        sections.extend(_split_large(section))
        buffer = []
        start_page = end_page

    for page_num in range(chapter.start_page, chapter.end_page + 1):
        text = _clean_page_text(reader.pages[page_num - 1].extract_text() or "")
        for line in (raw.strip() for raw in text.splitlines() if raw.strip()):
            norm = _normalize(line)
            if CHAPTER_RE.match(line) or norm == _normalize(chapter.title):
                continue
            if norm in subsection_lookup:
                flush()
                current_subsection = subsection_lookup[norm]
                current_heading = current_subsection
                start_page = page_num
                end_page = page_num
                continue
            if _looks_like_heading(line):
                flush()
                current_heading = line
                start_page = page_num
                end_page = page_num
                continue
            buffer.append(line)
            end_page = page_num
        buffer.append("")
    flush()
    return sections


def parse_pdf_semantically(pdf_path: Path) -> ParsedDocument:
    reader = PdfReader(str(pdf_path))
    chapters = _parse_toc(reader)
    _find_chapter_boundaries(reader, chapters)

    sections: list[DocumentSection] = []
    order_offset = 0
    for chapter in chapters:
        chapter_sections = _build_chapter_sections(reader, chapter, order_offset)
        sections.extend(chapter_sections)
        order_offset += len(chapter_sections)

    return ParsedDocument(
        title=pdf_path.stem,
        source_path=str(pdf_path),
        total_pages=len(reader.pages),
        chapters=chapters,
        sections=sections,
    )


def group_sections_for_llm(
    sections: list[DocumentSection],
    *,
    target_chars: int = 18000,
    max_chars: int = 26000,
) -> list[DocumentSection]:
    """Merge small parser sections into larger LLM batches.

    The parser keeps tiny headings because they are useful anchors, but LLM
    extraction works better when a whole rule cluster stays together.
    """
    batches: list[DocumentSection] = []
    current: list[DocumentSection] = []
    current_chars = 0
    batch_index = 0

    def same_context(a: DocumentSection, b: DocumentSection) -> bool:
        return a.chapter == b.chapter and a.subsection == b.subsection

    def render_piece(section: DocumentSection) -> str:
        return (
            f"[section_id={section.section_id}; pages={section.page_start}-{section.page_end}]\n"
            f"## {section.heading}\n"
            f"{section.text.strip()}"
        ).strip()

    def flush() -> None:
        nonlocal current, current_chars, batch_index
        if not current:
            return
        batch_index += 1
        first = current[0]
        last = current[-1]
        text = "\n\n".join(render_piece(section) for section in current)
        heading = first.subsection or first.chapter
        batches.append(
            DocumentSection(
                section_id=f"batch-{batch_index:04d}-{_slugify(heading)}",
                chapter=first.chapter,
                subsection=first.subsection,
                heading=heading,
                page_start=first.page_start,
                page_end=last.page_end,
                order=batch_index,
                text=text,
                headings_path=[first.chapter, first.subsection, heading],
            )
        )
        current = []
        current_chars = 0

    for section in sections:
        piece_len = len(render_piece(section))
        if current and not same_context(current[-1], section):
            flush()
        elif current and current_chars >= target_chars and current_chars + piece_len > max_chars:
            flush()

        current.append(section)
        current_chars += piece_len

        if current_chars > max_chars:
            flush()

    flush()
    return batches
