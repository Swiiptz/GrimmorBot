"""Découpage sémantique des documents (Markdown, TXT, PDF) en chunks RAG."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

# Approximation tokens. Pour MiniLM ~ 4 chars/token en moyenne FR.
CHARS_PER_TOKEN = 4
MAX_TOKENS_MD = 800
MIN_TOKENS = 50
OVERLAP_TOKENS = 80
MAX_TOKENS_PDF = 600


@dataclass
class Chunk:
    text: str
    source: str
    page: int = 0
    section: str = ""
    type: str = "rule"
    chunk_index: int = 0
    extra: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "source": self.source,
            "page": int(self.page),
            "section": self.section,
            "type": self.type,
            "chunk_index": int(self.chunk_index),
        }


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Sous-découpe un texte trop long en respectant les paragraphes.

    On agrège des paragraphes (séparés par lignes vides) jusqu'à atteindre
    max_tokens. Si un seul paragraphe dépasse, on coupe en phrases.
    L'overlap se fait sur les derniers tokens du chunk précédent.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if _approx_tokens(p) <= max_tokens:
            pieces.append(p)
        else:
            # Coupe en phrases
            sentences = re.split(r"(?<=[.!?])\s+", p)
            buffer: list[str] = []
            buf_tok = 0
            for s in sentences:
                tok = _approx_tokens(s)
                if buf_tok + tok > max_tokens and buffer:
                    pieces.append(" ".join(buffer))
                    buffer, buf_tok = [], 0
                buffer.append(s)
                buf_tok += tok
            if buffer:
                pieces.append(" ".join(buffer))

    chunks: list[str] = []
    current: list[str] = []
    cur_tok = 0
    for piece in pieces:
        tok = _approx_tokens(piece)
        if cur_tok + tok > max_tokens and current:
            chunks.append("\n\n".join(current))
            # Overlap : repartir sur la fin du chunk précédent
            tail = chunks[-1][-overlap_tokens * CHARS_PER_TOKEN :]
            current = [tail.strip()] if tail.strip() else []
            cur_tok = _approx_tokens(tail) if tail.strip() else 0
        current.append(piece)
        cur_tok += tok
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _merge_short(chunks: list[Chunk]) -> list[Chunk]:
    """Fusionne les chunks trop courts avec leur voisin suivant."""
    if not chunks:
        return chunks
    merged: list[Chunk] = []
    i = 0
    while i < len(chunks):
        c = chunks[i]
        if _approx_tokens(c.text) < MIN_TOKENS and i + 1 < len(chunks):
            nxt = chunks[i + 1]
            combined_section = c.section if c.section == nxt.section else (c.section or nxt.section)
            merged.append(
                Chunk(
                    text=c.text + "\n\n" + nxt.text,
                    source=c.source,
                    page=c.page or nxt.page,
                    section=combined_section,
                    type=c.type,
                    chunk_index=c.chunk_index,
                )
            )
            i += 2
        else:
            merged.append(c)
            i += 1
    return merged


def _detect_type(text: str, source_path: Path) -> str:
    parts = {p.lower() for p in source_path.parts}
    if "lore" in parts:
        return "lore"
    if re.search(r"\bex(?:emple)?\s*[:\.]|par exemple", text, re.IGNORECASE):
        return "example"
    # Tableau Markdown ou colonnes alignées de chiffres
    pipe_lines = sum(1 for line in text.splitlines() if line.count("|") >= 2)
    digit_lines = sum(
        1 for line in text.splitlines() if re.search(r"^\s*\d", line)
    )
    if pipe_lines >= 2 or (digit_lines >= 3 and len(text.splitlines()) >= 4):
        return "table"
    return "rule"


# ─── Markdown / TXT ───────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def chunk_markdown(text: str, source_path: Path) -> list[Chunk]:
    """Découpe un fichier MD/TXT par sections (titres `#`/`##`/...)."""
    headings = list(_HEADING_RE.finditer(text))

    sections: list[tuple[str, str]] = []  # (titre, contenu)
    if not headings:
        sections.append(("", text.strip()))
    else:
        # Préambule avant le 1er titre
        if headings[0].start() > 0:
            preamble = text[: headings[0].start()].strip()
            if preamble:
                sections.append(("", preamble))
        for i, m in enumerate(headings):
            title = m.group(2).strip()
            start = m.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            content = text[start:end].strip()
            if content:
                sections.append((title, content))

    raw_chunks: list[Chunk] = []
    for section_title, content in sections:
        if _approx_tokens(content) > MAX_TOKENS_MD:
            for sub in _split_long_text(content, MAX_TOKENS_MD, OVERLAP_TOKENS):
                raw_chunks.append(
                    Chunk(
                        text=sub.strip(),
                        source=source_path.name,
                        page=0,
                        section=section_title,
                    )
                )
        else:
            raw_chunks.append(
                Chunk(
                    text=content,
                    source=source_path.name,
                    page=0,
                    section=section_title,
                )
            )

    merged = _merge_short(raw_chunks)
    for i, c in enumerate(merged):
        c.chunk_index = i
        c.type = _detect_type(c.text, source_path)
    return merged


# ─── PDF ──────────────────────────────────────────────────────────────────────


def chunk_pdf(path: Path) -> list[Chunk]:
    reader = PdfReader(str(path))
    raw_chunks: list[Chunk] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        page_text = page_text.strip()
        if not page_text:
            continue

        blocks = [b.strip() for b in re.split(r"\n\s*\n", page_text) if b.strip()]
        buffer: list[str] = []
        buf_tok = 0
        for block in blocks:
            tok = _approx_tokens(block)
            if buf_tok + tok > MAX_TOKENS_PDF and buffer:
                raw_chunks.append(
                    Chunk(
                        text="\n\n".join(buffer),
                        source=path.name,
                        page=page_num,
                        section="",
                    )
                )
                buffer, buf_tok = [], 0
            if tok > MAX_TOKENS_PDF:
                # Bloc unique trop long : sous-découpe
                for sub in _split_long_text(block, MAX_TOKENS_PDF, OVERLAP_TOKENS):
                    raw_chunks.append(
                        Chunk(
                            text=sub,
                            source=path.name,
                            page=page_num,
                            section="",
                        )
                    )
            else:
                buffer.append(block)
                buf_tok += tok
        if buffer:
            raw_chunks.append(
                Chunk(
                    text="\n\n".join(buffer),
                    source=path.name,
                    page=page_num,
                    section="",
                )
            )

    merged = _merge_short(raw_chunks)
    for i, c in enumerate(merged):
        c.chunk_index = i
        c.type = _detect_type(c.text, path)
    return merged


# ─── Dispatcher ───────────────────────────────────────────────────────────────


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


def chunk_file(path: Path) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return chunk_markdown(text, path)
    if suffix == ".pdf":
        return chunk_pdf(path)
    return []


def iter_documents(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield p
