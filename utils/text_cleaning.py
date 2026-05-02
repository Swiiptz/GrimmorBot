"""Nettoyage de texte partage entre RAG, prompt et affichage Discord."""
from __future__ import annotations

import re


_CONFIDENTIAL_BLOCK_RE = re.compile(
    r"F\S*d\S*ration\s+Camarilla\s+Fran\S*aise\s+"
    r"Document\s+confidentiel\s*[-–—]\s*Diffusion\s+interdite\s+en\s+dehors\s+de\s+la\s+"
    r".{0,80}?\s+sans\s+auth?orisation\s+du\s+Bureau",
    re.IGNORECASE,
)
_CONFIDENTIAL_LINE_RE = re.compile(
    r"Document\s+confidentiel\s*[-–—]\s*Diffusion\s+interdite\s+en\s+dehors\s+de\s+la\s+"
    r".{0,80}?\s+sans\s+auth?orisation\s+du\s+Bureau",
    re.IGNORECASE,
)
_FCF_HEADER_LINE_RE = re.compile(
    r"^\s*F\S*d\S*ration\s+Camarilla\s+Fran\S*aise\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_pdf_boilerplate(text: str) -> str:
    """Retire les en-tetes/pieds de page non utiles issus du PDF source."""
    if not text:
        return ""
    cleaned = _CONFIDENTIAL_BLOCK_RE.sub("", text)
    cleaned = _CONFIDENTIAL_LINE_RE.sub("", cleaned)
    cleaned = _FCF_HEADER_LINE_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()
