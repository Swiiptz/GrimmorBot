"""Helpers de mise en forme des embeds Discord."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable

import discord

from utils.text_cleaning import strip_pdf_boilerplate

GRIMMOR_PURPLE = 0x7B2FBE
GRIMMOR_ORANGE = 0xE8A23A
GRIMMOR_RED = 0xC0392B
GRIMMOR_GREEN = 0x27AE60


def error_embed(message: str, *, title: str = "Erreur") -> discord.Embed:
    return discord.Embed(title=title, description=message, color=GRIMMOR_RED)


def warning_embed(message: str, *, title: str = "Attention") -> discord.Embed:
    return discord.Embed(title=title, description=message, color=GRIMMOR_ORANGE)


def success_embed(message: str, *, title: str = "OK") -> discord.Embed:
    return discord.Embed(title=title, description=message, color=GRIMMOR_GREEN)


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def format_sources(sources: Iterable[dict]) -> str:
    """Formate des sources simples (legacy)."""
    lines: list[str] = []
    for src in sources:
        source = src.get("source") or "?"
        page = src.get("page")
        section = src.get("section")
        bits: list[str] = [f"**{source}**"]
        if page and int(page) > 0:
            bits.append(f"p.{page}")
        if section:
            bits.append(f"_{truncate(str(section), 60)}_")
        lines.append("• " + ", ".join(bits))
    text = "\n".join(lines) if lines else "_aucune_"
    return truncate(text, 1024)


def format_duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    if days >= 1:
        return f"{days}j {hours}h" if hours else f"{days}j"
    if hours >= 1:
        return f"{hours}h {minutes}min" if minutes else f"{hours}h"
    if minutes >= 1:
        return f"{minutes}min"
    return f"{seconds}s"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ─── NotebookLM-style helpers ────────────────────────────────────────────────


def short_source_label(meta: dict) -> str:
    """Étiquette courte d'une source (sans excerpt). Ex: `regles.pdf · p.12 · Initiative`."""
    source = meta.get("source") or "?"
    bits: list[str] = [str(source)]
    page = meta.get("page")
    if page and int(page) > 0:
        bits.append(f"p.{page}")
    section = meta.get("section")
    if section:
        bits.append(truncate(str(section), 50))
    return " · ".join(bits)


def clean_excerpt(text: str, max_chars: int = 280) -> str:
    """Nettoie et tronque un extrait pour affichage en blockquote."""
    if not text:
        return ""
    cleaned = strip_pdf_boilerplate(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return truncate(cleaned, max_chars)


def format_citation_field(idx: int, meta: dict, quote: str | None) -> tuple[str, str]:
    """Champ embed pour une source numérotée. Retourne (name, value).

    name  : `[1] regles.pdf · p.12`
    value : `_Initiative et ordre de tour_\n> « extrait verbatim »`
    """
    source = meta.get("source") or "?"
    page = meta.get("page")
    bits: list[str] = [str(source)]
    if page and int(page) > 0:
        bits.append(f"p.{page}")
    name = truncate(f"[{idx}] " + " · ".join(bits), 256)

    parts: list[str] = []
    section = meta.get("section")
    if section:
        parts.append(f"_{truncate(str(section), 200)}_")
    if quote:
        excerpt = clean_excerpt(quote, max_chars=400)
        # Préfixe blockquote sur chaque ligne
        parts.append("> " + excerpt.replace("\n", "\n> "))
    value = "\n".join(parts) if parts else "_(extrait non disponible)_"
    return name, truncate(value, 1024)
