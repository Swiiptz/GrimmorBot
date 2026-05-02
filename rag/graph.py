"""Couche auxiliaire de précision : résolution d'entités depuis le graphe consolidé.

Le graphe `rules_pipeline/output/03_consolidated_graph.json` contient ~466
entités (powers, atouts, handicaps, clans, focus, …) avec leurs propriétés
structurées (level, mechanics, costs, prerequisites, source_quotes, section_ids,
focus relations).

Cette couche s'insère entre le retrieval RAG et la génération LLM :

1. `resolve(query)` matche les entités les plus pertinentes par nom, alias,
   keywords et niveau (« niveau 5 » → entité de type power avec level=5
   appartenant à la même discipline détectée dans la question).
2. `entity_card(entity)` formate une fiche compacte injectable dans le prompt.
3. `bonus_chunks_for(entities)` extrait les sections d'origine en chunks
   compatibles RAG (avec metadata source/page/section), pour que le LLM puisse
   citer normalement.

Singleton chargé une seule fois au démarrage du bot (~5 MB en RAM).
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_GRAPH_PATH = Path("./rules_pipeline/output/03_consolidated_graph.json")
PRIMARY_SOURCE_NAME = "FCF - Règles MET.pdf"

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LEVEL_RE = re.compile(
    r"\b(?:niveau|niv|lvl|level)\s*(?P<lvl>[1-5]|un|une|deux|trois|quatre|cinq|iv|iii|ii|v|i)\b",
    re.IGNORECASE,
)
_LEVEL_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5,
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5,
}
_STOPWORDS = {
    "les", "des", "une", "est", "sont", "que", "qui", "dont", "pour", "par",
    "sur", "dans", "avec", "sans", "comment", "quel", "quelle", "quels",
    "quelles", "quoi", "ce", "cette", "ces", "cet", "et", "ou", "du", "de",
    "la", "le", "un", "moi", "toi", "lui", "ils", "elle", "elles", "ils",
    "niveau", "niv", "lvl", "level", "max", "maximum", "donne", "donner",
    "detaille", "detailles", "detail", "details", "explique", "explication",
    "fonctionne", "fonctionnent", "expliquer", "info", "infos", "tout", "tous",
}

# Disciplines connues (utile pour le filtrage par discipline)
KNOWN_DISCIPLINES = {
    "animalisme", "auspex", "celerite", "celérité", "chimerie", "chimérie",
    "domination", "dissimulation", "force d'ame", "force d'âme", "necromancie",
    "nécromancie", "obtenebration", "obténébration", "presence", "présence",
    "proteisme", "protéisme", "puissance", "quietus", "quiétus", "serpentis",
    "thaumaturgie", "vicissitude", "alienation", "aliénation", "abysses",
}


def _normalize(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text or "")
    return nfkd.encode("ascii", "ignore").decode("ascii").lower()


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(_normalize(text)) if len(t) >= 3 and t not in _STOPWORDS}


def _query_levels(query: str) -> set[int]:
    return {_LEVEL_MAP[m.group("lvl").lower()] for m in _LEVEL_RE.finditer(query)
            if m.group("lvl").lower() in _LEVEL_MAP}


def _query_disciplines(query: str) -> set[str]:
    norm = _normalize(query)
    return {d for d in KNOWN_DISCIPLINES if d in norm}


# ─── Entité enrichie & résolution ────────────────────────────────────────────


@dataclass
class ResolvedEntity:
    entity: dict
    score: float
    matched_on: list[str]  # raisons du match (debug)


class RuleGraph:
    """Graphe consolidé in-memory + résolution d'entités."""

    def __init__(self, graph: dict):
        self.document = graph.get("document") or {}
        self.chapters = graph.get("chapters") or []
        self.sections: dict[str, dict] = graph.get("sections") or {}
        self.entities: list[dict] = graph.get("entities") or []
        self.relationships: list[dict] = graph.get("relationships") or []

        # Index pré-calculés (perf : fait au boot, pas à la query)
        self._by_id: dict[str, dict] = {e["id"]: e for e in self.entities if e.get("id")}
        self._by_name_norm: dict[str, list[dict]] = {}
        self._tokens_cache: dict[str, set[str]] = {}

        for e in self.entities:
            name_norm = _normalize(e.get("name", ""))
            if name_norm:
                self._by_name_norm.setdefault(name_norm, []).append(e)
            # Tokens utilisés au matching
            blob_parts = [e.get("name", ""), " ".join(e.get("aliases") or []),
                          " ".join(e.get("keywords") or []),
                          " ".join(e.get("source_quotes") or [])]
            self._tokens_cache[e.get("id") or name_norm] = _tokens(" ".join(blob_parts))

        # Index relations sortantes par nom de source (lookup rapide)
        self._rels_by_source: dict[str, list[dict]] = {}
        for r in self.relationships:
            src = _normalize(r.get("source_name", ""))
            if src:
                self._rels_by_source.setdefault(src, []).append(r)

        # Inférence discipline → section_ids par adjacence dans "Les Disciplines"
        # Le PDF MET liste les disciplines sous "Les Disciplines" : chaque heading
        # sans bullets (ex: "Animalisme") ouvre une discipline ; les headings
        # avec bullets (ex: "• • • • • Conquérir la Bête") sont ses pouvoirs.
        self._section_discipline: dict[str, str] = {}
        ordered = sorted(self.sections.values(), key=lambda s: s.get("order", 0))
        current_discipline: Optional[str] = None
        for sec in ordered:
            heading = (sec.get("heading") or "").strip()
            subsection = (sec.get("subsection") or "")
            if subsection != "Les Disciplines":
                current_discipline = None
                continue
            heading_norm = _normalize(heading)
            if heading_norm in KNOWN_DISCIPLINES:
                current_discipline = heading_norm
            elif current_discipline:
                self._section_discipline[sec["section_id"]] = current_discipline

        logger.info(
            "RuleGraph chargé : %d entités, %d relations, %d sections",
            len(self.entities), len(self.relationships), len(self.sections),
        )

    # ─── Résolution ──────────────────────────────────────────────────────

    def resolve(self, query: str, max_results: int = 4) -> list[ResolvedEntity]:
        """Renvoie les entités les plus pertinentes pour la query.

        Heuristique simple en pur Python (zero dépendance) :
        - exact name match (gros bonus)
        - substring name/alias match
        - token Jaccard sur (name + aliases + keywords + source_quotes)
        - filtres souples : level + discipline détectés dans la query
        """
        if not query.strip():
            return []

        q_tokens = _tokens(query)
        q_norm = _normalize(query)
        q_levels = _query_levels(query)
        q_disciplines = _query_disciplines(query)

        scored: list[ResolvedEntity] = []

        for entity in self.entities:
            name = entity.get("name", "")
            if not name:
                continue
            name_norm = _normalize(name)
            etype = (entity.get("entity_type") or "").lower()
            score = 0.0
            reasons: list[str] = []

            # Exact name match
            if name_norm == q_norm:
                score += 5.0
                reasons.append("exact_name")
            # Substring name (le name est dans la query, OU la query contient ce name).
            # Exige >= 4 chars pour éviter les faux positifs sur des mots courts (Vol, etc.).
            elif len(name_norm) >= 4 and (
                f" {name_norm} " in f" {q_norm} "
                or q_norm.startswith(name_norm + " ")
                or q_norm.endswith(" " + name_norm)
                or (len(name_norm) >= 5 and q_norm in name_norm)
            ):
                score += 2.5
                reasons.append("substring_name")

            # Aliases
            for alias in entity.get("aliases") or []:
                a_norm = _normalize(alias)
                if a_norm and (a_norm == q_norm or a_norm in q_norm):
                    score += 1.5
                    reasons.append(f"alias:{alias}")
                    break

            # Token overlap (Jaccard pondéré)
            ent_tokens = self._tokens_cache.get(entity.get("id") or name_norm, set())
            if q_tokens and ent_tokens:
                overlap = q_tokens & ent_tokens
                if overlap:
                    jacc = len(overlap) / len(q_tokens | ent_tokens)
                    score += 1.5 * jacc
                    reasons.append(f"jaccard:{','.join(list(overlap)[:3])}")

            # Level filter (si la query parle de niveau N)
            ent_level = entity.get("level")
            if q_levels:
                if ent_level in q_levels and etype in {"power", "pouvoir"}:
                    # Power du bon niveau : gros bonus, c'est l'intent principal
                    score += 3.5
                    reasons.append(f"power_level:{ent_level}")
                elif ent_level in q_levels:
                    score += 1.0
                    reasons.append(f"level:{ent_level}")
                elif ent_level is None:
                    # L'utilisateur cherche un niveau précis, et cette entité n'en a
                    # pas (ex: clan, lignée). Pénalise pour éviter qu'elle prenne
                    # la place du power pertinent.
                    score -= 1.5
                else:
                    # Mauvais niveau
                    score -= 0.5

            # Discipline filter (la query mentionne une discipline)
            if q_disciplines:
                # 1) discipline inférée depuis l'adjacence des sections
                ent_disciplines = {self._section_discipline.get(sid)
                                   for sid in entity.get("section_ids") or []}
                ent_disciplines.discard(None)
                if ent_disciplines & q_disciplines:
                    score += 2.5
                    reasons.append(f"discipline_section:{','.join(ent_disciplines & q_disciplines)}")
                else:
                    # 2) fallback : mention textuelle dans le blob
                    blob_parts = (entity.get("section_ids") or []) + \
                                 (entity.get("source_quotes") or []) + \
                                 (entity.get("keywords") or []) + \
                                 (entity.get("prerequisites") or [])
                    blob_norm = _normalize(" ".join(blob_parts))
                    if any(d in blob_norm for d in q_disciplines):
                        score += 1.0
                        reasons.append("discipline_blob")
                    elif etype in {"power", "pouvoir"} and ent_disciplines:
                        # Power d'une autre discipline : pénalise pour éviter
                        # qu'il sorte sur une question d'une discipline différente.
                        score -= 1.5

            if score > 0.4:
                scored.append(ResolvedEntity(entity=entity, score=score, matched_on=reasons))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:max_results]

    # ─── Formatage en cartes d'entité (pour le prompt LLM) ──────────────

    def entity_card(self, entity: dict, *, include_relations: bool = True) -> str:
        """Fiche compacte d'une entité pour injection dans le prompt."""
        lines: list[str] = []
        name = entity.get("name", "?")
        etype = entity.get("entity_type", "?")
        level = entity.get("level")
        header = f"• {name}  ({etype}"
        if level is not None:
            header += f", niveau {level}"
        header += ")"
        lines.append(header)

        # Discipline inférée par adjacence des sections
        ent_disc = {self._section_discipline.get(sid)
                    for sid in entity.get("section_ids") or []}
        ent_disc.discard(None)
        if ent_disc:
            disc_str = ", ".join(sorted(ent_disc)).title()
            lines.append(f"  Discipline : {disc_str}")

        if entity.get("aliases"):
            lines.append(f"  Alias : {', '.join(entity['aliases'][:5])}")
        if entity.get("summary"):
            lines.append(f"  Résumé : {entity['summary'][:300]}")
        if entity.get("prerequisites"):
            lines.append(f"  Prérequis : {' ; '.join(entity['prerequisites'][:5])}")
        if entity.get("costs"):
            lines.append(f"  Coût : {' ; '.join(entity['costs'][:5])}")
        if entity.get("mechanics"):
            mecha = " ; ".join(entity["mechanics"][:3])
            if len(mecha) > 400:
                mecha = mecha[:400].rstrip() + "…"
            lines.append(f"  Mécaniques : {mecha}")
        if entity.get("keywords"):
            lines.append(f"  Mots-clés : {', '.join(entity['keywords'][:6])}")

        # Pages d'origine (à partir de section_ids)
        pages: list[int] = []
        for sid in entity.get("section_ids") or []:
            sec = self.sections.get(sid) or {}
            p = sec.get("page_start")
            if p:
                pages.append(int(p))
        if pages:
            lines.append(f"  Source : {PRIMARY_SOURCE_NAME}, p.{', p.'.join(map(str, sorted(set(pages))[:5]))}")

        # Relations sortantes
        if include_relations:
            rels = self._rels_by_source.get(_normalize(name), [])
            if rels:
                rel_strs = []
                for r in rels[:5]:
                    rel = r.get("relation", "")
                    tgt = r.get("target_name", "")
                    if rel and tgt:
                        rel_strs.append(f"{rel}→{tgt}")
                if rel_strs:
                    lines.append(f"  Relations : {' ; '.join(rel_strs)}")

        return "\n".join(lines)

    # ─── Bonus chunks issus du graphe (pour citation par le LLM) ────────

    def bonus_chunks_for(
        self, resolved: list[ResolvedEntity], *, max_chunks: int = 3
    ) -> list[dict]:
        """Renvoie les sections d'origine des entités résolues comme chunks RAG.

        Format compatible avec le retriever : {chunk_id, text, metadata{source,
        page, section}, rrf_score}. Le LLM peut les citer normalement.
        """
        out: list[dict] = []
        seen_sections: set[str] = set()
        for r in resolved:
            for sid in r.entity.get("section_ids") or []:
                if sid in seen_sections:
                    continue
                seen_sections.add(sid)
                section = self.sections.get(sid)
                if not section or not section.get("text"):
                    continue
                page = section.get("page_start") or 0
                heading = section.get("heading") or ""
                out.append({
                    "chunk_id": f"graph::{sid}",
                    "text": section["text"],
                    "metadata": {
                        "source": PRIMARY_SOURCE_NAME,
                        "page": int(page) if page else 0,
                        "section": heading,
                        "type": "graph_section",
                        "chunk_index": 0,
                    },
                    "rrf_score": 0.05,  # boost moyen, à fusionner avec le RAG normal
                })
                if len(out) >= max_chunks:
                    return out
        return out


# ─── Singleton ───────────────────────────────────────────────────────────────


_GRAPH_SINGLETON: Optional[RuleGraph] = None


def get_graph(path: Path | None = None) -> Optional[RuleGraph]:
    """Singleton du graphe. Renvoie None si le fichier n'existe pas (mode dégradé)."""
    global _GRAPH_SINGLETON
    if _GRAPH_SINGLETON is None:
        target = path or DEFAULT_GRAPH_PATH
        if not target.exists():
            logger.warning("Graphe consolidé absent : %s — couche auxiliaire désactivée.", target)
            return None
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
            _GRAPH_SINGLETON = RuleGraph(data)
        except Exception:
            logger.exception("Échec chargement du graphe consolidé %s", target)
            return None
    return _GRAPH_SINGLETON


def reset_graph_singleton() -> None:
    global _GRAPH_SINGLETON
    _GRAPH_SINGLETON = None
