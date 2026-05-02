from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from rules_pipeline.models import ParsedDocument


def _key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", normalized).strip()


def consolidate_graph(parsed_document: ParsedDocument, regex_results: list[dict], llm_results: list[dict]) -> dict:
    entities: dict[tuple[str, str], dict] = {}
    edges: dict[tuple[str, str, str], dict] = {}
    origins: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add_entity(entity: dict, origin: str) -> None:
        entity_type = str(entity.get("entity_type", "")).strip()
        name = str(entity.get("name", "")).strip()
        if not entity_type or not name:
            return
        entity_key = (entity_type, _key(name))
        origins[entity_key].add(origin)
        current = entities.setdefault(
            entity_key,
            {
                "id": f"{entity_type}:{_key(name).replace(' ', '_')}",
                "entity_type": entity_type,
                "name": name,
                "aliases": [],
                "summary": "",
                "mechanics": [],
                "costs": [],
                "prerequisites": [],
                "keywords": [],
                "source_quotes": [],
                "section_ids": [],
                "origins": [],
            },
        )

        for field in ("aliases", "mechanics", "costs", "prerequisites", "keywords", "source_quotes"):
            for value in entity.get(field, []) or []:
                if value and value not in current[field]:
                    current[field].append(value)

        if entity.get("summary") and len(str(entity["summary"])) > len(current["summary"]):
            current["summary"] = str(entity["summary"])

        if "level" in entity and "level" not in current:
            current["level"] = entity["level"]

        section_id = entity.get("section_id")
        if section_id and section_id not in current["section_ids"]:
            current["section_ids"].append(section_id)

    def add_edge(edge: dict) -> None:
        source = str(edge.get("source_name", "")).strip()
        target = str(edge.get("target_name", "")).strip()
        relation = str(edge.get("relation", "")).strip()
        if not source or not target or not relation:
            return
        edge_key = (_key(source), _key(target), relation)
        current = edges.setdefault(
            edge_key,
            {
                "source_name": source,
                "target_name": target,
                "relation": relation,
                "evidence": [],
            },
        )
        evidence = str(edge.get("evidence", "")).strip()
        if evidence and evidence not in current["evidence"]:
            current["evidence"].append(evidence)

    active_power_name = ""
    for result in regex_results:
        for entity in result.get("nodes", []):
            add_entity(entity, "regex")
            if entity.get("entity_type") == "power":
                active_power_name = str(entity.get("name", "")).strip()
            elif entity.get("entity_type") == "focus" and active_power_name:
                add_edge(
                    {
                        "source_name": entity.get("name", ""),
                        "target_name": active_power_name,
                        "relation": "focus_for",
                        "evidence": (entity.get("source_quotes") or [""])[0],
                    }
                )
        for edge in result.get("edges", []):
            add_edge(edge)

    for result in llm_results:
        section_id = result.get("section_id", "")
        for entity in result.get("entities", []):
            enriched = dict(entity)
            enriched.setdefault("section_id", section_id)
            add_entity(enriched, "llm")
        for edge in result.get("relationships", []):
            add_edge(edge)

    for entity_key, entity in entities.items():
        entity["origins"] = sorted(origins[entity_key])

    return {
        "document": {
            "title": parsed_document.title,
            "source_path": parsed_document.source_path,
            "total_pages": parsed_document.total_pages,
        },
        "chapters": [chapter.to_dict() for chapter in parsed_document.chapters],
        "sections": {section.section_id: section.to_dict() for section in parsed_document.sections},
        "entities": list(entities.values()),
        "relationships": list(edges.values()),
    }
