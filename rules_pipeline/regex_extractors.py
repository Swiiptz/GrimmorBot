from __future__ import annotations

import re

from rules_pipeline.models import DocumentSection

BULLET_POWER_RE = re.compile(r"^(?P<bullets>•(?:\s*•){0,4})\s*(?P<name>.+)$")
FOCUS_RE = re.compile(r"^Focus\s*:\s*(?P<name>.+)$", re.IGNORECASE)
PREREQ_RE = re.compile(
    r"^(?:Prerequis|Prérequis|Conditions prealables|Conditions préalables)\s*:\s*(?P<value>.+)$",
    re.IGNORECASE,
)
METADATA_RE = re.compile(
    r"^(?P<key>Disciplines|Approbation|Surnom|Faiblesse de Clan|Faiblesse du clan)\s*:\s*(?P<value>.+)$",
    re.IGNORECASE,
)


def _extract_table_blocks(section: DocumentSection) -> list[dict]:
    tables: list[dict] = []
    blocks = [part.strip() for part in re.split(r"\n\s*\n", section.text) if part.strip()]
    for block in blocks:
        rows = [line.strip() for line in block.splitlines() if line.strip()]
        if len(rows) < 3:
            continue
        digit_rows = sum(1 for row in rows if re.search(r"\d", row))
        tabular_rows = sum(1 for row in rows if row.count("|") >= 2 or re.search(r"\s{2,}", row))
        if digit_rows >= 2 and tabular_rows >= 2:
            tables.append(
                {
                    "entity_type": "table",
                    "name": section.heading,
                    "summary": "",
                    "mechanics": rows,
                    "costs": [],
                    "prerequisites": [],
                    "keywords": ["table"],
                    "source_quotes": rows[:3],
                    "section_id": section.section_id,
                }
            )
    return tables


def extract_deterministic_entities(section: DocumentSection) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []
    last_power_name: str | None = None

    source_lines = [section.heading, *section.text.splitlines()]
    for line in (raw.strip() for raw in source_lines):
        if not line:
            continue

        bullet_match = BULLET_POWER_RE.match(line)
        if bullet_match:
            name = bullet_match.group("name").strip()
            if len(name) > 70 or name.endswith((".", "!", "?", ";")):
                continue
            last_power_name = name
            nodes.append(
                {
                    "entity_type": "power",
                    "name": name,
                    "aliases": [],
                    "level": bullet_match.group("bullets").count("•"),
                    "summary": "",
                    "mechanics": [],
                    "costs": [],
                    "prerequisites": [],
                    "keywords": [],
                    "source_quotes": [line],
                    "section_id": section.section_id,
                }
            )
            continue

        focus_match = FOCUS_RE.match(line)
        if focus_match:
            focus_name = focus_match.group("name").strip()
            nodes.append(
                {
                    "entity_type": "focus",
                    "name": focus_name,
                    "aliases": [],
                    "summary": "",
                    "mechanics": [],
                    "costs": [],
                    "prerequisites": [],
                    "keywords": [],
                    "source_quotes": [line],
                    "section_id": section.section_id,
                }
            )
            if last_power_name:
                edges.append(
                    {
                        "source_name": focus_name,
                        "target_name": last_power_name,
                        "relation": "focus_for",
                        "evidence": line,
                    }
                )
            continue

        prereq_match = PREREQ_RE.match(line)
        if prereq_match and last_power_name:
            edges.append(
                {
                    "source_name": last_power_name,
                    "target_name": prereq_match.group("value").strip(),
                    "relation": "requires",
                    "evidence": line,
                }
            )
            continue

        metadata_match = METADATA_RE.match(line)
        if metadata_match:
            nodes.append(
                {
                    "entity_type": "metadata",
                    "name": metadata_match.group("key").strip(),
                    "aliases": [],
                    "summary": metadata_match.group("value").strip(),
                    "mechanics": [],
                    "costs": [],
                    "prerequisites": [],
                    "keywords": [],
                    "source_quotes": [line],
                    "section_id": section.section_id,
                }
            )

    nodes.extend(_extract_table_blocks(section))
    return {"section_id": section.section_id, "nodes": nodes, "edges": edges}
