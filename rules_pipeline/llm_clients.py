from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod

import httpx
from google.api_core.exceptions import DeadlineExceeded, ResourceExhausted, ServiceUnavailable

from rules_pipeline.config import PipelineConfig
from rules_pipeline.models import DocumentSection

ENTITY_TYPES = [
    "chapter",
    "section",
    "discipline",
    "power",
    "ritual",
    "technique",
    "clan",
    "rule",
    "system",
    "focus",
    "trait",
    "advantage",
    "disadvantage",
    "weapon",
    "armor",
    "table",
    "metadata",
]

RELATION_TYPES = [
    "belongs_to",
    "requires",
    "modifies",
    "references",
    "focus_for",
    "variant_of",
    "exception_to",
]

EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "section_id": {"type": "string"},
        "section_title": {"type": "string"},
        "summary": {"type": "string"},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entity_type": {"type": "string", "enum": ENTITY_TYPES},
                    "name": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                    "mechanics": {"type": "array", "items": {"type": "string"}},
                    "costs": {"type": "array", "items": {"type": "string"}},
                    "prerequisites": {"type": "array", "items": {"type": "string"}},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "source_quotes": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "entity_type",
                    "name",
                    "aliases",
                    "summary",
                    "mechanics",
                    "costs",
                    "prerequisites",
                    "keywords",
                    "source_quotes",
                ],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source_name": {"type": "string"},
                    "target_name": {"type": "string"},
                    "relation": {"type": "string", "enum": RELATION_TYPES},
                    "evidence": {"type": "string"},
                },
                "required": ["source_name", "target_name", "relation", "evidence"],
            },
        },
        "open_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "section_id",
        "section_title",
        "summary",
        "entities",
        "relationships",
        "open_questions",
    ],
}


def _extract_json_blob(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start >= 0:
            depth = 0
            in_string = False
            escaped = False
            for index in range(start, len(text)):
                char = text[index]
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start : index + 1])
        raise


def _build_prompt(section: DocumentSection, deterministic: dict) -> str:
    deterministic_json = json.dumps(deterministic, ensure_ascii=False)
    return f"""Extrais les regles de jeu de role de la section PDF courante en JSON strict.

Regles:
- Reponds en francais pour tous les contenus textuels.
- Utilise uniquement le texte fourni dans la section.
- N'invente aucune information manquante.
- Garde les source_quotes courtes et verbatim, dans la langue exacte du document.
- Si un Focus, une exception, un prerequis ou un cout appartient clairement au pouvoir precedent, cree la relation.
- Les informations absentes doivent etre une chaine vide ou un tableau vide.
- Renvoie uniquement du JSON.
- Utilise exactement cette forme racine:
  {{"section_id": "...", "section_title": "...", "summary": "...", "entities": [], "relationships": [], "open_questions": []}}

Metadonnees de section:
section_id: {section.section_id}
chapitre: {section.chapter}
sous-section: {section.subsection}
titre: {section.heading}
pages: {section.page_start}-{section.page_end}

Extraction deterministe deja trouvee:
{deterministic_json}

Texte de section:
{section.text}
"""


class BaseStructuredClient(ABC):
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    @abstractmethod
    def extract(self, section: DocumentSection, deterministic: dict) -> dict:
        raise NotImplementedError


class OpenAIStructuredClient(BaseStructuredClient):
    def extract(self, section: DocumentSection, deterministic: dict) -> dict:
        if not self.cfg.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        payload = {
            "model": self.cfg.openai_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise JSON data extraction engine.",
                },
                {"role": "user", "content": _build_prompt(section, deterministic)},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "rules_section_extraction",
                    "strict": True,
                    "schema": EXTRACTION_SCHEMA,
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.openai_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.cfg.timeout_seconds) as client:
            response = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        parsed = _extract_json_blob(data["choices"][0]["message"]["content"])
        parsed.setdefault("section_id", section.section_id)
        parsed.setdefault("section_title", section.heading)
        return parsed


class GeminiStructuredClient(BaseStructuredClient):
    def _generate_once(self, section: DocumentSection, deterministic: dict) -> dict:
        import google.generativeai as genai

        genai.configure(api_key=self.cfg.gemini_api_key)
        model = genai.GenerativeModel(self.cfg.gemini_model)
        config_kwargs = {
            "temperature": self.cfg.temperature,
            "max_output_tokens": self.cfg.max_output_tokens,
        }
        # Gemma hosted through the Gemini API currently rejects JSON mode, so we
        # rely on prompt-only JSON and parse the returned object locally.
        if not self.cfg.gemini_model.startswith("gemma-"):
            config_kwargs["response_mime_type"] = "application/json"
        generation_config = genai.types.GenerationConfig(**config_kwargs)
        response = model.generate_content(
            _build_prompt(section, deterministic),
            generation_config=generation_config,
            request_options={"timeout": self.cfg.timeout_seconds},
        )
        parsed = _extract_json_blob(response.text or "")
        parsed.setdefault("section_id", section.section_id)
        parsed.setdefault("section_title", section.heading)
        return parsed

    def extract(self, section: DocumentSection, deterministic: dict) -> dict:
        if not self.cfg.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is missing.")
        for attempt in range(1, 6):
            try:
                return self._generate_once(section, deterministic)
            except ResourceExhausted as exc:
                message = str(exc)
                match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", message)
                wait_seconds = int(match.group(1)) + 3 if match else 20 * attempt
                if attempt >= 5:
                    raise
                print(
                    f"Gemini quota reached; waiting {wait_seconds}s before retry "
                    f"({attempt}/5).",
                    flush=True,
                )
                time.sleep(wait_seconds)
            except (DeadlineExceeded, ServiceUnavailable) as exc:
                wait_seconds = 10 * attempt
                if attempt >= 3:
                    raise
                print(
                    f"Gemini temporary failure ({exc.__class__.__name__}); "
                    f"waiting {wait_seconds}s before retry ({attempt}/3).",
                    flush=True,
                )
                time.sleep(wait_seconds)
        raise RuntimeError("Gemini extraction failed after retries.")


class GroqStructuredClient(BaseStructuredClient):
    def extract(self, section: DocumentSection, deterministic: dict) -> dict:
        if not self.cfg.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing.")
        payload = {
            "model": self.cfg.groq_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only a valid JSON object, without Markdown.",
                },
                {"role": "user", "content": _build_prompt(section, deterministic)},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.cfg.groq_api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self.cfg.timeout_seconds) as client:
            response = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        parsed = _extract_json_blob(data["choices"][0]["message"]["content"] or "")
        parsed.setdefault("section_id", section.section_id)
        parsed.setdefault("section_title", section.heading)
        return parsed


def build_structured_client(cfg: PipelineConfig) -> BaseStructuredClient:
    if cfg.provider == "openai":
        return OpenAIStructuredClient(cfg)
    if cfg.provider == "gemini":
        return GeminiStructuredClient(cfg)
    return GroqStructuredClient(cfg)
