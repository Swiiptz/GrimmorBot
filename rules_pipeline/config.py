from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _default_pdf_path() -> Path:
    docs_root = Path(os.getenv("DOCS_PATH", "./docs")).resolve()
    pdfs = sorted(docs_root.rglob("*.pdf"))
    if not pdfs:
        raise RuntimeError("No PDF found. Set RULES_PDF_PATH or add a PDF in docs/.")
    return pdfs[0]


@dataclass(frozen=True)
class PipelineConfig:
    pdf_path: Path
    output_dir: Path
    provider: str
    openai_api_key: str
    openai_model: str
    gemini_api_key: str
    gemini_model: str
    groq_api_key: str
    groq_model: str
    timeout_seconds: int
    max_output_tokens: int
    temperature: float
    batch_target_chars: int
    batch_max_chars: int


def load_pipeline_config() -> PipelineConfig:
    provider = os.getenv("STRUCTURED_LLM_PROVIDER", "openai").strip().lower() or "openai"
    if provider not in {"openai", "gemini", "groq"}:
        raise RuntimeError("STRUCTURED_LLM_PROVIDER must be openai, gemini, or groq.")

    pdf_raw = os.getenv("RULES_PDF_PATH", "").strip()
    pdf_path = Path(pdf_raw).resolve() if pdf_raw else _default_pdf_path()

    gemini_model = os.getenv(
        "GEMINI_STRUCTURED_MODEL",
        os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
    ).strip() or "gemini-2.5-flash-lite"
    batch_target = _get_int("RULES_BATCH_TARGET_CHARS", 18000)
    batch_max = _get_int("RULES_BATCH_MAX_CHARS", 26000)
    if provider == "gemini" and gemini_model.startswith("gemma-"):
        batch_target = min(batch_target, 8500)
        batch_max = min(batch_max, 12000)
    if provider == "groq":
        groq_model = os.getenv(
            "GROQ_STRUCTURED_MODEL",
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        ).strip() or "llama-3.3-70b-versatile"
        if groq_model == "llama-3.3-70b-versatile":
            batch_target = min(batch_target, 7000)
            batch_max = min(batch_max, 10000)

    return PipelineConfig(
        pdf_path=pdf_path,
        output_dir=Path(os.getenv("RULES_OUTPUT_DIR", "./rules_pipeline/output")).resolve(),
        provider=provider,
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini",
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=gemini_model,
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv(
            "GROQ_STRUCTURED_MODEL",
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        ).strip() or "llama-3.3-70b-versatile",
        timeout_seconds=_get_int("STRUCTURED_LLM_TIMEOUT_SECONDS", 180),
        max_output_tokens=_get_int("STRUCTURED_LLM_MAX_OUTPUT_TOKENS", 4000),
        temperature=_get_float("STRUCTURED_LLM_TEMPERATURE", 0.1),
        batch_target_chars=batch_target,
        batch_max_chars=batch_max,
    )
