"""Chargement et validation des variables d'environnement."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _parse_id_list(raw: str | None) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            continue
    return out


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    discord_token: str
    guild_id: int

    llm_provider: str = "gemini"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    cerebras_api_key: str = ""
    mistral_api_key: str = ""

    log_channel_id: int | None = None

    chroma_path: Path = field(default_factory=lambda: Path("./chroma_db"))
    docs_path: Path = field(default_factory=lambda: Path("./docs"))
    sqlite_path: Path = field(default_factory=lambda: Path("./grimmor.db"))
    archive_path: Path = field(default_factory=lambda: Path("./archives"))
    permissions_path: Path = field(default_factory=lambda: Path("./permissions.json"))

    rag_top_k_dense: int = 15
    rag_top_k_sparse: int = 15
    rag_top_k_final: int = 8
    rag_min_rrf_score: float = 0.015
    rag_rrf_k: int = 60
    # Multi-query expansion par LLM : coûte 1 appel + tokens en plus.
    # Désactivé par défaut depuis l'ajout de la couche graphe qui fournit
    # déjà des variantes "gratuites" (noms d'entités résolues).
    rag_use_multi_query_llm: bool = False

    gemini_model: str = "gemini-2.5-flash-lite"
    groq_model: str = "llama-3.3-70b-versatile"
    cerebras_model: str = "qwen-3-235b-a22b-instruct-2507"
    mistral_model: str = "mistral-small-latest"
    gemini_timeout_seconds: int = 30
    gemini_max_output_tokens: int = 2200

    # File d'attente LLM : limite la concurrence pour éviter les 429.
    # 1 = serial (un appel à la fois), 2-3 = autorise un peu de parallélisme.
    llm_max_concurrent: int = 1
    # Intervalle minimum entre 2 appels LLM (en secondes). 0 = pas d'enforcement.
    # Utile si le rate limit est en RPM (ex: 15 RPM → mets 4.0).
    llm_min_interval_seconds: float = 0.0

    asso_name: str = "Notre Association JDR"


def load_config() -> Config:
    discord_token = os.getenv("DISCORD_TOKEN", "").strip()
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower() or "gemini"
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    cerebras_api_key = os.getenv("CEREBRAS_API_KEY", "").strip()
    mistral_api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    guild_id_raw = os.getenv("GUILD_ID", "").strip()

    if not discord_token:
        raise RuntimeError("DISCORD_TOKEN manquant dans l'environnement.")
    if not guild_id_raw:
        raise RuntimeError("GUILD_ID manquant dans l'environnement.")
    if llm_provider not in {"gemini", "groq", "cerebras", "mistral"}:
        raise RuntimeError("LLM_PROVIDER doit valoir 'gemini', 'groq', 'cerebras' ou 'mistral'.")
    if llm_provider == "gemini" and not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY manquant dans l'environnement.")
    if llm_provider == "groq" and not groq_api_key:
        raise RuntimeError("GROQ_API_KEY manquant dans l'environnement.")
    if llm_provider == "cerebras" and not cerebras_api_key:
        raise RuntimeError("CEREBRAS_API_KEY manquant dans l'environnement.")
    if llm_provider == "mistral" and not mistral_api_key:
        raise RuntimeError("MISTRAL_API_KEY manquant dans l'environnement.")

    try:
        guild_id = int(guild_id_raw)
    except ValueError as exc:
        raise RuntimeError("GUILD_ID doit être un entier.") from exc

    log_channel_raw = os.getenv("LOG_CHANNEL_ID", "").strip()
    log_channel_id: int | None = None
    if log_channel_raw:
        try:
            log_channel_id = int(log_channel_raw)
        except ValueError:
            log_channel_id = None

    return Config(
        discord_token=discord_token,
        guild_id=guild_id,
        llm_provider=llm_provider,
        gemini_api_key=gemini_api_key,
        groq_api_key=groq_api_key,
        cerebras_api_key=cerebras_api_key,
        mistral_api_key=mistral_api_key,
        log_channel_id=log_channel_id,
        chroma_path=Path(os.getenv("CHROMA_PATH", "./chroma_db")).resolve(),
        docs_path=Path(os.getenv("DOCS_PATH", "./docs")).resolve(),
        sqlite_path=Path(os.getenv("SQLITE_PATH", "./grimmor.db")).resolve(),
        archive_path=Path(os.getenv("ARCHIVE_PATH", "./archives")).resolve(),
        permissions_path=Path(os.getenv("PERMISSIONS_PATH", "./permissions.json")).resolve(),
        rag_top_k_dense=_get_int("RAG_TOP_K_DENSE", 15),
        rag_top_k_sparse=_get_int("RAG_TOP_K_SPARSE", 15),
        rag_top_k_final=_get_int("RAG_TOP_K_FINAL", 8),
        rag_min_rrf_score=_get_float("RAG_MIN_RRF_SCORE", 0.015),
        rag_rrf_k=_get_int("RAG_RRF_K", 60),
        rag_use_multi_query_llm=os.getenv("RAG_USE_MULTI_QUERY_LLM", "false").strip().lower() in {"1", "true", "yes", "on"},
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip() or "gemini-2.5-flash-lite",
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile",
        cerebras_model=os.getenv("CEREBRAS_MODEL", "qwen-3-235b-a22b-instruct-2507").strip() or "qwen-3-235b-a22b-instruct-2507",
        mistral_model=os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip() or "mistral-small-latest",
        gemini_timeout_seconds=_get_int("GEMINI_TIMEOUT_SECONDS", 30),
        gemini_max_output_tokens=_get_int(
            "LLM_MAX_OUTPUT_TOKENS",
            _get_int("GEMINI_MAX_OUTPUT_TOKENS", 2200),
        ),
        llm_max_concurrent=_get_int("LLM_MAX_CONCURRENT", 1),
        llm_min_interval_seconds=_get_float("LLM_MIN_INTERVAL_SECONDS", 0.0),
        asso_name=os.getenv("ASSO_NAME", "Notre Association JDR"),
    )


_CONFIG_CACHE: Config | None = None


def get_config() -> Config:
    """Retourne le singleton de configuration, en chargeant à la première demande."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()
    return _CONFIG_CACHE
