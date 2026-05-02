"""Génération style NotebookLM avec sortie JSON structurée.

Le LLM reçoit des extraits numérotés et doit produire un objet JSON :
{
  "answer": "Texte avec citations [1] [2] inline.",
  "citations": [
    {"id": 1, "chunk_index": 3, "quote": "extrait verbatim ..."},
    ...
  ],
  "follow_ups": ["Question 1 ?", "Question 2 ?"]
}

Cette structure garantit que chaque citation pointe vers un chunk précis,
et qu'on peut afficher l'extrait verbatim côté Discord (mode NotebookLM).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import httpx

from config import get_config
from utils.text_cleaning import strip_pdf_boilerplate

logger = logging.getLogger(__name__)


class LLMRateLimitError(RuntimeError):
    """Rate limit temporaire cote provider LLM."""

    def __init__(self, provider: str, retry_after_seconds: float | None = None):
        self.provider = provider
        self.retry_after_seconds = retry_after_seconds
        suffix = ""
        if retry_after_seconds is not None:
            suffix = f" Reessayer dans {retry_after_seconds:.0f}s."
        super().__init__(f"Rate limit {provider}.{suffix}")


NO_ANSWER_SENTINEL = "Je n'ai pas trouvé cette règle dans mes documents."


# ─── File d'attente LLM (gate) ────────────────────────────────────────────────


class LLMGate:
    """Sémaphore async + rate limit minimal pour serializer les appels LLM.

    - `max_concurrent` : nombre d'appels LLM simultanés autorisés.
    - `min_interval_s` : durée minimale entre deux appels (RPM enforcement).
    - Expose `waiting` / `in_flight` pour la visibilité côté UX.
    """

    def __init__(self, max_concurrent: int = 1, min_interval_s: float = 0.0):
        self._sem = asyncio.Semaphore(max(1, max_concurrent))
        self._min_interval = max(0.0, float(min_interval_s))
        self._last_call_at: float = 0.0
        self.waiting: int = 0
        self.in_flight: int = 0
        self._lock = asyncio.Lock()  # protège _last_call_at

    @property
    def queue_position(self) -> int:
        """Combien d'appels passent ou attendent devant le prochain demandeur."""
        return self.waiting + self.in_flight

    async def __aenter__(self):
        self.waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self.waiting -= 1
        self.in_flight += 1
        # Enforcement du min interval (RPM)
        if self._min_interval > 0:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                wait = self._min_interval - (now - self._last_call_at)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_call_at = asyncio.get_running_loop().time()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.in_flight -= 1
        self._sem.release()


# Singleton de la gate (créée à la 1ère résolution du générateur, partagée).
_LLM_GATE: Optional[LLMGate] = None


def get_llm_gate() -> Optional[LLMGate]:
    return _LLM_GATE


# ─── Schéma JSON attendu en sortie de Gemini ─────────────────────────────────
# Compatible avec google-generativeai 0.8.x (protobuf Schema sous le capot).
# On utilise les types uppercase OpenAPI pour éviter toute ambiguïté.

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "citations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "INTEGER"},
                    "chunk_index": {"type": "INTEGER"},
                    "quote": {"type": "STRING"},
                },
                "required": ["id", "chunk_index", "quote"],
            },
        },
        "follow_ups": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["answer", "citations", "follow_ups"],
}


# Schéma pour la multi-query expansion
VARIANTS_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "variants": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
        },
    },
    "required": ["variants"],
}


def _build_variants_prompt(question: str, asso_name: str) -> str:
    return f"""Tu es un assistant de recherche pour {asso_name} (jeu de rôle).

L'utilisateur a posé la question suivante :
« {question} »

Génère exactement 3 reformulations alternatives de cette question, conçues pour
maximiser les chances de matcher le vocabulaire d'un manuel de règles JDR.

Règles :
- Reste dans le même thème, ne dévie pas du sujet.
- Varie les angles : synonymes, terminologie technique du JDR, noms de pouvoirs
  ou règles connus, paraphrases.
- Si la question parle d'un « niveau N » d'une discipline, propose une variante
  qui mentionne le nom du pouvoir typique de ce niveau (ex: niveau 5 d'Animalisme
  → « Conquérir la Bête »).
- Si la question est déjà très spécifique, reformule plus largement.
- Garde chaque reformulation en moins de 15 mots.
- Réponds en français.

Réponds STRICTEMENT en JSON : {{"variants": ["reformulation 1", "...", "..."]}}.
"""


# ─── Construction du prompt ───────────────────────────────────────────────────


def _build_prompt(
    question: str,
    chunks: list[dict],
    asso_name: str,
    entity_cards: list[str] | None = None,
) -> str:
    rules = f"""Tu es Grimmor, l'oracle des règles de {asso_name}.

RÈGLES ABSOLUES — à suivre sans exception :

1. Tu ne réponds QU'À PARTIR des EXTRAITS numérotés ci-dessous. Aucune
   connaissance externe, aucune inférence personnelle.

1bis. Dans les manuels Vampire/JDR, les titres de pouvoirs de Discipline
   peuvent être écrits avec des puces : `•` = niveau 1, `• •` = niveau 2,
   `• • •` = niveau 3, `• • • •` = niveau 4, `• • • • •` = niveau 5.
   Si un extrait contient par exemple `• • • • • Nom du pouvoir`, tu peux
   répondre que `Nom du pouvoir` est le pouvoir de niveau 5.

1ter. Les PDF peuvent couper un même pouvoir entre deux extraits consécutifs :
   le titre peut apparaître à la fin d'un extrait et le système au début du
   suivant. Si deux extraits se suivent dans le même fichier et sur des pages
   voisines, tu peux les utiliser ensemble, mais uniquement pour relier le
   titre au texte immédiatement adjacent.

1quater. Si la question porte sur un pouvoir, atout, poste, concept ou règle
   précis, distingue la règle principale des autres règles qui la mentionnent.
   Un extrait qui commence par le titre d'un autre pouvoir ne doit pas être
   présenté comme un effet du pouvoir demandé. Tu peux le mentionner seulement
   si l'utilisateur demande explicitement les interactions, synergies, autres
   pouvoirs liés ou exceptions.

2. Si la réponse n'est pas littéralement présente dans les extraits, ton
   `answer` doit être EXACTEMENT : « {NO_ANSWER_SENTINEL} » et `citations`
   doit être un tableau vide. N'invente jamais.

3. Chaque affirmation factuelle dans ton `answer` DOIT être suivie d'une ou
   plusieurs citations sous la forme [n], où n correspond à un id présent
   dans `citations`. Plusieurs sources possibles : [1][3].

4. Pour chaque citation utilisée dans `answer`, tu dois ajouter dans
   `citations` un objet contenant :
   - `id` : le numéro affiché dans answer (1, 2, 3, …)
   - `chunk_index` : l'index 1-based de l'EXTRAIT d'origine
   - `quote` : un extrait verbatim COURT (≈30-50 mots) recopié SANS
     modification depuis le chunk.

5. Ne paraphrase JAMAIS les chiffres, noms propres, termes techniques. Recopie
   exactement (1d10, Vigueur, attribut, etc.).

6. Réponse en français. Donne une réponse assez détaillée pour être utile en
   partie : en général 6 à 10 phrases, ou une courte liste structurée si la
   règle contient plusieurs conditions, étapes, coûts, exceptions ou effets.
   Si la question est très simple, reste plus court.

6bis. Pour une règle de jeu, cherche systématiquement dans les extraits :
   définition, coût, action requise, jet/challenge, durée, portée, prérequis,
   limites, exceptions, focus, plafonds, moyens d'augmentation, interactions et
   cas particuliers. N'invente pas : si une précision n'est pas dans les
   extraits, dis explicitement qu'elle n'a pas été trouvée.

6ter. Si plusieurs extraits donnent des éléments complémentaires fiables,
   synthétise-les ensemble au lieu de répondre seulement avec le premier
   extrait pertinent. Distingue clairement la règle principale des interactions
   ou règles voisines.

7. `follow_ups` : propose 2 ou 3 questions courtes pertinentes que
   l'utilisateur pourrait vouloir poser ensuite, à partir des extraits.

8. CARTES D'ENTITÉS : si une section [CARTES D'ENTITÉS DU CORPUS] est fournie
   avant les extraits, ces cartes sont des **fiches structurées de référence**
   pré-extraites du PDF (catalogue de pouvoirs, atouts, clans, etc.).

   Tu peux faire confiance aux **affirmations structurelles** des cartes
   (nom, type, niveau, discipline, focus, prérequis, coût listés) comme
   vérité de référence — c'est de la métadonnée extraite mécaniquement.
   Pour ces faits structurels, tu peux les énoncer en t'appuyant sur la
   carte ; la citation [n] doit alors pointer vers l'extrait qui décrit le
   pouvoir/atout (souvent un chunk de type `graph_section` qui a été ajouté).

   Pour la **description / les effets / le système de jeu**, tu DOIS continuer
   à citer un EXTRAIT verbatim normalement.

   Si la question porte sur "le niveau N de Discipline X" et qu'une carte
   liste un pouvoir avec exactement ce niveau et cette discipline, tu PEUX
   répondre en nommant ce pouvoir, suivi de sa description prise dans les
   extraits.

Réponds STRICTEMENT en JSON, sans Markdown, avec exactement cette structure :
{{"answer": "texte avec citations [1]", "citations": [{{"id": 1, "chunk_index": 1, "quote": "court extrait verbatim"}}], "follow_ups": ["question courte ?"]}}
"""

    cards_block = ""
    if entity_cards:
        cards_block = "CARTES D'ENTITÉS DU CORPUS (référence structurée, ne pas citer directement) :\n\n"
        cards_block += "\n\n".join(entity_cards)
        cards_block += "\n"

    extracts: list[str] = ["EXTRAITS DISPONIBLES :", ""]
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {}) or {}
        bits: list[str] = [f"source: {meta.get('source', '?')}"]
        page = meta.get("page")
        if page and int(page) > 0:
            bits.append(f"p.{page}")
        section = meta.get("section")
        if section:
            bits.append(f"section: {section}")
        text = strip_pdf_boilerplate(c.get("text", ""))
        extracts.append(f"=== EXTRAIT {i} ({', '.join(bits)}) ===")
        extracts.append(text.strip())
        extracts.append("")

    parts = [rules, ""]
    if cards_block:
        parts.append(cards_block)
    parts.extend(["\n".join(extracts), "QUESTION DE L'UTILISATEUR :", question.strip()])
    return "\n\n".join(parts)


# ─── Parsing tolérant de la réponse JSON ─────────────────────────────────────


def _extract_json_blob(text: str) -> Optional[dict]:
    """Tente de récupérer un objet JSON même si Gemini a entouré de texte."""
    if not text:
        return None
    text = text.strip()
    # 1) JSON pur
    try:
        return json.loads(text)
    except Exception:
        pass
    # 2) Bloc markdown ```json ... ```
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    # 3) Premier { ... } équilibré (best-effort)
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    return None


def _normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", strip_pdf_boilerplate(text or "")).strip().lower()


def _find_quote_chunk_index(quote: str, chunks: list[dict]) -> Optional[int]:
    quote_norm = _normalize_quote(quote)
    if len(quote_norm) < 20:
        return None
    for idx, chunk in enumerate(chunks, start=1):
        text_norm = _normalize_quote(chunk.get("text", ""))
        if quote_norm in text_norm:
            return idx
    return None


def _fallback_quote(chunk_text: str, max_words: int = 45) -> str:
    words = re.sub(r"\s+", " ", strip_pdf_boilerplate(chunk_text or "")).strip().split()
    return " ".join(words[:max_words])


def _validate_and_normalize(
    data: dict, chunks: list[dict]
) -> dict:
    """Valide la structure et garantit que les chunk_index sont valides."""
    answer = strip_pdf_boilerplate(str(data.get("answer", ""))).strip()
    citations_raw = data.get("citations") or []
    follow_ups_raw = data.get("follow_ups") or []

    citations: list[dict] = []
    seen_ids: set[int] = set()
    for c in citations_raw:
        if not isinstance(c, dict):
            continue
        try:
            cid = int(c.get("id"))
            chunk_idx = int(c.get("chunk_index"))
        except (TypeError, ValueError):
            continue
        if chunk_idx < 1 or chunk_idx > len(chunks):
            continue
        if cid in seen_ids:
            continue
        quote = strip_pdf_boilerplate(str(c.get("quote", ""))).strip()
        corrected_idx = _find_quote_chunk_index(quote, chunks)
        if corrected_idx is not None:
            chunk_idx = corrected_idx
        seen_ids.add(cid)
        chunk = chunks[chunk_idx - 1]
        chunk_text = strip_pdf_boilerplate(chunk.get("text", ""))
        if quote and _normalize_quote(quote) not in _normalize_quote(chunk_text):
            quote = _fallback_quote(chunk_text)
        elif not quote:
            quote = _fallback_quote(chunk_text)
        citations.append(
            {
                "id": cid,
                "chunk_index": chunk_idx,
                "quote": quote,
                "metadata": chunk.get("metadata", {}) or {},
                "chunk_id": chunk.get("chunk_id", ""),
                "chunk_text": chunk_text,
            }
        )

    follow_ups: list[str] = []
    for q in follow_ups_raw:
        if not isinstance(q, str):
            continue
        q = strip_pdf_boilerplate(q).strip()
        if q and q not in follow_ups:
            follow_ups.append(q)
        if len(follow_ups) >= 3:
            break

    return {
        "answer": answer,
        "citations": citations,
        "follow_ups": follow_ups,
    }


# ─── Wrapper Gemini ──────────────────────────────────────────────────────────


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Retourne le delai conseille par le provider quand il renvoie 429."""
    candidates: list[float] = []
    for name in (
        "retry-after",
        "x-ratelimit-reset-tokens-minute",
        "x-ratelimit-reset-requests-minute",
    ):
        raw = response.headers.get(name)
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0:
            candidates.append(value)
    if not candidates:
        return None
    return min(candidates)


class GeminiGenerator:
    """Génération NotebookLM-style via le provider LLM configuré."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model_name: str,
        timeout_seconds: int = 30,
        max_output_tokens: int = 1500,
        executor: Optional[ThreadPoolExecutor] = None,
        max_concurrent: int = 1,
        min_interval_s: float = 0.0,
    ):
        self._provider = provider
        self._api_key = api_key
        self._model_name = model_name
        self._timeout = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._executor = executor or ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="llm"
        )
        # Gate partagée pour serializer tous les appels LLM (cross-/ask).
        global _LLM_GATE
        if _LLM_GATE is None:
            _LLM_GATE = LLMGate(max_concurrent=max_concurrent, min_interval_s=min_interval_s)
        self._gate = _LLM_GATE

    def _is_gemma(self) -> bool:
        """Gemma ne supporte pas response_schema/JSON mode (400 InvalidArgument)."""
        return "gemma" in (self._model_name or "").lower()

    def _generate_sync(self, prompt: str) -> str:
        if self._provider in {"groq", "cerebras", "mistral"}:
            return self._generate_openai_compatible_sync(prompt, max_tokens=self._max_output_tokens)

        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model_name)
        kwargs = dict(
            temperature=0.15,
            max_output_tokens=self._max_output_tokens,
        )
        if not self._is_gemma():
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = RESPONSE_SCHEMA
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(**kwargs),
            request_options={"timeout": self._timeout},
        )
        return (response.text or "").strip()

    def _generate_variants_sync(self, prompt: str) -> str:
        if self._provider in {"groq", "cerebras", "mistral"}:
            return self._generate_openai_compatible_sync(prompt, max_tokens=400)

        import google.generativeai as genai

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model_name)
        kwargs = dict(
            temperature=0.5,
            max_output_tokens=400,
        )
        if not self._is_gemma():
            kwargs["response_mime_type"] = "application/json"
            kwargs["response_schema"] = VARIANTS_SCHEMA
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(**kwargs),
            request_options={"timeout": self._timeout},
        )
        return (response.text or "").strip()

    def _generate_openai_compatible_sync(self, prompt: str, *, max_tokens: int) -> str:
        if self._provider == "groq":
            endpoint = "https://api.groq.com/openai/v1/chat/completions"
            token_field = "max_tokens"
        elif self._provider == "cerebras":
            endpoint = "https://api.cerebras.ai/v1/chat/completions"
            token_field = "max_completion_tokens"
        elif self._provider == "mistral":
            endpoint = "https://api.mistral.ai/v1/chat/completions"
            token_field = "max_tokens"
        else:
            raise RuntimeError(f"Provider OpenAI-compatible non supporte: {self._provider}")

        payload = {
            "model": self._model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Tu es une API de règles JDR. Réponds uniquement avec un objet JSON valide, "
                        "sans Markdown, sans texte avant ou après le JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.15,
            token_field: max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=self._timeout) as client:
            for attempt in range(2):
                response = client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                )
                if response.status_code == 429:
                    retry_after = _parse_retry_after(response)
                    logger.warning(
                        "Rate limit %s (%s). retry_after=%s body=%s",
                        self._provider,
                        self._model_name,
                        retry_after,
                        response.text[:500],
                    )
                    if attempt == 0 and retry_after is not None and retry_after <= 75:
                        time.sleep(max(1.0, retry_after + 0.5))
                        continue
                    raise LLMRateLimitError(self._provider, retry_after)
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    logger.error(
                        "Erreur HTTP %s (%s): %s",
                        self._provider,
                        response.status_code,
                        response.text[:500],
                    )
                    raise
                data = response.json()
                break
        return (data["choices"][0]["message"]["content"] or "").strip()

    async def generate_query_variants(self, question: str) -> list[str]:
        """Génère 3 reformulations LLM de la question pour multi-query retrieval.

        Renvoie une liste vide si Gemini échoue, le retriever continuera avec
        la query originale seule.
        """
        cfg = get_config()
        prompt = _build_variants_prompt(question, cfg.asso_name)
        loop = asyncio.get_running_loop()
        try:
            async with self._gate:
                raw = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._generate_variants_sync, prompt),
                    timeout=self._timeout + 90,
                )
        except Exception:
            logger.exception("Échec génération de variantes")
            return []

        data = _extract_json_blob(raw) or {}
        variants_raw = data.get("variants") or []
        out: list[str] = []
        seen: set[str] = set()
        for v in variants_raw:
            if not isinstance(v, str):
                continue
            v = v.strip()
            if v and v.lower() not in seen and v.lower() != question.strip().lower():
                seen.add(v.lower())
                out.append(v)
            if len(out) >= 3:
                break
        return out

    async def generate(
        self,
        question: str,
        chunks: list[dict],
        *,
        entity_cards: list[str] | None = None,
    ) -> dict:
        cfg = get_config()

        if not chunks:
            return {
                "answer": NO_ANSWER_SENTINEL,
                "citations": [],
                "follow_ups": [],
                "chunks_used": 0,
            }

        prompt = _build_prompt(question, chunks, cfg.asso_name, entity_cards=entity_cards)
        loop = asyncio.get_running_loop()

        try:
            async with self._gate:
                raw = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, self._generate_sync, prompt),
                    timeout=self._timeout + 90,
                )
        except asyncio.TimeoutError:
            logger.warning("Timeout LLM sur la question: %s", question[:80])
            raise

        data = _extract_json_blob(raw)
        if data is None:
            logger.error("Le LLM n'a pas renvoyé de JSON exploitable: %r", raw[:300])
            return {
                "answer": "Désolé, ma réponse était mal formée. Réessaie en reformulant.",
                "citations": [],
                "follow_ups": [],
                "chunks_used": len(chunks),
            }

        normalized = _validate_and_normalize(data, chunks)
        normalized["chunks_used"] = len(chunks)
        return normalized


# ─── Singleton ───────────────────────────────────────────────────────────────


_GENERATOR_SINGLETON: Optional[GeminiGenerator] = None


def get_generator() -> GeminiGenerator:
    global _GENERATOR_SINGLETON
    if _GENERATOR_SINGLETON is None:
        cfg = get_config()
        if cfg.llm_provider == "groq":
            api_key = cfg.groq_api_key
            model_name = cfg.groq_model
        elif cfg.llm_provider == "cerebras":
            api_key = cfg.cerebras_api_key
            model_name = cfg.cerebras_model
        elif cfg.llm_provider == "mistral":
            api_key = cfg.mistral_api_key
            model_name = cfg.mistral_model
        else:
            api_key = cfg.gemini_api_key
            model_name = cfg.gemini_model
        _GENERATOR_SINGLETON = GeminiGenerator(
            provider=cfg.llm_provider,
            api_key=api_key,
            model_name=model_name,
            timeout_seconds=cfg.gemini_timeout_seconds,
            max_output_tokens=cfg.gemini_max_output_tokens,
            max_concurrent=cfg.llm_max_concurrent,
            min_interval_s=cfg.llm_min_interval_seconds,
        )
    return _GENERATOR_SINGLETON
