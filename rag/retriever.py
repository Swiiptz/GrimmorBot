"""Recherche hybride dense (ChromaDB) + sparse (BM25) avec fusion RRF.

Supporte un filtre par fichier source (pour l'option /ask source:).
Les requetes utilisateur sont envoyees telles quelles ; les enrichissements
semantiques doivent venir du graphe extrait du PDF ou du multi-query LLM.
"""
from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from config import get_config
from rag.bm25_index import BM25Index, get_bm25_index
from rag.ingest import COLLECTION_NAME, EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)


_LEVEL_TO_BULLETS = {
    "1": "•", "2": "••", "3": "•••", "4": "••••", "5": "•••••",
    "un": "•", "une": "•",
    "deux": "••",
    "trois": "•••",
    "quatre": "••••",
    "cinq": "•••••",
    "i": "•", "ii": "••", "iii": "•••", "iv": "••••", "v": "•••••",
}

# Détecte « niveau 5 », « lvl 3 », « niv III », etc.
_LEVEL_RE = re.compile(
    r"\b(?:niveau|niv|lvl|level)\s*(?P<lvl>[1-5]|un|une|deux|trois|quatre|cinq|iv|iii|ii|v|i)\b",
    re.IGNORECASE,
)
_BULLET_RUN_RE = re.compile(r"(?:•\s*){1,5}")
_MATCH_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "comment", "quelle", "quelles", "quels", "quel", "quoi", "qui", "dont",
    "dans", "avec", "sans", "pour", "par", "sur", "les", "des", "une",
    "est", "sont", "fonctionne", "fonctionnent", "faire", "fait", "niveau",
    "niv", "lvl", "level", "du", "de", "la", "le", "un", "au", "aux", "et",
    "ou", "ce", "cet", "cette", "ces",
}


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower()


def _meaningful_terms(query: str) -> list[str]:
    normalized = _normalize_for_match(query)
    terms = []
    for token in _MATCH_TOKEN_RE.findall(normalized):
        if len(token) < 3 or token in _STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _query_levels(query: str) -> set[int]:
    levels: set[int] = set()
    for match in _LEVEL_RE.finditer(query):
        bullets = _LEVEL_TO_BULLETS.get(match.group("lvl").lower())
        if bullets:
            levels.add(len(bullets))
    return levels


def _text_levels(text: str) -> set[int]:
    levels: set[int] = set()
    for match in _BULLET_RUN_RE.finditer(text or ""):
        level = match.group(0).count("•")
        if 1 <= level <= 5:
            levels.add(level)
    return levels


def _lexical_boost(text: str, queries: list[str]) -> float:
    """Petit boost de reranking pour les chunks qui matchent vraiment la question."""
    normalized_text = _normalize_for_match(text)
    levels_in_text = _text_levels(text)
    best = 0.0

    for query in queries:
        normalized_query = _normalize_for_match(query).strip()
        terms = _meaningful_terms(query)
        score = 0.0

        if len(normalized_query) >= 6 and normalized_query in normalized_text:
            score += 0.08

        if terms:
            hits = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", normalized_text))
            ratio = hits / len(terms)
            score += 0.045 * ratio
            if hits == len(terms) and hits >= 2:
                score += 0.04

        if _query_levels(query) & levels_in_text:
            score += 0.04

        best = max(best, score)

    return min(best, 0.16)


class HybridRetriever:
    """Singleton coordonnant l'embedder, ChromaDB et BM25."""

    def __init__(
        self,
        chroma_path: Path,
        bm25_path: Path,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        self.chroma_path = chroma_path
        self.bm25_path = bm25_path
        self._executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag")
        self._model = None
        self._collection = None
        self._bm25: Optional[BM25Index] = None
        # Cache id -> source pour le filtrage BM25 (rempli à warmup)
        self._chunk_id_to_source: dict[str, str] = {}
        # Liste des fichiers indexés (pour autocomplete)
        self._indexed_sources: list[str] = []

    def _load_sync(self) -> None:
        from sentence_transformers import SentenceTransformer
        import chromadb

        logger.info("Chargement du modèle d'embedding %s ...", EMBEDDING_MODEL_NAME)
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        logger.info("Connexion ChromaDB persistant: %s", self.chroma_path)
        client = chromadb.PersistentClient(path=str(self.chroma_path))
        self._collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self._bm25 = get_bm25_index(self.bm25_path)

        # Pré-calcule l'index id->source et la liste des sources
        try:
            full = self._collection.get(include=["metadatas"])
            ids = full.get("ids", []) or []
            metas = full.get("metadatas", []) or []
            sources_set: set[str] = set()
            for cid, meta in zip(ids, metas):
                source = (meta or {}).get("source", "")
                if source:
                    self._chunk_id_to_source[cid] = source
                    sources_set.add(source)
            self._indexed_sources = sorted(sources_set)
        except Exception:
            logger.exception("Échec pré-calcul des sources indexées")

        logger.info(
            "Retriever prêt — Chroma: %d chunks, BM25: %d chunks, sources: %d",
            self._collection.count(),
            len(self._bm25.chunk_ids),
            len(self._indexed_sources),
        )

    async def warmup(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._load_sync)

    def list_sources(self) -> list[str]:
        """Liste des fichiers indexés (pour l'autocomplete /ask)."""
        return list(self._indexed_sources)

    def _fetch_missing_texts(self, ids: list[str], chunks_by_id: dict[str, dict]) -> None:
        missing = [
            cid for cid in ids
            if "dense" not in chunks_by_id.get(cid, {})
        ]
        if not missing or self._collection is None:
            return
        try:
            fetched = self._collection.get(
                ids=missing,
                include=["documents", "metadatas"],
            )
            for cid, doc, meta in zip(
                fetched.get("ids", []) or [],
                fetched.get("documents", []) or [],
                fetched.get("metadatas", []) or [],
            ):
                chunks_by_id.setdefault(cid, {})["dense"] = {
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta or {},
                    "rank": -1,
                    "distance": None,
                }
        except Exception:
            logger.exception("Échec récupération des textes BM25-only")

    def _rerank_with_lexical_boost(
        self,
        ranked: list[tuple[str, float]],
        chunks_by_id: dict[str, dict],
        queries: list[str],
    ) -> list[tuple[str, float]]:
        cfg = get_config()
        candidate_limit = max(cfg.rag_top_k_final * 10, 40)
        candidate_ids = [cid for cid, _ in ranked[:candidate_limit]]
        self._fetch_missing_texts(candidate_ids, chunks_by_id)

        reranked: list[tuple[str, float]] = []
        for position, (cid, score) in enumerate(ranked):
            if position >= candidate_limit:
                reranked.append((cid, score))
                continue
            entry = chunks_by_id.get(cid, {})
            dense = entry.get("dense") or {}
            text = dense.get("text", "")
            reranked.append((cid, score + _lexical_boost(text, queries)))

        return sorted(reranked, key=lambda kv: kv[1], reverse=True)

    @staticmethod
    def _next_chunk_id(chunk_id: str) -> Optional[str]:
        try:
            prefix, idx = chunk_id.rsplit("::", 1)
            return f"{prefix}::{int(idx) + 1}"
        except (ValueError, TypeError):
            return None

    def _select_with_neighbors(
        self,
        ranked: list[tuple[str, float]],
        chunks_by_id: dict[str, dict],
        queries: list[str],
        limit: int,
    ) -> list[tuple[str, float]]:
        selected: list[tuple[str, float]] = []
        seen: set[str] = set()

        for cid, score in ranked:
            if cid not in seen:
                selected.append((cid, score))
                seen.add(cid)

            entry = chunks_by_id.get(cid, {})
            dense = entry.get("dense") or {}
            text = dense.get("text", "")
            should_add_next = bool(_text_levels(text)) and _lexical_boost(text, queries) >= 0.08
            if should_add_next:
                next_id = self._next_chunk_id(cid)
                if next_id and next_id not in seen:
                    self._fetch_missing_texts([next_id], chunks_by_id)
                    if "dense" in chunks_by_id.get(next_id, {}):
                        selected.append((next_id, score * 0.98))
                        seen.add(next_id)

            if len(selected) >= limit:
                break

        return selected[:limit]

    # ─── Recherche dense ──────────────────────────────────────────────────

    def _dense_search_sync(
        self, query: str, top_k: int, source_filter: Optional[str]
    ) -> list[dict]:
        if self._model is None or self._collection is None:
            return []
        embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()[0]
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if source_filter:
            kwargs["where"] = {"source": source_filter}
        result = self._collection.query(**kwargs)
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        out: list[dict] = []
        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            out.append(
                {
                    "chunk_id": cid,
                    "text": doc,
                    "metadata": meta or {},
                    "rank": rank,
                    "distance": float(dist) if dist is not None else None,
                }
            )
        return out

    # ─── Recherche sparse ─────────────────────────────────────────────────

    def _sparse_search_sync(
        self, query: str, top_k: int, source_filter: Optional[str]
    ) -> list[dict]:
        if self._bm25 is None:
            return []
        # On demande plus large si on filtre, pour ne pas se retrouver à zéro après filtre
        oversample = top_k * 4 if source_filter else top_k
        results = self._bm25.search(query, top_k=oversample)
        if source_filter:
            kept: list[dict] = []
            for r in results:
                if self._chunk_id_to_source.get(r["chunk_id"]) == source_filter:
                    kept.append(r)
                if len(kept) >= top_k:
                    break
            # Re-numérote les rangs après filtrage
            for i, r in enumerate(kept):
                r["rank"] = i
            return kept
        return results

    # ─── Pipeline complet ─────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """Recherche hybride. Retourne au plus `RAG_TOP_K_FINAL` chunks fusionnés.

        Si `source_filter` est passé, ne retourne que les chunks de ce fichier.
        """
        cfg = get_config()
        loop = asyncio.get_running_loop()

        dense_task = loop.run_in_executor(
            self._executor,
            self._dense_search_sync,
            query,
            cfg.rag_top_k_dense,
            source_filter,
        )
        sparse_task = loop.run_in_executor(
            self._executor,
            self._sparse_search_sync,
            query,
            cfg.rag_top_k_sparse,
            source_filter,
        )
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # ─── Fusion RRF ───────────────────────────────────────────────────
        k = cfg.rag_rrf_k
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict] = {}

        for r in dense_results:
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r["rank"])
            chunks_by_id.setdefault(cid, {})["dense"] = r

        for r in sparse_results:
            cid = r["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r["rank"])
            chunks_by_id.setdefault(cid, {})["sparse"] = r

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        ranked = self._rerank_with_lexical_boost(
            ranked,
            chunks_by_id,
            [query],
        )
        top_score = ranked[0][1]
        if top_score < cfg.rag_min_rrf_score:
            logger.info(
                "Aucun chunk au-dessus du seuil RRF (top=%.4f, seuil=%.4f)",
                top_score,
                cfg.rag_min_rrf_score,
            )
            return []

        selected = self._select_with_neighbors(
            ranked,
            chunks_by_id,
            [query],
            cfg.rag_top_k_final,
        )
        self._fetch_missing_texts([cid for cid, _ in selected], chunks_by_id)

        final: list[dict] = []
        for cid, score in selected:
            entry = chunks_by_id.get(cid, {})
            dense = entry.get("dense")
            if dense is None:
                continue
            final.append(
                {
                    "chunk_id": cid,
                    "text": dense.get("text", ""),
                    "metadata": dense.get("metadata", {}),
                    "rrf_score": score,
                }
            )
        return final


    async def search_multi(
        self,
        queries: list[str],
        *,
        source_filter: Optional[str] = None,
    ) -> list[dict]:
        """Recherche hybride multi-query : fusion RRF sur N variantes.

        Chaque variante de query est expansée puis envoyée à dense+sparse.
        Tous les rangs sont fusionnés en RRF (additif).
        """
        if not queries:
            return []
        if len(queries) == 1:
            return await self.search(queries[0], source_filter=source_filter)

        cfg = get_config()
        loop = asyncio.get_running_loop()

        scores: dict[str, float] = {}
        coverage: dict[str, int] = {}
        chunks_by_id: dict[str, dict] = {}
        k = cfg.rag_rrf_k

        for q in queries:
            dense_task = loop.run_in_executor(
                self._executor, self._dense_search_sync,
                q, cfg.rag_top_k_dense, source_filter,
            )
            sparse_task = loop.run_in_executor(
                self._executor, self._sparse_search_sync,
                q, cfg.rag_top_k_sparse, source_filter,
            )
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

            query_scores: dict[str, float] = {}
            for r in dense_results:
                cid = r["chunk_id"]
                query_scores[cid] = query_scores.get(cid, 0.0) + 1.0 / (k + r["rank"])
                chunks_by_id.setdefault(cid, {})["dense"] = r
            for r in sparse_results:
                cid = r["chunk_id"]
                query_scores[cid] = query_scores.get(cid, 0.0) + 1.0 / (k + r["rank"])
                chunks_by_id.setdefault(cid, {})["sparse"] = r

            for cid, score in query_scores.items():
                scores[cid] = max(scores.get(cid, 0.0), score)
                coverage[cid] = coverage.get(cid, 0) + 1

        if not scores:
            return []

        ranked = sorted(
            (
                (cid, score + min(max(coverage.get(cid, 1) - 1, 0), 3) * 0.003)
                for cid, score in scores.items()
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
        ranked = self._rerank_with_lexical_boost(ranked, chunks_by_id, queries)
        if ranked[0][1] < cfg.rag_min_rrf_score:
            return []

        selected = self._select_with_neighbors(
            ranked,
            chunks_by_id,
            queries,
            cfg.rag_top_k_final,
        )
        self._fetch_missing_texts([cid for cid, _ in selected], chunks_by_id)

        final: list[dict] = []
        for cid, score in selected:
            entry = chunks_by_id.get(cid, {})
            dense = entry.get("dense")
            if dense is None:
                continue
            final.append({
                "chunk_id": cid,
                "text": dense.get("text", ""),
                "metadata": dense.get("metadata", {}),
                "rrf_score": score,
            })
        return final


_RETRIEVER_SINGLETON: Optional[HybridRetriever] = None


async def get_retriever() -> HybridRetriever:
    global _RETRIEVER_SINGLETON
    if _RETRIEVER_SINGLETON is None:
        cfg = get_config()
        bm25_path = cfg.chroma_path / "bm25_index.pkl"
        _RETRIEVER_SINGLETON = HybridRetriever(cfg.chroma_path, bm25_path)
        await _RETRIEVER_SINGLETON.warmup()
    return _RETRIEVER_SINGLETON
