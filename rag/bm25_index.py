"""Index BM25 persistant pour la recherche sparse côté RAG."""
from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Tokenisation : lowercase, split sur espaces et ponctuation.
# On garde les bullets (•) parce que les manuels JDR notent les niveaux de pouvoir
# avec des points empilés (ex : « Animalisme ••••• » = niveau 5).
_TOKEN_RE = re.compile(r"[•A-Za-zÀ-ÖØ-öø-ÿ0-9]+")
_BULLET_RUN_RE = re.compile(r"(?:•\s*){1,5}")


def _bullet_level_tokens(text: str) -> list[str]:
    """Convertit `• • • • •` en token `niveau5` pour BM25."""
    tokens: list[str] = []
    for match in _BULLET_RUN_RE.finditer(text):
        level = match.group(0).count("•")
        if 1 <= level <= 5:
            tokens.append(f"niveau{level}")
    return tokens


def tokenize(text: str) -> list[str]:
    out: list[str] = _bullet_level_tokens(text)
    for t in _TOKEN_RE.findall(text):
        t = t.lower()
        # Tokens à 1 caractère : on garde les séquences de bullets, on jette le reste.
        if len(t) == 1 and t != "•":
            continue
        out.append(t)
    return out


class BM25Index:
    """Wrapper persistant autour de BM25Okapi.

    Le corpus est stocké aux côtés des chunk_ids et des métadonnées
    nécessaires pour fusionner avec ChromaDB.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: list[str] = []
        self.tokenized_corpus: list[list[str]] = []

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids et texts doivent avoir la même longueur")
        self.chunk_ids = list(chunk_ids)
        self.tokenized_corpus = [tokenize(t) for t in texts]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self.bm25 is None or not self.chunk_ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        # Top-k indices triés par score décroissant
        if len(scores) <= top_k:
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        else:
            # argpartition pour rapidité, puis tri sur le sous-ensemble
            import numpy as np

            part = np.argpartition(scores, -top_k)[-top_k:]
            indices = sorted(part.tolist(), key=lambda i: scores[i], reverse=True)
        results: list[dict] = []
        for rank, idx in enumerate(indices):
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(
                {
                    "chunk_id": self.chunk_ids[idx],
                    "score": score,
                    "rank": rank,
                }
            )
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunk_ids": self.chunk_ids,
                    "tokenized_corpus": self.tokenized_corpus,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("Index BM25 sauvegardé (%d chunks) -> %s", len(self.chunk_ids), path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        idx = cls()
        if not path.exists():
            logger.warning("Pickle BM25 introuvable: %s", path)
            return idx
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx.chunk_ids = list(data.get("chunk_ids", []))
        idx.tokenized_corpus = list(data.get("tokenized_corpus", []))
        if idx.tokenized_corpus:
            idx.bm25 = BM25Okapi(idx.tokenized_corpus)
        return idx


_BM25_SINGLETON: Optional[BM25Index] = None


def get_bm25_index(path: Path | None = None) -> BM25Index:
    """Singleton chargé une seule fois au démarrage du bot."""
    global _BM25_SINGLETON
    if _BM25_SINGLETON is None:
        if path is None:
            raise RuntimeError("Premier appel à get_bm25_index sans chemin.")
        _BM25_SINGLETON = BM25Index.load(path)
    return _BM25_SINGLETON


def reset_bm25_singleton() -> None:
    """Force le rechargement (utilisé après une ingestion en live)."""
    global _BM25_SINGLETON
    _BM25_SINGLETON = None
