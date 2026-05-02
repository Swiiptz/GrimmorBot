"""Script CLI d'ingestion des documents dans ChromaDB et BM25.

Usage : `python -m rag.ingest`
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from config import get_config
from rag.bm25_index import BM25Index
from rag.chunker import Chunk, chunk_file, iter_documents

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH = 32
COLLECTION_NAME = "grimmor_rules"
STATE_FILENAME = "ingest_state.json"
BM25_FILENAME = "bm25_index.pkl"


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {"hashes": {}}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {"hashes": {}}


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _chunk_id(rel_path: str, idx: int) -> str:
    return f"{rel_path}::{idx}"


def _batched(seq: list, size: int) -> Iterable[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )
    cfg = get_config()
    docs_root = cfg.docs_path
    chroma_path = cfg.chroma_path

    if not docs_root.exists():
        logger.error("Le dossier de documents n'existe pas: %s", docs_root)
        return 1

    chroma_path.mkdir(parents=True, exist_ok=True)
    state_path = chroma_path / STATE_FILENAME
    bm25_path = chroma_path / BM25_FILENAME
    state = _load_state(state_path)
    old_hashes: dict[str, str] = state.get("hashes", {})
    new_hashes: dict[str, str] = {}

    documents = list(iter_documents(docs_root))
    if not documents:
        logger.warning("Aucun document supporté dans %s", docs_root)
        return 0

    logger.info("Détection : %d document(s) sous %s", len(documents), docs_root)

    # ─── Chunking + détection de changement ───────────────────────────────
    files_to_index: list[tuple[Path, list[Chunk]]] = []
    skipped_files: list[Path] = []
    for path in tqdm(documents, desc="Hash/chunking", unit="fichier"):
        rel = str(path.relative_to(docs_root))
        h = _file_hash(path)
        new_hashes[rel] = h
        if old_hashes.get(rel) == h:
            skipped_files.append(path)
            continue
        chunks = chunk_file(path)
        if not chunks:
            logger.warning("Aucun chunk extrait de %s", path)
            continue
        files_to_index.append((path, chunks))

    # Documents supprimés (présents dans l'ancien état mais plus sur disque)
    removed_files = [rel for rel in old_hashes if rel not in new_hashes]

    logger.info(
        "À (ré)indexer: %d fichier(s) — inchangés: %d — supprimés: %d",
        len(files_to_index),
        len(skipped_files),
        len(removed_files),
    )

    # ─── Connexion ChromaDB ───────────────────────────────────────────────
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ─── Suppression des chunks des fichiers modifiés/supprimés ───────────
    sources_to_purge = [p.name for p, _ in files_to_index] + [
        Path(rel).name for rel in removed_files
    ]
    if sources_to_purge:
        for src in set(sources_to_purge):
            try:
                collection.delete(where={"source": src})
            except Exception:
                logger.exception("Échec suppression des anciens chunks pour %s", src)

    # ─── Embedding + insertion ────────────────────────────────────────────
    if files_to_index:
        from sentence_transformers import SentenceTransformer

        logger.info("Chargement du modèle d'embedding %s ...", EMBEDDING_MODEL_NAME)
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # Aplatit tous les chunks pour batcher proprement
        flat: list[tuple[str, Chunk]] = []
        for path, chunks in files_to_index:
            rel = str(path.relative_to(docs_root)).replace("\\", "/")
            for ch in chunks:
                cid = _chunk_id(rel, ch.chunk_index)
                flat.append((cid, ch))

        logger.info("Embedding et insertion: %d chunks", len(flat))
        for batch in tqdm(
            list(_batched(flat, EMBEDDING_BATCH)),
            desc="Embedding",
            unit="batch",
        ):
            ids = [cid for cid, _ in batch]
            texts = [ch.text for _, ch in batch]
            metas = [ch.to_metadata() for _, ch in batch]
            embeddings = model.encode(
                texts,
                batch_size=EMBEDDING_BATCH,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metas,
                embeddings=embeddings,
            )

    # ─── Reconstruction complète de l'index BM25 depuis ChromaDB ──────────
    logger.info("Reconstruction de l'index BM25 (corpus complet)")
    full = collection.get(include=["documents", "metadatas"])
    all_ids = full.get("ids", []) or []
    all_docs = full.get("documents", []) or []
    bm25 = BM25Index()
    bm25.build(all_ids, all_docs)
    bm25.save(bm25_path)

    # ─── Sauvegarde de l'état ─────────────────────────────────────────────
    _save_state(state_path, {"hashes": new_hashes})
    logger.info("Ingestion terminée. Total dans la collection : %d chunks", len(all_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
