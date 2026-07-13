
#ChromaDB persistence layer for the Indexer.

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)

# ChromaDB hard limit on upsert batch size
_CHROMA_MAX_BATCH = 512



# Collection management
def get_collection(
    persist_dir: str = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.CHROMA_COLLECTION_NAME,
) -> chromadb.Collection:

    # Return a persistent ChromaDB collection configured for cosine similarity.
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity for ANN search
    )

    logger.info(
        "ChromaDB collection '%s' opened at %s  (current size: %d)",
        collection_name,
        persist_dir,
        collection.count(),
    )
    return collection


# Incremental indexing helpers
def get_existing_ids(collection: chromadb.Collection) -> set[str]:
    #Return the set of document IDs already present in the collection.
    existing: set[str] = set()
    page_size = 10_000
    offset = 0

    while True:
        result = collection.get(
            limit=page_size,
            offset=offset,
            include=[],          # IDs only — no embeddings or metadata
        )
        ids = result.get("ids", [])
        if not ids:
            break
        existing.update(ids)
        offset += len(ids)
        if len(ids) < page_size:
            break

    logger.debug("Found %d existing IDs in collection.", len(existing))
    return existing


# Upsert
def upsert_batch(
    collection: chromadb.Collection,
    paths: List[Path],
    embeddings: List[List[float]],
    existing_ids: set[str] | None = None,
) -> tuple[int, int]:

    #Upsert a batch of image embeddings into *collection*.
    if len(paths) != len(embeddings):
        raise ValueError(
            f"paths ({len(paths)}) and embeddings ({len(embeddings)}) must be "
            "the same length."
        )

    ids_to_add: List[str] = []
    embs_to_add: List[List[float]] = []
    meta_to_add: List[dict] = []
    skipped = 0

    for path, emb in zip(paths, embeddings):
        doc_id = path.stem  # filename without extension
        if existing_ids is not None and doc_id in existing_ids:
            logger.debug("Skipping duplicate ID: %s", doc_id)
            skipped += 1
            continue
        ids_to_add.append(doc_id)
        embs_to_add.append(emb)
        meta_to_add.append({"file_path": str(path)})

    if not ids_to_add:
        return 0, skipped

    # Upsert in sub-batches to respect ChromaDB's internal size limit
    inserted = 0
    for start in range(0, len(ids_to_add), _CHROMA_MAX_BATCH):
        end = start + _CHROMA_MAX_BATCH
        collection.upsert(
            ids=ids_to_add[start:end],
            embeddings=embs_to_add[start:end],
            metadatas=meta_to_add[start:end],
        )
        inserted += len(ids_to_add[start:end])

    return inserted, skipped


def collection_size(collection: chromadb.Collection) -> int:
    #Return the total number of documents stored in *collection*.
    return collection.count()
