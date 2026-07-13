
# Core search logic: wires decomposer → encoder → ChromaDB query → ranked results.


from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from retriever.decomposer import decompose_query
from retriever.encoder import encode_query

logger = logging.getLogger(__name__)

# Type alias for a single search result record
SearchResult = Dict[str, Any]


def search(
    query: str,
    model: torch.nn.Module,
    tokenizer,
    collection,
    device: torch.device,
    k: int = config.TOP_K,
) -> List[SearchResult]:

    # Retrieve the top-k images most similar to *query*.

    # Input validation

    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    if len(query) > 512:
        raise ValueError(
            f"Query exceeds 512 characters (got {len(query)})."
        )
    if not (1 <= k <= 1000):
        raise ValueError(f"k must be between 1 and 1000, got {k}.")

    n_docs = collection.count()
    if n_docs == 0:
        raise RuntimeError(
            "ChromaDB collection is empty. Run the Indexer first."
        )

    # Clamp k to collection size to avoid ChromaDB raising an error
    effective_k = min(k, n_docs)
    if effective_k < k:
        logger.info(
            "Collection has only %d images; returning all of them (k=%d requested).",
            n_docs,
            k,
        )

    # Step 1: decompose the query into sub-segments
    segments = decompose_query(query)
    logger.info(
        "Query decomposed into %d segment(s): %s", len(segments), segments
    )


    # Step 2 & 3: encode segments → composite query vector
    query_embedding = encode_query(
        segments=segments,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

    # ChromaDB expects a list of lists (one query vector per query)
    query_vector: List[List[float]] = [query_embedding.cpu().float().tolist()]

    # Step 4: nearest-neighbour search in ChromaDB
    results = collection.query(
        query_embeddings=query_vector,
        n_results=effective_k,
        include=["metadatas", "distances"],
    )

    # results["ids"][0]        → list of doc IDs for query 0
    # results["distances"][0]  → list of cosine distances for query 0
    # results["metadatas"][0]  → list of metadata dicts for query 0
    ids: List[str] = results["ids"][0]
    distances: List[float] = results["distances"][0]
    metadatas: List[dict] = results["metadatas"][0]

    # Step 5: convert distances to similarity scores and assemble output
    ranked: List[SearchResult] = []
    for rank_idx, (doc_id, dist, meta) in enumerate(
        zip(ids, distances, metadatas), start=1
    ):
        # ChromaDB cosine distance = 1 - cosine_similarity
        score = float(1.0 - dist)
        ranked.append(
            {
                "rank": rank_idx,
                "file_path": meta.get("file_path", doc_id),
                "image_id": doc_id,
                "score": round(score, 6),
            }
        )

    logger.info(
        "Query '%s' → top-%d results, best score=%.4f",
        query[:60],
        len(ranked),
        ranked[0]["score"] if ranked else float("nan"),
    )

    return ranked
