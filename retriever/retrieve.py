
# Main entry point for the Retriever pipeline.


from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from indexer.model import get_model, get_device
from indexer.store import get_collection
from retriever.search import search, SearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Cached model/collection loader (module-level singletons)
# For interactive / API use: calling retrieve() multiple times should NOT
# reload the model each time.  We cache the heavy objects at module level
# behind a simple flag.

_model = None
_tokenizer = None
_device = None
_collection = None


def _ensure_loaded(persist_dir: str) -> None:
    #Load model and collection into module-level singletons if not yet done.
    global _model, _tokenizer, _device, _collection

    if _model is None:
        _device = get_device()
        logger.info("Loading NegCLIP model on %s …", _device)
        _model, _, _tokenizer = get_model(device=_device)
        logger.info("Model loaded.")

    if _collection is None or _collection._client._settings.chroma_db_impl != persist_dir:
        logger.info("Opening ChromaDB collection at %s …", persist_dir)
        _collection = get_collection(persist_dir=persist_dir)


# Public API
def retrieve(
    query: str,
    k: int = config.TOP_K,
    persist_dir: str = config.CHROMA_PERSIST_DIR,
) -> List[SearchResult]:

    # Search the indexed image collection with a natural-language *query*.
    _ensure_loaded(persist_dir)
    return search(
        query=query,
        model=_model,
        tokenizer=_tokenizer,
        collection=_collection,
        device=_device,
        k=k,
    )


# CLI
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retriever.retrieve",
        description=(
            "Query the NegCLIP fashion image search engine.\n\n"
            "Example evaluation queries:\n"
            "  'A person in a bright yellow raincoat.'\n"
            "  'Professional business attire inside a modern office.'\n"
            "  'Someone wearing a blue shirt sitting on a park bench.'\n"
            "  'Casual weekend outfit for a city walk.'\n"
            "  'A red tie and a white shirt in a formal setting.'"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Natural language search query (required).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        help=f"Number of results to return (1–1000). (default: {config.TOP_K})",
    )
    parser.add_argument(
        "--persist-dir",
        default=config.CHROMA_PERSIST_DIR,
        help=(
            f"ChromaDB persistence directory. "
            f"(default: {config.CHROMA_PERSIST_DIR})"
        ),
    )
    return parser


def _print_results(results: List[SearchResult], query: str) -> None:
    # Pretty-print ranked results to stdout.
    print(f"\nQuery : {query}")
    print(f"Results ({len(results)} found):")
    print("-" * 72)
    for r in results:
        print(f"  {r['rank']:3d}.  score={r['score']:.4f}  {r['file_path']}")
    print("-" * 72)


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not (1 <= args.top_k <= 1000):
        parser.error("--top-k must be between 1 and 1000.")

    results = retrieve(
        query=args.query,
        k=args.top_k,
        persist_dir=args.persist_dir,
    )

    _print_results(results, args.query)
