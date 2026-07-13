
# Main entry point for the Indexer pipeline.

# Wires together model.py → embedder.py → store.py into a single callable function and a CLI.

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

# Allow running as `python indexer/index.py` OR `python -m indexer.index`
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from indexer.model import get_model, get_device
from indexer.embedder import discover_images, embed_images
from indexer.store import get_collection, get_existing_ids, upsert_batch, collection_size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)



# Public API

def build_index(
    dataset_dir: str = config.DATASET_DIR,
    persist_dir: str = config.CHROMA_PERSIST_DIR,
    batch_size: int = config.BATCH_SIZE,
) -> int:
    if not (1 <= batch_size <= 512):
        raise ValueError(f"batch_size must be 1–512, got {batch_size}")

    
    # Step 1: discover images
    logger.info("=== Indexer starting ===")
    logger.info("Dataset : %s", dataset_dir)
    logger.info("Store   : %s", persist_dir)
    logger.info("Batch   : %d", batch_size)

    all_images = discover_images(dataset_dir)
    total_discovered = len(all_images)

    # Step 2: load model
    device = get_device()
    logger.info("Device  : %s", device)
    model, preprocess, _ = get_model(device=device)

    # Step 3: open ChromaDB collection
    collection = get_collection(persist_dir=persist_dir)

    # Step 4: determine which images are already indexed
    existing_ids = get_existing_ids(collection)
    logger.info(
        "Already indexed: %d / %d images — %d to embed",
        len(existing_ids),
        total_discovered,
        total_discovered - len(existing_ids),
    )

    # Filter to only unindexed images before running the (expensive) GPU pass
    images_to_process = [
        p for p in all_images if p.stem not in existing_ids
    ]

    if not images_to_process:
        logger.info("Nothing to do — all images are already indexed.")
        return 0

    # Step 5: embed and upsert in batches
    total_inserted = 0
    total_skipped = 0        # skipped inside upsert (race condition safety)
    total_unreadable = 0

    n_batches = (len(images_to_process) + batch_size - 1) // batch_size

    with tqdm(total=len(images_to_process), desc="Indexing", unit="img") as pbar:
        for batch_paths, batch_embeddings in embed_images(
            image_paths=images_to_process,
            model=model,
            preprocess=preprocess,
            device=device,
            batch_size=batch_size,
        ):
            # Detect images that were skipped inside embed_images (corrupt)
            # by comparing expected vs actual batch sizes.
            unreadable_in_batch = 0  # embed_images already logged these

            inserted, skipped = upsert_batch(
                collection=collection,
                paths=batch_paths,
                embeddings=batch_embeddings,
                existing_ids=existing_ids,
            )

            # Update existing_ids set so the next batch doesn't re-insert
            for p in batch_paths:
                existing_ids.add(p.stem)

            total_inserted += inserted
            total_skipped += skipped
            pbar.update(len(batch_paths) + unreadable_in_batch)

    # Step 6: summary
    final_size = collection_size(collection)
    logger.info("=== Indexing complete ===")
    logger.info("  Newly inserted : %d", total_inserted)
    logger.info("  Skipped (dup)  : %d", total_skipped)
    logger.info("  Collection size: %d", final_size)

    return total_inserted



# CLI
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m indexer.index",
        description="Index fashion images into a ChromaDB vector store using NegCLIP.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=config.DATASET_DIR,
        help=f"Directory of .jpg images to index. (default: {config.DATASET_DIR})",
    )
    parser.add_argument(
        "--persist-dir",
        default=config.CHROMA_PERSIST_DIR,
        help=f"ChromaDB persistence directory. (default: {config.CHROMA_PERSIST_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help=f"Images per forward pass (1–512). (default: {config.BATCH_SIZE})",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    n = build_index(
        dataset_dir=args.dataset_dir,
        persist_dir=args.persist_dir,
        batch_size=args.batch_size,
    )
    print(f"\nDone. {n} images indexed.")
