

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)



# Public helpers

def discover_images(dataset_dir: str) -> List[Path]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"Path is not a directory: {root}")

    extensions = {ext.lower() for ext in config.IMAGE_EXTENSIONS}
    images = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        raise ValueError(
            f"No images found in {root} "
            f"(looked for extensions: {config.IMAGE_EXTENSIONS})"
        )

    logger.info("Discovered %d images in %s", len(images), root)
    return images


def embed_images(
    image_paths: List[Path],
    model: torch.nn.Module,
    preprocess,
    device: torch.device,
    batch_size: int = config.BATCH_SIZE,
) -> Generator[Tuple[List[Path], List[List[float]]], None, None]:
    if not (1 <= batch_size <= 512):
        raise ValueError(f"batch_size must be between 1 and 512, got {batch_size}")

    total = len(image_paths)
    skipped_total = 0

    for start in range(0, total, batch_size):
        chunk = image_paths[start: start + batch_size]
        valid_paths: List[Path] = []
        tensors: List[torch.Tensor] = []

        for path in chunk:
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(preprocess(img))
                valid_paths.append(path)
            except (UnidentifiedImageError, OSError, Exception) as exc:
                logger.warning("Skipping unreadable image %s: %s", path.name, exc)
                skipped_total += 1
                continue

        if not valid_paths:
            # Entire batch was corrupt — nothing to yield
            continue

        # Stack into [B, 3, H, W] and move to device
        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad():
            # encode_image returns shape [B, embed_dim]
            embeddings = model.encode_image(batch_tensor)

        # L2-normalise: cosine_similarity(a, b) == dot(a_norm, b_norm)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Convert to CPU float32 lists for ChromaDB compatibility
        embeddings_list: List[List[float]] = embeddings.cpu().float().tolist()

        logger.debug(
            "Embedded batch %d–%d: %d ok, %d skipped in this batch",
            start,
            start + len(chunk) - 1,
            len(valid_paths),
            len(chunk) - len(valid_paths),
        )

        yield valid_paths, embeddings_list

    if skipped_total:
        logger.warning(
            "Total skipped (unreadable) images across all batches: %d", skipped_total
        )
