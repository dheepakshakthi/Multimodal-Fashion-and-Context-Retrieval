
# Text query encoder with fashion-specific prompt engineering.

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# Template selection
def _select_templates(segment: str) -> List[str]:

    # Return the list of prompt templates to apply to *segment*.
    seg_lower = segment.lower()

    templates = list(config.PROMPT_TEMPLATES)

    if any(kw in seg_lower for kw in config.CONTEXT_KEYWORDS):
        templates.extend(config.CONTEXT_PROMPT_TEMPLATES)

    if any(kw in seg_lower for kw in config.COLOR_KEYWORDS):
        templates.extend(config.COLOR_PROMPT_TEMPLATES)

    # De-duplicate while preserving order
    seen: set[str] = set()
    deduped: List[str] = []
    for t in templates:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return deduped

# Single-segment encoder
def encode_segment(
    segment: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:

    # Encode one query segment using template ensembling.

    segment = segment.strip()
    if not segment:
        raise ValueError("Query segment must not be empty.")
    if len(segment) > 512:
        raise ValueError(
            f"Query segment exceeds 512 characters (got {len(segment)})."
        )

    templates = _select_templates(segment)
    filled_prompts: List[str] = [t.format(query=segment) for t in templates]

    logger.debug(
        "Encoding segment '%s' with %d template(s)", segment, len(filled_prompts)
    )

    # Tokenise: shape [num_templates, context_length]
    tokens = tokenizer(filled_prompts).to(device)

    with torch.no_grad():
        # encode_text returns [num_templates, embed_dim]
        text_embeddings = model.encode_text(tokens)

    # L2-normalise each prompt embedding individually
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    # Average over templates → shape [embed_dim]
    avg_embedding = text_embeddings.mean(dim=0)

    # Re-normalise the average to keep it on the unit sphere
    avg_embedding = F.normalize(avg_embedding, p=2, dim=-1)

    return avg_embedding  # shape [embed_dim], float32



# Multi-segment (composite) encoder
def encode_query(
    segments: List[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:

    # Encode a list of query segments into a single composite embedding.

    if not segments:
        raise ValueError("segments list must not be empty.")

    segment_embeddings: List[torch.Tensor] = [
        encode_segment(seg, model, tokenizer, device)
        for seg in segments
    ]

    if len(segment_embeddings) == 1:
        return segment_embeddings[0]

    # Stack → [num_segments, embed_dim], mean over segments, re-normalise
    stacked = torch.stack(segment_embeddings, dim=0)   # [N, D]
    composite = stacked.mean(dim=0)                     # [D]
    composite = F.normalize(composite, p=2, dim=-1)     # [D]  unit vector

    logger.debug(
        "Composite embedding from %d segments, shape=%s",
        len(segments),
        tuple(composite.shape),
    )

    return composite
