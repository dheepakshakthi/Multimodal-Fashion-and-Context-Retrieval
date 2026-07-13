
# Compositional query decomposer.

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logger = logging.getLogger(__name__)


# Build a compiled regex from config tokens at import time
def _build_split_pattern() -> re.Pattern:

    # Compile a regex that matches any of the configured conjunction tokens.
    # Escape each token for regex safety, then OR them
    sorted_tokens = sorted(
        config.CONJUNCTION_TOKENS, key=len, reverse=True
    )
    escaped = [re.escape(tok) for tok in sorted_tokens]
    pattern = "|".join(escaped)
    return re.compile(pattern, flags=re.IGNORECASE)


_SPLIT_RE: re.Pattern = _build_split_pattern()


# Public function
def decompose_query(query: str) -> List[str]:

    # Split *query* into one or more semantically independent segments.
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")
    if len(query) > 512:
        raise ValueError(
            f"Query exceeds 512 characters (got {len(query)})."
        )

    # Split on conjunctions
    raw_segments = _SPLIT_RE.split(query)

    # Clean: strip whitespace and drop fragments that are too short to be
    # meaningful (e.g. trailing "and" leaves an empty string)
    segments: List[str] = [
        seg.strip()
        for seg in raw_segments
        if len(seg.strip()) >= 2
    ]

    if not segments:
        # Fallback: treat the whole query as a single segment
        logger.warning(
            "decompose_query: splitting produced no valid segments for '%s'; "
            "falling back to full query.",
            query,
        )
        return [query]

    # Cap at MAX_SEGMENTS
    if len(segments) > config.MAX_SEGMENTS:
        logger.debug(
            "decompose_query: capping %d segments to MAX_SEGMENTS=%d for '%s'",
            len(segments),
            config.MAX_SEGMENTS,
            query,
        )
        segments = segments[: config.MAX_SEGMENTS]

    logger.debug(
        "decompose_query: '%s'  →  %d segment(s): %s",
        query,
        len(segments),
        segments,
    )

    return segments


# Attribute detection helpers (used by encoder.py template routing)
def has_color_attribute(query: str) -> bool:
    #Return True if the query contains at least one color keyword.
    lower = query.lower()
    return any(kw in lower for kw in config.COLOR_KEYWORDS)


def has_context_attribute(query: str) -> bool:
    #Return True if the query contains at least one environment/context keyword.
    lower = query.lower()
    return any(kw in lower for kw in config.CONTEXT_KEYWORDS)
