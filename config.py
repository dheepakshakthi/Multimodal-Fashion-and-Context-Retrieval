
# Shared configuration for the fashion image search engine. Both the Indexer and Retriever import from this module. All tuneable parameters live here — no hardcoded constants elsewhere.

import os
from pathlib import Path


# Paths

# Root of the repository (directory that contains this file)
ROOT_DIR = Path(__file__).parent.resolve()

# Raw image dataset
DATASET_DIR: str = str(ROOT_DIR / "val_test2020" / "test")

# Where ChromaDB persists its on-disk index
CHROMA_PERSIST_DIR: str = str(ROOT_DIR / "chroma_store")


# ChromaDB

CHROMA_COLLECTION_NAME: str = "fashion_images"


# Model

MODEL_ARCH: str = "ViT-B-32"
PRETRAINED_TAG: str = "openai"
NEGCLIP_WEIGHTS_PATH: str | None = os.environ.get("NEGCLIP_WEIGHTS_PATH", None)
EMBED_DIM: int = 512


# Indexing

# Number of images processed in one GPU/CPU batch
BATCH_SIZE: int = 64

# Image extensions to include
IMAGE_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".webp")


# Retrieval

# Default number of results returned
TOP_K: int = 5


# Prompt templates

# Each template must contain exactly one {query} placeholder.
# Templates are applied to the full query and to every decomposed sub-query.
# The resulting embeddings are averaged before similarity search.

PROMPT_TEMPLATES: list[str] = [
    "a photo of {query}",
    "a fashion photo of {query}",
    "a full-body fashion image showing {query}",
    "street style photo of {query}",
    "an outfit featuring {query}",
]

# Additional templates applied when a query is identified as
# environment/context-dominant.
CONTEXT_PROMPT_TEMPLATES: list[str] = [
    "a person dressed in {query}",
    "someone wearing {query} in a real-world setting",
    "a candid photo of {query}",
]

# Additional templates applied when a query is identified as color-dominant.
COLOR_PROMPT_TEMPLATES: list[str] = [
    "a fashion photo featuring {query} colored clothing",
    "an outfit with {query} garments",
]


# Query decomposition


# Keywords that indicate an environment/context-dominant query segment
CONTEXT_KEYWORDS: tuple = (
    "office", "park", "street", "home", "indoor", "outdoor",
    "city", "urban", "bench", "room", "setting", "modern",
    "formal setting", "business", "casual setting",
)

# Common color words used to detect color-dominant segments
COLOR_KEYWORDS: tuple = (
    "red", "blue", "green", "yellow", "black", "white", "grey", "gray",
    "pink", "purple", "orange", "brown", "beige", "navy", "teal",
    "maroon", "olive", "turquoise", "lavender", "crimson", "bright",
)

# Conjunctions used to split compositional queries
CONJUNCTION_TOKENS: tuple = (" and ", " with ", " wearing ", " in ")

# Maximum number of sub-query segments after decomposition
MAX_SEGMENTS: int = 3
