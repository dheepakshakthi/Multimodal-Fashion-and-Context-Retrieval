# Multimodal Fashion & Context Retrieval

A zero-shot, natural language image search engine for fashion images. Given a plain English description of an outfit, environment, or style, the system retrieves the most relevant images from a 3,200-image Fashionpedia dataset — without any labelled training data.

Built for the Glance ML Internship Assignment.

---

## Table of Contents

1. [Overview](#overview)
2. [Why This Is Better Than Vanilla CLIP](#why-this-is-better-than-vanilla-clip)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Dataset](#dataset)
6. [Setup](#setup)
7. [Running the Indexer](#running-the-indexer)
8. [Running the Retriever](#running-the-retriever)
9. [Configuration Reference](#configuration-reference)
10. [Evaluation Queries](#evaluation-queries)
11. [Approaches Considered](#approaches-considered)
12. [Chosen Approach: Deep Dive](#chosen-approach-deep-dive)
13. [Shortcomings & Limitations](#shortcomings--limitations)
14. [Future Work](#future-work)

---

## Overview

The system is split into two independent, self-contained modules:

| Module | Directory | Purpose |
|--------|-----------|---------|
| **Indexer** | `indexer/` | Extracts NegCLIP image embeddings and stores them in ChromaDB |
| **Retriever** | `retriever/` | Accepts a natural language query and returns top-k ranked image paths |

Both modules share a single `config.py` at the repository root — no hardcoded constants anywhere else.

---

## Why This Is Better Than Vanilla CLIP

Vanilla CLIP has two well-documented failure modes for fashion retrieval:

### 1. Compositionality failure
CLIP is trained with in-batch random negatives. It learns that "red shirt" and "blue pants" co-occur but does not reliably bind the attribute _red_ to _shirt_ and _blue_ to _pants_ separately. A query like _"red tie and white shirt"_ produces an embedding that partially matches _"white tie and red shirt"_ as well.

### 2. Fine-grained attribute insensitivity
Generic CLIP training data is dominated by object-level descriptions. Fashion-specific nuances — fabric, cut, layering, colour shades — are underrepresented.

This system addresses both problems:

| Technique | What it does | Where |
|-----------|-------------|-------|
| **NegCLIP backbone** | Fine-tuned with hard negatives (attribute-swapped pairs). Text encoder is significantly more sensitive to attribute-object binding. | `indexer/model.py` |
| **Compositional decomposition** | Splits queries at conjunctions ("and", "with", …) into independent sub-queries, each encoded separately. Scores are averaged per image. | `retriever/decomposer.py` |
| **Prompt template ensembling** | Encodes 5+ fashion-specific template variants of every query and averages their embeddings. Reduces sensitivity to exact phrasing. | `retriever/encoder.py` |
| **Context-aware template routing** | Detects environment keywords (office, park, street, …) and adds location-specific templates. Detects colour keywords and adds colour-specific templates. | `retriever/encoder.py` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                       INDEXER                           │
│                                                         │
│  val_test2020/test/*.jpg                                │
│         │                                               │
│         ▼                                               │
│  embedder.discover_images()                             │
│         │                                               │
│         ▼  mini-batches (default: 64 images)            │
│  model.get_model()  ──►  NegCLIP ViT-B/32               │
│  (encode_image)                                         │
│         │  L2-normalised 512-d vectors                  │
│         ▼                                               │
│  store.upsert_batch()  ──►  ChromaDB (cosine HNSW)      │
│                              chroma_store/              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                      RETRIEVER                          │
│                                                         │
│  "A red tie and a white shirt in a formal setting"      │
│         │                                               │
│         ▼                                               │
│  decomposer.decompose_query()                           │
│    → ["A red tie", "a white shirt in a formal setting"] │
│         │                                               │
│         ▼  per segment                                  │
│  encoder.encode_segment()                               │
│    ├── select templates (base + context + color)        │
│    ├── fill & tokenise all templates                    │
│    ├── NegCLIP encode_text  →  [N_templates, 512]       │
│    ├── L2-normalise each                                │
│    └── mean + L2-normalise  →  [512]                    │
│         │                                               │
│         ▼  average segment vectors + L2-normalise       │
│  composite query vector  [512]                          │
│         │                                               │
│         ▼                                               │
│  ChromaDB.query()  (HNSW nearest-neighbour)             │
│         │  cosine distance → similarity score           │
│         ▼                                               │
│  ranked list: [{rank, file_path, image_id, score}, …]  │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
glance/
├── config.py                   # Shared settings — single source of truth
├── requirements.txt
├── README.md
│
├── indexer/
│   ├── model.py                # NegCLIP / OpenCLIP model loader
│   ├── embedder.py             # Image discovery + batch embedding generator
│   ├── store.py                # ChromaDB collection management & upsert
│   └── index.py                # CLI entry point + build_index() API
│
├── retriever/
│   ├── decomposer.py           # Conjunctive query splitter
│   ├── encoder.py              # Template ensembling + composite embedding
│   ├── search.py               # Core search function
│   └── retrieve.py             # CLI entry point + retrieve() API (cached)
│
├── val_test2020/
│   └── test/                   # 3,200 Fashionpedia .jpg images
│
└── chroma_store/               # Created on first indexer run (ChromaDB files)
```

---

## Dataset

The dataset used is the **Fashionpedia validation set** (`val_test2020/test/`), containing **3,200 JPEG images**. It provides variation across the three required axes:

| Axis | Examples |
|------|---------|
| **Environment** | Office interiors, urban streets, parks, home settings |
| **Clothing type** | Formal (blazers, button-downs), Casual (hoodies, t-shirts), Outerwear |
| **Color** | Full palette — primary, secondary, neutral, and bright tones |

Images are identified by their MD5-hash filename (e.g. `003d41dd20f271d27219fe7ee6de727d.jpg`). The filename stem is used as the ChromaDB document ID.

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended for indexing (CPU works but takes ~15 min for 3,200 images; GPU takes ~2 min)

### Install dependencies

```bash
pip install -r requirements.txt
```

### (Optional) Use NegCLIP weights

By default the system loads the original CLIP ViT-B/32 weights from OpenAI as a working baseline. To use the actual NegCLIP fine-tuned checkpoint:

1. Download the weights from [Hugging Face — laion/negclip-ViT-B-32](https://huggingface.co/laion/negclip-ViT-B-32) (file: `negclip.pt`)
2. Set the environment variable before running any command:

```bash
# Windows CMD
set NEGCLIP_WEIGHTS_PATH=D:\models\negclip.pt

# Windows PowerShell
$env:NEGCLIP_WEIGHTS_PATH = "D:\models\negclip.pt"

# Linux / macOS
export NEGCLIP_WEIGHTS_PATH=/path/to/negclip.pt
```

The model loader in `indexer/model.py` detects this variable and overrides the OpenAI weights with the NegCLIP checkpoint automatically.

---

## Running the Indexer

The Indexer scans the dataset, extracts NegCLIP image embeddings in batches, and persists them in a ChromaDB vector store. **Run this once before any retrieval.**

### Basic run (uses defaults from `config.py`)

```bash
python -m indexer.index
```

### With explicit arguments

```bash
python -m indexer.index \
    --dataset-dir val_test2020/test \
    --persist-dir chroma_store \
    --batch-size 64
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-dir` | `val_test2020/test` | Directory containing `.jpg` images |
| `--persist-dir` | `chroma_store` | Where ChromaDB writes its files |
| `--batch-size` | `64` | Images per GPU forward pass (1–512) |

### Expected output

```
09:14:22  INFO  indexer.embedder  Discovered 3200 images in val_test2020\test
09:14:22  INFO  indexer.model     Loading model arch=ViT-B-32 on device=cuda
09:14:28  INFO  indexer.store     ChromaDB collection 'fashion_images' opened (current size: 0)
09:14:28  INFO  indexer.index     Already indexed: 0 / 3200 — 3200 to embed
Indexing: 100%|████████████████| 3200/3200 [01:47<00:00, 29.7img/s]
09:16:15  INFO  indexer.index     Newly inserted : 3200
09:16:15  INFO  indexer.index     Collection size: 3200

Done. 3200 images indexed.
```

**Incremental indexing:** Running the indexer a second time skips images already in the collection. Only new images are embedded and stored.

---

## Running the Retriever

The Retriever accepts a natural language query and returns the top-k matching images from the indexed collection.

### CLI

```bash
python -m retriever.retrieve --query "A person in a bright yellow raincoat"
```

```bash
python -m retriever.retrieve \
    --query "A red tie and a white shirt in a formal setting" \
    --top-k 10
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--query` | *(required)* | Natural language search string |
| `--top-k` | `5` | Number of results (1–1000) |
| `--persist-dir` | `chroma_store` | ChromaDB directory (must exist) |

### Expected output

```
Query : A red tie and a white shirt in a formal setting
Results (5 found):
------------------------------------------------------------------------
    1.  score=0.3241  D:\glance\val_test2020\test\3f18271f3f7d101e2f9a27b0b12b034b.jpg
    2.  score=0.3187  D:\glance\val_test2020\test\8c54a0c4b1c47d32a13f2bcc0f4c24bb.jpg
    3.  score=0.3102  D:\glance\val_test2020\test\a7309a892ac28a7ea5f3a7af1d9e798a.jpg
    4.  score=0.3056  D:\glance\val_test2020\test\e4432c751c332f2f5829e6a1ebb7301d.jpg
    5.  score=0.2998  D:\glance\val_test2020\test\b2c35c8f0ca1e87a90dbe33a0c62d87e.jpg
------------------------------------------------------------------------
```

### Python API

The retriever also exposes a Python function for programmatic use. The model and collection are loaded once and cached across calls within the same process:

```python
from retriever.retrieve import retrieve

# Single query
results = retrieve("Professional business attire inside a modern office", k=5)
for r in results:
    print(f"{r['rank']:2d}  score={r['score']:.4f}  {r['file_path']}")

# Subsequent calls reuse the cached model — no reload
results2 = retrieve("Casual weekend outfit for a city walk", k=10)
```

Each result dict contains:

| Key | Type | Description |
|-----|------|-------------|
| `rank` | `int` | 1-based rank (1 = best match) |
| `file_path` | `str` | Absolute path to the image file |
| `image_id` | `str` | Filename stem used as ChromaDB document ID |
| `score` | `float` | Cosine similarity in [−1, 1]; higher is better |

---

## Configuration Reference

All tuneable parameters live in `config.py`. Neither the Indexer nor the Retriever contains any hardcoded constants.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_DIR` | `val_test2020/test` | Image source directory |
| `CHROMA_PERSIST_DIR` | `chroma_store` | ChromaDB on-disk location |
| `CHROMA_COLLECTION_NAME` | `fashion_images` | Collection name |
| `MODEL_ARCH` | `ViT-B-32` | OpenCLIP model architecture |
| `PRETRAINED_TAG` | `openai` | Fallback weights if no NegCLIP checkpoint |
| `NEGCLIP_WEIGHTS_PATH` | `None` | Path to local NegCLIP `.pt` file |
| `EMBED_DIM` | `512` | Embedding dimensionality (ViT-B/32) |
| `BATCH_SIZE` | `64` | Images per indexer forward pass |
| `TOP_K` | `5` | Default retrieval result count |
| `MAX_SEGMENTS` | `3` | Max sub-queries after decomposition |
| `PROMPT_TEMPLATES` | 5 templates | Base fashion prompt templates |
| `CONTEXT_PROMPT_TEMPLATES` | 3 templates | Added for environment-heavy queries |
| `COLOR_PROMPT_TEMPLATES` | 2 templates | Added for color-heavy queries |
| `CONJUNCTION_TOKENS` | `" and ", " with ", " wearing ", " in "` | Query split boundaries |
| `CONTEXT_KEYWORDS` | office, park, street, … | Triggers context templates |
| `COLOR_KEYWORDS` | red, blue, green, … | Triggers color templates |

---

## Evaluation Queries

The five assignment evaluation queries and how the system handles each:

| # | Query | Decomposition | Template route |
|---|-------|---------------|----------------|
| 1 | *"A person in a bright yellow raincoat."* | 1 segment | base + color (yellow, bright) |
| 2 | *"Professional business attire inside a modern office."* | 1 segment | base + context (office, modern) |
| 3 | *"Someone wearing a blue shirt sitting on a park bench."* | 2 segments: `"Someone"` + `"a blue shirt sitting on a park bench"` | base + color (blue) + context (park) |
| 4 | *"Casual weekend outfit for a city walk."* | 1 segment | base + context (city) |
| 5 | *"A red tie and a white shirt in a formal setting."* | 2 segments: `"A red tie"` + `"a white shirt in a formal setting"` | base + color (red/white) + context (formal setting) |

Query 5 is the hardest compositional case: vanilla CLIP would give similar scores to "white tie and red shirt". The NegCLIP backbone + sub-query decomposition directly addresses this.

---

## Approaches Considered

### Option A — Vanilla CLIP zero-shot retrieval
Encode all images with CLIP ViT-B/32, encode queries, compute cosine similarity, rank.

**Pros:** Simple, no fine-tuning needed, strong baseline.  
**Cons:** Poor compositionality, treats queries as bags-of-words, generic prompting misses fashion nuance.

### Option B — Fine-tuned fashion CLIP (e.g. FashionCLIP)
Use a CLIP model fine-tuned on e-commerce fashion images.

**Pros:** Better at garment types and attributes.  
**Cons:** Requires a specific fine-tuned checkpoint, may overfit to catalogue-style images, less generalisation to lifestyle/candid shots, heavy download (~1 GB+).

### Option C — NegCLIP + prompt ensembling + query decomposition *(chosen)*
Use the NegCLIP ViT-B/32 checkpoint (same size as vanilla CLIP) with:
- Fashion-specific prompt template ensembling
- Compositional query decomposition
- Context/color-aware template routing

**Pros:** Directly targets both known CLIP failure modes; zero-shot; same model size as vanilla CLIP; no training data needed; all improvements are interpretable.  
**Cons:** Heuristic conjunctive splitting may mis-segment unusual phrasings; NegCLIP checkpoint requires a download.

### Option D — Dense captioning + BM25 / keyword search
Generate image captions (e.g. with BLIP-2), store as text, retrieve with BM25.

**Pros:** Highly interpretable results.  
**Cons:** Not truly semantic; caption quality limits recall; no zero-shot for novel descriptions; two-model pipeline (captioner + retriever).

### Option E — Multi-modal LLM reranking (e.g. GPT-4V)
Use a large vision-language model to re-score top-k CLIP candidates.

**Pros:** Very high precision.  
**Cons:** Extremely slow and expensive at scale, API dependency, not viable for 1M images.

**Summary:**

| Approach | Compositionality | Zero-shot | Scalability | Complexity |
|----------|-----------------|-----------|-------------|------------|
| A — Vanilla CLIP | ✗ | ✓ | ✓✓ | Low |
| B — FashionCLIP | ✗ | Partial | ✓✓ | Low |
| **C — NegCLIP + decomp.** | **✓** | **✓✓** | **✓✓** | **Medium** |
| D — Captions + BM25 | ✗ | ✗ | ✓ | High |
| E — LLM rerank | ✓✓ | ✓✓ | ✗ | Very High |

---

## Chosen Approach: Deep Dive

### NegCLIP

NegCLIP (Yuksekgonul et al., ICLR 2023, *"When and Why Vision-Language Models Behave like Bags-Of-Words, and What to Do About It"*) fine-tunes CLIP with additional hard negative text pairs constructed by swapping noun phrases and attributes within captions. For example:

- Positive: *"a red tie and a white shirt"*
- Hard negative: *"a white tie and a red shirt"*

This forces the text encoder to become sensitive to the *binding* between attributes and objects, not just their co-occurrence. The image encoder is unchanged; only the text encoder is updated. The result is a model that occupies the same embedding space and is a drop-in replacement for vanilla CLIP ViT-B/32 weights.

### Prompt Template Ensembling

Radford et al. (CLIP, 2021) showed that ensembling multiple prompt templates improves zero-shot classification by up to 3.5%. We apply the same principle to retrieval:

```
"a photo of {query}"
"a fashion photo of {query}"
"a full-body fashion image showing {query}"
"street style photo of {query}"
"an outfit featuring {query}"
```

For a query like *"bright yellow raincoat"*, all five are encoded, their 512-d vectors are L2-normalised individually, then averaged and re-normalised. This smooths over vocabulary mismatches between the query and the image training distribution.

**Context routing** adds three more templates when environment keywords are detected (`office`, `park`, `street`, `home`, `city`, `urban`, `bench`, `modern`, …).

**Color routing** adds two more templates when color words appear (`red`, `blue`, `yellow`, `bright`, …).

### Compositional Query Decomposition

Queries containing conjunctions are split into independent sub-queries:

```
"A red tie and a white shirt in a formal setting"
    → ["A red tie", "a white shirt in a formal setting"]
```

Each segment is encoded independently using the full template-ensembling pipeline. The resulting unit vectors are averaged and re-normalised to form the composite query vector. This forces the similarity search to consider both attributes rather than blending them into a single holistic vector that may lose binding information.

Splitting is capped at 3 segments (configurable via `MAX_SEGMENTS`).

### ChromaDB for Scalable Vector Storage

ChromaDB is an embedded vector database that runs in-process (no server required). It uses an HNSW (Hierarchical Navigable Small World) index for approximate nearest-neighbour search with sub-linear (O(log N)) query time. The same code handles 3,200 images and 1,000,000 images identically — no architectural changes needed.

Each image is stored with:
- **ID:** filename stem (e.g. `003d41dd20f271d27219fe7ee6de727d`)
- **Embedding:** 512-d L2-normalised float32 vector
- **Metadata:** absolute file path for direct result display

---

## Shortcomings & Limitations

1. **Heuristic query splitting** — The conjunction-based splitter can mis-segment queries with unusual structure. A learned constituency parser would be more robust but adds a dependency.

2. **No spatial understanding** — Neither CLIP nor NegCLIP understands spatial relationships well (e.g. "shirt tucked into pants" vs "shirt worn over pants"). This requires spatial-aware models like FLAVA or StructuredCLIP.

3. **Fashionpedia dataset bias** — The dataset is catalogue/editorial photography, not candid lifestyle images. Retrieval for queries like "casual weekend outfit at home" may be limited by dataset coverage, not model capability.

4. **No negative query support** — Queries like "not red" or "without a jacket" are not handled. True NegCLIP training provides some implicit negation sensitivity but no explicit support is implemented.

5. **Equal sub-query weighting** — All decomposed segments are weighted equally. A query like "blue shirt in an office" might benefit from weighting "blue shirt" higher than "office" as a filtering criterion.

---

## Future Work

### a. Extending for location (cities, places) and weather

**Approach 1 — Metadata enrichment at index time:**  
Use a scene classifier (e.g. MIT Places365) or a geolocation model to tag each indexed image with location type and season/weather. Store these as ChromaDB metadata fields and add pre-filtering before ANN search.

**Approach 2 — Extended prompt vocabulary:**  
Add city-specific and weather-specific prompt templates to `config.py`. For example:
- `"a fashion photo taken in a European city street, {weather}, {query}"`
- `"street style in {city}, {query}"`

Fill these with detected location/weather tokens from the query using a lightweight NER model (spaCy) and add them to the template ensemble.

**Approach 3 — Multi-modal index with geographic metadata:**  
If images have GPS metadata (EXIF), extract coordinates and map to city/country. Store as filterable ChromaDB metadata. At query time, parse city/country from the query and pre-filter the collection before ANN search.

### b. Improving precision

**1. Re-ranking with a cross-encoder:**  
Use a small vision-language model (e.g. BLIP-2, LLaVA-7B) to re-score the top-50 CLIP candidates. Cross-encoders see the image and text together (not independently embedded) and can reason about fine-grained attribute binding. Run only on the shortlist to keep latency manageable.

**2. Query expansion:**  
Use an LLM to expand the user query into 3–5 paraphrases before encoding. E.g. *"casual weekend outfit"* → also encode *"relaxed everyday clothing"*, *"informal leisure wear"*, *"laid-back street style"*. Average all embeddings. Increases recall significantly for short or ambiguous queries.

**3. Fashion-domain fine-tuning:**  
Fine-tune NegCLIP on a fashion-specific image-caption dataset (e.g. FashionGen, DeepFashion-MultiModal) with additional hard negatives targeting color-garment binding. This directly improves precision for the attribute-specific and compositional query types.

**4. Attribute-disentangled embeddings:**  
Train separate lightweight adapter heads on top of CLIP for each attribute axis (color, garment type, environment). At retrieval time, compute weighted combination of attribute scores. This gives explicit control over which attributes matter most for a given query.

**5. User feedback loop (RLHF-style):**  
Collect relevance feedback from users (click-through, explicit ratings) and use it to fine-tune the prompt templates or a lightweight re-ranking layer. Even a small amount of fashion-domain preference data dramatically improves precision over pure zero-shot retrieval.

---

## References

- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP). OpenAI.
- Yuksekgonul, M. et al. (2023). *When and Why Vision-Language Models Behave like Bags-Of-Words, and What to Do About It* (NegCLIP). ICLR 2023.
- Kirillov, A. et al. (2021). *Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset*.
- Ilharco, G. et al. (2021). *OpenCLIP*. [github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
