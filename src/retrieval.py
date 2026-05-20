# src/retrieval.py
# Persistent vector store interface built on ChromaDB.
# Direct port of vector_store.py — same heading/comment style preserved.
# All constants now sourced from config/settings.yaml.

import yaml
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Load config ──────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"

def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = _load_config()
_vs  = _cfg["vector_store"]
_emb = _cfg["embedding"]

PERSIST_DIR      = _vs["persist_dir"]
COLLECTION_NAME  = _vs["collection_name"]
EMBED_MODEL_NAME = _emb["model_name"]
PASSAGE_PREFIX   = _emb["passage_prefix"]
QUERY_PREFIX     = _emb["query_prefix"]

# ── Load embedding model ─────────────────────────────────────────────
print(f"Loading embedding model: {EMBED_MODEL_NAME}")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
print("Model loaded ✓")


# ── Persistent ChromaDB client ───────────────────────────────────────
def get_client() -> chromadb.ClientAPI:
    """
    Returns a persistent ChromaDB client.
    Data is saved to disk — survives restarts.
    """
    return chromadb.PersistentClient(path=PERSIST_DIR)


def get_or_create_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    """
    Gets existing collection or creates a new one.
    Safe to call every run — won't duplicate data.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": _vs["distance_metric"]}
    )


# ── Embedding ─────────────────────────────────────────────────────────
def embed_in_batches(texts: list[str],
                     batch_size: int = None,
                     show_progress: bool = True) -> list[list[float]]:
    """
    Embed a large list of texts in batches.

    batch_size=64 is the sweet spot for most GPUs and CPUs.
    Too large → memory errors. Too small → slow.
    """
    if batch_size is None:
        batch_size = _emb["batch_size"]

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1

        if show_progress:
            print(f"  Embedding batch {batch_num}/{total_batches} "
                  f"({len(batch)} chunks)...", end="\r")

        # BGE models need a special prefix for better retrieval accuracy
        # This is a BGE-specific trick — tells the model these are passages
        prefixed = [f"{PASSAGE_PREFIX}{t}" for t in batch]

        embeddings = embed_model.encode(
            prefixed,
            normalize_embeddings=_emb["normalize"],  # important for cosine similarity
            show_progress_bar=False
        )
        all_embeddings.extend(embeddings.tolist())

    if show_progress:
        print(f"\n  Done — embedded {len(texts)} chunks ✓")

    return all_embeddings


# ── Ingestion ─────────────────────────────────────────────────────────
def ingest_chunks(collection: chromadb.Collection,
                  chunks: list[dict],
                  source_name: str = "document") -> int:
    """
    Ingest chunks into ChromaDB.
    Skips chunks that are already stored (by ID).
    Returns number of NEW chunks added.

    chunks: list of dicts with at least {"id": str, "text": str}
    """

    # Check which IDs are already in the collection
    existing = collection.get(include=[])  # just get IDs, no content
    existing_ids = set(existing["ids"])

    # Filter to only new chunks
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]

    if not new_chunks:
        print(f"All {len(chunks)} chunks already in DB — skipping")
        return 0

    print(f"Ingesting {len(new_chunks)} new chunks "
          f"({len(chunks) - len(new_chunks)} already exist)...")

    texts     = [c["text"] for c in new_chunks]
    ids       = [c["id"] for c in new_chunks]
    metadatas = [
        {
            "source":      c.get("source", source_name),
            "chunk_index": c.get("chunk_index", i),
            "word_count":  c.get("word_count", len(c["text"].split())),
            "entity_types": c.get("entity_types", "")
        }
        for i, c in enumerate(new_chunks)
    ]

    # Embed in batches
    embeddings = embed_in_batches(texts)

    # Add to ChromaDB in batches too (ChromaDB has its own limits)
    batch_size = 500
    for i in range(0, len(new_chunks), batch_size):
        collection.add(
            documents =texts[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            ids       =ids[i:i+batch_size],
            metadatas =metadatas[i:i+batch_size]
        )

    print(f"Stored {len(new_chunks)} chunks ✓")
    return len(new_chunks)


# ── Retrieval ─────────────────────────────────────────────────────────
def retrieve_with_scores(question: str,
                         collection: chromadb.Collection,
                         top_k: int = None) -> list[dict]:
    """
    Retrieve top_k chunks with their similarity scores and metadata.
    Returns structured results — essential for the eval dashboard.
    """
    if top_k is None:
        top_k = _cfg["retrieval"]["top_k"]

    # BGE queries also need the prefix
    prefixed_question = f"{QUERY_PREFIX}{question}"
    question_embedding = embed_model.encode(
        [prefixed_question],
        normalize_embeddings=_emb["normalize"]
    ).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # Format into clean dicts
    retrieved = []
    for i in range(len(results["documents"][0])):
        # ChromaDB returns distance (lower=better), convert to similarity
        distance   = results["distances"][0][i]
        similarity = 1 - distance  # cosine: similarity = 1 - distance

        retrieved.append({
            "text":             results["documents"][0][i],
            "metadata":         results["metadatas"][0][i],
            "similarity_score": round(similarity, 4),
            "rank":             i + 1
        })

    return retrieved


# ── Token utilities ───────────────────────────────────────────────────
def count_tokens_approx(text: str) -> int:
    """
    Rough token count without needing a tokenizer library.
    Rule of thumb: 1 token ≈ 0.75 words for English text.
    Good enough for benchmarking; use tiktoken for precision.
    """
    return int(len(text.split()) / 0.75)


def count_tokens_precise(text: str) -> int:
    """
    Precise token count using Google's tokenizer approach.
    Install: pip install tiktoken
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer, close enough
        return len(enc.encode(text))
    except ImportError:
        return count_tokens_approx(text)
