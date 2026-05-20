# src/ingestion.py
# Loads biomedical .txt documents from dataset_for_project/, chunks them,
# and prepares them for storage in the vector store.
# Mirrors the load_text() logic from rag_pipeline.py and chunking strategies
# from chunking.py — same heading/comment style preserved.

import re
import yaml
from pathlib import Path

# ── Load config ──────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"

def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Document loader ──────────────────────────────────────────────────
def load_txt_file(filepath: str) -> str:
    """
    Load a plain-text document from disk.
    Mirrors load_text() from rag_pipeline.py — kept as a single-purpose function.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def parse_ann_file(ann_path: str) -> list[dict]:
    """
    Parse a .ann annotation file (Brat standoff format).
    Returns a list of entity dicts: {id, type, start, end, text}
    These can be attached as metadata to chunks that overlap the span.
    """
    entities = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            entity_id = parts[0]
            span_info = parts[1].split()
            entity_type = span_info[0]
            try:
                start = int(span_info[1])
                end = int(span_info[-1])
            except (ValueError, IndexError):
                continue
            entity_text = parts[2]
            entities.append({
                "id": entity_id,
                "type": entity_type,
                "start": start,
                "end": end,
                "text": entity_text
            })
    return entities


# ── Chunking (recursive strategy — from chunking.py) ─────────────────
def _split_on_separators(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """
    Internal recursive helper — same logic as chunk_recursive() in chunking.py.
    """
    if not separators:
        return [text]

    sep = separators[0]
    splits = re.split(sep, text)

    good_chunks = []
    current = ""

    for split in splits:
        split = split.strip()
        if not split:
            continue

        test = (current + " " + split).strip()
        if len(test.split()) <= chunk_size:
            current = test
        else:
            if current:
                good_chunks.append(current)
            if len(split.split()) > chunk_size:
                good_chunks.extend(
                    _split_on_separators(split, separators[1:], chunk_size)
                )
            else:
                current = split

    if current:
        good_chunks.append(current)

    return good_chunks


def chunk_recursive(text: str, chunk_size: int = 150, overlap: int = 20) -> list[dict]:
    """
    Recursive chunking by separators: paragraphs → sentences → words.
    Returns dicts with text, strategy, chunk_index, word_count.
    Identical behaviour to chunk_recursive() in chunking.py.
    """
    separators = [r"\n\n", r"(?<=[.!?])\s+", r"\s+"]
    raw_chunks = _split_on_separators(text.strip(), separators, chunk_size)

    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if i + 1 < len(raw_chunks):
            next_words = " ".join(raw_chunks[i + 1].split()[:overlap])
            chunk_with_overlap = chunk_text + " " + next_words
        else:
            chunk_with_overlap = chunk_text

        chunks.append({
            "text": chunk_with_overlap.strip(),
            "strategy": "recursive",
            "chunk_index": i,
            "word_count": len(chunk_with_overlap.split()),
        })

    return chunks


# ── Dataset ingestion pipeline ────────────────────────────────────────
def load_dataset_files(raw_dir: str) -> list[dict]:
    """
    Scan raw_dir for all .txt files and load them.
    Each entry: {doc_id, text, ann_path (if exists)}
    """
    raw_path = Path(raw_dir)
    documents = []

    for txt_file in sorted(raw_path.glob("*.txt")):
        doc_id = txt_file.stem
        text = load_txt_file(str(txt_file))

        ann_file = txt_file.with_suffix(".ann")
        entities = parse_ann_file(str(ann_file)) if ann_file.exists() else []

        documents.append({
            "doc_id": doc_id,
            "text": text,
            "entities": entities,
            "source_path": str(txt_file)
        })

    print(f"Loaded {len(documents)} documents from {raw_dir}")
    return documents


def prepare_chunks(documents: list[dict], chunk_size: int = 150, overlap: int = 20) -> list[dict]:
    """
    Chunk all loaded documents.
    Assigns unique IDs and attaches source metadata to each chunk.
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_recursive(doc["text"], chunk_size=chunk_size, overlap=overlap)

        for chunk in chunks:
            chunk["id"] = f"{doc['doc_id']}_chunk_{chunk['chunk_index']}"
            chunk["source"] = doc["doc_id"]
            # Attach entity types present in the document as searchable metadata
            entity_types = list({e["type"] for e in doc["entities"]})
            chunk["entity_types"] = ",".join(entity_types) if entity_types else ""

        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


def ingest_dataset(raw_dir: str = None) -> list[dict]:
    """
    Full ingestion pipeline:
      1. Load config
      2. Load all .txt files from dataset dir
      3. Chunk them
      4. Return chunk list ready for retrieval.ingest_chunks()
    """
    cfg = _load_config()
    if raw_dir is None:
        raw_dir = cfg["dataset"]["raw_dir"]

    documents = load_dataset_files(raw_dir)
    chunks = prepare_chunks(
        documents,
        chunk_size=cfg["chunking"]["chunk_size"],
        overlap=cfg["chunking"]["overlap"]
    )
    return chunks
