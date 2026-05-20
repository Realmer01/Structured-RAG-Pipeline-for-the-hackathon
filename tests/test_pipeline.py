# tests/test_pipeline.py
# Unit tests for the RAG pipeline modules.
# All tests are fully offline — no API calls, no ChromaDB I/O.
# Run with: python -m pytest tests/ -v

import sys
import os
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Module-level patches ─────────────────────────────────────────────
# Patch SentenceTransformer so retrieval.py doesn't download the BGE model.
_mock_embed_model = MagicMock()
_mock_embed_model.encode.side_effect = lambda texts, **kw: np.array(
    [[0.1, 0.2, 0.3]] * len(texts)
)

# Apply patches BEFORE retrieval/generation are imported anywhere in this file.
_st_patch  = patch("sentence_transformers.SentenceTransformer", return_value=_mock_embed_model)
_gen_patch  = patch("google.genai.Client", return_value=MagicMock())
_st_patch.start()
_gen_patch.start()


# ─────────────────────────────────────────────────────────────────────
# ingestion.py tests
# ─────────────────────────────────────────────────────────────────────

def test_load_txt_file(tmp_path):
    """ingestion.load_txt_file reads a .txt file correctly."""
    from src.ingestion import load_txt_file

    sample = "Patient presented with palpitations and dyspnea."
    txt = tmp_path / "sample.txt"
    txt.write_text(sample, encoding="utf-8")

    result = load_txt_file(str(txt))
    assert result == sample


def test_chunk_produces_ids(tmp_path):
    """ingestion.prepare_chunks assigns required keys to every chunk."""
    from src.ingestion import prepare_chunks

    docs = [{
        "doc_id":   "test_doc",
        "text":     "Photosynthesis is the process used by plants. " * 20,
        "entities": [],
        "source_path": str(tmp_path / "test_doc.txt")
    }]
    chunks = prepare_chunks(docs, chunk_size=30, overlap=5)

    assert len(chunks) > 0
    for chunk in chunks:
        assert "id"         in chunk, "chunk missing 'id'"
        assert "text"       in chunk, "chunk missing 'text'"
        assert "word_count" in chunk, "chunk missing 'word_count'"
        assert chunk["id"].startswith("test_doc_chunk_")


def test_parse_ann_file(tmp_path):
    """ingestion.parse_ann_file extracts entity records from a .ann file."""
    from src.ingestion import parse_ann_file

    ann_content = (
        "T1\tAge 8 19\t28-year-old\n"
        "T2\tSign_symptom 31 38\thealthy\n"
    )
    ann_file = tmp_path / "sample.ann"
    ann_file.write_text(ann_content, encoding="utf-8")

    entities = parse_ann_file(str(ann_file))
    assert len(entities) == 2
    assert entities[0]["type"] == "Age"
    assert entities[1]["type"] == "Sign_symptom"


# ─────────────────────────────────────────────────────────────────────
# retrieval.py tests  (SentenceTransformer patched at module level above)
# ─────────────────────────────────────────────────────────────────────

def test_embed_in_batches_shape():
    """retrieval.embed_in_batches returns one embedding per input text."""
    import src.retrieval as retrieval_mod

    texts = [f"sentence {i}" for i in range(5)]
    result = retrieval_mod.embed_in_batches(texts, batch_size=3, show_progress=False)

    assert len(result) == 5


def test_count_tokens_approx():
    """retrieval.count_tokens_approx returns sensible estimate."""
    from src.retrieval import count_tokens_approx

    text = " ".join(["word"] * 75)   # 75 words → ~100 tokens
    result = count_tokens_approx(text)
    assert 90 <= result <= 110, f"Unexpected token count: {result}"


# ─────────────────────────────────────────────────────────────────────
# generation.py tests
# ─────────────────────────────────────────────────────────────────────

def test_build_prompt():
    """generation.build_prompt fills the template with context and question."""
    from src.generation import build_prompt

    question = "What is Ebstein's anomaly?"
    chunks   = ["Ebstein's anomaly is a congenital heart defect."]
    prompt   = build_prompt(question, chunks, template_key="rag_answer")

    assert "Ebstein's anomaly" in prompt
    assert "What is Ebstein's anomaly?" in prompt
    assert "CONTEXT" in prompt


# ─────────────────────────────────────────────────────────────────────
# evaluation.py tests
# ─────────────────────────────────────────────────────────────────────

def test_score_retrieval_weak():
    """evaluation.score_retrieval flags weak retrieval correctly."""
    from src.evaluation import score_retrieval

    chunks = [
        {"rank": 1, "similarity_score": 0.10, "text": "some text", "metadata": {}},
        {"rank": 2, "similarity_score": 0.12, "text": "more text", "metadata": {}},
    ]
    result = score_retrieval(chunks)
    assert result["weak_retrieval"] is True
    assert result["chunks_above_threshold"] == 0


def test_score_retrieval_strong():
    """evaluation.score_retrieval correctly identifies strong chunks."""
    from src.evaluation import score_retrieval

    chunks = [
        {"rank": 1, "similarity_score": 0.85, "text": "relevant text", "metadata": {}},
        {"rank": 2, "similarity_score": 0.72, "text": "also relevant", "metadata": {}},
    ]
    result = score_retrieval(chunks)
    assert result["weak_retrieval"] is False
    assert result["avg_similarity"] > 0.5
