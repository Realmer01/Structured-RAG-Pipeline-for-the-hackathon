# src/evaluation.py
# Evaluation utilities for the RAG pipeline.
# Drawn from token-counting helpers in vector_store.py.
# Provides retrieval quality metrics, context budget checks,
# and basic answer faithfulness assessment.

import yaml
from pathlib import Path

# ── Load config ──────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"

def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = _load_config()
_eval_cfg = _cfg["evaluation"]


# ── Token counting ────────────────────────────────────────────────────
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


# ── Retrieval quality metrics ─────────────────────────────────────────
def score_retrieval(retrieved_chunks: list[dict]) -> dict:
    """
    Summarise quality of a single retrieval call.

    Input: output from retrieval.retrieve_with_scores()
    Returns: {
        num_chunks, avg_similarity, max_similarity, min_similarity,
        chunks_above_threshold, weak_retrieval (bool)
    }
    """
    if not retrieved_chunks:
        return {
            "num_chunks": 0,
            "avg_similarity": 0.0,
            "max_similarity": 0.0,
            "min_similarity": 0.0,
            "chunks_above_threshold": 0,
            "weak_retrieval": True
        }

    scores = [r["similarity_score"] for r in retrieved_chunks]
    threshold = _eval_cfg["min_similarity_threshold"]
    above = [s for s in scores if s >= threshold]

    return {
        "num_chunks":             len(scores),
        "avg_similarity":         round(sum(scores) / len(scores), 4),
        "max_similarity":         round(max(scores), 4),
        "min_similarity":         round(min(scores), 4),
        "chunks_above_threshold": len(above),
        "weak_retrieval":         len(above) == 0
    }


# ── Context budget check ──────────────────────────────────────────────
def check_context_budget(retrieved_chunks: list[dict],
                          precise: bool = False) -> dict:
    """
    Count tokens across all retrieved chunks and warn if over budget.

    Returns: {total_tokens, budget, over_budget (bool), chunk_token_counts}
    """
    count_fn = count_tokens_precise if precise else count_tokens_approx
    budget   = _eval_cfg["max_context_tokens"]

    chunk_counts = []
    for r in retrieved_chunks:
        tokens = count_fn(r["text"])
        chunk_counts.append({
            "rank":   r["rank"],
            "tokens": tokens,
            "source": r.get("metadata", {}).get("source", "unknown")
        })

    total = sum(c["tokens"] for c in chunk_counts)

    if total > budget:
        print(f"⚠️  Context budget exceeded: {total} tokens > {budget} limit")
    else:
        print(f"✅ Context token count: {total} / {budget}")

    return {
        "total_tokens":       total,
        "budget":             budget,
        "over_budget":        total > budget,
        "chunk_token_counts": chunk_counts
    }


# ── Basic faithfulness check ──────────────────────────────────────────
def check_faithfulness(answer: str, retrieved_chunks: list[dict],
                        top_n_terms: int = 10) -> dict:
    """
    Lightweight check: does the answer contain key terms from the retrieved chunks?
    Not a substitute for an LLM-as-judge evaluation — useful as a quick smoke test.

    Returns: {key_terms, terms_found_in_answer, coverage_ratio}
    """
    import re

    # Extract unique content words (length > 4) from top retrieved chunks
    all_text = " ".join(r["text"] for r in retrieved_chunks)
    words    = re.findall(r"\b[a-zA-Z]{5,}\b", all_text.lower())

    # Simple frequency ranking
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Skip stopwords (basic list)
    _STOP = {"which", "their", "there", "about", "these", "those", "where",
             "after", "before", "other", "using", "between", "during",
             "provide", "however", "within", "through", "without"}
    key_terms = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])
                 if w not in _STOP][:top_n_terms]

    answer_lower = answer.lower()
    found = [t for t in key_terms if t in answer_lower]

    return {
        "key_terms":            key_terms,
        "terms_found_in_answer": found,
        "coverage_ratio":       round(len(found) / len(key_terms), 2) if key_terms else 0.0
    }


# ── Full evaluation report ────────────────────────────────────────────
def evaluate(question: str,
             retrieved_chunks: list[dict],
             generation_result: dict) -> dict:
    """
    Run all evaluation checks and return a combined report dict.
    Call this after retrieval + generation to get a full picture.
    """
    print(f"\n📋 Evaluating: '{question[:60]}...'")

    retrieval_scores = score_retrieval(retrieved_chunks)
    budget_report    = check_context_budget(retrieved_chunks)
    faithfulness     = check_faithfulness(
        generation_result.get("answer", ""), retrieved_chunks
    )

    report = {
        "question":       question,
        "retrieval":      retrieval_scores,
        "context_budget": budget_report,
        "faithfulness":   faithfulness,
        "model":          generation_result.get("model"),
        "prompt_tokens":  generation_result.get("prompt_tokens"),
    }

    print(
        f"  Retrieval — avg: {retrieval_scores['avg_similarity']}, "
        f"weak: {retrieval_scores['weak_retrieval']}\n"
        f"  Context  — {budget_report['total_tokens']} tokens, "
        f"over_budget: {budget_report['over_budget']}\n"
        f"  Faithful — {faithfulness['coverage_ratio']} "
        f"({len(faithfulness['terms_found_in_answer'])}/{len(faithfulness['key_terms'])} key terms)"
    )
    return report
