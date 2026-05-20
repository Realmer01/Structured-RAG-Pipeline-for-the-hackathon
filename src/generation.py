# src/generation.py
# LLM generation module — calls Gemini with a context-grounded prompt.
# Drawn from generate_answer() in rag_pipeline.py.
# Prompt templates are loaded from src/prompts.yaml.

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv()  # reads .env and loads GEMINI_API_KEY into os.environ

# ── Load config and prompts ──────────────────────────────────────────
_ROOT          = Path(__file__).parent.parent
_CONFIG_PATH   = _ROOT / "config" / "settings.yaml"
_PROMPTS_PATH  = Path(__file__).parent / "prompts.yaml"

def _load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_prompts() -> dict:
    with open(_PROMPTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg     = _load_config()
_prompts = _load_prompts()
_GEN_CFG = _cfg["generation"]

# ── Gemini client ─────────────────────────────────────────────────────
_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


# ── Prompt builder ────────────────────────────────────────────────────
def build_prompt(question: str,
                 context_chunks: list[str],
                 template_key: str = None) -> str:
    """
    Fill a named prompt template with context and question.
    Falls back to the template key defined in config/settings.yaml.
    """
    if template_key is None:
        template_key = _GEN_CFG["prompt_template"]

    template = _prompts.get(template_key)
    if template is None:
        raise ValueError(f"Prompt template '{template_key}' not found in prompts.yaml")

    context = "\n\n---\n\n".join(context_chunks)
    return template.format(context=context, question=question)


# ── Generation ────────────────────────────────────────────────────────
def generate_answer(question: str,
                    context_chunks: list[str],
                    template_key: str = None) -> dict:
    """
    Generate an answer using Gemini given a question and retrieved chunks.

    Returns a dict with:
      - answer:       the LLM's text response
      - prompt_tokens: approximate input token count
      - model:        the Gemini model used
      - template:     which prompt template was used
    """
    template_key = template_key or _GEN_CFG["prompt_template"]
    prompt = build_prompt(question, context_chunks, template_key)

    response = _client.models.generate_content(
        model=_GEN_CFG["model"],
        contents=prompt
    )

    prompt_tokens = len(prompt.split())  # approximation — same as rag_pipeline.py
    print(f"📊 Input tokens (approx): {prompt_tokens}")

    return {
        "answer":        response.text,
        "prompt_tokens": prompt_tokens,
        "model":         _GEN_CFG["model"],
        "template":      template_key
    }


# ── Full RAG generation step ──────────────────────────────────────────
def run_generation(question: str,
                   retrieved_chunks: list[dict],
                   similarity_threshold: float = None) -> dict:
    """
    High-level generation step called by the pipeline.
    Accepts output from retrieval.retrieve_with_scores().

    Automatically falls back to out_of_scope prompt if all retrieved
    chunks are below the similarity threshold.
    """
    if similarity_threshold is None:
        similarity_threshold = _cfg["evaluation"]["min_similarity_threshold"]

    # Check if any chunk clears the threshold
    strong_chunks = [r for r in retrieved_chunks
                     if r["similarity_score"] >= similarity_threshold]

    if not strong_chunks:
        print("⚠️  No chunks above similarity threshold — using out_of_scope prompt")
        return generate_answer(question, [], template_key="out_of_scope")

    texts = [r["text"] for r in strong_chunks]
    return generate_answer(question, texts)
