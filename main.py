import argparse
import sys
from pathlib import Path

from src.ingestion import ingest_dataset
from src.preprocessing import preprocess_dataset
from src.retrieval import (
    get_client,
    get_or_create_collection,
    ingest_chunks,
    retrieve_with_scores,
)
from src.generation import run_generation
from src.evaluation import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the biomedical RAG pipeline end to end."
    )

    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run CSV/JSON preprocessing before ingestion.",
    )
    parser.add_argument(
        "--preprocess-config",
        type=str,
        default=None,
        help="Optional YAML config file for preprocessing.",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest the dataset into ChromaDB before querying.",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="The question to ask the RAG system.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override the number of retrieval chunks to use.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Ask a sample question instead of providing one explicitly.",
    )
    return parser.parse_args()


def ingest_if_requested(collection, should_ingest: bool, preprocessed_chunks=None) -> None:
    if not should_ingest:
        return

    print("\n=== Ingesting dataset into ChromaDB ===")
    if preprocessed_chunks is not None:
        chunks = preprocessed_chunks
    else:
        chunks = ingest_dataset()

    added = ingest_chunks(collection, chunks, source_name="pubmed_biomedical_dataset")
    print(f"Ingestion complete. New chunks added: {added}\n")


def run_question_pipeline(collection, question: str, top_k: int | None = None) -> None:
    print("\n=== Running query pipeline ===")
    print(f"Question: {question}\n")

    retrieved = retrieve_with_scores(question, collection, top_k=top_k)
    if not retrieved:
        print("No retrieval results returned from the vector store.")
        return

    generation_result = run_generation(question, retrieved)
    evaluation_report = evaluate(question, retrieved, generation_result)

    print("\n=== Result ===")
    print(generation_result["answer"])

    print("\n=== Evaluation ===")
    print(f"Model: {evaluation_report['model']}")
    print(f"Prompt tokens: {evaluation_report['prompt_tokens']}")
    print(f"Retrieval avg similarity: {evaluation_report['retrieval']['avg_similarity']}")
    print(f"Weak retrieval: {evaluation_report['retrieval']['weak_retrieval']}")
    print(f"Context tokens: {evaluation_report['context_budget']['total_tokens']}")
    print(f"Faithfulness coverage: {evaluation_report['faithfulness']['coverage_ratio']}\n")


def main() -> int:
    args = parse_args()

    if not args.question and not args.sample and not args.preprocess:
        print("Error: select --question, --sample, or --preprocess to run the pipeline.")
        return 1

    question = args.question
    if args.sample:
        question = (
            question
            or "What are the most common symptoms described for pulmonary embolism in the dataset?"
        )

    preprocessed_chunks = None
    if args.preprocess:
        print("\n=== Preprocessing input files ===")
        preprocessed_chunks = preprocess_dataset(args.preprocess_config)
        print(f"Preprocessing complete. Generated {len(preprocessed_chunks)} chunks.\n")
        if not args.ingest and not question:
            return 0

    client = get_client()
    collection = get_or_create_collection(client)

    ingest_if_requested(collection, args.ingest, preprocessed_chunks)

    if question:
        run_question_pipeline(collection, question, top_k=args.top_k)
        return 0

    print("No question was provided.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
