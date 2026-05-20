# Biomedical RAG Pipeline

A Retrieval-Augmented Generation (RAG) system built over a corpus of **200 clinical case reports** from PubMed. Documents are chunked, embedded with BGE, stored in a persistent ChromaDB vector store, and answered with **Gemini 2.5 Flash**.

---

## Project Structure

```
RAG/
├── .github/
│   └── workflows/
│       └── ci.yml              ← GitHub Actions: runs pytest on push/PR
├── config/
│   └── settings.yaml           ← All tunable parameters (models, chunk sizes, etc.)
├── dataset_for_project/        ← 200 biomedical .txt + .ann files (source dataset)
├── src/
│   ├── ingestion.py            ← Load .txt/.ann files, chunk text
│   ├── retrieval.py            ← ChromaDB vector store + BGE embeddings
│   ├── generation.py           ← Gemini LLM call + prompt management
│   ├── evaluation.py           ← Retrieval scoring, token budget, faithfulness
│   └── prompts.yaml            ← Named prompt templates
├── tests/
│   └── test_pipeline.py        ← Offline unit tests (no API keys required)
├── main.py                     ← Entrypoint for ingestion, retrieval, generation, evaluation
├── archive_prototype/         ← Legacy prototype scripts (not part of the core pipeline)
├── chroma_db/                  ← Persistent vector DB (gitignored)
├── .env                        ← API keys (gitignored — never commit this)
├── .gitignore
├── requirements.txt
└── README.md
```

Note: `main.py` and the `src/` package are the recommended runtime path. Legacy root-level prototype scripts have been moved to the `archive_prototype/` folder for reference.


---

## Setup

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd RAG

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
# Optional: set HUGGINGFACE_TOKEN if your embedding model requires authentication
# for private or gated models.
(echo GEMINI_API_KEY=your_key_here > .env) && \
(echo HUGGINGFACE_TOKEN=your_hf_token_here >> .env)
```

---

## Usage

### Ingest the dataset

```python
from src.ingestion import ingest_dataset
from src.retrieval import get_client, get_or_create_collection, ingest_chunks

# Load and chunk all 200 clinical case reports
chunks = ingest_dataset()

# Store in ChromaDB (deduplication built in — safe to re-run)
client     = get_client()
collection = get_or_create_collection(client)
ingest_chunks(collection, chunks)
```

### Preprocess CSV/JSON datasets

```bash
python main.py --preprocess
```

By default, preprocessing reads files from `./data/raw` and writes results to `./data/processed`.
If you have a custom YAML config, pass it with `--preprocess-config`:

```bash
python main.py --preprocess --preprocess-config config/preprocessing_dataset2.yaml
```

The preprocessing module loads raw CSV or JSON files, cleans and chunks text, writes a standardized output file, and generates a manifest.

### Ask a question

```python
from src.retrieval   import get_client, get_or_create_collection, retrieve_with_scores
from src.generation  import run_generation
from src.evaluation  import evaluate

client     = get_client()
collection = get_or_create_collection(client)

question = "What are the symptoms of Ebstein's anomaly?"
retrieved = retrieve_with_scores(question, collection)
result    = run_generation(question, retrieved)
report    = evaluate(question, retrieved, result)

print(result["answer"])
```

### Run the pipeline from `main.py`

```bash
# Ingest the dataset and ask a question in one command
python main.py --ingest --question "What are the symptoms of Ebstein's anomaly?"

# Run a single sample query without specifying a question
python main.py --sample

# Preprocess raw CSV/JSON input with a custom config, then ingest the results
python main.py --preprocess --preprocess-config config/preprocessing_dataset2.yaml --ingest

# Override the retrieval top-k value
python main.py --question "What is the treatment for arrhythmia?" --top-k 3
```

---

## How to run this pipeline

1. Create and activate the Python virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your Gemini API key in `.env`:

```bash
echo GEMINI_API_KEY=your_key_here > .env
```

4. Run preprocessing for CSV/JSON inputs:

```bash
python main.py --preprocess
```

5. Optionally preprocess with a custom config and ingest the results:

```bash
python main.py --preprocess --preprocess-config config/preprocessing_dataset2.yaml --ingest
```

6. Run the retrieval + generation pipeline on a question:

```bash
python main.py --question "What are the most common symptoms described for pulmonary embolism in the dataset?"
```

7. Run a quick demo using the sample question path:

```bash
python main.py --sample
```

### Exhibition checklist

- Start by showing the repository structure and explain that `main.py` is the entrypoint.
- Demonstrate `python main.py --preprocess --preprocess-config config/preprocessing_dataset2.yaml --ingest`.
- Show the generated `./data/processed/*.jsonl` output and the manifest file.
- Run `python main.py --sample` to display an actual retrieval + Gemini answer.
- Optionally show `python -c "from src.retrieval import get_client,get_or_create_collection; c=get_or_create_collection(get_client()); print('count', c.count())"` to confirm vector store size.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests run **fully offline** — no API key or ChromaDB writes needed.

---

## Dataset

The `dataset_for_project/` directory contains **200 PubMed clinical case reports** in [Brat standoff format](https://brat.nlplab.org/standoff.html):

| File type | Content |
|-----------|---------|
| `.txt` | Raw clinical case report text |
| `.ann` | Named entity annotations (Age, Sign_symptom, Disease_disorder, Diagnostic_procedure, Therapeutic_procedure, etc.) |

Entity types from `.ann` files are stored as chunk metadata in ChromaDB, enabling filtered retrieval in future iterations.

---

## Configuration

All parameters live in [`config/settings.yaml`](config/settings.yaml) — no hardcoded values in source files:

| Section | Key parameters |
|---------|---------------|
| `dataset` | raw_dir, processed_dir |
| `embedding` | model_name (`BAAI/bge-base-en-v1.5`), batch_size, BGE prefixes |
| `vector_store` | persist_dir, collection_name, distance_metric |
| `chunking` | strategy, chunk_size (words), overlap |
| `retrieval` | top_k |
| `generation` | model (`gemini-2.5-flash`), prompt_template |
| `evaluation` | max_context_tokens, min_similarity_threshold |

---

## What's Done

- [x] **Ingestion** (`src/ingestion.py`) — loads 200 `.txt`/`.ann` files, recursive chunking, entity metadata extraction
- [x] **Retrieval** (`src/retrieval.py`) — persistent ChromaDB, BGE embeddings, batch ingestion with deduplication, similarity-scored retrieval
- [x] **Generation** (`src/generation.py`) — Gemini 2.5 Flash, YAML prompt templates, automatic out-of-scope fallback
- [x] **Evaluation** (`src/evaluation.py`) — retrieval quality scoring, context token budget checks, basic faithfulness assessment
- [x] **Config** (`config/settings.yaml`) — all tunable parameters centralised
- [x] **Prompts** (`src/prompts.yaml`) — `rag_answer` and `out_of_scope` templates
- [x] **CI** (`.github/workflows/ci.yml`) — pytest on push/PR to `main`
- [x] **Tests** (`tests/test_pipeline.py`) — 7 offline unit tests

## What's Next

- [ ] **Evaluation dashboard** — visualise retrieval scores and token budgets across a question set
- [ ] **Annotation-aware retrieval** — filter by entity type (e.g. only retrieve chunks containing `Disease_disorder`)
- [ ] **Hierarchical chunking** — implement parent-child indexing from `chunking.py` into the pipeline
- [ ] **GraphRAG** — build a knowledge graph over entities in `.ann` files for graph-based retrieval
- [ ] **Batch question runner** — evaluate the pipeline over a fixed QA set and report aggregate metrics
- [ ] **Streamlit UI** — simple interface for querying the RAG system interactively
