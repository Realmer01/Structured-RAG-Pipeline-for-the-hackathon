import asyncio
import csv
import html
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import yaml

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

_DEFAULT_CONFIG = {
    "input": {
        "paths": ["./data/raw"],
        "file_types": ["csv", "json"],
        "csv": {
            "delimiter": ",",
            "encoding": "utf-8",
            "text_fields": ["text"],
            "metadata_fields": [],
            "required_fields": ["text"],
        },
        "json": {
            "encoding": "utf-8",
            "record_path": None,
            "text_fields": ["text"],
            "metadata_fields": [],
            "required_fields": ["text"],
        },
    },
    "output": {
        "format": "jsonl",
        "dir": "./data/processed",
        "filename": "preprocessed_chunks.jsonl",
        "manifest_path": "./data/processed/manifest.json",
        "write_manifest": True,
    },
    "chunking": {
        "chunk_size": 150,
        "overlap": 20,
        "min_chunk_words": 20,
    },
    "cleanup": {
        "html": True,
        "remove_non_printable": True,
        "normalize_whitespace": True,
    },
    "tokenizer": {
        "type": "approx",
        "approx_ratio": 0.75,
    },
}


def _load_config(config_path: str | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "preprocessing.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return _DEFAULT_CONFIG.copy()

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    config = _DEFAULT_CONFIG.copy()
    config.update(loaded)

    # Deep merge minimal nested structures.
    for key in ["input", "output", "chunking", "cleanup", "tokenizer"]:
        if key in loaded:
            config[key].update(loaded[key] or {})

    if "csv" in loaded.get("input", {}):
        config["input"]["csv"].update(loaded["input"]["csv"] or {})
    if "json" in loaded.get("input", {}):
        config["input"]["json"].update(loaded["input"]["json"] or {})

    return config


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _remove_non_printable(text: str) -> str:
    return "".join(ch for ch in text if ch.isprintable())


def _strip_html(text: str) -> str:
    if BeautifulSoup is not None:
        try:
            return BeautifulSoup(text, "lxml").get_text(separator=" ")
        except Exception:
            return BeautifulSoup(text, "html.parser").get_text(separator=" ")

    # Fall back to a simple regex-based cleanup.
    text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def clean_text(text: str,
               html_clean: bool = True,
               remove_non_printable: bool = True,
               normalize_whitespace: bool = True) -> str:
    if html_clean:
        text = _strip_html(text)
    if remove_non_printable:
        text = _remove_non_printable(text)
    if normalize_whitespace:
        text = _normalize_whitespace(text)
    return html.unescape(text).strip()


def count_tokens_approx(text: str, ratio: float = 0.75) -> int:
    return max(1, int(len(text.split()) / ratio))


def count_tokens_precise(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return count_tokens_approx(text)


def chunk_text(text: str,
               chunk_size: int = 150,
               overlap: int = 20,
               min_chunk_words: int = 20) -> list[dict[str, Any]]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) >= min_chunk_words or not chunks:
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "chunk_index": len(chunks),
                "word_count": len(chunk_words),
            })
        start += max(1, chunk_size - overlap)
    return chunks


def _extract_json_records(data: Any, record_path: str | None) -> list[dict[str, Any]]:
    if record_path is None:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    current = data
    for part in record_path.split("."):
        if isinstance(current, dict):
            current = current.get(part, {})
        else:
            current = {}
    if isinstance(current, list):
        return current
    if isinstance(current, dict):
        return [current]
    return []


async def _read_file_text(path: Path, encoding: str) -> str:
    async with aiofiles.open(path, "r", encoding=encoding) as f:
        return await f.read()


async def _load_csv(path: Path, encoding: str, delimiter: str) -> list[dict[str, Any]]:
    raw = await _read_file_text(path, encoding)
    reader = csv.DictReader(raw.splitlines(), delimiter=delimiter)
    return [dict(row) for row in reader]


async def _load_json(path: Path, encoding: str, record_path: str | None) -> list[dict[str, Any]]:
    raw = await _read_file_text(path, encoding)
    data = json.loads(raw)
    return _extract_json_records(data, record_path)


async def _find_input_files(config: dict) -> list[Path]:
    input_cfg = config["input"]
    file_types = set(input_cfg["file_types"])
    paths = input_cfg["paths"]
    files: list[Path] = []

    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for ext in file_types:
                files.extend(sorted(path.rglob(f"*.{ext}")))
        elif path.is_file():
            files.append(path)
        else:
            logging.warning("Input path does not exist: %s", raw_path)

    return sorted(files)


def _resolve_record_id(record: dict[str, Any], metadata_fields: list[str], file_meta: dict[str, Any]) -> str:
    for key in ["id", "source_id"] + metadata_fields:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"{file_meta['source_name']}_{file_meta['record_index']}"


async def _process_record(record: dict[str, Any],
                          file_meta: dict[str, Any],
                          config: dict) -> list[dict[str, Any]]:
    source_cfg = config["input"]
    file_type = file_meta["file_type"]
    format_cfg = source_cfg.get(file_type, {})

    text_fields = format_cfg.get("text_fields", [])
    metadata_fields = format_cfg.get("metadata_fields", [])
    required_fields = format_cfg.get("required_fields", [])

    text_parts = [str(record.get(field, "")) for field in text_fields if record.get(field)]
    text = "\n".join(text_parts).strip()
    if not text:
        return []

    missing = [field for field in required_fields if not str(record.get(field, "")).strip()]
    if missing:
        logging.debug("Skipping record because required fields are missing: %s", missing)
        return []

    cleanup_cfg = config["cleanup"]
    cleaned = clean_text(
        text,
        html_clean=cleanup_cfg["html"],
        remove_non_printable=cleanup_cfg["remove_non_printable"],
        normalize_whitespace=cleanup_cfg["normalize_whitespace"],
    )

    chunks = chunk_text(
        cleaned,
        chunk_size=config["chunking"]["chunk_size"],
        overlap=config["chunking"]["overlap"],
        min_chunk_words=config["chunking"]["min_chunk_words"],
    )

    source_id = _resolve_record_id(record, metadata_fields, file_meta)
    metadata = {
        "source_file": file_meta["source_path"],
        "source_name": file_meta["source_name"],
        "source_type": file_type,
        "source_id": source_id,
    }

    for field in metadata_fields:
        if field in record:
            metadata[field] = record[field]

    result_chunks = []
    for chunk in chunks:
        chunk_id = f"{source_id}_chunk_{chunk['chunk_index']}"
        token_count = count_tokens_approx(chunk["text"], config["tokenizer"]["approx_ratio"])
        result_chunks.append({
            "id": chunk_id,
            "text": chunk["text"],
            "metadata": metadata.copy(),
            "chunk_index": chunk["chunk_index"],
            "word_count": chunk["word_count"],
            "token_count": token_count,
        })

    return result_chunks


async def _write_output(chunks: list[dict[str, Any]], config: dict) -> Path:
    output_cfg = config["output"]
    output_dir = Path(output_cfg["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_cfg["filename"]

    if output_cfg["format"] == "json":
        async with aiofiles.open(output_path, "w", encoding="utf-8") as out:
            await out.write(json.dumps(chunks, indent=2, ensure_ascii=False))
    else:
        async with aiofiles.open(output_path, "w", encoding="utf-8") as out:
            for chunk in chunks:
                await out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return output_path


async def _write_manifest(chunks: list[dict[str, Any]], input_files: list[Path], config: dict) -> Path:
    manifest_path = Path(config["output"]["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "output_file": str(Path(config["output"]["dir"]) / config["output"]["filename"]),
        "output_format": config["output"]["format"],
        "input_paths": [str(p) for p in input_files],
        "file_count": len(input_files),
        "chunk_count": len(chunks),
        "token_count": sum(c.get("token_count", 0) for c in chunks),
        "chunking": config["chunking"],
        "cleanup": config["cleanup"],
    }

    async with aiofiles.open(manifest_path, "w", encoding="utf-8") as out:
        await out.write(json.dumps(manifest, indent=2, ensure_ascii=False))

    return manifest_path


async def preprocess_dataset_async(config_path: str | None = None) -> list[dict[str, Any]]:
    config = _load_config(config_path)
    logging.basicConfig(level=logging.INFO, format="[preprocess] %(message)s")

    input_files = await _find_input_files(config)
    if not input_files:
        logging.warning("No CSV or JSON input files found for preprocessing.")
        return []

    chunks: list[dict[str, Any]] = []
    for file_index, file_path in enumerate(input_files):
        file_type = file_path.suffix.lstrip(".").lower()
        if file_type not in config["input"]["file_types"]:
            continue

        logging.info("Loading %s", file_path)
        if file_type == "csv":
            records = await _load_csv(file_path, config["input"]["csv"]["encoding"], config["input"]["csv"]["delimiter"])
        else:
            records = await _load_json(file_path, config["input"]["json"]["encoding"], config["input"]["json"]["record_path"])

        for record_index, record in enumerate(records):
            file_meta = {
                "source_path": str(file_path),
                "source_name": file_path.stem,
                "file_type": file_type,
                "file_index": file_index,
                "record_index": record_index,
            }
            record_chunks = await _process_record(record, file_meta, config)
            chunks.extend(record_chunks)

    output_path = await _write_output(chunks, config)
    logging.info("Wrote %s chunks to %s", len(chunks), output_path)

    if config["output"]["write_manifest"]:
        manifest_path = await _write_manifest(chunks, input_files, config)
        logging.info("Manifest written to %s", manifest_path)

    return chunks


def preprocess_dataset(config_path: str | None = None) -> list[dict[str, Any]]:
    return asyncio.run(preprocess_dataset_async(config_path))
