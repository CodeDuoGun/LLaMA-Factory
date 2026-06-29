#!/usr/bin/env python3
"""Index Wu Weiping prescription JSONL into Elasticsearch."""

from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
from typing import Any

from medical.config import config
from medical.rag.es.es_processor import (
    DEFAULT_INDEX_NAME,
    DEFAULT_WUWEIPING_JSONL_FILE,
    _load_elasticsearch,
    bulk_upsert,
    build_wuweiping_embedding_text,
    build_wuweiping_prescription_doc,
    create_index,
    embed_prescription_query,
    iter_jsonl,
)


def infer_dims(embedding_model_type: str) -> int:
    if embedding_model_type == "doubao":
        return 4096
    if embedding_model_type == "bge_code":
        return 1536
    return 1024


def make_client() -> Any:
    es_class = _load_elasticsearch()
    host = getattr(config, "ES_HOST", "localhost")
    port = getattr(config, "ES_PORT", "9200")
    user = getattr(config, "ES_USER", "")
    password = getattr(config, "ES_AUTH", "")
    auth = (user, password) if user else None
    return es_class(f"http://{host}:{port}", basic_auth=auth, request_timeout=120)


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def has_required_wuweiping_fields(row: dict[str, Any]) -> bool:
    diagnosis_sickness = str(row.get("diagnosis_sickness") or "").strip()
    diagnosis_disease = str(row.get("diagnosis_disease") or "").strip()
    prescriptions = row.get("ps") or []
    if isinstance(prescriptions, str):
        prescriptions = [prescriptions]
    has_prescription = any(str(item or "").strip() for item in prescriptions)
    return bool(diagnosis_sickness and diagnosis_disease and has_prescription)


def index_wuweiping_prescriptions(
    input_path: Path,
    index_name: str,
    dims: int,
    analyzer: str,
    recreate: bool,
    embedding_model_type: str,
    batch_size: int,
    limit: int | None = None,
) -> int:
    client = make_client()
    create_index(client, index_name, dims=dims, text_analyzer=analyzer, recreate=recreate)

    rows_iter = iter_jsonl(input_path)
    raw_rows = list(islice(rows_iter, limit)) if limit else list(rows_iter)
    rows = [row for row in raw_rows if has_required_wuweiping_fields(row)]
    skipped = len(raw_rows) - len(rows)
    if skipped:
        print(f"skipped {skipped} docs without diagnosis_sickness, diagnosis_disease, or ps")
    count = 0
    for row_batch in batched(rows, batch_size):
        docs = []
        for row in row_batch:
            embedding_text = build_wuweiping_embedding_text(row)
            embedding = embed_prescription_query(embedding_text, embedding_model_type=embedding_model_type)
            docs.append(build_wuweiping_prescription_doc(row, embedding=[float(item) for item in embedding]))
        bulk_upsert(client, index_name, docs)
        break
        count += len(docs)
        print(f"indexed {count}/{len(rows)} docs")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index Wu Weiping prescription JSONL into Elasticsearch.")
    parser.add_argument("--input", type=Path, default=DEFAULT_WUWEIPING_JSONL_FILE)
    parser.add_argument("--index", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--embedding-model", default="doubao", choices=["doubao", "bge", "bge_large_zh", "bge_code", "bgem3"])
    parser.add_argument("--dims", type=int, default=None)
    parser.add_argument("--analyzer", default="standard", help="Use ik_max_word if your ES cluster has IK installed.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dims = args.dims or infer_dims(args.embedding_model)
    count = index_wuweiping_prescriptions(
        input_path=args.input,
        index_name=args.index,
        dims=dims,
        analyzer=args.analyzer,
        recreate=args.recreate,
        embedding_model_type=args.embedding_model,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(f"finished indexing {count} docs into {args.index}")


if __name__ == "__main__":
    main()
