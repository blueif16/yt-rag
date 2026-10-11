"""
Re-embed stored chunks with the current embedding model, in place.

Embedding models are not interchangeable (gemini-embedding-001 and -2 vectors
live in different spaces), so a store must be converted as a whole. This reads
each chunk's stored text — nothing is downloaded again — embeds it with
EMBEDDING_MODEL (or --model), and records metadata.embed_model. Rows already
on the target model are skipped, so it can be stopped and resumed.

    .venv/bin/python scripts/reembed.py [--model gemini-embedding-2] [--namespace after-effects]
"""
from __future__ import annotations
import argparse
import sys
import time

from yt_rag import RAGStore
from yt_rag.gemini_embeddings import GeminiEmbeddings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="default: EMBEDDING_MODEL or gemini-embedding-2")
    ap.add_argument("--namespace", default=None, help="default: every collection")
    ap.add_argument("--page", type=int, default=100)
    a = ap.parse_args()

    rag = RAGStore(namespace="_meta")
    emb = GeminiEmbeddings(model=a.model, output_dimensionality=rag.config.embedding_dim)
    model = emb.model

    def pending(count: bool = False):
        q = rag.client.table("documents").select(
            "id, content, metadata", count="exact" if count else None).or_(
            f'metadata->>embed_model.is.null,metadata->>embed_model.neq."{model}"')
        if a.namespace:
            q = q.eq("namespace", a.namespace)
        return q.order("id").limit(a.page).execute()

    first = pending(count=True)
    total, done, t0 = first.count, 0, time.time()
    print(f"{total} chunks to re-embed with {model}", file=sys.stderr)
    rows = first.data
    while rows:
        vecs = emb.embed_documents([r["content"] for r in rows],
                                   titles=[(r["metadata"] or {}).get("title") for r in rows])
        for r, v in zip(rows, vecs):
            rag.client.table("documents").update({
                "embedding": v,
                "metadata": {**(r["metadata"] or {}), "embed_model": model},
            }).eq("id", r["id"]).execute()
        done += len(rows)
        print(f"  {done}/{total}  ({time.time() - t0:.0f}s)", file=sys.stderr)
        rows = pending().data
    print(f"done: {done} chunks now on {model}")


if __name__ == "__main__":
    main()
