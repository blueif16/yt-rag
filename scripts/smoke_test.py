"""
End-to-end smoke test for the scaffold.

Validates, in order:
  1. .env is filled in (no FILL_IN_ placeholders left).
  2. Gemini embedding call succeeds (key + model reachable).
  3. Supabase insert+search succeeds (URL + key + migration applied).

Uses an ephemeral namespace and cleans up after itself. Idempotent.

Run:
    .venv/bin/python scripts/smoke_test.py
"""
from __future__ import annotations
import os
import sys
import uuid

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def check_env() -> None:
    required = {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_SECRET_KEY": os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    }
    missing = [k for k, v in required.items() if not v or str(v).startswith("FILL_IN_")]
    if missing:
        fail(f".env not filled in: {missing}")
    print("OK: .env has all required keys")


def check_gemini() -> None:
    from yt_rag.gemini_embeddings import GeminiEmbeddings
    emb = GeminiEmbeddings(model=os.getenv("EMBEDDING_MODEL") or None,
                           output_dimensionality=int(os.getenv("EMBEDDING_DIM", "768")))
    vec = emb.embed_query("smoke test")
    if not vec or len(vec) != int(os.getenv("EMBEDDING_DIM", "768")):
        fail(f"Gemini returned unexpected vector (len={len(vec) if vec else 0})")
    print(f"OK: Gemini embedding returned {len(vec)}-dim vector")


def check_supabase() -> None:
    from yt_rag import RAGStore
    ns = f"_smoke_{uuid.uuid4().hex[:8]}"
    rag = RAGStore(namespace=ns)
    try:
        r = rag.ingest(
            "Smoke test content about reciprocal rank fusion and graph traversal in retrieval.",
            source="smoke_test", type="note",
        )
        if r.get("status") != "created":
            fail(f"ingest unexpected status: {r}")
        hits = rag.search("rank fusion retrieval")
        if not hits:
            fail("search returned 0 results for known content (migration applied?)")
        print(f"OK: Supabase ingest+search round-trip succeeded ({len(hits)} hits)")
    finally:
        rag.delete_all()


if __name__ == "__main__":
    check_env()
    check_gemini()
    check_supabase()
    print("\nAll checks passed. Scaffold is live.")
