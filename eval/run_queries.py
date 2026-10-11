"""
Run a question set against one collection and record what search returns.

Plumbing only — scoring is done by a judge (agent) reading the output, or by
gold chunk ids when the question set has them (`gold_ids`).

    .venv/bin/python eval/run_queries.py QUESTIONS.json OUT.json --collection after-effects

QUESTIONS.json: [{"id", "query", ...}]  (extra fields are copied through)
OUT.json: per question, the top-k hits for three modes — "agent" (the MCP
search tool exactly as agents call it: search_chunks, per-source cap,
relevance), "hybrid" (search_hybrid, --keyword-mode all|any) and "vector"
(vector only) — with
timings: embed_ms (Gemini query embedding) and rpc_ms (Supabase search).
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path

from yt_rag import RAGStore
from yt_rag.mcp_server import search as agent_search


# An RRF hit found by only one retriever scores at most 1/(rrf_k+1) = 1/61.
ONE_RETRIEVER_MAX = 1 / 61 + 1e-9


def _hit(r: dict, rank: int) -> dict:
    m = r.get("metadata") or {}
    out = {"rank": rank, "id": r.get("id"), "type": m.get("type"), "title": m.get("title"),
            "author": m.get("author") or m.get("channel"), "locator": m.get("locator"),
            "source_url": m.get("source"), "content": r.get("content", "")}
    if "score" in r:  # hybrid only: did keyword (BM25) AND vector both find it?
        out["score"] = round(r["score"], 5)
        out["both_retrievers"] = r["score"] > ONE_RETRIEVER_MAX
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--filter-meta", default=None,
                    help='JSON containment filter for the vector run, e.g. {"batch": "pilot"}')
    ap.add_argument("--keyword-mode", choices=["all", "any"], default="all",
                    help="BM25 half: all query words must match (all) or any (any)")
    a = ap.parse_args()

    qs = json.loads(Path(a.questions).read_text())
    rag = RAGStore(namespace=a.collection)
    results = []
    for q in qs:
        t0 = time.perf_counter()
        emb = rag._embeddings.embed_query(q["query"])
        t1 = time.perf_counter()
        hybrid = rag.client.rpc("search_hybrid", {
            "query_text": q["query"], "query_embedding": emb, "match_count": a.top_k,
            "rrf_k": rag.config.rrf_k, "graph_depth": rag.config.graph_depth,
            "filter_namespace": a.collection, "keyword_mode": a.keyword_mode,
        }).execute().data
        t2 = time.perf_counter()
        if a.filter_meta:  # vector-only over a subset (search_chunks without its keyword half)
            vector = rag.client.rpc("search_chunks", {
                "query_text": "", "query_embedding": emb, "match_count": a.top_k,
                "filter_namespace": a.collection, "filter_meta": json.loads(a.filter_meta),
            }).execute().data
        else:
            vector = rag.client.rpc("search_vector", {
                "query_embedding": emb, "match_count": a.top_k, "filter_namespace": a.collection,
            }).execute().data
        hybrid = hybrid[:a.top_k]  # best-first since migration 006
        t3 = time.perf_counter()
        # exactly what an agent's MCP search() call returns (per-source cap, relevance)
        agent = agent_search(q["query"], collection=a.collection, top_k=a.top_k)
        rec = {**q,
               "embed_ms": round((t1 - t0) * 1000), "rpc_ms": round((t2 - t1) * 1000),
               "agent_ms": round((time.perf_counter() - t3) * 1000),
               "agent": [{"rank": i + 1, **h} for i, h in enumerate(agent)],
               "hybrid": [_hit(r, i + 1) for i, r in enumerate(hybrid)],
               "vector": [_hit(r, i + 1) for i, r in enumerate(vector)]}
        if q.get("gold_ids"):
            gold = set(q["gold_ids"])
            for mode in ("hybrid", "vector"):
                ranks = [h["rank"] for h in rec[mode] if h["id"] in gold]
                rec[f"{mode}_gold_rank"] = min(ranks) if ranks else None
        results.append(rec)
        print(f"{q['id']}: embed {rec['embed_ms']} ms, rpc {rec['rpc_ms']} ms, "
              f"top types {[h['type'] for h in rec['hybrid']]}")

    Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    emb_ms = [r["embed_ms"] for r in results]
    rpc_ms = [r["rpc_ms"] for r in results]
    kw = sum(1 for r in results if any(h.get("both_retrievers") for h in r["hybrid"]))
    print(f"\nkeyword+vector agreed on at least one top hit in {kw}/{len(results)} queries")
    print(f"{len(results)} queries · embed median {statistics.median(emb_ms)} ms · "
          f"rpc median {statistics.median(rpc_ms)} ms (max {max(rpc_ms)})")
    if any("hybrid_gold_rank" in r for r in results):
        for mode in ("hybrid", "vector"):
            ranks = [r.get(f"{mode}_gold_rank") for r in results]
            n = len(ranks)
            print(f"{mode}: hit@1 {sum(1 for x in ranks if x == 1)}/{n} · "
                  f"hit@{a.top_k} {sum(1 for x in ranks if x)}/{n} · "
                  f"MRR {sum(1 / x for x in ranks if x) / n:.2f}")


if __name__ == "__main__":
    main()
