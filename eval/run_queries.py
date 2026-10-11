"""
Run a question set against one collection and record what search returns.

Plumbing only — scoring is done by a judge (agent) reading the output, or by
gold chunk ids when the question set has them (`gold_ids`).

    .venv/bin/python eval/run_queries.py QUESTIONS.json OUT.json --collection after-effects
        [--terms TERMS.json]

QUESTIONS.json: [{"id", "query", ...}]  (extra fields are copied through)
TERMS.json (optional): {id: "term query"} — the second, term-based search the
ae-library skill asks agents to run; recorded as "agent_terms".
OUT.json: per question, the passages an agent gets from the MCP search tool
with its defaults ("agent": search_passages, the per-source cap, passages of
adjacent chunks; each hit lists its chunk `ids`), and the plain vector top-k
("vector"), with timings: search_ms (embedding + Supabase).
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
from pathlib import Path

from yt_rag import RAGStore
from yt_rag.mcp_server import _passage_hit, search_passages


def _hit(r: dict, rank: int) -> dict:
    m = r.get("metadata") or {}
    return {"rank": rank, "id": r.get("id"), "type": m.get("type"), "title": m.get("title"),
            "author": m.get("author") or m.get("channel"), "locator": m.get("locator"),
            "source_url": m.get("source"), "content": r.get("content", "")}


def agent_hits(query: str, collection: str, top_k: int) -> list[dict]:
    """Exactly what an agent's search() call returns, plus each passage's chunk ids."""
    _, passages = search_passages(query, collection, top_k)
    return [{"rank": n + 1, "ids": [r["id"] for r in p], **_passage_hit(p, collection)}
            for n, p in enumerate(passages)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("out")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--terms", default=None, help="JSON {id: term query} for the second search")
    a = ap.parse_args()

    qs = json.loads(Path(a.questions).read_text())
    terms = json.loads(Path(a.terms).read_text()) if a.terms else {}
    rag = RAGStore(namespace=a.collection)
    results = []
    for q in qs:
        t0 = time.perf_counter()
        agent = agent_hits(q["query"], a.collection, a.top_k)
        t1 = time.perf_counter()
        vector = rag.search_vector(q["query"], top_k=a.top_k)
        rec = {**q, "search_ms": round((t1 - t0) * 1000), "agent": agent,
               "vector": [_hit(r, i + 1) for i, r in enumerate(vector)]}
        if q["id"] in terms:
            rec["terms"] = terms[q["id"]]
            rec["agent_terms"] = agent_hits(terms[q["id"]], a.collection, a.top_k)
        if q.get("gold_ids"):
            gold = set(q["gold_ids"])
            ranks = [h["rank"] for h in agent if gold & set(h["ids"])]
            rec["agent_gold_rank"] = min(ranks) if ranks else None
        results.append(rec)
        print(f"{q['id']}: search {rec['search_ms']} ms, "
              f"{sum(len(h['ids']) for h in agent)} chunks in {len(agent)} passages")

    Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    ms = [r["search_ms"] for r in results]
    print(f"\n{len(results)} queries · search median {statistics.median(ms)} ms (max {max(ms)})")
    if any("agent_gold_rank" in r for r in results):
        ranks = [r.get("agent_gold_rank") for r in results]
        n = len(ranks)
        print(f"agent: hit@1 {sum(1 for x in ranks if x == 1)}/{n} · "
              f"hit@{a.top_k} {sum(1 for x in ranks if x)}/{n} · "
              f"MRR {sum(1 / x for x in ranks if x) / n:.2f}")


if __name__ == "__main__":
    main()
