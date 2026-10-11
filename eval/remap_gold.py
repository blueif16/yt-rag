"""
Map gold chunk ids from one ingest of a collection onto a re-ingest (re-chunking
changes ids). A new chunk inherits a gold id when it is from the same video/page
and overlaps it: time overlap for transcripts, shared word 5-grams for text.

    .venv/bin/python eval/remap_gold.py OLD_CHUNKS.jsonl NEW_COLLECTION QUESTIONS.json OUT.json
"""
from __future__ import annotations
import json
import re
import sys

from yt_rag import RAGStore


def grams(text: str, n: int = 5) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def body(c: str) -> str:
    return c.split("\n", 1)[1] if c.startswith("[") and "\n" in c else c


def main() -> None:
    old_path, collection, q_path, out_path = sys.argv[1:5]
    old = {c["id"]: c for c in map(json.loads, open(old_path))}
    rag = RAGStore(namespace=collection)
    new = rag.client.table("documents").select("id, content, metadata").eq(
        "namespace", collection).order("id").execute().data
    by_doc: dict[str, list[dict]] = {}
    for r in new:
        by_doc.setdefault(r["metadata"].get("doc_id"), []).append(r)

    def mapped(oid: int) -> list[int]:
        o = old[oid]
        hits = []
        for r in by_doc.get(o["doc_id"], []):
            m = r["metadata"]
            if o.get("chunk_start_sec") is not None and m.get("chunk_start_sec") is not None:
                a = max(o["chunk_start_sec"], m["chunk_start_sec"])
                b = min(o["chunk_end_sec"], m["chunk_end_sec"])
                span = max(1, o["chunk_end_sec"] - o["chunk_start_sec"])
                if (b - a) / span >= 0.3:
                    hits.append(r["id"])
            else:
                g = grams(body(o["content"]))
                if g and len(g & grams(body(r["content"]))) / len(g) >= 0.3:
                    hits.append(r["id"])
        return hits

    qs = json.loads(open(q_path).read())
    lost = 0
    for q in qs:
        new_ids = sorted({n for oid in q["gold_ids"] for n in mapped(oid)})
        q["gold_ids_v1"], q["gold_ids"] = q["gold_ids"], new_ids
        if not new_ids:
            lost += 1
            print(f"{q['id']}: gold lost (v1 {q['gold_ids_v1']})")
    open(out_path, "w").write(json.dumps(qs, indent=1, ensure_ascii=False))
    print(f"{len(qs) - lost}/{len(qs)} questions mapped")


if __name__ == "__main__":
    main()
