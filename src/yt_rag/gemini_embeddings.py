"""
Gemini embeddings (LangChain-compatible interface: embed_query / embed_documents).

Two model generations, chosen by EMBEDDING_MODEL (default gemini-embedding-2):
- gemini-embedding-2: no task_type; the task goes into the text itself —
  queries as "task: search result | query: …", documents as
  "title: … | text: …". Several strings in one `contents` would be merged into
  ONE embedding, so each document is sent as its own Content.
- gemini-embedding-001: task_type RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT.
The two models' vectors are incompatible: a collection must be embedded with
one model throughout (scripts/reembed.py converts a store).
"""
import os
import re
import sys
import time
from typing import List
import numpy as np
from google import genai
from google.genai import errors, types

DEFAULT_MODEL = "gemini-embedding-2"
RETRY_IN_RE = re.compile(r"retry in ([\d.]+)s", re.I)


def _with_retry(call, attempts: int = 8):
    """Run a Gemini call, waiting out 429 rate limits.

    The free tier counts every text in a batch as one request (100/min), so
    bulk ingest hits it routinely; the error says how long to wait."""
    for i in range(attempts):
        try:
            return call()
        except errors.ClientError as e:
            if e.code != 429 or i == attempts - 1:
                raise
            if "PerDay" in str(e):  # a daily quota does not clear by waiting
                raise RuntimeError("Gemini daily embedding quota exhausted — resume tomorrow "
                                   "or use a paid key (ingest is idempotent)") from e
            m = RETRY_IN_RE.search(str(e))
            wait = min(float(m.group(1)) + 2, 90) if m else 30 * (i + 1)
            quota = re.findall(r"quotaId': '([^']+)'", str(e))
            print(f"  gemini 429 {quota or ''}: waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)


def is_v2(model: str) -> bool:
    return model.startswith("gemini-embedding-2")


def format_query_v2(text: str) -> str:
    return f"task: search result | query: {text}"


def format_doc_v2(text: str, title: str | None = None) -> str:
    return f"title: {title or 'none'} | text: {text}"


class GeminiEmbeddings:
    MAX_BATCH = 20  # texts per request: keeps one request under the per-minute token quota

    def __init__(
        self,
        model: str | None = None,
        output_dimensionality: int = 768,
        task_type: str = "RETRIEVAL_DOCUMENT",
        api_key: str | None = None
    ):
        self.model = model or os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type  # gemini-embedding-001 only
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Truncated (non-3072) vectors must be unit length for cosine search.
        gemini-embedding-2 already normalizes; doing it again is harmless."""
        if self.output_dimensionality == 3072:
            return embedding
        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm == 0:
            return embedding
        return (embedding_np / norm).tolist()

    def _config(self, task_type: str | None) -> types.EmbedContentConfig:
        if is_v2(self.model):
            return types.EmbedContentConfig(output_dimensionality=self.output_dimensionality)
        return types.EmbedContentConfig(task_type=task_type,
                                        output_dimensionality=self.output_dimensionality)

    def embed_query(self, text: str) -> List[float]:
        contents = format_query_v2(text) if is_v2(self.model) else text
        result = _with_retry(lambda: self.client.models.embed_content(
            model=self.model, contents=contents, config=self._config("RETRIEVAL_QUERY")))
        return self._normalize_embedding(result.embeddings[0].values)

    def embed_documents(self, texts: List[str],
                        titles: List[str | None] | None = None) -> List[List[float]]:
        """One vector per text. `titles` (optional, same length) are used by
        gemini-embedding-2's "title: … | text: …" document format."""
        titles = titles or [None] * len(texts)
        out: List[List[float]] = []
        for i in range(0, len(texts), self.MAX_BATCH):
            out.extend(self._embed_documents(texts[i:i + self.MAX_BATCH],
                                             titles[i:i + self.MAX_BATCH]))
        return out

    def _embed_documents(self, texts: List[str], titles: List[str | None]) -> List[List[float]]:
        if is_v2(self.model):
            # each text its own Content — a list of plain strings would be
            # merged into a single embedding by gemini-embedding-2
            contents = [types.Content(parts=[types.Part(text=format_doc_v2(t, ti))])
                        for t, ti in zip(texts, titles)]
        else:
            contents = texts
        result = _with_retry(lambda: self.client.models.embed_content(
            model=self.model, contents=contents, config=self._config(self.task_type)))
        if len(result.embeddings) != len(texts):
            raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(result.embeddings)}")
        return [self._normalize_embedding(e.values) for e in result.embeddings]
