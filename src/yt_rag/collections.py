"""
Collections: the only grouping unit of the corpus.

A collection is defined by UTILITY — one topic / one job the videos serve
(e.g. "agentic-rag", "crypto-staking") — never by who uploaded them. Channels
are just a source of videos; one collection can draw from many channels.

Rules (enforced here, so every ingest path obeys them):
  1. Names are topic slugs: lowercase kebab-case, 3-40 chars.
  2. Names may not be channel-derived ("yt_<handle>", "@handle") or catch-alls
     ("default", "misc", ...) — those are how the corpus sprawls untracked.
  3. Every collection carries a one-sentence purpose, stored with it.
  4. Ingest only into a collection that already exists. Creating one is a
     separate, deliberate step (create_collection), so a caller always sees
     the existing collections first and reuses one when it fits.

The collection name is stored as the `namespace` column of `documents`.
"""
from __future__ import annotations
import re

NAME_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
CATCH_ALL = {"default", "misc", "general", "other", "stuff", "temp", "test", "videos"}
MIN_PURPOSE_CHARS = 20


class CollectionError(ValueError):
    pass


def validate_name(name: str) -> None:
    if name.startswith(("yt_", "yt-", "@")):
        raise CollectionError(
            f"'{name}' looks channel-derived. Collections group by topic/utility, "
            "not by channel — name it for what the videos teach (e.g. 'agentic-rag')."
        )
    if not (3 <= len(name) <= 40) or not NAME_RE.match(name):
        raise CollectionError(
            f"'{name}' is not a valid collection name: use lowercase kebab-case, "
            "3-40 chars (e.g. 'llm-evals')."
        )
    if name in CATCH_ALL:
        raise CollectionError(f"'{name}' is a catch-all; name the specific topic instead.")


def validate_purpose(purpose: str) -> None:
    if len((purpose or "").strip()) < MIN_PURPOSE_CHARS:
        raise CollectionError(
            f"purpose must be a sentence (>= {MIN_PURPOSE_CHARS} chars) saying what "
            "these videos are for, e.g. 'Tutorials on building agentic RAG pipelines'."
        )


class SupabaseBackend:
    """`collections` table (migrations/003_collections.sql)."""

    def __init__(self, client):
        self.client = client

    def get(self, name):
        r = self.client.table("collections").select("name, purpose").eq("name", name).execute()
        return r.data[0] if r.data else None

    def insert(self, name, purpose):
        self.client.table("collections").insert({"name": name, "purpose": purpose}).execute()

    def list(self):
        return self.client.table("collections").select("name, purpose").order("name").execute().data


class CollectionRegistry:
    def __init__(self, backend):
        self.backend = backend

    def list(self) -> list[dict]:
        return self.backend.list()

    def create(self, name: str, purpose: str) -> dict:
        validate_name(name)
        validate_purpose(purpose)
        if self.backend.get(name):
            raise CollectionError(f"collection '{name}' already exists.")
        self.backend.insert(name, purpose.strip())
        return {"name": name, "purpose": purpose.strip()}

    def require(self, name: str) -> dict:
        row = self.backend.get(name)
        if row:
            return row
        existing = "\n".join(f"  - {c['name']}: {c['purpose']}" for c in self.list()) or "  (none)"
        raise CollectionError(
            f"collection '{name}' does not exist. Reuse one whose purpose fits:\n"
            f"{existing}\n"
            "Only if none fits, call create_collection(name, purpose) first."
        )
