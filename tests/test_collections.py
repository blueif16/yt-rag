"""Collection rules: a collection is one utility/topic, never a channel."""
import pytest

from yt_rag.collections import (
    CollectionError,
    CollectionRegistry,
    validate_name,
    validate_purpose,
)


class FakeBackend:
    """In-memory stand-in for the Supabase `collections` table."""

    def __init__(self):
        self.rows: dict[str, str] = {}

    def get(self, name):
        return {"name": name, "purpose": self.rows[name]} if name in self.rows else None

    def insert(self, name, purpose):
        self.rows[name] = purpose

    def list(self):
        return [{"name": n, "purpose": p} for n, p in sorted(self.rows.items())]


PURPOSE = "Tutorials on building agentic RAG pipelines"


# ---- naming ----

@pytest.mark.parametrize("name", ["agentic-rag", "crypto-staking", "llm-evals", "abc"])
def test_accepts_topic_slugs(name):
    validate_name(name)


@pytest.mark.parametrize("name", ["yt_lewiswjackson", "yt-lewiswjackson", "@LewisWJackson"])
def test_rejects_channel_derived_names(name):
    with pytest.raises(CollectionError, match="channel"):
        validate_name(name)


@pytest.mark.parametrize("name", ["Agentic RAG", "agentic_rag", "ab", "-rag", "rag-", "a" * 41])
def test_rejects_malformed_slugs(name):
    with pytest.raises(CollectionError):
        validate_name(name)


@pytest.mark.parametrize("name", ["default", "misc", "general"])
def test_rejects_catch_all_names(name):
    with pytest.raises(CollectionError):
        validate_name(name)


# ---- purpose ----

def test_rejects_vague_purpose():
    with pytest.raises(CollectionError, match="purpose"):
        validate_purpose("  videos  ")


def test_accepts_sentence_purpose():
    validate_purpose(PURPOSE)


# ---- registry ----

def test_create_then_require_succeeds():
    reg = CollectionRegistry(FakeBackend())
    reg.create("agentic-rag", PURPOSE)
    assert reg.require("agentic-rag")["purpose"] == PURPOSE


def test_create_duplicate_is_rejected():
    reg = CollectionRegistry(FakeBackend())
    reg.create("agentic-rag", PURPOSE)
    with pytest.raises(CollectionError, match="exists"):
        reg.create("agentic-rag", "A different purpose that is long enough")


def test_create_validates_before_writing():
    backend = FakeBackend()
    with pytest.raises(CollectionError):
        CollectionRegistry(backend).create("yt_somechannel", PURPOSE)
    assert backend.rows == {}


def test_require_unknown_lists_existing_collections_and_purposes():
    reg = CollectionRegistry(FakeBackend())
    reg.create("agentic-rag", PURPOSE)
    with pytest.raises(CollectionError) as e:
        reg.require("rag-stuff")
    msg = str(e.value)
    assert "agentic-rag" in msg and PURPOSE in msg and "create_collection" in msg
