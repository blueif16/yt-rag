-- ============================================================================
-- Topic collections registry.
--
-- A collection groups videos by UTILITY (what they teach / what they're for),
-- never by channel. documents.namespace holds the collection name.
-- Rules are enforced in src/yt_rag/collections.py; this table just persists
-- each collection's purpose so the corpus stays self-describing.
-- ============================================================================

CREATE TABLE IF NOT EXISTS collections (
  name TEXT PRIMARY KEY CHECK (name ~ '^[a-z][a-z0-9]*(-[a-z0-9]+)*$' AND length(name) BETWEEN 3 AND 40),
  purpose TEXT NOT NULL CHECK (length(btrim(purpose)) >= 20),
  created_at TIMESTAMPTZ DEFAULT NOW()
);
