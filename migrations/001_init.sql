-- ============================================================================
-- 2026 SOTA: Supabase GraphRAG Schema (Gemini Embeddings Edition)
--
-- Philosophy:
-- - Content is rich and self-describing (gets embedded + FTS indexed)
-- - Metadata is minimal (just source/type for reference, not filtering)
-- - Power comes from superior indexing: RRF (BM25 + Vector) + Graph
-- - Gemini embeddings: 768 dimensions (balanced performance & quality)
-- ============================================================================

-- 1. Extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Documents Table
CREATE TABLE IF NOT EXISTS documents (
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,

  -- The knowledge (rich, self-describing)
  content TEXT NOT NULL,

  -- Minimal metadata (NOT for filtering, just reference)
  metadata JSONB DEFAULT '{}',
  -- Expected: {"source": "...", "type": "..."} - that's it

  -- Isolation
  namespace TEXT DEFAULT 'default',

  -- Indexing
  content_hash TEXT,  -- Deduplication
  embedding vector(768),  -- Gemini embeddings

  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),

  CONSTRAINT documents_content_check CHECK (content IS NOT NULL)
);

-- 3. Graph Edges (for relationships between knowledge)
CREATE TABLE IF NOT EXISTS doc_relations (
  id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
  source_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
  target_id BIGINT REFERENCES documents(id) ON DELETE CASCADE,
  type TEXT NOT NULL,  -- "relates_to", "contradicts", "contains", "complements"
  properties JSONB DEFAULT '{}',  -- weight, notes
  namespace TEXT DEFAULT 'default',
  created_at TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE(source_id, target_id, type, namespace)
);

-- 4. Indexes (THE POWER SOURCE)

-- Vector similarity (semantic search)
CREATE INDEX IF NOT EXISTS idx_docs_embedding
  ON documents USING hnsw (embedding vector_cosine_ops);

-- Full-text search (keyword/BM25-style)
CREATE INDEX IF NOT EXISTS idx_docs_fts
  ON documents USING GIN (to_tsvector('english', content));

-- Namespace isolation
CREATE INDEX IF NOT EXISTS idx_docs_namespace ON documents(namespace);

-- Deduplication
CREATE INDEX IF NOT EXISTS idx_docs_hash ON documents(content_hash);

-- Graph traversal
CREATE INDEX IF NOT EXISTS idx_rels_source ON doc_relations(source_id);
CREATE INDEX IF NOT EXISTS idx_rels_target ON doc_relations(target_id);

-- ============================================================================
-- 5. SOTA Search: Hybrid RRF + Graph Traversal
--
-- This is where the magic happens. Query by feeling/semantics,
-- get results from both keyword matches AND meaning matches,
-- fused with Reciprocal Rank Fusion, then expanded via graph.
-- ============================================================================

CREATE OR REPLACE FUNCTION search_context_mesh(
  query_text TEXT,
  query_embedding vector(768),
  match_count INT DEFAULT 5,
  rrf_k INT DEFAULT 60,
  graph_depth INT DEFAULT 2,
  filter_namespace TEXT DEFAULT NULL
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  metadata JSONB,
  source_type TEXT,  -- 'seed' or relation type
  score FLOAT,
  depth INT
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
  RETURN QUERY
  WITH RECURSIVE

  -- A. Full-Text Search (keyword matching)
  fts AS (
    SELECT
      d.id,
      ts_rank_cd(to_tsvector('english', d.content), plainto_tsquery('english', query_text)) AS rank
    FROM documents d
    WHERE to_tsvector('english', d.content) @@ plainto_tsquery('english', query_text)
      AND (filter_namespace IS NULL OR d.namespace = filter_namespace)
    ORDER BY rank DESC
    LIMIT match_count * 3
  ),
  fts_ranked AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY rank DESC) AS rank_pos FROM fts
  ),

  -- B. Vector Search (semantic matching)
  vec AS (
    SELECT
      d.id,
      d.embedding <=> query_embedding AS dist
    FROM documents d
    WHERE filter_namespace IS NULL OR d.namespace = filter_namespace
    ORDER BY dist
    LIMIT match_count * 3
  ),
  vec_ranked AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY dist) AS rank_pos FROM vec
  ),

  -- C. Reciprocal Rank Fusion
  rrf AS (
    SELECT
      COALESCE(f.id, v.id) AS id,
      COALESCE(1.0 / (rrf_k + f.rank_pos), 0.0) +
      COALESCE(1.0 / (rrf_k + v.rank_pos), 0.0) AS rrf_score
    FROM fts_ranked f
    FULL OUTER JOIN vec_ranked v ON f.id = v.id
  ),
  seeds AS (
    SELECT id, rrf_score FROM rrf ORDER BY rrf_score DESC LIMIT match_count
  ),

  -- D. Graph Expansion
  graph AS (
    -- Seeds
    SELECT
      d.id, d.content, d.metadata,
      'seed'::TEXT AS source_type,
      s.rrf_score AS score,
      0 AS depth,
      ARRAY[d.id] AS path
    FROM seeds s
    JOIN documents d ON s.id = d.id

    UNION ALL

    -- Traverse
    SELECT
      d.id, d.content, d.metadata,
      r.type AS source_type,
      g.score * 0.8,  -- Decay
      g.depth + 1,
      g.path || d.id
    FROM graph g
    JOIN doc_relations r ON r.source_id = g.id
    JOIN documents d ON d.id = r.target_id
    WHERE g.depth < graph_depth
      AND NOT d.id = ANY(g.path)
      AND (filter_namespace IS NULL OR d.namespace = filter_namespace)
  )

  SELECT DISTINCT ON (graph.id)
    graph.id,
    graph.content,
    graph.metadata,
    graph.source_type,
    graph.score::FLOAT,
    graph.depth
  FROM graph
  ORDER BY graph.id, graph.score DESC;
END;
$$;

-- ============================================================================
-- 6. Simple Vector Search (fast mode, skip graph)
-- ============================================================================

CREATE OR REPLACE FUNCTION search_vector(
  query_embedding vector(768),
  match_count INT DEFAULT 5,
  filter_namespace TEXT DEFAULT NULL
)
RETURNS TABLE (id BIGINT, content TEXT, metadata JSONB, similarity FLOAT)
LANGUAGE sql STABLE
AS $$
  SELECT
    id, content, metadata,
    (1 - (embedding <=> query_embedding))::FLOAT AS similarity
  FROM documents
  WHERE filter_namespace IS NULL OR namespace = filter_namespace
  ORDER BY embedding <=> query_embedding
  LIMIT match_count;
$$;
