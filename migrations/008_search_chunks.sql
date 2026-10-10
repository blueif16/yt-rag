-- ============================================================================
-- search_chunks: the search the MCP server uses from now on.
--
-- Same hybrid as search_hybrid (BM25 'all' + vector, RRF, graph expansion),
-- plus what the after-effects pilot showed agents need:
--   similarity       absolute cosine similarity of each hit to the query, so a
--                    caller can tell "the library has nothing" from a real
--                    answer (RRF scores depend only on rank).
--   namespace        returned per row (no extra lookup for cross-collection search).
--   filter_meta      JSONB containment on the tag envelope, e.g.
--                    {"scenarios": ["ui-promo"], "type": "doc"}.
--   tools_available  only sources whose required plugins are all in this list
--                    (untagged sources pass).
-- The vector half is an exact scan (the "+ 0" keeps the planner off the HNSW
-- index): an index scan applies filters afterwards and can return too few rows
-- for a selective filter, and Supabase does not let a function set
-- hnsw.iterative_scan. Exact scan is a few ms at tens of thousands of chunks.
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_docs_metadata ON documents USING gin (metadata jsonb_path_ops);

CREATE OR REPLACE FUNCTION public.search_chunks(
  query_text TEXT,
  query_embedding extensions.vector(768),
  match_count INT DEFAULT 5,
  rrf_k INT DEFAULT 60,
  graph_depth INT DEFAULT 2,
  filter_namespace TEXT DEFAULT NULL,
  filter_meta JSONB DEFAULT NULL,
  tools_available JSONB DEFAULT NULL,
  keyword_mode TEXT DEFAULT 'all'
)
RETURNS TABLE (
  id BIGINT,
  namespace TEXT,
  content TEXT,
  metadata JSONB,
  source_type TEXT,
  score FLOAT,
  similarity FLOAT,
  depth INT
)
LANGUAGE plpgsql STABLE
SET search_path = public, extensions
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  WITH RECURSIVE
  kq AS (
    SELECT CASE WHEN keyword_mode = 'any'
      THEN replace(plainto_tsquery('english', query_text)::text, ' & ', ' | ')::tsquery
      ELSE plainto_tsquery('english', query_text)
    END AS tsq
  ),
  pool AS NOT MATERIALIZED (  -- inlined, so the vector/FTS indexes stay usable
    SELECT d.id, d.content, d.embedding FROM documents d
    WHERE (filter_namespace IS NULL OR d.namespace = filter_namespace)
      AND (filter_meta IS NULL OR d.metadata @> filter_meta)
      AND (tools_available IS NULL OR NOT (d.metadata ? 'tools')
           OR (d.metadata->'tools') <@ tools_available)
  ),
  fts AS (
    SELECT p.id, ts_rank_cd(to_tsvector('english', p.content), kq.tsq) AS rank
    FROM pool p, kq
    WHERE to_tsvector('english', p.content) @@ kq.tsq
    ORDER BY rank DESC
    LIMIT match_count * 3
  ),
  fts_ranked AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY rank DESC) AS rank_pos FROM fts
  ),
  vec AS (
    SELECT p.id, (p.embedding <=> query_embedding) + 0 AS dist
    FROM pool p
    ORDER BY dist
    LIMIT match_count * 3
  ),
  vec_ranked AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY dist) AS rank_pos FROM vec
  ),
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
  graph AS (
    SELECT
      d.id, d.namespace, d.content, d.metadata,
      'seed'::TEXT AS source_type,
      s.rrf_score AS score,
      (1 - (d.embedding <=> query_embedding))::FLOAT AS similarity,
      0 AS depth,
      ARRAY[d.id] AS path
    FROM seeds s
    JOIN documents d ON s.id = d.id

    UNION ALL

    SELECT
      d.id, d.namespace, d.content, d.metadata,
      r.type AS source_type,
      g.score * 0.8,
      (1 - (d.embedding <=> query_embedding))::FLOAT,
      g.depth + 1,
      g.path || d.id
    FROM graph g
    JOIN doc_relations r ON r.source_id = g.id
    JOIN documents d ON d.id = r.target_id
    WHERE g.depth < graph_depth
      AND NOT d.id = ANY(g.path)
      AND (filter_namespace IS NULL OR d.namespace = filter_namespace)
  ),
  best AS (
    SELECT DISTINCT ON (graph.id)
      graph.id, graph.namespace, graph.content, graph.metadata, graph.source_type,
      graph.score::FLOAT AS score, graph.similarity, graph.depth
    FROM graph
    ORDER BY graph.id, graph.score DESC
  )
  SELECT best.id, best.namespace, best.content, best.metadata, best.source_type,
         best.score, best.similarity, best.depth
  FROM best
  ORDER BY best.score DESC, best.depth;
END;
$$;
