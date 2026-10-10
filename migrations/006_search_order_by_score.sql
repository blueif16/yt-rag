-- ============================================================================
-- Fix: search_context_mesh returned its hits in id order, not best-first.
--
-- The final SELECT DISTINCT ON (graph.id) ... ORDER BY graph.id is needed to
-- keep each document's best path through the graph, but it also sets the
-- output order — so callers saw the top-k sorted by row id. Wrap it and
-- order by score. (CREATE OR REPLACE drops SET clauses, so the pinned
-- search_path from 005 is restated here.)
-- ============================================================================

CREATE OR REPLACE FUNCTION public.search_context_mesh(
  query_text TEXT,
  query_embedding extensions.vector(768),
  match_count INT DEFAULT 5,
  rrf_k INT DEFAULT 60,
  graph_depth INT DEFAULT 2,
  filter_namespace TEXT DEFAULT NULL
)
RETURNS TABLE (
  id BIGINT,
  content TEXT,
  metadata JSONB,
  source_type TEXT,
  score FLOAT,
  depth INT
)
LANGUAGE plpgsql STABLE
SET search_path = public, extensions
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  WITH RECURSIVE
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
      d.id, d.content, d.metadata,
      'seed'::TEXT AS source_type,
      s.rrf_score AS score,
      0 AS depth,
      ARRAY[d.id] AS path
    FROM seeds s
    JOIN documents d ON s.id = d.id

    UNION ALL

    SELECT
      d.id, d.content, d.metadata,
      r.type AS source_type,
      g.score * 0.8,
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
      graph.id, graph.content, graph.metadata, graph.source_type,
      graph.score::FLOAT AS score, graph.depth
    FROM graph
    ORDER BY graph.id, graph.score DESC
  )
  SELECT best.id, best.content, best.metadata, best.source_type, best.score, best.depth
  FROM best
  ORDER BY best.score DESC, best.depth;
END;
$$;
