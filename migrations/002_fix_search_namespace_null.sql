-- Fix: search_context_mesh throws "column reference 'id' is ambiguous"
-- when called with filter_namespace = NULL.
--
-- Root cause: the function's RETURNS TABLE declares an `id` output column,
-- which PL/pgSQL then treats as a variable in the function body. The
-- unqualified `id` reference inside the `fts_ranked` and `vec_ranked` CTEs
-- becomes ambiguous between that variable and the underlying column. The
-- query planner only takes the offending path when no namespace filter
-- narrows the candidate set.
--
-- Fix: tell PL/pgSQL to prefer columns over variables for name conflicts.

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
  source_type TEXT,
  score FLOAT,
  depth INT
)
LANGUAGE plpgsql STABLE
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
