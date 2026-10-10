-- ============================================================================
-- search_hybrid: search_context_mesh + keyword_mode for the BM25 half.
--
-- 'all' (default, unchanged): plainto_tsquery — every query word must appear
--   in the chunk. Agents search in full sentences, so this almost never
--   matches: in the after-effects pilot the keyword half contributed to 0/24
--   queries and hybrid search was really vector-only.
-- 'any': the same words OR-ed; ts_rank_cd still ranks chunks that contain more
--   of them, closer together, higher.
-- A new name rather than a new overload of search_context_mesh: an overload
-- with a defaulted extra parameter makes PostgREST calls ambiguous, and
-- replacing it would need a DROP. search_context_mesh stays as it was.
-- ============================================================================

CREATE OR REPLACE FUNCTION public.search_hybrid(
  query_text TEXT,
  query_embedding extensions.vector(768),
  match_count INT DEFAULT 5,
  rrf_k INT DEFAULT 60,
  graph_depth INT DEFAULT 2,
  filter_namespace TEXT DEFAULT NULL,
  keyword_mode TEXT DEFAULT 'all'
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
  kq AS (
    SELECT CASE WHEN keyword_mode = 'any'
      THEN replace(plainto_tsquery('english', query_text)::text, ' & ', ' | ')::tsquery
      ELSE plainto_tsquery('english', query_text)
    END AS tsq
  ),
  fts AS (
    SELECT
      d.id,
      ts_rank_cd(to_tsvector('english', d.content), kq.tsq) AS rank
    FROM documents d, kq
    WHERE to_tsvector('english', d.content) @@ kq.tsq
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
