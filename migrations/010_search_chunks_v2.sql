-- ============================================================================
-- search_chunks_v2: vector similarity + a small boost for rare query words.
--
-- Why not search_chunks (008): its keyword half ANDs every query word
-- (plainto_tsquery), so for the sentence-length questions agents ask it never
-- matched, and search was vector-only. Making it OR every word and fusing with
-- RRF was measured on the after-effects question set (72 questions) and is
-- worse: RRF throws away the similarity values, vector ranks 1-10 differ by
-- only 0.002 in RRF score, so any keyword weight reshuffles the top 10 and
-- answered questions lost their answers.
--
-- What this does instead:
--   rare words  the query's lexemes (english config) that occur in fewer than
--               rare_df of the searched chunks: effect, plugin and trade words
--               ("stagger", "bt.709", "posterizetim"), not "effect" or "layer".
--   keyword     per chunk: idf of the rare words it contains / idf of all the
--               query's rare words (0-1; 1 = it has every rare word).
--   score       similarity + keyword_boost * keyword. With the default 0.01 a
--               chunk holding all the rare words gains 0.01 cosine: it moves up
--               a few ranks, never past a clearly better match.
-- Candidates are the match_count nearest chunks plus every chunk with a rare
-- word (their similarity is computed exactly), so a strong term match outside
-- the vector pool can still enter. The vector half is an exact scan (the "+ 0"
-- keeps the planner off the HNSW index; see 008).
-- No graph expansion: doc_relations is empty in every collection.
-- A new name, so callers of search_chunks keep working unchanged.
-- ============================================================================

CREATE OR REPLACE FUNCTION public.search_chunks_v2(
  query_text TEXT,
  query_embedding extensions.vector(768),
  match_count INT DEFAULT 100,
  filter_namespace TEXT DEFAULT NULL,
  filter_meta JSONB DEFAULT NULL,
  tools_available JSONB DEFAULT NULL,
  keyword_boost FLOAT DEFAULT 0.01,
  rare_df FLOAT DEFAULT 0.2
)
RETURNS TABLE (
  id BIGINT,
  namespace TEXT,
  content TEXT,
  metadata JSONB,
  score FLOAT,
  similarity FLOAT,
  keyword FLOAT
)
LANGUAGE plpgsql STABLE
SET search_path = public, extensions
AS $$
#variable_conflict use_column
BEGIN
  RETURN QUERY
  WITH
  pool AS NOT MATERIALIZED (  -- inlined, so the vector/FTS indexes stay usable
    SELECT d.id, d.namespace, d.content, d.metadata, d.embedding FROM documents d
    WHERE (filter_namespace IS NULL OR d.namespace = filter_namespace)
      AND (filter_meta IS NULL OR d.metadata @> filter_meta)
      AND (tools_available IS NULL OR NOT (d.metadata ? 'tools')
           OR (d.metadata->'tools') <@ tools_available)
  ),
  n AS (SELECT count(*)::FLOAT AS n FROM pool),
  lex AS (
    SELECT DISTINCT l AS lexeme, quote_literal(l)::tsquery AS q
    FROM unnest(tsvector_to_array(to_tsvector('english', query_text))) AS l
  ),
  df AS (
    SELECT lex.lexeme, lex.q,
           (SELECT count(*) FROM pool p WHERE to_tsvector('english', p.content) @@ lex.q)::FLOAT AS df
    FROM lex
  ),
  rare AS (
    SELECT df.lexeme, df.q, ln(n.n / df.df) AS idf
    FROM df, n
    WHERE keyword_boost > 0 AND df.df > 0 AND df.df < rare_df * n.n
  ),
  total AS (SELECT sum(rare.idf) AS s FROM rare),
  kw AS (
    SELECT p.id, (sum(r.idf) / min(t.s))::FLOAT AS kw
    FROM rare r
    JOIN pool p ON to_tsvector('english', p.content) @@ r.q
    CROSS JOIN total t
    GROUP BY p.id
  ),
  vec AS (
    SELECT p.id FROM pool p
    ORDER BY (p.embedding <=> query_embedding) + 0
    LIMIT match_count
  ),
  cand AS (
    SELECT vec.id FROM vec
    UNION
    SELECT kw.id FROM kw
  ),
  scored AS (
    SELECT d.id, d.namespace, d.content, d.metadata,
           (1 - (d.embedding <=> query_embedding))::FLOAT AS similarity,
           COALESCE(kw.kw, 0)::FLOAT AS keyword
    FROM cand
    JOIN documents d ON d.id = cand.id
    LEFT JOIN kw ON kw.id = cand.id
  )
  SELECT s.id, s.namespace, s.content, s.metadata,
         s.similarity + keyword_boost * s.keyword AS score,
         s.similarity, s.keyword
  FROM scored s
  ORDER BY s.similarity + keyword_boost * s.keyword DESC
  LIMIT match_count;
END;
$$;
