-- ============================================================================
-- set_doc_tags: (re)tag every chunk of one source in one call.
--
-- Tags live on each chunk (so search can filter without a join). When a
-- source is curated twice, or its tags are revised, ingest skips the
-- unchanged chunks — this sets the tags on them anyway. Merges into the
-- existing metadata; returns the number of chunks updated.
-- ============================================================================

CREATE OR REPLACE FUNCTION public.set_doc_tags(p_namespace TEXT, p_doc_id TEXT, p_tags JSONB)
RETURNS INT
LANGUAGE sql
SET search_path = public
AS $$
  WITH u AS (
    UPDATE documents SET metadata = metadata || p_tags
    WHERE namespace = p_namespace AND metadata->>'doc_id' = p_doc_id
    RETURNING 1
  )
  SELECT count(*)::INT FROM u;
$$;

-- Only the server's secret (service-role) key may call it.
REVOKE EXECUTE ON FUNCTION public.set_doc_tags(TEXT, TEXT, JSONB) FROM PUBLIC, anon, authenticated;
