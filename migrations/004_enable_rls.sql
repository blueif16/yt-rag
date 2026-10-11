-- ============================================================================
-- Lock the tables to the server's secret key.
--
-- Tables in `public` are exposed through PostgREST, so without RLS anyone
-- holding the publishable (anon) key could read or overwrite the corpus.
-- RLS on with no policies = no access for anon/authenticated; the secret
-- (service-role) key the server uses bypasses RLS, so ingest/search are
-- unaffected.
-- ============================================================================

ALTER TABLE documents     ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_relations ENABLE ROW LEVEL SECURITY;
ALTER TABLE collections   ENABLE ROW LEVEL SECURITY;
