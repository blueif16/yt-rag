-- ============================================================================
-- Clear the two Supabase security-advisor warnings.
--
-- 1. extension_in_public: move pgvector out of the API-exposed `public` schema
--    into `extensions` (Supabase's convention). Existing vector columns and the
--    HNSW index keep working — they reference the type by OID.
-- 2. function_search_path_mutable: pin each search function's search_path so
--    a caller's search_path cannot redirect table or operator lookups.
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS extensions;
ALTER EXTENSION vector SET SCHEMA extensions;

ALTER FUNCTION public.search_context_mesh(text, extensions.vector, int, int, int, text)
  SET search_path = public, extensions;
ALTER FUNCTION public.search_vector(extensions.vector, int, text)
  SET search_path = public, extensions;
