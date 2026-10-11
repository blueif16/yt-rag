"""
RAGStore - 2026 SOTA with Minimal Metadata Philosophy
======================================================

Content is rich and self-describing → embeddings capture everything
Metadata is minimal → just source/type for reference
Power from indexing → RRF (BM25 + Vector) + Graph traversal

Usage:
    rag = RAGStore(namespace="video_styles")
    rag.ingest("Product Hunt hook pacing: Fast energetic cuts, 1.5-2s in first 10 seconds...")
    results = rag.search("energetic fast opening style")
"""
from __future__ import annotations
import os
import json
import hashlib
from dataclasses import dataclass, field

from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client
from .gemini_embeddings import DEFAULT_MODEL, GeminiEmbeddings

# Load THIS package's own .env by absolute path — never CWD-discovered. The MCP
# client spawns this process with an arbitrary working directory (e.g. another
# project's root), and a foreign .env there would otherwise shadow ours and point
# us at the wrong Supabase project (→ 401 Invalid authentication credentials).
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)


@dataclass
class RAGConfig:
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY", ""))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    embedding_dim: int = 768

    # Search defaults
    match_count: int = 5
    rrf_k: int = 60
    graph_depth: int = 2


class RAGStore:
    """
    Minimal, powerful RAG store.
    
    - Rich content (self-describing, natural text)
    - Minimal metadata (source, type - that's it)
    - SOTA search via RRF + graph
    """
    
    def __init__(
        self,
        namespace: str = "default",
        supabase_url: str | None = None,
        supabase_key: str | None = None,
        **kwargs
    ):
        self.namespace = namespace
        self.config = RAGConfig(
            supabase_url=supabase_url or os.getenv("SUPABASE_URL", ""),
            supabase_key=supabase_key or os.getenv("SUPABASE_SECRET_KEY") or os.getenv("SUPABASE_KEY", ""),
            **{k: v for k, v in kwargs.items() if hasattr(RAGConfig, k)}
        )
        
        self.client: Client = create_client(self.config.supabase_url, self.config.supabase_key)
        self._embeddings = GeminiEmbeddings(
            model=self.config.embedding_model,
            output_dimensionality=self.config.embedding_dim
        )
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    def get_setup_sql(self) -> str:
        """Get SQL to run in Supabase. Copy from migrations/20260125_init_graph_schema.sql"""
        sql_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", 
                                "supabase", "migrations", "20260125_init_graph_schema.sql")
        try:
            with open(sql_path) as f:
                return f.read()
        except FileNotFoundError:
            return "-- See supabase/migrations/20260125_init_graph_schema.sql"
    
    # =========================================================================
    # INGEST
    # =========================================================================
    
    def _hash(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def ingest(
        self,
        content: str,
        source: str | None = None,
        type: str | None = None,
        skip_duplicates: bool = True
    ) -> dict:
        """
        Ingest a piece of knowledge.
        
        Args:
            content: Rich, self-describing text
            source: Where this came from (optional)
            type: Category like "pattern", "guide", "note" (optional)
        
        Returns:
            {"id": int, "status": "created" | "skipped"}
        """
        content = content.strip()
        content_hash = self._hash(content)
        
        # Check duplicate
        if skip_duplicates:
            existing = self.client.table("documents").select("id").eq(
                "content_hash", content_hash
            ).eq("namespace", self.namespace).execute()
            if existing.data:
                return {"id": existing.data[0]["id"], "status": "skipped"}
        
        # Embed
        embedding = self._embeddings.embed_query(content)
        
        # Minimal metadata (+ which model embedded it — models are incompatible)
        metadata = {"embed_model": self._embeddings.model}
        if source:
            metadata["source"] = source
        if type:
            metadata["type"] = type
        
        # Insert
        result = self.client.table("documents").insert({
            "content": content,
            "content_hash": content_hash,
            "embedding": embedding,
            "metadata": metadata,
            "namespace": self.namespace
        }).execute()
        
        return {"id": result.data[0]["id"], "status": "created"}
    
    def ingest_batch(
        self,
        items: list[str | dict],
        source: str | None = None,
        type: str | None = None,
        batch_size: int = 50
    ) -> dict:
        """
        Batch ingest.
        
        Items can be:
        - str: Just content
        - dict: {"content": "...", "source": "...", "type": "..."}
        """
        created, skipped, ids = 0, 0, []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Normalize
            normalized = []
            for item in batch:
                if isinstance(item, str):
                    normalized.append({"content": item.strip()})
                else:
                    normalized.append({
                        "content": item.get("content", "").strip(),
                        "source": item.get("source", source),
                        "type": item.get("type", type)
                    })
            
            # Hash and check duplicates
            hashes = [self._hash(n["content"]) for n in normalized]
            existing = self.client.table("documents").select("content_hash").in_(
                "content_hash", hashes
            ).eq("namespace", self.namespace).execute()
            existing_hashes = {r["content_hash"] for r in existing.data}
            
            # Filter new
            new_items = []
            new_hashes = []
            for n, h in zip(normalized, hashes):
                if h in existing_hashes:
                    skipped += 1
                elif n["content"]:
                    new_items.append(n)
                    new_hashes.append(h)
            
            if not new_items:
                continue
            
            # Embed
            embeddings = self._embeddings.embed_documents([n["content"] for n in new_items])
            
            # Build records
            records = []
            for n, emb, h in zip(new_items, embeddings, new_hashes):
                metadata = {"embed_model": self._embeddings.model}
                if n.get("source"):
                    metadata["source"] = n["source"]
                if n.get("type"):
                    metadata["type"] = n["type"]
                
                records.append({
                    "content": n["content"],
                    "content_hash": h,
                    "embedding": emb,
                    "metadata": metadata,
                    "namespace": self.namespace
                })
            
            result = self.client.table("documents").insert(records).execute()
            created += len(result.data)
            ids.extend([r["id"] for r in result.data])
        
        return {"created": created, "skipped": skipped, "ids": ids}
    
    # =========================================================================
    # SEARCH
    # =========================================================================
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        graph_depth: int | None = None,
        filter_meta: dict | None = None,
        tools_available: list[str] | None = None,
    ) -> list[dict]:
        """
        SOTA search: Hybrid RRF (BM25 + Vector) + Graph traversal
        (search_chunks, migrations/008). Rows come best-first, each with
        `similarity` (cosine, 0-1) and `namespace`.

        filter_meta: JSONB containment on chunk metadata, e.g.
            {"scenarios": ["ui-promo"]}; tools_available: keep only sources
            whose required plugins are all in this list.
        """
        k = top_k or self.config.match_count
        depth = graph_depth or self.config.graph_depth
        embedding = self._embeddings.embed_query(query)

        try:
            result = self.client.rpc("search_chunks", {
                "query_text": query,
                "query_embedding": embedding,
                "match_count": k,
                "rrf_k": self.config.rrf_k,
                "graph_depth": depth,
                "filter_namespace": None if self.namespace == "*" else self.namespace,
                "filter_meta": filter_meta,
                "tools_available": tools_available,
            }).execute()
            return result.data
        except Exception:
            # Fallback to vector only
            return self.search_vector(query, top_k=k)

    def search_vector(self, query: str, top_k: int | None = None) -> list[dict]:
        """Fast vector-only search (skip graph)."""
        k = top_k or self.config.match_count
        embedding = self._embeddings.embed_query(query)
        
        try:
            result = self.client.rpc("search_vector", {
                "query_embedding": embedding,
                "match_count": k,
                "filter_namespace": self.namespace
            }).execute()
            return result.data
        except Exception:
            # Direct query fallback
            result = self.client.table("documents").select(
                "id, content, metadata"
            ).eq("namespace", self.namespace).limit(k).execute()
            return result.data
    
    # Aliases
    search_context_mesh = search
    
    # =========================================================================
    # GRAPH
    # =========================================================================
    
    def add_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str = "relates_to",
        properties: dict | None = None
    ) -> dict:
        """
        Add graph edge between documents.
        
        Types: "relates_to", "contradicts", "contains", "complements", "requires"
        """
        result = self.client.table("doc_relations").upsert({
            "source_id": source_id,
            "target_id": target_id,
            "type": relation_type,
            "properties": properties or {},
            "namespace": self.namespace
        }).execute()
        return result.data[0] if result.data else {}
    
    # =========================================================================
    # UTILS
    # =========================================================================
    
    def stats(self) -> dict:
        docs = self.client.table("documents").select("id", count="exact").eq(
            "namespace", self.namespace
        ).execute()
        rels = self.client.table("doc_relations").select("id", count="exact").eq(
            "namespace", self.namespace
        ).execute()
        return {"namespace": self.namespace, "documents": docs.count, "relations": rels.count}
    
    def delete_all(self) -> dict:
        """Delete all in namespace."""
        self.client.table("doc_relations").delete().eq("namespace", self.namespace).execute()
        result = self.client.table("documents").delete().eq("namespace", self.namespace).execute()
        return {"deleted": len(result.data) if result.data else 0}
    
    def get(self, id: int) -> dict | None:
        """Get single document by ID."""
        result = self.client.table("documents").select("*").eq("id", id).execute()
        return result.data[0] if result.data else None
