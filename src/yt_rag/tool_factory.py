"""
LangGraph Tool Factory - Simple, Clean
======================================

Create tools from RAGStore for use in LangGraph agents.
"""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING

from langchain_core.tools import tool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .rag_store import RAGStore


class SearchInput(BaseModel):
    query: str = Field(description="What to search for")
    top_k: int = Field(default=5, description="Number of results")


class IngestInput(BaseModel):
    content: str = Field(description="Knowledge to store (rich, self-describing)")
    source: str | None = Field(default=None, description="Where this came from")


def create_search_tool(
    rag: "RAGStore",
    name: str = "search_knowledge",
    description: str = "Search the knowledge base"
):
    """Create a search tool for LangGraph agents."""
    
    @tool(name, args_schema=SearchInput)
    def search_tool(query: str, top_k: int = 5) -> str:
        results = rag.search(query, top_k=top_k)
        if not results:
            return "No relevant knowledge found."
        return "\n\n---\n\n".join([r.get("content", "") for r in results])
    
    search_tool.__doc__ = description
    return search_tool


def create_ingest_tool(
    rag: "RAGStore",
    name: str = "store_knowledge",
    description: str = "Store new knowledge"
):
    """Create an ingest tool for LangGraph agents."""
    
    @tool(name, args_schema=IngestInput)
    def ingest_tool(content: str, source: str | None = None) -> str:
        result = rag.ingest(content, source=source)
        return f"Stored (ID: {result['id']}, status: {result['status']})"
    
    ingest_tool.__doc__ = description
    return ingest_tool


def create_search_fn(rag: "RAGStore", top_k: int = 5) -> Callable[[str], str]:
    """
    Create a simple search function for LangGraph nodes.
    
    Usage in workflow:
        search = create_search_fn(rag)
        
        def retrieve_node(state):
            context = search(state["question"])
            return {"context": context}
    """
    def search(query: str) -> str:
        results = rag.search(query, top_k=top_k)
        if not results:
            return ""
        return "\n\n".join([r.get("content", "") for r in results])
    return search


def create_search_fn_raw(rag: "RAGStore", top_k: int = 5) -> Callable[[str], list[dict]]:
    """Return raw results (list of dicts) instead of formatted string."""
    def search(query: str) -> list[dict]:
        return rag.search(query, top_k=top_k)
    return search
