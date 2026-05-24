"""
Data Adapters - Extract content from various formats
=====================================================

Focus: Get clean text content. That's it.
Metadata is minimal, handled at ingest time.
"""
from __future__ import annotations
import json
import csv
from pathlib import Path
from typing import Any, Iterator


class DataAdapter:
    """Extract content from various data formats."""
    
    @staticmethod
    def _get_nested(obj: dict, path: str) -> Any:
        """Get nested value: _get_nested({"a": {"b": 1}}, "a.b") → 1"""
        for key in path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                return None
        return obj
    
    @staticmethod
    def from_list(items: list, content_field: str = "content") -> list[str]:
        """
        Extract content from list of dicts.
        
        Returns list of strings ready for rag.ingest_batch()
        """
        results = []
        for item in items:
            if isinstance(item, str):
                results.append(item)
            elif isinstance(item, dict):
                content = DataAdapter._get_nested(item, content_field)
                if content:
                    results.append(str(content))
        return results
    
    @staticmethod
    def from_json_file(path: str | Path, content_field: str = "content", items_path: str | None = None) -> list[str]:
        """Load content from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if items_path:
            data = DataAdapter._get_nested(data, items_path) or []
        if not isinstance(data, list):
            data = [data]
        
        return DataAdapter.from_list(data, content_field)
    
    @staticmethod
    def from_api_response(response: dict | list, content_field: str = "content", items_path: str | None = None) -> list[str]:
        """Extract content from API response."""
        if isinstance(response, list):
            items = response
        elif items_path:
            items = DataAdapter._get_nested(response, items_path) or []
        else:
            # Auto-detect common patterns
            for key in ["data", "results", "items", "records"]:
                if key in response and isinstance(response[key], list):
                    items = response[key]
                    break
            else:
                items = [response]
        
        return DataAdapter.from_list(items, content_field)
    
    @staticmethod
    def from_csv(path: str | Path, content_column: str = "content") -> list[str]:
        """Extract content from CSV."""
        results = []
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                content = row.get(content_column, "")
                if content:
                    results.append(content)
        return results
    
    @staticmethod
    def from_text_chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
        """Split long text into chunks. Returns list of strings."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # Try to break at paragraph/sentence
            if end < len(text):
                for sep in ["\n\n", ".\n", ". ", "\n"]:
                    pos = chunk.rfind(sep)
                    if pos > chunk_size * 0.5:
                        end = start + pos + len(sep)
                        chunk = text[start:end]
                        break
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def from_langchain_docs(documents: list) -> list[str]:
        """Extract content from LangChain Document objects."""
        return [getattr(d, "page_content", str(d)) for d in documents]


class StreamingAdapter:
    """Stream large files in batches."""
    
    @staticmethod
    def stream_jsonl(path: str | Path, content_field: str = "content", batch_size: int = 100) -> Iterator[list[str]]:
        """Yield batches of content from JSONL file."""
        batch = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    content = DataAdapter._get_nested(item, content_field) if isinstance(item, dict) else str(item)
                    if content:
                        batch.append(str(content))
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
        if batch:
            yield batch
    
    @staticmethod
    def stream_csv(path: str | Path, content_column: str = "content", batch_size: int = 100) -> Iterator[list[str]]:
        """Yield batches of content from CSV."""
        batch = []
        with open(path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                content = row.get(content_column, "")
                if content:
                    batch.append(content)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch
