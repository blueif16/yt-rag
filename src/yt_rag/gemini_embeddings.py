"""
Gemini Embeddings 封装类
兼容 LangChain Embeddings 接口
"""
from typing import List
import numpy as np
from google import genai
from google.genai import types


class GeminiEmbeddings:
    """
    Gemini embeddings 封装，兼容 LangChain 接口

    使用 gemini-embedding-001 模型
    默认输出维度: 768 (推荐值，平衡性能和质量)
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        output_dimensionality: int = 768,
        task_type: str = "RETRIEVAL_DOCUMENT",
        api_key: str | None = None
    ):
        self.model = model
        self.output_dimensionality = output_dimensionality
        self.task_type = task_type

        # 初始化 Gemini 客户端
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            self.client = genai.Client()

    def _normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
        归一化 embedding 向量
        对于非 3072 维度的 embeddings，需要归一化以确保语义相似度计算准确
        """
        if self.output_dimensionality == 3072:
            return embedding

        embedding_np = np.array(embedding)
        norm = np.linalg.norm(embedding_np)
        if norm == 0:
            return embedding
        return (embedding_np / norm).tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        为查询文本生成 embedding
        使用 RETRIEVAL_QUERY 任务类型优化查询
        """
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.output_dimensionality
            )
        )

        embedding = result.embeddings[0].values
        return self._normalize_embedding(embedding)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        为文档列表生成 embeddings
        使用配置的任务类型（默认 RETRIEVAL_DOCUMENT）
        """
        result = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=self.task_type,
                output_dimensionality=self.output_dimensionality
            )
        )

        embeddings = [emb.values for emb in result.embeddings]
        return [self._normalize_embedding(emb) for emb in embeddings]
