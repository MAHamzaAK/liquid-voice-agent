"""
Pinecone Client - Vector database operations

Phase II: Semantic search for situations and principles using Pinecone.
"""

import os
from typing import Optional
from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec


@dataclass
class SituationMatch:
    """Result from situation similarity search."""
    situation_id: str
    signal_text: str
    score: float
    priority: int
    typical_stage: str
    applicable_principles: list[str]


@dataclass
class PrincipleMatch:
    """Result from principle similarity search."""
    principle_id: str
    name: str
    score: float
    author: str
    book: str


class PineconeClient:
    """Client for Pinecone vector database operations."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
    ):
        """
        Initialize Pinecone client.

        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Index name (defaults to PINECONE_INDEX_NAME env var)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "sales-coach-embeddings")

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not set")

        self.pc = Pinecone(api_key=self.api_key)
        self._index = None

    @property
    def index(self):
        """Lazy load the index."""
        if self._index is None:
            self._index = self.pc.Index(self.index_name)
        return self._index

    def create_index_if_not_exists(self, dimension: int = 384):
        """
        Create the index if it doesn't exist.

        Args:
            dimension: Embedding dimension (384 for BGE-small)
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Created index: {self.index_name}")
        else:
            print(f"Index already exists: {self.index_name}")

    def upsert_situations(
        self,
        situations: dict,
        embed_fn: callable
    ) -> int:
        """
        Upsert situation signals into Pinecone.

        Args:
            situations: Dictionary of situations from situations.json
            embed_fn: Function to generate embeddings

        Returns:
            Number of vectors upserted
        """
        vectors = []
        for situation_id, data in situations.items():
            for i, signal in enumerate(data.get("signals", [])):
                vector_id = f"sit_{situation_id}_{i}"
                embedding = embed_fn(signal)
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "situation_id": situation_id,
                        "signal_text": signal,
                        "priority": data.get("priority", 0),
                        "typical_stage": data.get("typical_stage", "unknown"),
                        "applicable_principles": ",".join(data.get("applicable_principles", [])),
                    }
                })

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace="situations")

        return len(vectors)

    def upsert_principles(
        self,
        principles: dict,
        embed_fn: callable
    ) -> int:
        """
        Upsert principles into Pinecone.

        Args:
            principles: Dictionary of principles (principle_id -> data)
            embed_fn: Function to generate embeddings

        Returns:
            Number of vectors upserted
        """
        vectors = []
        for principle_id, data in principles.items():
            # Combine definition and intervention for richer embedding
            text = f"{data.get('definition', '')} {data.get('intervention', '')}"
            embedding = embed_fn(text)

            source = data.get("source", {})
            vectors.append({
                "id": f"prin_{principle_id}",
                "values": embedding,
                "metadata": {
                    "principle_id": principle_id,
                    "name": data.get("name", ""),
                    "author": source.get("author", ""),
                    "book": source.get("book", ""),
                    "triggers": ",".join(data.get("triggers", [])),
                }
            })

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace="principles")

        return len(vectors)

    def query_situations(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        min_score: float = 0.5
    ) -> list[SituationMatch]:
        """
        Query for similar situations.

        Args:
            query_embedding: Embedding of the transcript
            top_k: Number of results to return
            min_score: Minimum similarity score threshold

        Returns:
            List of SituationMatch objects
        """
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace="situations",
            include_metadata=True
        )

        matches = []
        for match in results.matches:
            if match.score >= min_score:
                metadata = match.metadata
                matches.append(SituationMatch(
                    situation_id=metadata.get("situation_id", ""),
                    signal_text=metadata.get("signal_text", ""),
                    score=match.score,
                    priority=int(metadata.get("priority", 0)),
                    typical_stage=metadata.get("typical_stage", "unknown"),
                    applicable_principles=metadata.get("applicable_principles", "").split(","),
                ))

        return matches

    def query_principles(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filter_ids: Optional[list[str]] = None,
        min_score: float = 0.3
    ) -> list[PrincipleMatch]:
        """
        Query for similar principles.

        Args:
            query_embedding: Embedding of the transcript
            top_k: Number of results to return
            filter_ids: Optional list of principle IDs to filter by
            min_score: Minimum similarity score threshold

        Returns:
            List of PrincipleMatch objects
        """
        # Build filter if principle IDs provided
        filter_dict = None
        if filter_ids:
            filter_dict = {"principle_id": {"$in": filter_ids}}

        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace="principles",
            include_metadata=True,
            filter=filter_dict
        )

        matches = []
        for match in results.matches:
            if match.score >= min_score:
                metadata = match.metadata
                matches.append(PrincipleMatch(
                    principle_id=metadata.get("principle_id", ""),
                    name=metadata.get("name", ""),
                    score=match.score,
                    author=metadata.get("author", ""),
                    book=metadata.get("book", ""),
                ))

        return matches

    def delete_all(self, namespace: str):
        """Delete all vectors in a namespace."""
        self.index.delete(delete_all=True, namespace=namespace)

    def get_stats(self) -> dict:
        """Get index statistics."""
        return self.index.describe_index_stats()
