"""Base class for vector stores."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Document:
    """Document representation for vector indexing."""
    id: str
    content: str
    metadata: dict[str, Any] | None = None
    vector: list[float] | None = None


@dataclass
class SearchResult:
    """Search result from vector store."""
    document: Document
    score: float


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add_documents(self, documents: list[Document], collection_name: str) -> bool:
        """Add documents to the vector store."""

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        collection_name: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents using vector similarity."""

    @abstractmethod
    async def search_by_text(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents using text query."""

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
