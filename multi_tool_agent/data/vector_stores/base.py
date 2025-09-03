"""Base class for vector stores."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Optional


@dataclass
class Document:
    """Document representation for vector indexing."""
    id: str
    content: str
    metadata: Optional[dict[str, Any]] = None
    vector: Optional[list[float]] = None


@dataclass
class SearchResult:
    """Search result from vector store."""
    document: Document
    score: float


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def add_documents(self, documents: list[Document], collection_name: str) -> bool:
        """
        Add documents to the vector store.

        Args:
            documents: List of documents to add
            collection_name: Name of the collection to add documents to

        Returns:
            True if successful, False otherwise
        """

    @abstractmethod
    async def search(
        self,
        query_vector: list[float],
        collection_name: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents using vector similarity.

        Args:
            query_vector: Query vector for similarity search
            collection_name: Name of the collection to search in
            top_k: Maximum number of results to return
            filters: Optional filters to apply to the search

        Returns:
            List of search results ordered by similarity score
        """

    @abstractmethod
    async def search_by_text(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents using text query.

        Args:
            query_text: Text query to search for
            collection_name: Name of the collection to search in
            top_k: Maximum number of results to return
            filters: Optional filters to apply to the search

        Returns:
            List of search results ordered by similarity score
        """

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if successful, False otherwise
        """
