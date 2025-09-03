from typing import Any
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter
from qdrant_client.models import MatchValue
from qdrant_client.models import PointIdsList
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams

from multi_tool_agent.data.embeddings.base import EmbeddingService
from multi_tool_agent.data.vector_stores.base import Document
from multi_tool_agent.data.vector_stores.base import SearchResult
from multi_tool_agent.data.vector_stores.base import VectorStore


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of the vector store interface."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        host: str = 'localhost',
        port: int = 6333,
    ) -> None:
        """
        Initialize Qdrant vector store.

        Args:
            embedding_service: Service for generating embeddings
            host: Qdrant server host
            port: Qdrant server port
        """
        self.embedding_service = embedding_service
        self.host = host
        self.port = port
        self._client = None

    @property
    def client(self) -> QdrantClient:
        """
        Lazy initialization of Qdrant client.

        Returns:
            QdrantClient instance
        """
        if self._client is None:
            self._client = QdrantClient(
                host=self.host, port=self.port, check_compatibility=False,
            )
        return self._client

    def _ensure_collection(self, collection_name: str) -> bool:
        """
        Ensure the collection exists with the correct configuration.

        Args:
            collection_name: Name of the collection to ensure exists

        Returns:
            True if successful, False otherwise
        """
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_service.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
            return True
        except Exception as e:
            print(f'Error ensuring collection: {e}')
            return False

    async def add_documents(self, documents: list[Document], collection_name: str = '') -> bool:
        """
        Add documents to Qdrant.

        Args:
            documents: List of documents to add to the vector store
            collection_name: Name of the collection to add documents to

        Returns:
            True if successful, False otherwise

        Raises:
            ValueError: If the collection doesn't exist
        """
        if not self._ensure_collection(collection_name):
            raise ValueError(
                f"Collection '{collection_name}' does not exist in Qdrant.",
            )
        try:
            points = []
            for doc in documents:
                # Generate embedding if not provided
                if doc.vector is None:
                    doc.vector = await self.embedding_service.embed_text(doc.content)

                point = PointStruct(
                    id=doc.id,
                    vector=doc.vector,
                    payload={
                        'content': doc.content,
                        **(doc.metadata or {}),
                    },
                )
                points.append(point)

            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            return True
        except Exception as e:
            print(f'Error adding documents to Qdrant: {e}')
            return False

    async def search(
        self,
        query_vector: list[float],
        collection_name: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents in Qdrant.

        Args:
            query_vector: Vector representation of the query
            collection_name: Name of the collection to search in
            top_k: Maximum number of results to return
            filters: Optional filters to apply to the search

        Returns:
            List of search results ordered by similarity score
        """
        try:
            query_filter = None
            if filters:
                # Convert filters to Qdrant filter format
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value),
                        )
                        for key, value in filters.items()
                    ],
                )

            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

            results = []
            for hit in search_result:
                payload = hit.payload if hit.payload else {}
                content = payload.pop('content', '')

                document = Document(
                    id=str(hit.id),
                    content=content,
                    metadata=payload,
                    vector=query_vector,  # We don't store the vector in payload
                )

                results.append(
                    SearchResult(
                        document=document,
                        score=hit.score,
                    ),
                )

            return results
        except Exception as e:
            print(f'Error searching in Qdrant: {e}')
            return []

    async def search_by_text(
        self,
        query_text: str,
        collection_name: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[SearchResult]:
        """
        Search by text using embedding generation.

        Args:
            query_text: Text query to search for
            collection_name: Name of the collection to search in
            top_k: Maximum number of results to return
            filters: Optional filters to apply to the search

        Returns:
            List of search results ordered by similarity score
        """
        try:
            # Generate embedding for query text
            query_vector = await self.embedding_service.embed_text(query_text)

            # Use vector search with correct parameter order
            return await self.search(query_vector, collection_name, top_k, filters)
        except Exception as e:
            print(f'Error in text search: {e}')
            return []

    async def delete_document(self, document_id: str, collection_name: str) -> bool:
        """
        Delete a document from Qdrant.

        Args:
            document_id: ID of the document to delete
            collection_name: Name of the collection containing the document

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(
                    points=[document_id],
                ),
            )
            return True
        except Exception as e:
            print(f'Error deleting document from Qdrant: {e}')
            return False
