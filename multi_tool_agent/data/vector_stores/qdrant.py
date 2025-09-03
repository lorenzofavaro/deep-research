from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from qdrant_client.models import FieldCondition
from qdrant_client.models import Filter
from qdrant_client.models import MatchValue
from qdrant_client.models import PointIdsList
from qdrant_client.models import PointStruct
from qdrant_client.models import VectorParams

from multi_tool_agent.data.embeddings.openai_embeddings import EmbeddingService
from multi_tool_agent.data.vector_stores.base import Document
from multi_tool_agent.data.vector_stores.base import SearchResult
from multi_tool_agent.data.vector_stores.base import VectorStore


class QdrantVectorStore(VectorStore):
    def __init__(
        self,
        embedding_service: EmbeddingService,
        host: str = 'localhost',
        port: int = 6333,
    ):
        self.embedding_service = embedding_service
        self.host = host
        self.port = port
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            self._client = QdrantClient(
                host=self.host, port=self.port, check_compatibility=False,
            )
        return self._client

    def _ensure_collection(self, collection_name):
        """Ensure the collection exists with the correct configuration."""
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
        """Add documents to Qdrant."""
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
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents in Qdrant."""
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
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search by text using embedding generation."""
        try:
            # Generate embedding for query text
            query_vector = await self.embedding_service.embed_text(query_text)

            # Use vector search with collection_name parameter
            return await self.search(query_vector, top_k, filters, collection_name)
        except Exception as e:
            print(f'Error in text search: {e}')
            return []

    async def delete_document(self, document_id: str, collection_name: str) -> bool:
        """Delete a document from Qdrant."""
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
