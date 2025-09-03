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
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of the vector store interface."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        host: str = 'localhost',
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
    ) -> None:
        """
        Initialize Qdrant vector store.

        Args:
            embedding_service: Service for generating embeddings
            host: Qdrant server host
            port: Qdrant server HTTP port
            grpc_port: Qdrant server gRPC port
            prefer_grpc: Whether to prefer gRPC over HTTP when available
        """
        self.embedding_service = embedding_service
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self._client = None
        logger.debug(f'Initialized Qdrant vector store at {host}:{port} (HTTP) / {host}:{grpc_port} (gRPC)')

    @property
    def client(self) -> QdrantClient:
        """
        Lazy initialization of Qdrant client with gRPC preference.

        Returns:
            QdrantClient instance
        """
        if self._client is None:
            if self.prefer_grpc:
                try:
                    # Try gRPC connection first
                    logger.debug(f'Attempting gRPC connection to {self.host}:{self.grpc_port}')
                    self._client = QdrantClient(
                        host=self.host, 
                        port=self.grpc_port, 
                        prefer_grpc=True,
                        check_compatibility=False,
                    )
                    # Test the connection
                    self._client.get_collections()
                    logger.info(f'Successfully connected to Qdrant via gRPC on {self.host}:{self.grpc_port}')
                except Exception as e:
                    logger.warning(f'gRPC connection failed: {e}. Falling back to HTTP on port {self.port}')
                    self._client = None
            
            # Fall back to HTTP if gRPC failed or not preferred
            if self._client is None:
                logger.debug(f'Using HTTP connection to {self.host}:{self.port}')
                self._client = QdrantClient(
                    host=self.host, 
                    port=self.port, 
                    prefer_grpc=False,
                    check_compatibility=False,
                )
                logger.info(f'Successfully connected to Qdrant via HTTP on {self.host}:{self.port}')
        
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
            logger.error(
                f'Error ensuring collection {collection_name}: {e}', exc_info=True,
            )
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
            logger.debug(
                f'Successfully added {len(documents)} documents to collection {collection_name}',
            )
            return True
        except Exception as e:
            logger.error(
                f'Error adding documents to Qdrant collection {collection_name}: {e}', exc_info=True,
            )
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
            logger.error(
                f'Error searching in Qdrant collection {collection_name}: {e}', exc_info=True,
            )
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
            logger.error(
                f'Error in text search for collection {collection_name}: {e}', exc_info=True,
            )
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
            logger.debug(
                f'Successfully deleted document {document_id} from collection {collection_name}',
            )
            return True
        except Exception as e:
            logger.error(
                f'Error deleting document {document_id} from Qdrant collection {collection_name}: {e}', exc_info=True,
            )
            return False
