import re
import uuid
from collections.abc import Generator
from typing import Any
from typing import Optional

from multi_tool_agent.data.embedding import create_embedding_service
from multi_tool_agent.data.embeddings.base import EmbeddingService
from multi_tool_agent.data.vector_store import create_vector_store
from multi_tool_agent.data.vector_stores.base import Document
from multi_tool_agent.data.vector_stores.base import VectorStore
from multi_tool_agent.utils.config import Config
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentIngestionService:
    """Service that coordinates document storage and vector indexing."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
    ) -> None:
        """
        Initialize the document ingestion service.

        Args:
            vector_store: Vector store for document storage and retrieval
            embedding_service: Service for generating text embeddings
        """
        self.vector_store = vector_store
        self.embedding = embedding_service
        logger.debug('DocumentIngestionService initialized successfully')

    def process_text(self, text: str) -> str:
        """
        Process and clean text for document ingestion.

        Args:
            text: Raw text to process

        Returns:
            Cleaned and processed text
        """
        original_length = len(text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        logger.debug(
            f'Text processing: {original_length} -> {len(text)} characters',
        )
        return text

    def chunk_text(self, text: str, size: int = 300) -> Generator[str, None, None]:
        """
        Split text into chunks of specified size.

        Args:
            text: Text to split into chunks
            size: Size of each chunk in characters

        Yields:
            Text chunks of the specified size
        """
        for i in range(0, len(text), size):
            yield text[i:i + size]

    async def ingest_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        collection_name: str = '',
        max_length: int = 5000,
    ) -> bool:
        """
        Ingest a single document into both storage and vector store.

        Args:
            document_id: Unique identifier for the document
            content: Text content of the document
            metadata: Optional metadata to associate with the document
            collection_name: Name of the collection to store the document in
            max_length: Maximum length of content to process

        Returns:
            True if ingestion was successful, False otherwise
        """
        try:
            processed_content = self.process_text(content)[:max_length]
            chunks = list(self.chunk_text(processed_content))
            logger.info(
                f'Document {document_id}: processing {len(processed_content)} characters into {len(chunks)} chunks',
            )

            success = True
            i = 0
            for chunk in chunks:
                embedding = await self.embedding.embed_text(chunk)
                vector_document = Document(
                    id=str(uuid.uuid4()),
                    content=chunk,
                    metadata=metadata,
                    vector=embedding,
                )

                success &= await self.vector_store.add_documents([vector_document], collection_name)
                logger.info(
                    f'Ingested chunk {i+1} into collection {collection_name}',
                )
                i += 1

            return success

        except Exception as e:
            logger.error(
                f'Error ingesting document {document_id}: {e}', exc_info=True,
            )
            return False

    async def search_documents(
        self,
        query_text: str,
        top_k: int = 10,
        include_content: bool = True,
        filters: Optional[dict[str, Any]] = None,
        collection_name: str = '',
    ) -> list[dict[str, Any]]:
        """
        Search documents using text query.

        Args:
            query_text: Text query to search for
            top_k: Maximum number of results to return
            include_content: Whether to include document content in results
            filters: Optional filters to apply to the search
            collection_name: Name of the collection to search in

        Returns:
            List of dictionaries containing search results
        """
        try:
            query_embedding = await self.embedding.embed_text(query_text)

            search_results = await self.vector_store.search(
                query_vector=query_embedding,
                collection_name=collection_name,
                top_k=top_k,
                filters=filters,
            )

            results = []
            for result in search_results:
                doc_data = {
                    'id': result.document.id,
                    'score': result.score,
                    'metadata': result.document.metadata or {},
                }

                if include_content and result.document.content:
                    doc_data['content'] = result.document.content

                results.append(doc_data)

            return results

        except Exception as e:
            logger.error(f'Error searching documents: {e}', exc_info=True)
            return []


def create_document_service(config: Config) -> DocumentIngestionService:
    """
    Create and return a fully configured document ingestion service.

    Args:
        config: Configuration object containing service settings

    Returns:
        Configured DocumentIngestionService instance
    """
    vector_store = create_vector_store(config)
    embedding_service = create_embedding_service(config)

    return DocumentIngestionService(
        vector_store=vector_store,
        embedding_service=embedding_service,
    )


document_service = create_document_service(config)
