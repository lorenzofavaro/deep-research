import re
import uuid
from typing import Any

from multi_tool_agent.data.embedding import create_embedding_service
from multi_tool_agent.data.embeddings.base import EmbeddingService
from multi_tool_agent.data.vector_store import create_vector_store
from multi_tool_agent.data.vector_stores.base import Document
from multi_tool_agent.data.vector_stores.base import VectorStore
from multi_tool_agent.utils.config import Config
from multi_tool_agent.utils.config import config


class DocumentIngestionService:
    """Service that coordinates document storage and vector indexing."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
    ):
        self.vector_store = vector_store
        self.embedding = embedding_service

    def process_text(self, text: str):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def chunk_text(self, text: str, size=300):
        """Split text into chunks of specified size."""
        for i in range(0, len(text), size):
            yield text[i:i + size]

    async def ingest_document(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        collection_name: str | None = '',
        max_length=5000,
    ) -> bool:
        """Ingest a single document into both storage and vector store."""
        try:
            processed_content = self.process_text(content)[:max_length]
            success = True
            i = 0
            for chunk in self.chunk_text(processed_content):
                embedding = await self.embedding.embed_text(chunk)
                vector_document = Document(
                    id=str(uuid.uuid4()),
                    content=chunk,
                    metadata=metadata,
                    vector=embedding,
                )

                success &= await self.vector_store.add_documents([vector_document], collection_name)
                print(f'ingested chunk {i+1} inside {collection_name}')
                i += 1

            return success

        except Exception as e:
            print(f'Error ingesting document {document_id}: {e}')
            return False

    async def search_documents(
        self,
        query_text: str,
        top_k: int = 10,
        include_content: bool = True,
        filters: dict[str, Any] | None = None,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search documents using text query."""
        try:
            query_embedding = await self.embedding.embed_text(query_text)

            search_results = await self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                filters=filters,
                collection_name=collection_name,
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
            print(f'Error searching documents: {e}')
            return []


def create_document_service(config: Config) -> DocumentIngestionService:
    """Create and return a fully configured document ingestion service."""
    vector_store = create_vector_store(config)
    embedding_service = create_embedding_service(config)

    return DocumentIngestionService(
        vector_store=vector_store,
        embedding_service=embedding_service,
    )


document_service = create_document_service(config)
