from multi_tool_agent.data.embedding import create_embedding_service
from multi_tool_agent.data.vector_stores.base import VectorStore
from multi_tool_agent.data.vector_stores.qdrant import QdrantVectorStore
from multi_tool_agent.utils.config import Config


def create_vector_store(config: Config) -> VectorStore:
    """
    Create and return the configured vector store.

    Args:
        config: Configuration object containing vector store settings

    Returns:
        An instance of the configured vector store

    Raises:
        ValueError: If the vector store type is not supported
    """
    if config.vector_store_type == 'qdrant':
        # Create embedding service for Qdrant
        embedding_service = create_embedding_service(config)
        return QdrantVectorStore(
            embedding_service=embedding_service,
            host=config.qdrant_host,
            port=config.qdrant_port,
        )
    else:
        raise ValueError(
            f'Unsupported vector store type: {config.vector_store_type}',
        )
