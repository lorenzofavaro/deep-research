from multi_tool_agent.data.embedding import create_embedding_service
from multi_tool_agent.data.vector_stores.base import VectorStore
from multi_tool_agent.data.vector_stores.qdrant import QdrantVectorStore
from multi_tool_agent.utils.config import Config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


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
    logger.debug(f'Creating vector store of type: {config.vector_store_type}')
    if config.vector_store_type == 'qdrant':
        # Create embedding service for Qdrant
        embedding_service = create_embedding_service(config)
        store = QdrantVectorStore(
            embedding_service=embedding_service,
            host=config.qdrant_host,
            port=config.qdrant_port,
            grpc_port=config.qdrant_grpc_port,
            prefer_grpc=config.qdrant_prefer_grpc,
        )
        logger.debug(
            f'Created Qdrant vector store at {config.qdrant_host}:{config.qdrant_port} (HTTP) / {config.qdrant_host}:{config.qdrant_grpc_port} (gRPC)',
        )
        return store
    else:
        logger.error(
            f'Unsupported vector store type: {config.vector_store_type}',
        )
        raise ValueError(
            f'Unsupported vector store type: {config.vector_store_type}',
        )
