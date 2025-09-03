from multi_tool_agent.data.embeddings.base import EmbeddingService
from multi_tool_agent.data.embeddings.openai_embeddings import OpenAIEmbedding
from multi_tool_agent.utils.config import Config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


def create_embedding_service(config: Config) -> EmbeddingService:
    """
    Create and return the configured embedding service.

    Args:
        config: Configuration object containing embedding settings

    Returns:
        An instance of the configured embedding service

    Raises:
        ValueError: If the embedding type is not supported
    """
    logger.debug(
        f'Creating embedding service of type: {config.embedding_type}',
    )
    if config.embedding_type == 'openai':
        service = OpenAIEmbedding(
            model_name=config.embedding_model or 'text-embedding-3-small',
            api_key=config.openai_api_key,
        )
        logger.debug(
            f'Created OpenAI embedding service with model: {config.embedding_model}',
        )
        return service
    else:
        logger.error(f'Unsupported embedding type: {config.embedding_type}')
        raise ValueError(
            f'Unsupported embedding type: {config.embedding_type}',
        )
