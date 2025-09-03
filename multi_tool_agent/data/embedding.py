from multi_tool_agent.data.embeddings.base import EmbeddingService
from multi_tool_agent.data.embeddings.openai_embeddings import OpenAIEmbedding
from multi_tool_agent.utils.config import Config


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
    if config.embedding_type == 'openai':
        return OpenAIEmbedding(
            model_name=config.embedding_model or 'text-embedding-3-small',
            api_key=config.openai_api_key,
        )
    else:
        raise ValueError(
            f'Unsupported embedding type: {config.embedding_type}',
        )
