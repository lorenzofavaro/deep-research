from typing import Optional

from multi_tool_agent.data.embeddings.base import EmbeddingService
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIEmbedding(EmbeddingService):
    """OpenAI embedding service for generating text embeddings."""

    def __init__(self, model_name: str = 'text-embedding-3-small', api_key: Optional[str] = None) -> None:
        """
        Initialize OpenAI embedding service.

        Args:
            model_name: Name of the OpenAI embedding model to use
            api_key: OpenAI API key (if None, will use environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key
        self._client = None
        logger.debug(
            f'Initialized OpenAI embedding service with model: {model_name}',
        )

    @property
    def client(self):
        """
        Lazy initialization of OpenAI client.

        Returns:
            AsyncOpenAI client instance

        Raises:
            ImportError: If openai package is not installed
        """
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    'openai is required for OpenAI embeddings. '
                    'Install with: pip install openai',
                )
        return self._client

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text using OpenAI.

        Args:
            text: Input text to embed

        Returns:
            List of float values representing the text embedding

        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(
                f'Error generating embedding for text: {e}', exc_info=True,
            )
            raise

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors, one for each input text

        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(
                f'Error generating embeddings for {len(texts)} texts: {e}', exc_info=True,
            )
            raise

    @property
    def vector_size(self) -> int:
        """
        Get the size of the embedding vectors.

        Returns:
            Dimension size of the embedding vectors (1536 for text-embedding-3-small)
        """
        # text-embedding-3-small produces 1536-dimensional vectors
        return 1536
