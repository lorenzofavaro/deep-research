from abc import ABC
from abc import abstractmethod


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of float values representing the text embedding
        """

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors, one for each input text
        """

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """
        Get the size of the embedding vectors.

        Returns:
            Dimension size of the embedding vectors
        """
