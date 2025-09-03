from abc import ABC
from abc import abstractmethod


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Get the size of the embedding vectors."""
