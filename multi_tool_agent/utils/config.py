import os
from typing import Any


class Config:
    """Configuration class for managing environment variables and settings."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.vector_store_type = os.getenv(
            'VECTOR_STORE_TYPE', 'qdrant',
        )  # qdrant, vertex_ai
        self.qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
        self.qdrant_port = int(os.getenv('QDRANT_PORT', '6333'))
        self.qdrant_collection = os.getenv('QDRANT_COLLECTION', 'documents')

        self.embedding_type = os.getenv('EMBEDDING_TYPE', 'openai')  # openai
        self.embedding_model = os.getenv(
            'EMBEDDING_MODEL', 'text-embedding-3-small',
        )
        self.openai_api_key = os.getenv('OPENAI_API_KEY')

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary format.

        Returns:
            Dictionary containing all configuration values
        """
        return {
            'vector_store_type': self.vector_store_type,
            'qdrant_host': self.qdrant_host,
            'qdrant_port': self.qdrant_port,
            'qdrant_collection': self.qdrant_collection,
            'embedding_type': self.embedding_type,
            'embedding_model': self.embedding_model,
            'openai_api_key': self.openai_api_key,
        }


config = Config()
