from multi_tool_agent.data.embeddings.base import EmbeddingService


class OpenAIEmbedding(EmbeddingService):
    """OpenAI embedding service."""

    def __init__(self, model_name: str = 'text-embedding-3-small', api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
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
        """Generate embedding for a single text using OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f'Error generating embedding: {e}')
            raise

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f'Error generating embeddings: {e}')
            raise

    @property
    def vector_size(self) -> int:
        """Get the size of the embedding vectors."""
        # text-embedding-3-small produces 1536-dimensional vectors
        return 1536
