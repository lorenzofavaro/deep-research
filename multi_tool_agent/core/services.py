# multi_tool_agent/core/services.py
from typing import Optional

from multi_tool_agent.data.document_service import create_document_service
from multi_tool_agent.data.document_service import DocumentIngestionService
from multi_tool_agent.utils.config import Config


class ServiceContainer:
    """Container for managing service dependencies."""

    def __init__(self, config: Config) -> None:
        """
        Initialize the service container.

        Args:
            config: Configuration object containing service settings
        """
        self.config = config
        self._document_service: Optional[DocumentIngestionService] = None

    @property
    def document_service(self) -> DocumentIngestionService:
        """
        Lazy-loaded document service.

        Returns:
            DocumentIngestionService instance
        """
        if self._document_service is None:
            self._document_service = create_document_service(self.config)
        return self._document_service
