from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from multi_tool_agent.core.tools.arxiv import get_arxiv_paper
from multi_tool_agent.data.document_service import DocumentIngestionService
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


class PaperIDs(BaseModel):
    """Model for holding a list of ArXiv paper IDs."""
    ids: list[str] = Field(
        default_factory=list,
        description='List of Arxiv paper ids',
    )


class IngestStep(BaseAgent):
    """Agent step for ingesting ArXiv papers into the document store."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    document_service: DocumentIngestionService
    name: str = ''
    description: str = ''
    run_id: str = ''

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the IngestStep agent.

        Args:
            **kwargs: Keyword arguments including document_service and configuration
        """
        super().__init__(**kwargs)

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the paper ingestion step.

        Args:
            context: Invocation context containing session state and paper IDs

        Yields:
            Event indicating completion of the ingestion process
        """
        content = context.session.state['paper_ids']
        collection_name = context.session.state[f'collection_name:{self.run_id}']
        paper_ids = PaperIDs(**content)

        for paper_id in paper_ids.ids:
            logger.info(f'Starting ingestion of paper {paper_id}')
            document_id, text, metadata = get_arxiv_paper(paper_id)
            await self.document_service.ingest_document(document_id, text, metadata, collection_name, max_length=3000)
            logger.info(f'Completed ingestion of paper {paper_id}')

        result_message = f'Successfully ingested {len(paper_ids.ids)} papers'
        logger.info(result_message)
        yield Event(
            author=self.name,
            content=types.Content(
                role='assistant',
                parts=[types.Part(text=result_message)],
            ),
        )
