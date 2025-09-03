import json
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events import EventActions
from google.genai import types
from pydantic import ConfigDict

from multi_tool_agent.data.document_service import DocumentIngestionService
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


class RAGStep(BaseAgent):
    """Agent step for performing Retrieval-Augmented Generation (RAG) on ingested documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    name: str = ''
    description: str = ''
    run_id: str = ''
    agent_id: str = ''
    document_service: DocumentIngestionService

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the RAGStep agent.

        Args:
            **kwargs: Keyword arguments including document_service and configuration
        """
        super().__init__(**kwargs)

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the RAG step to retrieve relevant documents.

        Args:
            context: Invocation context containing session state and query information

        Yields:
            Event containing the retrieved document results
        """
        query = context.session.state[f'query:{self.run_id}:{self.agent_id}']
        collection_name = context.session.state[f'collection_name:{self.run_id}']

        logger.debug(
            f'Executing RAG search for query: "{query}" in collection: {collection_name}',
        )

        docs = await self.document_service.search_documents(query, collection_name=collection_name)
        docs_text = json.dumps(docs, ensure_ascii=False, indent=2)

        logger.info(f'RAG search returned {len(docs)} documents')

        step_delta: dict[str, object] = {
            f'results:{self.run_id}:{self.agent_id}': docs_text,
        }
        yield Event(
            author=self.name,
            content=types.Content(
                role='assistant',
                parts=[
                    types.Part(
                        text=f'Retrieved {len(docs)} relevant documents from knowledge base',
                    ),
                ],
            ),
            actions=EventActions(state_delta=step_delta),
        )
