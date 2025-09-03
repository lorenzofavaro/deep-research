from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from multi_tool_agent.core.agents.research.arxiv.filter import filter_agent
from multi_tool_agent.core.agents.research.arxiv.find import FindStep
from multi_tool_agent.core.agents.research.arxiv.ingest import IngestStep
from multi_tool_agent.core.agents.research.arxiv.rag import RAGStep
from multi_tool_agent.data.document_service import DocumentIngestionService
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


class ArxivAgent(BaseAgent):
    """Agent for coordinating ArXiv paper research workflow."""

    def __init__(
        self,
        *,
        name: str,
        run_id: str,
        agent_id: str,
        document_service: DocumentIngestionService,
    ) -> None:
        """
        Initialize the ArXiv agent.

        Args:
            name: Name of the agent
            run_id: Unique identifier for the current run
            agent_id: Unique identifier for this agent instance
            document_service: Service for document ingestion and retrieval
        """
        super().__init__(name=name)
        self._run_id = run_id
        self._agent_id = agent_id
        self._document_service = document_service
        self._find_step = FindStep(
            name='find_step', description='Find papers on arxiv', run_id=run_id, agent_id=agent_id,
        )
        self._ingest_step = IngestStep(
            name='ingest_step', description='Ingest documents', run_id=run_id, document_service=document_service,
        )
        self._rag_step = RAGStep(
            name='rag_step', description='Perform RAG',
            run_id=run_id, agent_id=agent_id, document_service=document_service,
        )

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the complete ArXiv research workflow.

        Args:
            context: Invocation context containing session state and configuration

        Yields:
            Events from each step of the workflow
        """
        logger.debug(
            f'Starting ArXiv research workflow for agent {self._agent_id}',
        )

        logger.debug('Starting ArXiv paper search step')
        async for event in self._find_step.run_async(context):
            yield event

        logger.debug('Starting paper filtering step')
        async for event in filter_agent.run_async(context):
            yield event

        logger.debug('Starting paper ingestion step')
        async for event in self._ingest_step.run_async(context):
            yield event

        logger.debug('Starting RAG step')
        async for event in self._rag_step.run_async(context):
            yield event

        logger.info(
            f'Completed ArXiv research workflow for agent {self._agent_id}',
        )
