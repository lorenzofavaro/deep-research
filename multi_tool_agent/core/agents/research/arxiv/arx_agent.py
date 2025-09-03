from google.adk.agents import BaseAgent

from multi_tool_agent.core.agents.research.arxiv.filter import filter_agent
from multi_tool_agent.core.agents.research.arxiv.find import FindStep
from multi_tool_agent.core.agents.research.arxiv.ingest import IngestStep
from multi_tool_agent.core.agents.research.arxiv.rag import RAGStep
from multi_tool_agent.data.document_service import DocumentIngestionService


class ArxivAgent(BaseAgent):
    def __init__(
        self, *, name: str, run_id: str, agent_id: str,
        document_service: DocumentIngestionService,
    ):
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

    async def _run_async_impl(self, context):
        async for event in self._find_step.run_async(context):
            yield event

        async for event in filter_agent.run_async(context):
            yield event

        async for event in self._ingest_step.run_async(context):
            yield event

        async for event in self._rag_step.run_async(context):
            yield event
