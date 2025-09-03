from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from multi_tool_agent.core.tools.arxiv import get_arxiv_paper
from multi_tool_agent.data.document_service import DocumentIngestionService


class PaperIDs(BaseModel):
    ids: list[str] = Field(
        default_factory=list,
        description='List of Arxiv paper ids',
    )


class IngestStep(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    document_service: DocumentIngestionService
    name: str = ''
    description: str = ''
    run_id: str = ''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _run_async_impl(self, context):
        content = context.session.state['paper_ids']
        collection_name = context.session.state[f'collection_name:{self.run_id}']
        paper_ids = PaperIDs(**content)

        for paper_id in paper_ids.ids:
            print(f'Reading paper {paper_id}...')
            document_id, text, metadata = get_arxiv_paper(paper_id)
            await self.document_service.ingest_document(document_id, text, metadata, collection_name, max_length=3000)

        result_message = f'Successfully ingested {len(paper_ids.ids)} papers'
        yield Event(
            author=self.name,
            content=types.Content(
                role='assistant',
                parts=[types.Part(text=result_message)],
            ),
        )
