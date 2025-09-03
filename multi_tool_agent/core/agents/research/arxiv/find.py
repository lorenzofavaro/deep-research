import json

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from multi_tool_agent.core.tools.arxiv import search_arxiv


class FindStep(BaseAgent):
    name: str = ''
    description: str = ''
    run_id: str = ''
    agent_id: str = ''

    def __init__(self, name: str = 'find_step', description: str = 'Find papers on arxiv', run_id: str = '0', agent_id: str = '0'):
        super().__init__(name=name, description=description)
        self.run_id = run_id
        self.agent_id = agent_id

    async def _run_async_impl(self, context: InvocationContext):
        query = context.session.state[f'query:{self.run_id}:{self.agent_id}']
        papers_meta = search_arxiv(query)
        yield Event(
            author=self.name,
            content=types.Content(
                role='assistant',
                parts=[types.Part(text=json.dumps(papers_meta))],
            ),
        )


find_step = FindStep()
