import json
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types

from multi_tool_agent.core.tools.arxiv import search_arxiv
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


class FindStep(BaseAgent):
    """Agent step for finding papers on ArXiv based on a query."""

    name: str = ''
    description: str = ''
    run_id: str = ''
    agent_id: str = ''

    def __init__(
        self,
        name: str = 'find_step',
        description: str = 'Find papers on arxiv',
        run_id: str = '0',
        agent_id: str = '0',
    ) -> None:
        """
        Initialize the FindStep agent.

        Args:
            name: Name of the agent step
            description: Description of what this step does
            run_id: Unique identifier for the current run
            agent_id: Unique identifier for this agent instance
        """
        super().__init__(name=name, description=description)
        self.run_id = run_id
        self.agent_id = agent_id

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the ArXiv search step.

        Args:
            context: Invocation context containing session state and other data

        Yields:
            Event containing the search results
        """
        query = context.session.state[f'query:{self.run_id}:{self.agent_id}']
        logger.debug(f'Executing ArXiv search for query: "{query}"')

        papers_meta = search_arxiv(query)
        logger.info(f'ArXiv search returned {len(papers_meta)} papers')

        yield Event(
            author=self.name,
            content=types.Content(
                role='assistant',
                parts=[types.Part(text=json.dumps(papers_meta))],
            ),
        )


find_step = FindStep()
