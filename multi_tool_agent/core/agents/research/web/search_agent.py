from collections.abc import AsyncGenerator

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from multi_tool_agent.core.agents.research.base import message
from multi_tool_agent.core.agents.research.base import ResearchStepAgent
from multi_tool_agent.core.models import Source
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.core.tools.tavily import tavily_search
from multi_tool_agent.utils.config import config


class WebSearchAgent(ResearchStepAgent):
    """Research step on the web: search with Tavily, then extract findings from the text of each page."""

    async def _research(self, context: InvocationContext, result: StepResult) -> AsyncGenerator[Event, None]:
        """
        Execute the web search workflow.

        Args:
            context: Invocation context
            result: Step result to fill

        Yields:
            Events from the search and the extraction
        """
        pages = await tavily_search(self._step.query, max_results=config.web_results)
        yield message(self.name, f'web `{self._step.query}`: {len(pages)} pages')

        documents = [
            (
                Source(
                    key=page['url'], kind='web', title=page['title'],
                    url=page['url'], published=page['published'],
                ),
                page['content'][:config.page_max_chars],
            )
            for page in pages
            if page['content']
        ]
        async for event in self._extract(context, result, documents):
            yield event
