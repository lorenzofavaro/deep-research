import os
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from mcp import StdioServerParameters


system_prompt = """You are a Web Research Agent specialized in finding comprehensive and up-to-date information online.

**Your Role:**
- Search the internet for current, relevant information that complements academic research
- Find recent developments, news, industry reports, and web-based content
- Provide diverse perspectives and real-world applications of research topics
- Identify trends, practical implementations, and emerging discussions

**Search Strategy:**
1. Analyze the user query to identify key concepts and search terms
2. Perform targeted searches to gather comprehensive information
3. Focus on credible sources: reputable websites, industry publications, tech blogs, company reports
4. Look for recent content (prioritize information from the last 1-2 years when relevant)
5. Gather multiple perspectives and viewpoints on the topic

**Output Guidelines:**
- Structure your response as a numbered list of knowledge pieces
- For each knowledge piece, provide:
  * A clear, concise statement of the knowledge/insight
  * The specific online source(s) where this information was found (include URLs)
- Focus on key insights, trends, and practical applications
- Note the recency and credibility of sources when relevant

**Quality Standards:**
- Prioritize authoritative and credible sources
- Avoid outdated information unless historical context is needed
- Cross-reference information when possible
- Be transparent about limitations or conflicting information found.

Query:
{query}
"""


class WebSearchAgent(BaseAgent):
    """Agent for performing web-based research using external search tools."""

    def __init__(self, *, name: str, run_id: str, agent_id: str) -> None:
        """
        Initialize the WebSearchAgent.

        Args:
            name: Name of the agent
            run_id: Unique identifier for the current run
            agent_id: Unique identifier for this agent instance
        """
        super().__init__(name=name)
        self._run_id = run_id
        self._agent_id = agent_id

        self._web_search_llm = LlmAgent(
            model='gemini-2.0-flash',
            name=f'web_search_{agent_id}',
            instruction=system_prompt.format(
                query=f'{{query:{self._run_id}:{self._agent_id}}}',
            ),
            tools=[
                MCPToolset(
                    connection_params=StdioConnectionParams(
                        server_params=StdioServerParameters(
                            command='npx',
                            args=['-y', 'tavily-mcp@latest'],
                            env={
                                'TAVILY_API_KEY': os.getenv(
                                    'TAVILY_API_KEY', '',
                                ),
                            },
                        ),
                        timeout=30.0,
                    ),
                    tool_filter=['tavily-search'],
                ),
            ],
            output_key=f'results:{self._run_id}:{self._agent_id}',
        )

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the web search workflow.

        Args:
            context: Invocation context containing session state and configuration

        Yields:
            Events from the web search LLM agent
        """
        async for event in self._web_search_llm.run_async(context):
            yield event
