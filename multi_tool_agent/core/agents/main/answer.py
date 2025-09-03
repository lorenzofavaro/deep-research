from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event


system_prompt = """Synthesize the research results to provide a direct, comprehensive answer to the user's query.

Research Results:
{results}

User Query:
{query}

Instructions:
- Use ALL research findings to answer the query completely
- Support your response with specific evidence from the results
- For EVERY piece of information you include, cite the source with proper references
- When referencing ArXiv papers, use the format: [Author et al., Year] (arXiv:ID)
- When referencing web sources, use the format: [Title] (URL)
- Include a "References" section at the end listing all sources used
- Structure your answer clearly and concisely with proper citations throughout
- Address the query directly without unnecessary elaboration
- Ensure each claim or statement is backed by a specific source reference"""


class AnswerAgent(BaseAgent):
    """Agent for synthesizing research results into a comprehensive answer."""

    def __init__(self, *, name: str, run_id: str) -> None:
        """
        Initialize the AnswerAgent.

        Args:
            name: Name of the agent
            run_id: Unique identifier for the current run
        """
        super().__init__(name=name)
        self._run_id = run_id

        self._answer_llm = LlmAgent(
            model='gemini-2.0-flash',
            name=f'answer',
            instruction=system_prompt.format(
                results=f'{{results:{self._run_id}}}',
                query='{query}',
            ),
        )

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the answer generation process.

        Args:
            context: Invocation context containing research results and query

        Yields:
            Events from the answer generation LLM
        """
        async for event in self._answer_llm.run_async(context):
            yield event
