from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents import ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events import EventActions
from google.genai import types

from multi_tool_agent.core.llm import GuardedAgent
from multi_tool_agent.core.llm import isolated_llm_agent
from multi_tool_agent.core.models import Findings
from multi_tool_agent.core.models import ResearchStep
from multi_tool_agent.core.models import Source
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


extract_prompt = """You are a Research Extraction Agent. You read ONE source and extract the knowledge it contains \
that helps answer a research question.

Rules:
- Extract only what the source itself states. Never add outside knowledge.
- Focus on the step goal, but also keep facts that directly answer the overall research question.
- Each claim must be self-contained: include the concrete names, numbers, conditions and comparisons.
- For each claim, copy a short supporting quote (max ~300 characters) verbatim from the source as evidence.
- Return at most 8 findings, the most informative first.
- If the source is not relevant, return an empty list."""


def step_result_key(run_id: str, step_id: str) -> str:
    """State key holding the StepResult of a research step."""
    return f'step_{run_id}_{step_id}'


def message(author: str, text: str, **kwargs) -> Event:
    """Build an event carrying a short progress message."""
    return Event(
        author=author,
        content=types.Content(role='model', parts=[types.Part(text=text)]),
        **kwargs,
    )


class ResearchStepAgent(BaseAgent):
    """Base agent for a research step: collects sources, then extracts findings from each one in parallel."""

    def __init__(self, *, name: str, run_id: str, step_id: str, step: ResearchStep, question: str) -> None:
        """
        Initialize the research step agent.

        Args:
            name: Unique name of the agent
            run_id: Unique identifier for the current run
            step_id: Unique identifier for this step
            step: The research step to execute
            question: The overall research question
        """
        super().__init__(name=name)
        self._run_id = run_id
        self._step_id = step_id
        self._step = step
        self._question = question

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the step and store its StepResult in the state, even when it fails.

        Args:
            context: Invocation context

        Yields:
            Events from the step
        """
        result = StepResult(step=self._step)
        try:
            async for event in self._research(context, result):
                yield event
        except Exception as e:
            logger.error(
                f'Research step {self._step_id} failed: {e}', exc_info=True,
            )
            result.error = str(e)

        found = sum(len(findings) for findings in result.findings.values())
        summary = f'{self._step.tool} `{self._step.query}`: {found} findings from {len(result.findings)} sources'
        if result.error:
            summary += f' (error: {result.error})'
        yield message(
            self.name,
            summary,
            actions=EventActions(
                state_delta={
                    step_result_key(
                        self._run_id, self._step_id,
                    ): result.model_dump(),
                },
            ),
        )

    async def _research(self, context: InvocationContext, result: StepResult) -> AsyncGenerator[Event, None]:
        """Collect sources and findings into `result`. Implemented by subclasses."""
        raise NotImplementedError
        yield  # pragma: no cover

    async def _extract(
        self, context: InvocationContext, result: StepResult, documents: list[tuple[Source, str]],
    ) -> AsyncGenerator[Event, None]:
        """
        Extract findings from each document in parallel and add them to `result`.

        Args:
            context: Invocation context
            result: Step result to fill
            documents: Sources paired with their text

        Yields:
            Events from the extraction agents
        """
        # Search engines can return the same page twice
        documents = list({
            source.key: (source, text)
            for source, text in documents
        }.values())
        if not documents:
            return
        output_keys: dict[str, str] = {}
        agents: list[BaseAgent] = []
        for i, (source, text) in enumerate(documents):
            output_key = f'extract_{self._run_id}_{self._step_id}_{i}'
            output_keys[source.key] = output_key
            agents.append(
                GuardedAgent(
                    agent=isolated_llm_agent(
                        name=f'extract_{self._step_id}_{i}',
                        model=config.fast_model,
                        instruction=extract_prompt,
                        build_input=lambda _, source=source, text=text: self._extract_input(
                            source, text,
                        ),
                        output_schema=Findings,
                        output_key=output_key,
                    ),
                ),
            )

        async for event in ParallelAgent(name=f'extract_{self._step_id}', sub_agents=agents).run_async(context):
            yield event

        result.sources = [source for source, _ in documents]
        for source in result.sources:
            output = context.session.state.get(output_keys[source.key])
            findings = Findings.model_validate(
                output,
            ).findings if output else []
            if findings:
                result.findings[source.key] = findings

    def _extract_input(self, source: Source, text: str) -> str:
        details = [f'Title: {source.title}', f'URL: {source.url}']
        if source.authors:
            details.append(f'Authors: {", ".join(source.authors[:10])}')
        if source.published:
            details.append(f'Published: {source.published}')
        return (
            f'Research question: {self._question}\n'
            f'Step goal: {self._step.goal}\n\n'
            + '\n'.join(details)
            + f'\n\n<source>\n{text}\n</source>'
        )
