from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents import ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from multi_tool_agent.core.agents.research.arxiv.arx_agent import ArxivAgent
from multi_tool_agent.core.agents.research.base import step_result_key
from multi_tool_agent.core.agents.research.web.search_agent import WebSearchAgent
from multi_tool_agent.core.models import ResearchStep
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.utils.logger import get_logger
from multi_tool_agent.utils.utils import valid_uuid

logger = get_logger(__name__)


def collect_results(state: dict, run_id: str) -> list[StepResult]:
    """
    Read the results of all the research steps of a run from the session state.

    Args:
        state: Session state
        run_id: Unique identifier of the run

    Returns:
        The step results, in execution order
    """
    prefix = step_result_key(run_id, '')
    return [StepResult.model_validate(value) for key, value in state.items() if key.startswith(prefix)]


class ResearchAgent(BaseAgent):
    """Agent for executing a batch of research steps in parallel."""

    def __init__(self, *, name: str, run_id: str, steps: list[ResearchStep], question: str) -> None:
        """
        Initialize the ResearchAgent.

        Args:
            name: Name of the agent
            run_id: Unique identifier for the current run
            steps: Research steps to execute
            question: The overall research question
        """
        super().__init__(name=name)
        self._run_id = run_id
        self._steps = steps
        self._question = question

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the research steps with parallel agents, each storing a StepResult in the state.

        Args:
            context: Invocation context

        Yields:
            Events from the research execution process
        """
        sub_agents: list[BaseAgent] = []
        for step in self._steps:
            step_id = valid_uuid()
            logger.debug(f'Processing research step {step_id}: {step}')
            agent_class = ArxivAgent if step.tool == 'arxiv' else WebSearchAgent
            sub_agents.append(
                agent_class(
                    # Names must be unique: ParallelAgent derives the branch of each sub-agent from its name
                    name=f'{step.tool}_{step_id}',
                    run_id=self._run_id,
                    step_id=step_id,
                    step=step,
                    question=self._question,
                ),
            )

        parallel = ParallelAgent(
            name=f'parallel_{valid_uuid()}', sub_agents=sub_agents,
        )
        async for event in parallel.run_async(context):
            yield event
