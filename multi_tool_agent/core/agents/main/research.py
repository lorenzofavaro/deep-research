import asyncio

from google.adk.agents.context import Context
from google.adk.workflow import node

from multi_tool_agent.core.agents.research.arxiv.arx_agent import research_arxiv
from multi_tool_agent.core.agents.research.base import StepTask
from multi_tool_agent.core.agents.research.web.search_agent import research_web
from multi_tool_agent.core.models import ResearchStep
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


@node(name='research_step', rerun_on_resume=True)
async def research_step(ctx: Context, node_input: StepTask) -> StepResult:
    """
    Execute one research step. Failures are recorded in the result instead of raised, so a failing step does not
    affect the others.

    Args:
        ctx: Node context
        node_input: The step and the research question

    Returns:
        The step result
    """
    result = StepResult(step=node_input.step)
    research = research_arxiv if node_input.step.tool == 'arxiv' else research_web
    try:
        await research(ctx, node_input, result)
    except Exception as e:
        logger.error(
            f'Research step `{node_input.step.query}` failed: {e}', exc_info=True,
        )
        result.error = str(e)
    return result


async def run_research_round(ctx: Context, steps: list[ResearchStep], question: str) -> list[StepResult]:
    """
    Execute research steps in parallel, each in its own branch.

    Args:
        ctx: Context of the calling node
        steps: Research steps to execute
        question: The overall research question

    Returns:
        The step results, in the order of the steps
    """
    outputs = await asyncio.gather(
        *(
            ctx.run_node(
                research_step, node_input=StepTask(step=step, question=question).model_dump(), use_sub_branch=True,
            )
            for step in steps
        ),
        return_exceptions=True,
    )
    return [
        StepResult(step=step, error=str(output)) if isinstance(output, BaseException)
        else StepResult.model_validate(output)
        for step, output in zip(steps, outputs)
    ]
