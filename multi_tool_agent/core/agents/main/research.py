import asyncio

from google.adk.agents.context import Context

from multi_tool_agent.core.agents.research.researcher import research_task
from multi_tool_agent.core.agents.research.researcher import TaskInput
from multi_tool_agent.core.models import ResearchTask
from multi_tool_agent.core.models import StepResult


async def run_research_round(
    ctx: Context, tasks: list[ResearchTask], question: str, complexity: str,
) -> list[StepResult]:
    """
    Run one researcher per task, in parallel, each in its own branch.

    Args:
        ctx: Context of the calling node
        tasks: Research tasks to execute
        question: The overall research question
        complexity: Complexity of the question, which sets each researcher's budget

    Returns:
        The task results, in the order of the tasks
    """
    outputs = await asyncio.gather(
        *(
            ctx.run_node(
                research_task,
                node_input=TaskInput(
                    task=task, question=question, complexity=complexity,
                ).model_dump(),
                use_sub_branch=True,
            )
            for task in tasks
        ),
        return_exceptions=True,
    )
    return [
        StepResult(step=task, error=str(output)) if isinstance(output, BaseException)
        else StepResult.model_validate(output)
        for task, output in zip(tasks, outputs)
    ]
