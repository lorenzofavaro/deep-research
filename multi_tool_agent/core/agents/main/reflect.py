from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.agents.main.plan import DELEGATION_GUIDE
from multi_tool_agent.core.llm import llm_agent
from multi_tool_agent.core.models import ResearchTask
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.utils import today


class Reflection(BaseModel):
    """Assessment of the research done so far."""
    sufficient: bool = Field(
        description='True if the findings are enough to answer the question well',
    )
    missing: str = Field(
        description='What is still missing or contradictory; empty if sufficient',
    )
    follow_up_tasks: list[ResearchTask] = Field(
        default_factory=list,
        description='New research tasks that fill the gaps; empty if sufficient',
    )


system_prompt = f"""You are a Research Reflection Agent. You review the findings collected so far for a research \
question and decide whether more research is needed.

**Process:**
1. Check whether the findings answer every part of the question, with enough evidence and recency
2. Identify gaps: unanswered sub-questions, missing recent data, contradictions, tasks that found nothing
3. If there are important gaps, delegate follow-up tasks that fill them

{DELEGATION_GUIDE}

**Requirements:**
- Mark the research as sufficient when the remaining gaps are minor: more research costs time
- At most {config.max_steps} follow-up tasks, independent from each other and focused on the gaps
- Never repeat a task that was already done; if a task found nothing, reformulate it (narrower objective, other \
sources)"""


def _format_results(results: list[StepResult]) -> str:
    lines = []
    for result in results:
        lines.append(
            f'## Task ({result.step.lens}, {", ".join(result.step.sources)}): {result.step.objective}',
        )
        if result.error:
            lines.append(f'Failed: {result.error}')
        if result.brief:
            lines.append(f'Researcher brief: {result.brief}')
        titles = {source.key: source.title for source in result.sources}
        for key, findings in result.findings.items():
            lines.append(f'- Source: {titles.get(key, key)}')
            lines += [f'  - {finding.claim}' for finding in findings]
        if not result.findings and not result.error:
            lines.append('No findings.')
    return '\n'.join(lines)


def reflect_input(question: str, results: list[StepResult]) -> str:
    """
    Build the reflection input from the results of all the tasks executed so far.

    Args:
        question: The research question
        results: Results of the executed tasks

    Returns:
        The reflection input
    """
    return f'Current date: {today()}\nResearch question: {question}\n\n# Findings so far\n{_format_results(results)}'


def reflect_agent() -> LlmAgent:
    """Create the agent that reviews the findings and plans follow-up tasks (input: `reflect_input`)."""
    return llm_agent(
        name='ReflectAgent',
        model=config.model,
        instruction=system_prompt,
        output_schema=Reflection,
    )
