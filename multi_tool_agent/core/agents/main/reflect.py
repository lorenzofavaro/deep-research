from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.agents.main.plan import TOOLS_GUIDE
from multi_tool_agent.core.llm import isolated_llm_agent
from multi_tool_agent.core.models import ResearchStep
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
    follow_up_steps: list[ResearchStep] = Field(
        default_factory=list,
        description='New research steps that fill the gaps; empty if sufficient',
    )


system_prompt = f"""You are a Research Reflection Agent. You review the findings collected so far for a research \
question and decide whether more research is needed.

**Process:**
1. Check whether the findings answer every part of the question, with enough evidence and recency
2. Identify gaps: unanswered sub-questions, missing recent data, contradictions, steps that returned nothing
3. If there are important gaps, plan follow-up steps that fill them

{TOOLS_GUIDE}

**Requirements:**
- Mark the research as sufficient when the remaining gaps are minor: more research costs time
- At most {config.max_steps} follow-up steps, independent from each other
- Never repeat a query that was already executed; if a query returned nothing, reformulate it (e.g. broader terms, \
the other tool)"""


def _format_results(results: list[StepResult]) -> str:
    lines = []
    for result in results:
        lines.append(
            f'## Step [{result.step.tool}] `{result.step.query}` (goal: {result.step.goal})',
        )
        if result.error:
            lines.append(f'Failed: {result.error}')
        titles = {source.key: source.title for source in result.sources}
        for key, findings in result.findings.items():
            lines.append(f'- Source: {titles.get(key, key)}')
            lines += [f'  - {finding.claim}' for finding in findings]
        if not result.findings and not result.error:
            lines.append('No findings.')
    return '\n'.join(lines)


def reflect_agent(*, question: str, results: list[StepResult]) -> LlmAgent:
    """
    Create the agent that reviews the findings and plans follow-up steps.

    Args:
        question: The research question
        results: Results of all the steps executed so far

    Returns:
        The configured LlmAgent
    """
    user_input = f'Current date: {today()}\nResearch question: {question}\n\n# Findings so far\n{_format_results(results)}'
    return isolated_llm_agent(
        name='ReflectAgent',
        model=config.model,
        instruction=system_prompt,
        build_input=lambda _: user_input,
        output_schema=Reflection,
        output_key='reflection',
    )
