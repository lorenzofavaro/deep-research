from collections import Counter

from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.agents.main.plan import DELEGATION_GUIDE
from multi_tool_agent.core.agents.main.plan import ResearchPlan
from multi_tool_agent.core.llm import llm_agent
from multi_tool_agent.core.models import ResearchTask
from multi_tool_agent.core.models import Section
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
    revised_outline: list[Section] = Field(
        default_factory=list,
        description='The outline, revised if the findings show it should change (new or merged sections); empty to '
                    'keep the current one',
    )


system_prompt = f"""You are a Research Reflection Agent. You check the coverage of a research report in progress: \
its outline, its quality constraints and the findings collected so far, and decide whether more research is needed.

**Process:**
1. For each section, check whether its findings answer its questions, with enough evidence and recency
2. Check each quality constraint against the findings
3. Identify gaps: thin or empty sections, unmet constraints, contradictions, tasks that found nothing
4. If there are important gaps, delegate follow-up tasks that fill them, linked to the sections they serve
5. Revise the outline only if the findings show it is wrong (e.g. a sub-topic that matters but has no section)

{DELEGATION_GUIDE}

**Requirements:**
- Mark the research as sufficient when the remaining gaps are minor: more research costs time
- At most {config.max_steps} follow-up tasks, independent from each other and focused on the gaps
- Never repeat a task that was already done; if a task found nothing, reformulate it (narrower objective, other \
sources)"""


def _format_coverage(plan: ResearchPlan, results: list[StepResult]) -> str:
    counts = Counter(
        finding.section for result in results for findings in result.findings.values() for finding in findings
    )
    lines = ['# Outline coverage']
    for section in plan.outline:
        lines.append(
            f'- `{section.id}` {section.title}: {counts[section.id]} findings — ' + '; '.join(
                section.questions,
            ),
        )
    if plan.quality_constraints:
        lines += [
            '', '# Quality constraints', *
            (f'- {constraint}' for constraint in plan.quality_constraints),
        ]
    return '\n'.join(lines)


def _format_results(results: list[StepResult]) -> str:
    lines = ['# Findings so far']
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
            lines += [
                f'  - [{finding.section or "no section"}] {finding.claim}' for finding in findings
            ]
        if not result.findings and not result.error:
            lines.append('No findings.')
    return '\n'.join(lines)


def reflect_input(question: str, plan: ResearchPlan, results: list[StepResult]) -> str:
    """
    Build the reflection input: the outline with its coverage, the quality constraints and the findings.

    Args:
        question: The research question
        plan: The current plan (outline and quality constraints)
        results: Results of the executed tasks

    Returns:
        The reflection input
    """
    return (
        f'Current date: {today()}\nResearch question: {question}\n\n'
        f'{_format_coverage(plan, results)}\n\n{_format_results(results)}'
    )


def reflect_agent() -> LlmAgent:
    """Create the agent that checks coverage and plans follow-up tasks (input: `reflect_input`)."""
    return llm_agent(
        name='ReflectAgent',
        model=config.model,
        instruction=system_prompt,
        output_schema=Reflection,
    )
