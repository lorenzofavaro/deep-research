from typing import Optional

from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.llm import llm_agent
from multi_tool_agent.core.models import ResearchTask
from multi_tool_agent.core.models import Section
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.utils import today


class ResearchPlan(BaseModel):
    """Outline of the report and the research tasks that gather its evidence, which run in parallel."""
    outline: list[Section] = Field(
        description='Sections of the report, in reading order',
    )
    quality_constraints: list[str] = Field(
        default_factory=list,
        description='Checkable requirements for a good answer, e.g. "quantitative comparison of the main methods"',
    )
    tasks: list[ResearchTask] = Field(
        description='Independent research tasks, one per researcher',
    )


# Shared with the reflection agent, which delegates the follow-up tasks
DELEGATION_GUIDE = f"""**Researchers**
Each task goes to an autonomous researcher that searches, reads sources and records findings on its own. Write \
tasks like a lead researcher briefing a colleague:
- `objective`: a specific question or goal, with the context needed to understand it
- `sources`: "papers" (arXiv: methods, benchmarks, established research) and/or "web" (news, products, \
industry, current facts)
- `lens`: the angle — evidence (core facts and results), comparator (alternatives on concrete criteria), critic \
(limitations, counter-evidence, disagreements), horizon (latest developments, open problems)
- `boundaries`: what other researchers cover, so efforts do not overlap
- `section_ids`: the outline sections the task gathers evidence for; every section needs at least one task

**Effort scaling** (the complexity of the question is given)
- simple (a single fact or definition): 1 task, 1 section
- moderate: 2-3 tasks
- complex (comparisons, broad topics, several sub-questions): up to {config.max_steps} tasks, with different lenses"""

system_prompt = f"""You are a Research Planning Agent, the architect of a research report. Given a research \
question (and, when available, a scout's quick survey of the landscape), you design:

1. **The outline**: the sections the answer needs, in reading order, each with the questions it must answer. Base \
it on what the scout found (the sub-topics and terms that actually exist in the sources), not on generic headings.
2. **Quality constraints**: 2-4 checkable requirements a good answer must meet (e.g. "compares at least 3 methods \
with numbers", "covers developments of the last 12 months", "cites primary sources for each claim").
3. **The research tasks** that gather the evidence for the sections.

{DELEGATION_GUIDE}

**Requirements:**
- At most {config.max_steps} tasks; tasks run in parallel, so they must be independent
- Together, the tasks must cover every section; do not re-research what the scout already established well
- Use papers only when academic literature is relevant; news and current facts need web
- Consider the current date when the question is about recent or latest developments"""


def _format_scout(scout: StepResult) -> str:
    lines = ['# Scout survey', scout.brief or '(no brief)']
    titles = {source.key: source.title for source in scout.sources}
    for key, findings in scout.findings.items():
        lines.append(f'- {titles.get(key, key)}')
        lines += [f'  - {finding.claim}' for finding in findings]
    return '\n'.join(lines)


def plan_input(
    question: str, complexity: str, scout: Optional[StepResult] = None, variant: int = 1,
    variants: int = 1,
) -> str:
    """
    Build the planner input: the question, its complexity and the scout's survey, if any.

    Args:
        question: The research question
        complexity: Its complexity
        scout: The scout's result, if a scout ran
        variant: Number of this plan, when several independent research runs are merged (ensemble)
        variants: Number of independent runs

    Returns:
        The planner input
    """
    text = f'Current date: {today()}\nResearch question: {question}\nComplexity: {complexity}'
    if scout:
        text += f'\n\n{_format_scout(scout)}'
    if variants > 1:
        text += (
            f'\n\nThis is independent research run {variant} of {variants}, whose findings will be merged: '
            + (
                'plan the most direct approach.' if variant == 1 else
                'choose different angles, sources and sub-questions than the most obvious plan would.'
            )
        )
    return text


def plan_agent() -> LlmAgent:
    """Create the agent that plans the outline and the research tasks (input: `plan_input`)."""
    return llm_agent(
        name='PlanAgent',
        model=config.model,
        instruction=system_prompt,
        output_schema=ResearchPlan,
    )
