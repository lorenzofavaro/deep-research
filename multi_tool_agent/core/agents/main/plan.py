from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.llm import llm_agent
from multi_tool_agent.core.models import ResearchTask
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.utils import today


class ResearchPlan(BaseModel):
    """Complete research plan, whose tasks run in parallel."""
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

**Effort scaling** (the complexity of the question is given)
- simple (a single fact or definition): 1 task
- moderate: 2-3 tasks
- complex (comparisons, broad topics, several sub-questions): up to {config.max_steps} tasks, with different lenses"""

system_prompt = f"""You are a Research Planning Agent: you split a research question into tasks for parallel \
researchers.

{DELEGATION_GUIDE}

**Requirements:**
- At most {config.max_steps} tasks; tasks run in parallel, so they must be independent
- Together, the tasks must cover every part of the question
- Use papers only when academic literature is relevant; news and current facts need web
- Consider the current date when the question is about recent or latest developments"""


def plan_input(question: str, complexity: str) -> str:
    """Build the planner input for a research question."""
    return f'Current date: {today()}\nResearch question: {question}\nComplexity: {complexity}'


def plan_agent() -> LlmAgent:
    """Create the agent that plans the research tasks for a question (input: `plan_input`)."""
    return llm_agent(
        name='PlanAgent',
        model=config.model,
        instruction=system_prompt,
        output_schema=ResearchPlan,
    )
