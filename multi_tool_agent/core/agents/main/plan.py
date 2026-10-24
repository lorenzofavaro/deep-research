from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.llm import llm_agent
from multi_tool_agent.core.models import ResearchStep
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.utils import today


class ResearchPlan(BaseModel):
    """Complete research plan, whose steps run in parallel."""
    steps: list[ResearchStep] = Field(
        description='List of independent research steps to execute',
    )


# Shared with the reflection agent, which plans the follow-up steps
TOOLS_GUIDE = """**Available Tools:**
1. **arxiv** - Academic papers: theoretical foundations, methods, benchmarks, established research. The full text of \
the most relevant papers is read.
   The query MUST use arXiv API syntax, because plain sentences match almost everything:
   - fields: `ti:` (title), `abs:` (abstract), `au:` (author), `cat:` (category, e.g. cat:cs.CL), `all:` (any field)
   - phrases in double quotes, combined with AND / OR / ANDNOT and parentheses
   - example: `abs:"retrieval augmented generation" AND (abs:evaluation OR abs:benchmark)`
   - keep it to 2-3 key concepts: over-constrained queries return nothing
2. **web** - Recent developments, news, industry applications, products, current facts. The query is a concise \
search engine query (keywords, no instructions)."""

system_prompt = f"""You are a Research Planning Agent that turns a research question into a plan of search steps.

{TOOLS_GUIDE}

**Process:**
1. Break the question into its key components
2. For each component pick the tool that is most likely to contain the answer
3. Write a targeted query and state the goal of the step

**Requirements:**
- At most {config.max_steps} steps; use fewer when the question is narrow (e.g. a single current fact needs one web step)
- Steps run in parallel, so they must be independent and must not overlap
- Use arxiv only when academic literature is relevant; news and current facts need web
- Consider the current date when the question is about recent or latest developments"""


def plan_input(question: str) -> str:
    """Build the planner input for a research question."""
    return f'Current date: {today()}\nResearch question: {question}'


def plan_agent() -> LlmAgent:
    """Create the agent that plans the research steps for a question (input: `plan_input`)."""
    return llm_agent(
        name='PlanAgent',
        model=config.model,
        instruction=system_prompt,
        output_schema=ResearchPlan,
    )
