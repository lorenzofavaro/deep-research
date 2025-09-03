from typing import Literal

from google.adk.agents import LlmAgent
from google.adk.planners.plan_re_act_planner import PlanReActPlanner
from pydantic import BaseModel
from pydantic import Field


class ClassificationResult(BaseModel):
    type: Literal[
        'valid', 'general',
        'need-more-info',
    ] = Field(description='Classification result')
    user_intent: str | None = Field(
        description="Precise user intent for 'valid' classifications", default=None,
    )
    next_message: str | None = Field(
        description="Next message to send to user for 'general' or 'need-more-info' classifications", default=None,
    )


class ResearchStep(BaseModel):
    action: Literal['arxiv_search', 'web_search'] = Field(
        description='Type of action to perform',
    )
    query: str = Field(
        description='Search query and description of what this step should accomplish',
    )


class ResearchPlan(BaseModel):
    steps: list[ResearchStep] = Field(
        description='List of research steps to execute',
    )


GEMINI_MODEL = 'gemini-1.5-flash'

system_prompt = """You are a Research Planning Agent that creates step-by-step research plans from the user_intent.

**Available Tools:**
1. **ArXiv Search** - Academic papers, theoretical foundations, established research
2. **Web Search** - Recent developments, industry applications, current trends

**Process:**
1. Start from the provided user_intent
2. Break it into key research components
3. Create targeted search queries using appropriate tools
4. Sequence steps logically to build comprehensive understanding

**Requirements:**
- Maximum 3 steps
- Each step must directly address the user's intent
- Build knowledge progressively from foundations to current applications
"""

plan_agent = LlmAgent(
    name='PlanAgent',
    model=GEMINI_MODEL,
    instruction=system_prompt,
    planner=PlanReActPlanner(),
    description='Plan a list of steps to gather all the information needed to answer the user query',
    input_schema=ClassificationResult,
    output_schema=ResearchPlan,
    output_key='research_plan',
)
