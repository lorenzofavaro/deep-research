import json

from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.llm import isolated_llm_agent
from multi_tool_agent.core.models import ResearchStep
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.utils import today


class PaperIDs(BaseModel):
    """Collection of selected paper IDs after filtering."""
    ids: list[str] = Field(
        default_factory=list,
        description='IDs of the selected papers, most relevant first',
    )


system_prompt = """You are a Research Search Agent.
Your role is to select, among candidate arXiv papers, the ones whose full text is most worth reading for a research step.

**Step 1 — Analyze candidates:**
- Review each candidate's title, abstract, publication date and citation count.
- Assess how directly it addresses the step goal and the overall research question.

**Step 2 — Select results:**
- Choose **up to {papers_num}** papers, most relevant first.
- Use citations as a signal of influence: for established methods, comparisons and overviews, prefer the well-cited \
papers that define the field over niche variants.
- Citation counts lag behind: recent papers have few citations, so when the goal is about the latest developments, \
judge them by relevance and recency instead.
- Only include papers that provide direct, specific insights on the topic.
- If fewer are relevant, return only those (never add irrelevant ones). An empty list is a valid answer.
- Use only IDs that appear among the candidates."""


def filter_agent(*, name: str, output_key: str, step: ResearchStep, question: str, candidates: list[dict]) -> LlmAgent:
    """
    Create the agent that selects which candidate papers to read.

    Args:
        name: Unique name of the agent
        output_key: State key for the selected PaperIDs
        step: The research step being executed
        question: The overall research question
        candidates: Papers returned by the arXiv search

    Returns:
        The configured LlmAgent
    """
    listing = [
        {
            'id': paper['id'],
            'title': paper['title'],
            'published': paper['published'],
            'authors': paper['authors'][:3],
            'citations': paper.get('citations', 'unknown'),
            'abstract': paper['abstract'][:1500],
        }
        for paper in candidates
    ]
    user_input = (
        f'Current date: {today()}\n'
        f'Research question: {question}\n'
        f'Step goal: {step.goal}\n\n'
        f'Candidates:\n{json.dumps(listing, ensure_ascii=False, indent=1)}'
    )
    return isolated_llm_agent(
        name=name,
        model=config.fast_model,
        instruction=system_prompt.format(papers_num=config.papers_per_step),
        build_input=lambda _: user_input,
        output_schema=PaperIDs,
        output_key=output_key,
    )
