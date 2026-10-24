import asyncio

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from pydantic import BaseModel

from multi_tool_agent.core.llm import llm_agent
from multi_tool_agent.core.models import Finding
from multi_tool_agent.core.models import Findings
from multi_tool_agent.core.models import ResearchStep
from multi_tool_agent.core.models import Source
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


extract_prompt = """You are a Research Extraction Agent. You read ONE source and extract the knowledge it contains \
that helps answer a research question.

Rules:
- Extract only what the source itself states. Never add outside knowledge.
- Focus on the step goal, but also keep facts that directly answer the overall research question.
- Each claim must be self-contained: include the concrete names, numbers, conditions and comparisons.
- For each claim, copy a short supporting quote (max ~300 characters) verbatim from the source as evidence.
- Return at most 8 findings, the most informative first.
- If the source is not relevant, return an empty list."""


class StepTask(BaseModel):
    """Input of a research step node."""
    step: ResearchStep
    question: str


def extract_agent() -> LlmAgent:
    """Create the agent that extracts findings from one source (input: `extract_input`)."""
    return llm_agent(
        name='ExtractAgent',
        model=config.fast_model,
        instruction=extract_prompt,
        output_schema=Findings,
    )


def extract_input(task: StepTask, source: Source, text: str) -> str:
    """Build the extraction input: the research question, the step goal and the source with its metadata."""
    details = [f'Title: {source.title}', f'URL: {source.url}']
    if source.authors:
        details.append(f'Authors: {", ".join(source.authors[:10])}')
    if source.published:
        details.append(f'Published: {source.published}')
    return (
        f'Research question: {task.question}\n'
        f'Step goal: {task.step.goal}\n\n'
        + '\n'.join(details)
        + f'\n\n<source>\n{text}\n</source>'
    )


async def extract_findings(
    ctx: Context, task: StepTask, documents: list[tuple[Source, str]],
) -> tuple[list[Source], dict[str, list[Finding]]]:
    """
    Extract findings from each document in parallel.

    A failed extraction (model error, invalid output) is logged and skipped.

    Args:
        ctx: Context of the calling node
        task: The research step
        documents: Sources paired with their text

    Returns:
        The deduplicated sources, and the findings by source key (sources without findings are left out)
    """
    # Search engines can return the same page twice
    documents = list({
        source.key: (source, text)
        for source, text in documents
    }.values())
    outputs = await asyncio.gather(
        *(
            ctx.run_node(
                extract_agent(), node_input=extract_input(
                    task, source, text,
                ), use_sub_branch=True,
            )
            for source, text in documents
        ),
        return_exceptions=True,
    )
    findings: dict[str, list[Finding]] = {}
    for (source, _), output in zip(documents, outputs):
        if isinstance(output, BaseException):
            logger.warning(f'Extraction from {source.url} failed: {output}')
            continue
        extracted = Findings.model_validate(output).findings if output else []
        if extracted:
            findings[source.key] = extracted
    return [source for source, _ in documents], findings
