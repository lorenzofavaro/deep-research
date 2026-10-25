import re

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.agents.main.answer import describe_source
from multi_tool_agent.core.agents.main.answer import RegistryEntry
from multi_tool_agent.core.agents.main.verify import CITATION
from multi_tool_agent.core.llm import llm_agent
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)

# A refined report must keep at least this share of the original citation markers, or it is discarded
MIN_KEPT_CITATIONS = 0.9


class RefinedReport(BaseModel):
    """The refined report."""
    report: str = Field(
        description='The whole report in Markdown, with its [n] citations',
    )


system_prompt = """You are a Report Refiner. You improve a research report with targeted, structured edits, using \
only the evidence of its sources.

Edits to make, where the evidence allows:
- **Quantify**: replace vague statements ("much faster", "widely used") with the numbers, dates and names the \
evidence gives, keeping the citations
- **Compare**: when a section compares several items (methods, products, results), add a Markdown table with the \
compared values, each row cited
- **Explain**: make causal links explicit (why something happens, what it implies), only when the evidence states them

Rules:
- Do not add facts that are not in the evidence below; do not remove sections or claims
- Keep every citation marker ([n]) attached to the claims it supports; never invent citation numbers
- Keep the structure (title, sections, conclusion) and the language of the report
- Return the whole report"""


def refine_input(report: str, registry: list[RegistryEntry]) -> str:
    """Build the refiner input: the report and the evidence of its sources."""
    lines = ['<report>', report, '</report>', '', '# Evidence of the sources']
    for number, (source, findings) in enumerate(registry, start=1):
        lines.append(f'[{number}] {describe_source(source)}')
        lines += [
            f'  - {finding.claim} (evidence: "{finding.evidence}")' for finding in findings
        ]
    return '\n'.join(lines)


def refine_agent() -> LlmAgent:
    """Create the agent that refines a report (input: `refine_input`)."""
    return llm_agent(name='RefineAgent', model=config.model, instruction=system_prompt, output_schema=RefinedReport)


def _citations(text: str) -> int:
    return sum(len(group.split(',')) for group in CITATION.findall(text))


async def refine_report(ctx: Context, report: str, registry: list[RegistryEntry]) -> str:
    """
    Refine a report (quantify, add comparison tables, explain causal links), keeping the original when the
    refinement fails or drops citations.

    Args:
        ctx: Context of the calling node
        report: The report
        registry: Sources with their findings, numbered from 1

    Returns:
        The refined report, or the original one
    """
    try:
        output = await ctx.run_node(refine_agent(), node_input=refine_input(report, registry))
        refined = RefinedReport.model_validate(output).report.strip()
        # Models sometimes echo the wrapper of the input
        refined = re.sub(
            r'^\s*<report>\s*|\s*</report>\s*$',
            '', refined,
        ).strip()
    except Exception as e:
        logger.warning(f'Refinement failed, keeping the report: {e}')
        return report
    if _citations(refined) < MIN_KEPT_CITATIONS * _citations(report):
        logger.warning(
            'The refined report dropped citations, keeping the original',
        )
        return report
    return refined
