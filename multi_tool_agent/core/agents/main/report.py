import asyncio

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.agents.main.answer import describe_source
from multi_tool_agent.core.agents.main.answer import RegistryEntry
from multi_tool_agent.core.llm import llm_agent
from multi_tool_agent.core.models import Section
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger
from multi_tool_agent.utils.utils import today

logger = get_logger(__name__)

# Findings without a section (e.g. the scout's) offered to every section writer, at most
MAX_SHARED_FINDINGS = 20


class SectionDraft(BaseModel):
    """The text of one report section."""
    body: str = Field(
        description='The section text in Markdown, without its heading',
    )


class ReportFrame(BaseModel):
    """The parts of the report around the sections."""
    title: str
    introduction: str = Field(
        description='Executive summary: the answer in a few sentences, with citations',
    )
    conclusion: str = Field(
        description='Conclusions: synthesis, open questions and evidence gaps, with citations',
    )


section_prompt = """You are a Report Section Writer. You write ONE section of a research report, using only the \
evidence given for it.

Instructions:
- Answer the section's questions directly, with specific evidence (numbers, names, results, dates)
- Use ONLY the findings provided; do not add outside knowledge
- Cite the source of EVERY claim with its number in square brackets, e.g. [2] or [1, 3], right after the claim; \
never cite a number that is not listed
- Compare and connect the findings: agreements, contradictions, trends; use a table when comparing several items
- If the evidence does not answer a question, say so briefly instead of guessing
- Write the body only (no section heading, no references list), in the same language as the user's message
- Be dense and concrete: no filler, no repetition of the question"""

frame_prompt = """You are a Report Editor. Given the drafted sections of a research report, write its title, an \
introduction (an executive summary that answers the research question in a few sentences) and a conclusion \
(synthesis across sections, open questions, evidence gaps).

Instructions:
- Base everything on the drafted sections and keep their citations ([n]) for each claim you reuse
- Do not introduce new facts
- Write in the same language as the user's message"""


def _format_findings(registry: list[RegistryEntry], keep) -> list[str]:
    lines = []
    for number, (source, findings) in enumerate(registry, start=1):
        selected = [finding for finding in findings if keep(finding)]
        if not selected:
            continue
        lines.append(f'[{number}] {describe_source(source)}')
        for finding in selected:
            lines.append(f'  - {finding.claim}')
            lines.append(f'    Evidence: "{finding.evidence}"')
    return lines


def section_input(
    user_message: str, question: str, section: Section, registry: list[RegistryEntry], section_ids: set[str],
) -> str:
    """
    Build a section writer's input: the section, its findings and the findings that belong to no section.

    Args:
        user_message: The latest user message, which sets the language
        question: The research question
        section: The section to write
        registry: Sources with their findings, numbered from 1
        section_ids: Ids of all the outline sections

    Returns:
        The section writer input
    """
    own = _format_findings(
        registry, lambda finding: finding.section == section.id,
    )
    shared = _format_findings(registry, lambda finding: finding.section not in section_ids)[
        :3 * MAX_SHARED_FINDINGS
    ]
    lines = [
        f'Current date: {today()}',
        f'User message: {user_message}',
        f'Research question: {question}',
        f'Section: {section.title}',
        'Questions to answer: ' + '; '.join(section.questions),
        '',
        '# Findings for this section',
        *(own or ['(none)']),
    ]
    if shared:
        lines += [
            '',
            '# Other findings (use only those relevant to this section)', *shared,
        ]
    return '\n'.join(lines)


def section_agent() -> LlmAgent:
    """Create the agent that writes one report section (input: `section_input`)."""
    return llm_agent(name='SectionWriter', model=config.model, instruction=section_prompt, output_schema=SectionDraft)


def frame_agent() -> LlmAgent:
    """Create the agent that writes the title, introduction and conclusion (input: drafted sections)."""
    return llm_agent(name='ReportEditor', model=config.model, instruction=frame_prompt, output_schema=ReportFrame)


async def write_report(
    ctx: Context, user_message: str, question: str, outline: list[Section], registry: list[RegistryEntry],
) -> str:
    """
    Write a report section by section: the sections in parallel, each from its own evidence, then the title,
    introduction and conclusion from the drafted sections.

    Args:
        ctx: Context of the calling node
        user_message: The latest user message, which sets the language
        question: The research question
        outline: The report outline
        registry: Sources with their findings, numbered from 1 (the citation numbers)

    Returns:
        The report in Markdown, without the references
    """
    section_ids = {section.id for section in outline}
    drafts = await asyncio.gather(
        *(
            ctx.run_node(
                section_agent(), node_input=section_input(user_message, question, section, registry, section_ids),
                use_sub_branch=True,
            )
            for section in outline
        ),
        return_exceptions=True,
    )
    sections = []
    for section, draft in zip(outline, drafts):
        if isinstance(draft, BaseException):
            logger.warning(
                f'Writing section {section.title!r} failed: {draft}',
            )
            continue
        sections.append(
            (section, SectionDraft.model_validate(draft).body.strip()),
        )

    drafted = '\n\n'.join(
        f'## {section.title}\n\n{body}' for section, body in sections
    )
    frame = ReportFrame.model_validate(
        await ctx.run_node(
            frame_agent(),
            node_input=f'User message: {user_message}\nResearch question: {question}\n\n# Drafted sections\n\n{drafted}',
        ),
    )
    return '\n\n'.join([
        f'# {frame.title.strip()}',
        frame.introduction.strip(),
        drafted,
        f'## Conclusion\n\n{frame.conclusion.strip()}',
    ])
