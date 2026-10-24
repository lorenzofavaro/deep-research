import re

from google.adk.agents import LlmAgent

from multi_tool_agent.core.llm import isolated_llm_agent
from multi_tool_agent.core.models import Finding
from multi_tool_agent.core.models import Source
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.utils import today

# A source with the findings collected from it; its citation number is its position in the registry + 1
RegistryEntry = tuple[Source, list[Finding]]

system_prompt = """Synthesize the research findings to provide a direct, comprehensive answer to the user's query.

Instructions:
- Use ONLY the findings provided below; do not add outside knowledge
- Use all the relevant findings to answer the query completely, with specific evidence (numbers, names, results)
- Cite the source of EVERY claim with its number in square brackets, e.g. [2] or [1, 3], right after the claim
- Never cite a number that is not in the source list
- Point out contradictions between sources, and say what the findings do not cover
- Structure your answer clearly (short paragraphs, headings or bullet points when helpful), without unnecessary elaboration
- Do NOT write a references section: it is appended automatically
- Write in the same language as the user's message
- If there are no findings, say that the research found no relevant information"""


def build_registry(results: list[StepResult]) -> list[RegistryEntry]:
    """
    Merge the step results into a list of unique sources with their findings.

    Args:
        results: Results of all the research steps

    Returns:
        Sources (deduplicated by key) paired with their findings
    """
    registry: dict[str, RegistryEntry] = {}
    for result in results:
        sources = {source.key: source for source in result.sources}
        for key, findings in result.findings.items():
            _, collected = registry.setdefault(key, (sources[key], []))
            claims = {finding.claim for finding in collected}
            collected.extend(
                finding for finding in findings if finding.claim not in claims
            )
    return list(registry.values())


def _describe(source: Source) -> str:
    authors = ''
    if source.authors:
        authors = source.authors[0] + \
            (' et al.' if len(source.authors) > 1 else '') + ', '
    published = f', {source.published}' if source.published else ''
    return f'{authors}"{source.title}"{published} — {source.url}'


def answer_agent(*, user_message: str, question: str, registry: list[RegistryEntry]) -> LlmAgent:
    """
    Create the agent that writes the final answer.

    Args:
        user_message: The latest user message, which sets the answer language
        question: The research question
        registry: Sources with their findings, numbered from 1

    Returns:
        The configured LlmAgent
    """
    lines = [
        f'Current date: {today()}',
        f'User message: {user_message}',
        f'Research question: {question}',
        '',
        '# Sources and findings',
    ]
    for number, (source, findings) in enumerate(registry, start=1):
        lines.append(f'[{number}] {_describe(source)}')
        for finding in findings:
            lines.append(f'  - {finding.claim}')
            lines.append(f'    Evidence: "{finding.evidence}"')
    if not registry:
        lines.append('(no findings)')
    user_input = '\n'.join(lines)

    return isolated_llm_agent(
        name='AnswerAgent',
        model=config.model,
        instruction=system_prompt,
        build_input=lambda _: user_input,
        output_key='answer',
    )


def format_references(answer: str, registry: list[RegistryEntry]) -> str:
    """
    Build the references section for the sources the answer actually cites.

    Args:
        answer: The answer text, with [n] citations
        registry: Sources with their findings, numbered from 1

    Returns:
        The references section, or an empty string if nothing valid is cited
    """
    cited = {
        int(number)
        for group in re.findall(r'\[(\d+(?:\s*,\s*\d+)*)\]', answer)
        for number in group.split(',')
    }
    lines = [
        f'[{number}] {_describe(registry[number - 1][0])}'
        for number in sorted(cited)
        if 1 <= number <= len(registry)
    ]
    return '**References**\n\n' + '\n'.join(lines) if lines else ''
