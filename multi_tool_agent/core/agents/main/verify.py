import asyncio
import re
from typing import Literal

from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.agents.main.answer import RegistryEntry
from multi_tool_agent.core.llm import llm_agent
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)

CITATION = re.compile(r'\[(\d+(?:\s*,\s*\d+)*)\]')
# Claims checked per verifier call
BATCH_SIZE = 20


class Verdict(BaseModel):
    """The verification of one claim."""
    id: int = Field(description='Number of the claim')
    verdict: Literal['supported', 'partial', 'unsupported']
    fix: str = Field(
        default='',
        description='For partial or unsupported claims: the sentence rewritten to state only what the evidence '
                    'supports, keeping its citations; empty to remove the sentence',
    )


class Verdicts(BaseModel):
    """Verifications of a batch of claims."""
    verdicts: list[Verdict]


class Verification(BaseModel):
    """Outcome of the verification of an answer."""
    checked: int = 0
    supported: int = 0
    partial: int = 0
    unsupported: int = 0
    rewritten: int = 0
    removed: int = 0
    invalid_citations: int = 0
    error: str = ''

    @property
    def supported_rate(self) -> float:
        return self.supported / self.checked if self.checked else 1.0


system_prompt = """You are a Citation Verifier. For each claim of a research answer, you check whether the evidence \
of the sources it cites supports it.

For each claim:
- **supported**: every fact in the claim (names, numbers, dates, comparisons, causal links) is stated by the \
evidence of the cited sources
- **partial**: the core is supported, but some detail is not (a different number, an overgeneralization, an \
unsupported comparison or qualifier)
- **unsupported**: the cited evidence does not support the claim

For partial and unsupported claims, write `fix`: the claim rewritten to state only what the evidence supports, \
in the same language and style, keeping the citation markers ([n]) that support it. Leave `fix` empty when nothing \
in the evidence supports it, so the claim is removed. Judge only against the evidence given, not your own knowledge."""


def strip_invalid_citations(text: str, sources: int) -> tuple[str, int]:
    """
    Remove citation numbers that point to no source.

    Args:
        text: The answer
        sources: Number of sources in the registry

    Returns:
        The text without invalid numbers, and how many were removed
    """
    removed = 0

    def clean(match: re.Match) -> str:
        nonlocal removed
        numbers = [number.strip() for number in match.group(2).split(',')]
        valid = [number for number in numbers if 1 <= int(number) <= sources]
        removed += len(numbers) - len(valid)
        # A marker left empty goes away with the space before it
        return f'{match.group(1)}[{", ".join(valid)}]' if valid else ''

    return re.sub(r'(\s?)' + CITATION.pattern, clean, text), removed


def split_claims(text: str) -> list[str]:
    """
    Split an answer into the units that carry citations: sentences of paragraphs, list items and table rows.

    Args:
        text: The answer in Markdown

    Returns:
        The cited units, verbatim from the text, in order and without duplicates
    """
    claims: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        is_unit = stripped.startswith(
            ('|', '-', '*'),
        ) or re.match(r'\d+\.\s', stripped)
        units = [stripped] if is_unit else re.split(
            r'(?<=[.!?])\s+(?=[A-Z0-9*"(\[])', stripped,
        )
        for unit in units:
            if CITATION.search(unit) and unit not in claims:
                claims.append(unit)
    return claims


def _evidence(claim: str, registry: list[RegistryEntry]) -> list[str]:
    lines = []
    numbers = sorted({
        int(n) for group in CITATION.findall(claim)
        for n in group.split(',')
    })
    for number in numbers:
        source, findings = registry[number - 1]
        lines.append(f'  [{number}] {source.title}')
        lines += [
            f'    - {finding.claim} (evidence: "{finding.evidence}")' for finding in findings
        ]
    return lines


def verify_input(claims: list[tuple[int, str]], registry: list[RegistryEntry]) -> str:
    """Build the verifier input: each numbered claim with the findings of the sources it cites."""
    lines = []
    for number, claim in claims:
        lines += [
            f'## Claim {number}', claim,
            'Cited evidence:', *_evidence(claim, registry), '',
        ]
    return '\n'.join(lines)


def verify_agent() -> LlmAgent:
    """Create the agent that checks claims against their cited evidence (input: `verify_input`)."""
    return llm_agent(name='VerifyAgent', model=config.fast_model, instruction=system_prompt, output_schema=Verdicts)


def _replace(text: str, claim: str, fix: str) -> str:
    if fix:
        return text.replace(claim, fix.strip(), 1)
    # Remove the claim: a whole line for list items and table rows, else the sentence and its trailing space
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == claim and claim.startswith(('|', '-', '*')):
            return '\n'.join(lines[:i] + lines[i + 1:])
    return re.sub(re.escape(claim) + r'[ \t]?', '', text, count=1)


async def verify_answer(ctx: Context, answer: str, registry: list[RegistryEntry]) -> tuple[str, Verification]:
    """
    Check every cited claim of an answer against the evidence of its sources, and correct the answer: partial
    claims are rewritten, unsupported ones rewritten or removed, citations to unknown sources dropped.

    A failing verification leaves the answer as it is (apart from the invalid citations).

    Args:
        ctx: Context of the calling node
        answer: The answer or report
        registry: Sources with their findings, numbered from 1

    Returns:
        The corrected answer, and the verification statistics
    """
    answer, invalid = strip_invalid_citations(answer, len(registry))
    report = Verification(invalid_citations=invalid)
    claims = list(enumerate(split_claims(answer), start=1))
    if not claims:
        return answer, report

    batches = [
        claims[i:i + BATCH_SIZE]
        for i in range(0, len(claims), BATCH_SIZE)
    ]
    outputs = await asyncio.gather(
        *(
            ctx.run_node(
                verify_agent(), node_input=verify_input(
                    batch,
                    registry,
                ), use_sub_branch=True,
            ) for batch in batches
        ),
        return_exceptions=True,
    )
    by_id = dict(claims)
    for batch, output in zip(batches, outputs):
        if isinstance(output, BaseException):
            logger.warning(
                f'Verification of {len(batch)} claims failed: {output}',
            )
            report.error = str(output)
            continue
        for verdict in Verdicts.model_validate(output).verdicts:
            claim = by_id.get(verdict.id)
            if claim is None:
                continue
            report.checked += 1
            setattr(
                report, verdict.verdict, getattr(
                    report, verdict.verdict,
                ) + 1,
            )
            if verdict.verdict == 'supported':
                continue
            answer = _replace(answer, claim, verdict.fix)
            if verdict.fix:
                report.rewritten += 1
            else:
                report.removed += 1
    return answer, report
