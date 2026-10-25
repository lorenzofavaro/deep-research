from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

LENSES = {
    'evidence': 'gather the core facts, data and results that answer the objective',
    'comparator': 'compare the alternatives (methods, products, positions) on concrete criteria and numbers',
    'critic': 'look for limitations, failures, counter-evidence and disagreements between sources',
    'horizon': 'find the most recent developments, open problems and where the field is heading',
}


class ResearchTask(BaseModel):
    """A research task delegated to one researcher."""
    objective: str = Field(
        description='What the researcher must find out, as a specific question or goal',
    )
    sources: list[Literal['papers', 'web']] = Field(
        description='Where to look: "papers" (arXiv) for academic literature, "web" for current information',
    )
    lens: Literal['evidence', 'comparator', 'critic', 'horizon'] = Field(
        default='evidence',
        description='Angle of the research: ' +
        '; '.join(f'{name}: {text}' for name, text in LENSES.items()),
    )
    boundaries: str = Field(
        default='',
        description='What NOT to cover because other researchers handle it; empty if nothing to exclude',
    )


class Source(BaseModel):
    """A document the research collected findings from."""
    key: str = Field(
        description='Stable identifier: "arxiv:<id>" or the page URL',
    )
    kind: Literal['arxiv', 'web']
    title: str
    url: str
    authors: list[str] = Field(default_factory=list)
    published: str = ''


class Finding(BaseModel):
    """A single piece of knowledge extracted from a source."""
    claim: str = Field(
        description='Self-contained statement relevant to the research goal',
    )
    evidence: str = Field(
        description='Short verbatim quote from the source that supports the claim',
    )


class StepResult(BaseModel):
    """Outcome of a research task."""
    step: ResearchTask
    sources: list[Source] = Field(default_factory=list)
    # Findings grouped by Source.key
    findings: dict[str, list[Finding]] = Field(default_factory=dict)
    # The researcher's closing summary: what was found, what is missing
    brief: str = ''
    # Short progress notes, e.g. number of tool calls
    notes: list[str] = Field(default_factory=list)
    error: Optional[str] = None
