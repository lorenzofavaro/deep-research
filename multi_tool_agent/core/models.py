from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field

LENSES = {
    'scout': 'map the landscape quickly: main sub-topics, key terms, landmark papers and authoritative sources',
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
    lens: Literal['evidence', 'comparator', 'critic', 'horizon', 'scout'] = Field(
        default='evidence',
        description='Angle of the research: '
        + '; '.join(
            f'{name}: {text}' for name,
            text in LENSES.items() if name != 'scout'
        ),
    )
    boundaries: str = Field(
        default='',
        description='What NOT to cover because other researchers handle it; empty if nothing to exclude',
    )
    section_ids: list[str] = Field(
        default_factory=list, description='Ids of the outline sections this task gathers evidence for',
    )


class Section(BaseModel):
    """A section of the report outline."""
    id: str = Field(description='Short identifier, e.g. "s1"')
    title: str
    questions: list[str] = Field(description='What the section must answer')


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
    # Id of the outline section the finding is evidence for, if any
    section: str = ''


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
