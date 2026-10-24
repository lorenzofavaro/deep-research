from typing import Literal
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class ResearchStep(BaseModel):
    """Individual step in a research plan."""
    tool: Literal['arxiv', 'web'] = Field(
        description='Source to search: "arxiv" for academic papers, "web" for current information',
    )
    query: str = Field(
        description='Search query in the syntax of the chosen tool (arXiv API syntax or web search keywords)',
    )
    goal: str = Field(
        description='What this step must find out to help answer the research question',
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


class Findings(BaseModel):
    """Findings extracted from one source."""
    findings: list[Finding] = Field(
        default_factory=list,
        description='Relevant findings; empty if the source does not help with the goal',
    )


class StepResult(BaseModel):
    """Outcome of a research step, stored in the session state."""
    step: ResearchStep
    sources: list[Source] = Field(default_factory=list)
    # Findings grouped by Source.key
    findings: dict[str, list[Finding]] = Field(default_factory=dict)
    error: Optional[str] = None
