from typing import Literal
from typing import Optional

from multi_tool_agent.core.models import Finding
from multi_tool_agent.core.models import Source

# Model calls allowed after the budget is exhausted, to record the last findings and write the brief
WRAP_UP_MODEL_CALLS = 3


class ResearchStore:
    """
    What one researcher has found: the sources it saw, their text once read, and the findings it recorded.

    Raw text stays here and never enters the orchestrator's context: only findings and the brief are returned.
    """

    def __init__(self, budget: int) -> None:
        """
        Initialize the store.

        Args:
            budget: Maximum number of search and read calls; at most half of them can be searches, so that reading
                is never starved
        """
        self.budget = budget
        self.search_limit = max(1, budget // 2)
        self.calls = 0
        self.searches = 0
        self.model_calls = 0
        # Model call at which the budget ran out
        self.exhausted_at: Optional[int] = None
        self.sources: dict[str, Source] = {}
        # Short description shown before a source is read (abstract or snippet)
        self.previews: dict[str, str] = {}
        self.texts: dict[str, str] = {}
        self.findings: dict[str, list[Finding]] = {}

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.budget

    def use_call(self, kind: Literal['search', 'read']) -> Optional[str]:
        """Count a search or read call; returns why it is refused, or None when it is allowed."""
        if self.exhausted:
            return 'Search budget exhausted: record your remaining findings, then write your final brief.'
        if kind == 'search' and self.searches >= self.search_limit:
            return (
                'Search limit reached: read the most promising sources you found with read_source, and record '
                'findings from them.'
            )
        self.calls += 1
        self.searches += kind == 'search'
        return None

    def add_source(self, source: Source, preview: str = '', text: str = '') -> None:
        self.sources.setdefault(source.key, source)
        if preview:
            self.previews.setdefault(source.key, preview)
        if text:
            self.texts.setdefault(source.key, text)

    def add_finding(self, source_key: str, finding: Finding) -> bool:
        """Record a finding; returns False if the same claim was already recorded for the source."""
        findings = self.findings.setdefault(source_key, [])
        if any(existing.claim == finding.claim for existing in findings):
            return False
        findings.append(finding)
        return True

    def cited_sources(self) -> list[Source]:
        """Sources with at least one finding."""
        return [self.sources[key] for key in self.findings if key in self.sources]
