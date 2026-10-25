from multi_tool_agent.core.models import Finding
from multi_tool_agent.core.models import Source


class ResearchStore:
    """
    What one researcher has found: the sources it saw, their text once read, and the findings it recorded.

    Raw text stays here and never enters the orchestrator's context: only findings and the brief are returned.
    """

    def __init__(self, budget: int) -> None:
        """
        Initialize the store.

        Args:
            budget: Maximum number of search and read calls
        """
        self.budget = budget
        self.calls = 0
        self.model_calls = 0
        self.sources: dict[str, Source] = {}
        # Short description shown before a source is read (abstract or snippet)
        self.previews: dict[str, str] = {}
        self.texts: dict[str, str] = {}
        self.findings: dict[str, list[Finding]] = {}

    @property
    def exhausted(self) -> bool:
        return self.calls >= self.budget

    def use_call(self) -> bool:
        """Count a search or read call; returns False when the budget is exhausted."""
        if self.exhausted:
            return False
        self.calls += 1
        return True

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
