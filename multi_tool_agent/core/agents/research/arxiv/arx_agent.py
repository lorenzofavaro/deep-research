import asyncio
import re
from collections.abc import AsyncGenerator

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from multi_tool_agent.core.agents.research.arxiv.filter import filter_agent
from multi_tool_agent.core.agents.research.arxiv.filter import PaperIDs
from multi_tool_agent.core.agents.research.base import message
from multi_tool_agent.core.agents.research.base import ResearchStepAgent
from multi_tool_agent.core.models import Source
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.core.tools.arxiv import get_arxiv_paper_text
from multi_tool_agent.core.tools.arxiv import get_arxiv_papers
from multi_tool_agent.core.tools.arxiv import relax_query
from multi_tool_agent.core.tools.arxiv import search_arxiv
from multi_tool_agent.core.tools.openalex import get_citation_counts
from multi_tool_agent.core.tools.openalex import search_cited_arxiv_papers
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


class ArxivAgent(ResearchStepAgent):
    """Research step on arXiv: search, select the most relevant papers, read them in full and extract findings."""

    async def _research(self, context: InvocationContext, result: StepResult) -> AsyncGenerator[Event, None]:
        """
        Execute the ArXiv research workflow.

        Args:
            context: Invocation context
            result: Step result to fill

        Yields:
            Events from each stage of the workflow
        """
        query = self._step.query
        candidates, cited = await asyncio.gather(
            search_arxiv(query, max_results=config.arxiv_candidates),
            search_cited_arxiv_papers(
                query, max_results=config.cited_candidates,
            ),
        )
        if not candidates and (relaxed := relax_query(query)) and relaxed != query:
            logger.info(
                f'No ArXiv results for "{query}", retrying with "{relaxed}"',
            )
            query = relaxed
            candidates = await search_arxiv(query, max_results=config.arxiv_candidates)
        candidates = await self._merge_with_citations(candidates, cited)
        yield message(
            self.name, f'arXiv `{query}`: {len(candidates)} candidates, {len(cited)} of the most cited from OpenAlex',
        )
        if not candidates:
            return

        output_key = f'select_{self._run_id}_{self._step_id}'
        selector = filter_agent(
            name=f'select_{self._step_id}', output_key=output_key,
            step=self._step, question=self._question, candidates=candidates,
        )
        async for event in selector.run_async(context):
            yield event

        # The model may add version suffixes or invent IDs: keep only known candidates
        papers = {paper['id']: paper for paper in candidates}
        selected_ids = PaperIDs.model_validate(
            context.session.state.get(output_key) or {},
        ).ids
        selected = []
        for paper_id in selected_ids:
            paper = papers.get(re.sub(r'v\d+$', '', paper_id.strip()))
            if paper and paper not in selected:
                selected.append(paper)
        selected = selected[:config.papers_per_step]
        if not selected:
            return

        texts = await asyncio.gather(
            *(
                get_arxiv_paper_text(
                    paper['id'], max_chars=config.paper_max_chars,
                )
                for paper in selected
            ),
            return_exceptions=True,
        )
        documents = []
        for paper, text in zip(selected, texts):
            if isinstance(text, BaseException):
                logger.warning(f'Skipping ArXiv paper {paper["id"]}: {text}')
                continue
            source = Source(
                key=f'arxiv:{paper["id"]}', kind='arxiv', title=paper['title'], url=paper['url'],
                authors=paper['authors'], published=paper['published'],
            )
            documents.append((source, text))

        async for event in self._extract(context, result, documents):
            yield event

    async def _merge_with_citations(self, candidates: list[dict], cited: list[dict]) -> list[dict]:
        """
        Merge the arXiv results with the most cited papers from OpenAlex, and add citation counts to all of them.

        Metadata always comes from arXiv, which is authoritative and carries the preprint date.

        Args:
            candidates: Papers from the arXiv search
            cited: Most cited papers from OpenAlex, with citation counts

        Returns:
            The merged candidates, each with a `citations` count when OpenAlex knows the paper
        """
        merged = {paper['id']: paper for paper in candidates}
        # OpenAlex can attach the title of another work to an arXiv ID: take the metadata of the papers only
        # OpenAlex found from arXiv itself, and drop those arXiv does not confirm
        cited_only = [
            paper['id']
            for paper in cited if paper['id'] not in merged
        ]
        try:
            confirmed = {paper['id']: paper for paper in await get_arxiv_papers(cited_only)}
        except Exception as e:
            logger.warning(
                f'Could not confirm OpenAlex papers on arXiv, skipping them: {e}',
            )
            confirmed = {}
        for paper in cited:
            if paper['id'] in merged:
                merged[paper['id']]['citations'] = paper['citations']
            elif paper['id'] in confirmed:
                merged[paper['id']] = {
                    **confirmed[paper['id']], 'citations': paper['citations'],
                }
        missing = [
            paper_id for paper_id,
            paper in merged.items() if 'citations' not in paper
        ]
        counts = await get_citation_counts(missing) if missing else {}
        for paper_id, count in counts.items():
            if paper_id in merged:
                merged[paper_id]['citations'] = count
        return list(merged.values())
