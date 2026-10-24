import asyncio
import re

from google.adk.agents.context import Context

from multi_tool_agent.core.agents.research.arxiv.filter import filter_agent
from multi_tool_agent.core.agents.research.arxiv.filter import filter_input
from multi_tool_agent.core.agents.research.arxiv.filter import PaperIDs
from multi_tool_agent.core.agents.research.base import extract_findings
from multi_tool_agent.core.agents.research.base import StepTask
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


async def research_arxiv(ctx: Context, task: StepTask, result: StepResult) -> None:
    """
    Research step on arXiv: search, select the most relevant papers, read them in full and extract findings.

    Args:
        ctx: Context of the step node
        task: The research step
        result: Step result to fill
    """
    query = task.step.query
    candidates, cited = await asyncio.gather(
        search_arxiv(query, max_results=config.arxiv_candidates),
        search_cited_arxiv_papers(query, max_results=config.cited_candidates),
    )
    if not candidates and (relaxed := relax_query(query)) and relaxed != query:
        logger.info(
            f'No ArXiv results for "{query}", retrying with "{relaxed}"',
        )
        query = relaxed
        candidates = await search_arxiv(query, max_results=config.arxiv_candidates)
        result.notes.append(f'relaxed query `{query}`')
    candidates = await merge_with_citations(candidates, cited)
    result.notes.append(
        f'{len(candidates)} candidates ({len(cited)} most cited from OpenAlex)',
    )
    if not candidates:
        return

    output = await ctx.run_node(filter_agent(), node_input=filter_input(task.step, task.question, candidates))
    # The model may add version suffixes or invent IDs: keep only known candidates
    papers = {paper['id']: paper for paper in candidates}
    selected = []
    for paper_id in PaperIDs.model_validate(output or {}).ids:
        paper = papers.get(re.sub(r'v\d+$', '', paper_id.strip()))
        if paper and paper not in selected:
            selected.append(paper)
    selected = selected[:config.papers_per_step]
    result.notes.append(f'{len(selected)} papers read')
    if not selected:
        return

    texts = await asyncio.gather(
        *(
            get_arxiv_paper_text(paper['id'], max_chars=config.paper_max_chars)
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
    result.sources, result.findings = await extract_findings(ctx, task, documents)


async def merge_with_citations(candidates: list[dict], cited: list[dict]) -> list[dict]:
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
