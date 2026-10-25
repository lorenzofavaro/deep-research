import asyncio
from typing import Any

from multi_tool_agent.core.tools.arxiv import get_arxiv_papers
from multi_tool_agent.core.tools.arxiv import relax_query
from multi_tool_agent.core.tools.arxiv import search_arxiv
from multi_tool_agent.core.tools.openalex import get_citation_counts
from multi_tool_agent.core.tools.openalex import search_cited_arxiv_papers
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


async def search_papers(query: str) -> list[dict[str, Any]]:
    """
    Search arXiv papers: text-relevance results from arXiv plus the most cited matching papers from OpenAlex,
    with citation counts. A query that matches nothing on arXiv is retried in a relaxed form.

    Args:
        query: Query in arXiv API syntax

    Returns:
        Candidate papers with id, title, abstract, authors, published date, url and citations (when known)
    """
    candidates, cited = await asyncio.gather(
        search_arxiv(query, max_results=config.arxiv_candidates),
        search_cited_arxiv_papers(query, max_results=config.cited_candidates),
    )
    if not candidates and (relaxed := relax_query(query)) and relaxed != query:
        logger.info(
            f'No ArXiv results for "{query}", retrying with "{relaxed}"',
        )
        candidates = await search_arxiv(relaxed, max_results=config.arxiv_candidates)
    return await merge_with_citations(candidates, cited)


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
