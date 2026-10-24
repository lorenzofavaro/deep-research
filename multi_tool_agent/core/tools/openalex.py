import re
from typing import Any
from typing import Optional

import httpx

from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)

API_URL = 'https://api.openalex.org/works'
# OpenAlex source ID of arXiv
ARXIV_SOURCE = 'S4306400194'
# Values OR-ed in a single filter (OpenAlex accepts up to 100)
BATCH_SIZE = 50
OPERATORS = r'(?:AND|OR|NOT)'


def _collapse_operators(match: re.Match) -> str:
    operators = match.group(0).split()
    return 'NOT' if 'NOT' in operators else operators[0]


def to_boolean_search(query: str) -> str:
    """
    Convert an arXiv API query into an OpenAlex boolean search over title and abstract.

    Field prefixes are dropped, while phrases, AND/OR/NOT and parentheses are kept. Categories and authors have no
    title/abstract equivalent, so they are removed along with the operators they leave dangling.

    Args:
        query: Query in arXiv API syntax

    Returns:
        The OpenAlex search string, or an empty string if nothing searchable is left
    """
    text = re.sub(r'\b(?:cat|au):(?:"[^"]*"|[^\s()]+)', ' ', query)
    text = re.sub(r'\b(?:ti|abs|co|jr|rn|all):', '', text)
    # Commas separate filters in the OpenAlex URL syntax
    text = text.replace('ANDNOT', 'NOT').replace(',', ' ')
    previous = None
    while previous != text:
        previous = text
        text = re.sub(rf'\(\s*(?:{OPERATORS}\s*)*\)', ' ', text)
        # Keep the exclusion in sequences like `AND NOT`, otherwise the first operator
        text = re.sub(
            rf'\b{OPERATORS}(\s+{OPERATORS}\b)+',
            _collapse_operators, text,
        )
        text = re.sub(rf'\(\s*(?:AND|OR)\s+', '(', text)
        text = re.sub(rf'\s+{OPERATORS}\s*\)', ')', text)
        text = re.sub(
            rf'^\s*{OPERATORS}\b|\b{OPERATORS}\s*$', '', text.strip(),
        )
    return ' '.join(text.split())


def _abstract(inverted_index: Optional[dict[str, list[int]]]) -> str:
    """Rebuild an abstract from the inverted index OpenAlex stores it as."""
    if not inverted_index:
        return ''
    positions = {
        position: word for word, indexes in inverted_index.items()
        for position in indexes
    }
    return ' '.join(positions[i] for i in sorted(positions))


def _arxiv_id(work: dict[str, Any]) -> Optional[str]:
    """Find the arXiv ID of a work from its DOI or its locations."""
    urls = [work.get('doi') or ''] + [
        location.get('landing_page_url')
        or '' for location in work.get('locations', [])
    ]
    for url in urls:
        match = re.search(
            r'(?:arxiv\.org/abs/|10\.48550/arxiv\.)([\w.\-/]+?)(?:v\d+)?$', url, re.IGNORECASE,
        )
        if match:
            return match.group(1)
    return None


async def _get_works(params: dict[str, Any]) -> list[dict[str, Any]]:
    if config.openalex_api_key:
        params['api_key'] = config.openalex_api_key
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(API_URL, params=params)
        resp.raise_for_status()
        return resp.json().get('results', [])


async def search_cited_arxiv_papers(query: str, max_results: int = 15) -> list[dict[str, Any]]:
    """
    Find the most cited arXiv papers matching a query.

    It complements the arXiv search, which ranks by text relevance and often misses the influential papers of a
    topic. Failures are logged and produce no results, since this source is optional.

    Args:
        query: Query in arXiv API syntax
        max_results: Maximum number of papers to return

    Returns:
        Papers in the same format as `search_arxiv`, with a `citations` count
    """
    search = to_boolean_search(query)
    if not search:
        return []
    try:
        works = await _get_works({
            'filter': f'title_and_abstract.search:{search},locations.source.id:{ARXIV_SOURCE}',
            'sort': 'cited_by_count:desc',
            'per-page': max_results,
            'select': 'doi,title,publication_date,cited_by_count,authorships,abstract_inverted_index,locations',
        })
    except Exception as e:
        logger.warning(f'OpenAlex search failed for "{search}": {e}')
        return []

    papers = []
    for work in works:
        arxiv_id = _arxiv_id(work)
        if arxiv_id is None or not work.get('title'):
            continue
        papers.append({
            'id': arxiv_id,
            'title': work['title'],
            'abstract': _abstract(work.get('abstract_inverted_index')),
            'authors': [a['author']['display_name'] for a in work.get('authorships', []) if a.get('author')],
            'published': work.get('publication_date') or '',
            'url': f'https://arxiv.org/abs/{arxiv_id}',
            'citations': work.get('cited_by_count', 0),
        })

    logger.info(
        f'OpenAlex returned {len(papers)} cited arXiv papers for "{search}"',
    )
    return papers


async def get_citation_counts(arxiv_ids: list[str]) -> dict[str, int]:
    """
    Look up how many times arXiv papers are cited, according to OpenAlex.

    The counts cover the arXiv record of each paper, so they underestimate papers also published in a venue, but
    they are comparable across candidates. Failures are logged and produce no counts.

    Args:
        arxiv_ids: arXiv IDs without version

    Returns:
        Citation count by arXiv ID, for the papers OpenAlex knows
    """
    counts: dict[str, int] = {}
    for start in range(0, len(arxiv_ids), BATCH_SIZE):
        batch = arxiv_ids[start:start + BATCH_SIZE]
        dois = '|'.join(f'10.48550/arxiv.{arxiv_id}' for arxiv_id in batch)
        try:
            works = await _get_works({
                'filter': f'doi:{dois}',
                'per-page': BATCH_SIZE,
                'select': 'doi,cited_by_count,locations',
            })
        except Exception as e:
            logger.warning(f'OpenAlex citation lookup failed: {e}')
            continue
        for work in works:
            arxiv_id = _arxiv_id(work)
            if arxiv_id:
                counts[arxiv_id] = work.get('cited_by_count', 0)
    return counts
