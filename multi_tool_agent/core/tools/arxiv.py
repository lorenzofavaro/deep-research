import asyncio
import re
import time
from io import BytesIO
from typing import Any
from typing import Optional

import feedparser
import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)

API_URL = 'https://export.arxiv.org/api/query'
# arXiv asks API clients to make at most one request every three seconds
API_INTERVAL = 3.0
MAX_ATTEMPTS = 3
HEADERS = {
    'User-Agent': 'deep-research (https://github.com/lorenzofavaro/deep-research)',
}

_api_lock = asyncio.Lock()
_last_api_call = 0.0
_download_slots = asyncio.Semaphore(3)


async def _get(url: str, params: Optional[dict[str, Any]] = None) -> httpx.Response:
    """
    GET a URL, retrying with exponential backoff on rate limits, server errors and network failures.

    Args:
        url: URL to fetch
        params: Optional query parameters

    Returns:
        The successful response (2xx or 404, which callers handle)
    """
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True, headers=HEADERS) as client:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = await client.get(url, params=params)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f'HTTP {resp.status_code}', request=resp.request, response=resp,
                    )
                if resp.status_code != 404:
                    resp.raise_for_status()
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == MAX_ATTEMPTS:
                    raise
                delay = 2 ** attempt
                logger.warning(f'GET {url} failed ({e}), retrying in {delay}s')
                await asyncio.sleep(delay)
    raise AssertionError('unreachable')


def _strip_version(arxiv_id: str) -> str:
    return re.sub(r'v\d+$', '', arxiv_id)


async def search_arxiv(query: str, max_results: int = 30) -> list[dict[str, Any]]:
    """
    Search for papers on arXiv using the arXiv API.

    Args:
        query: Query in arXiv API syntax, e.g. `abs:"state space model" AND cat:cs.LG`
        max_results: Maximum number of results to return

    Returns:
        List of dictionaries with id, title, abstract, authors, published date and url
    """
    global _last_api_call
    logger.debug(
        f'Searching ArXiv with query: "{query}", max_results: {max_results}',
    )

    async with _api_lock:
        wait = _last_api_call + API_INTERVAL - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            resp = await _get(API_URL, params={'search_query': query, 'start': 0, 'max_results': max_results})
        finally:
            _last_api_call = time.monotonic()

    feed = feedparser.parse(resp.text)
    papers = []
    for entry in feed.entries:
        arxiv_id = _strip_version(str(entry.id).split('/abs/')[-1])
        papers.append({
            'id': arxiv_id,
            'title': ' '.join(str(entry.title).split()),
            'abstract': ' '.join(str(entry.summary).split()),
            'authors': [author.name for author in entry.get('authors', [])],
            'published': str(entry.get('published', ''))[:10],
            'url': f'https://arxiv.org/abs/{arxiv_id}',
        })

    logger.info(f'Found {len(papers)} papers from ArXiv search')
    return papers


def relax_query(query: str) -> str:
    """
    Turn a query into a looser one, used when the original returns nothing.

    Categories, field prefixes, operators and quotes are dropped and the first content words are AND-ed across
    all fields.

    Args:
        query: Original arXiv query

    Returns:
        Relaxed arXiv query, or an empty string if no words are left
    """
    text = re.sub(r'\bcat:[\w.\-]+', ' ', query)
    text = re.sub(r'\b(ti|au|abs|co|jr|rn|all):', ' ', text)
    words = [
        w for w in re.findall(
            r'[\w\-]+', text,
        ) if w not in {'AND', 'OR', 'ANDNOT'} and len(w) > 2
    ]
    return ' AND '.join(f'all:{w}' for w in words[:4])


def _html_to_text(html: str) -> Optional[str]:
    soup = BeautifulSoup(html, 'html.parser')
    article = soup.find('article')
    if article is None:
        return None
    for tag in article.select('.ltx_bibliography, .ltx_page_footer, nav, script, style'):
        tag.decompose()
    # Keep formulas readable by replacing MathML with its LaTeX source
    for math in article.find_all('math'):
        math.replace_with(f' ${math.get("alttext", "")}$ ')
    lines = (
        ' '.join(line.split())
        for line in article.get_text('\n').splitlines()
    )
    return '\n'.join(line for line in lines if line)


def _pdf_to_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    return '\n'.join(page.extract_text() or '' for page in reader.pages)


async def get_arxiv_paper_text(arxiv_id: str, max_chars: int = 100_000) -> str:
    """
    Fetch the full text of an arXiv paper.

    The HTML rendering (arxiv.org/html) is preferred because it keeps the section structure and the LaTeX of
    formulas; the PDF is the fallback for papers without one.

    Args:
        arxiv_id: The arXiv paper ID (e.g., "2301.07041")
        max_chars: Maximum number of characters to return

    Returns:
        The paper text, truncated to max_chars
    """
    async with _download_slots:
        resp = await _get(f'https://arxiv.org/html/{arxiv_id}')
        text = await asyncio.to_thread(_html_to_text, resp.text) if resp.status_code == 200 else None
        if not text:
            logger.debug(f'No HTML version of {arxiv_id}, falling back to PDF')
            resp = await _get(f'https://arxiv.org/pdf/{arxiv_id}')
            if resp.status_code == 404:
                raise ValueError(f'ArXiv paper {arxiv_id} not found')
            text = await asyncio.to_thread(_pdf_to_text, resp.content)

    logger.info(
        f'Extracted {len(text)} characters from ArXiv paper {arxiv_id}',
    )
    return text[:max_chars]
