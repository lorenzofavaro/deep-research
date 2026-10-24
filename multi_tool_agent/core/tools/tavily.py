from typing import Any

from tavily import AsyncTavilyClient

from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


async def tavily_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search the web with Tavily, including the extracted text of each page.

    Args:
        query: The search query string
        max_results: Maximum number of results to return

    Returns:
        List of results with title, url, published date (when known) and content
    """
    logger.debug(
        f'Searching Tavily with query: "{query}", max_results: {max_results}',
    )
    client = AsyncTavilyClient(config.tavily_api_key)
    response = await client.search(
        query=query,
        max_results=max_results,
        include_raw_content='text',
        timeout=60,
    )

    results = []
    for result in response.get('results', []):
        results.append({
            'title': result.get('title') or result['url'],
            'url': result['url'],
            'published': (result.get('published_date') or '')[:10],
            # Fall back to the snippet when the page text could not be extracted
            'content': result.get('raw_content') or result.get('content') or '',
        })

    logger.info(f'Tavily search returned {len(results)} results')
    return results
