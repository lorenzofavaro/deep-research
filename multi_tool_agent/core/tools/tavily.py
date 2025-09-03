import os
from typing import Any
from typing import Optional

from tavily import TavilyClient

from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


def tavily_search(
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """
    Search for information using the Tavily API.

    Args:
        query: The search query string
        start_date: Optional start date for filtering results (YYYY-MM-DD format)
        end_date: Optional end date for filtering results (YYYY-MM-DD format)
        max_results: Maximum number of results to return

    Returns:
        Dictionary containing search results from Tavily API
    """
    logger.debug(
        f'Searching Tavily with query: "{query}", max_results: {max_results}',
    )
    if start_date:
        logger.debug(f'Search start_date: {start_date}')
    if end_date:
        logger.debug(f'Search end_date: {end_date}')

    client = TavilyClient(os.getenv('TAVILY_API_KEY'))

    # Build parameters dict, excluding None values
    search_params = {'query': query, 'max_results': max_results}
    if start_date is not None:
        search_params['start_date'] = start_date
    if end_date is not None:
        search_params['end_date'] = end_date

    try:
        response = client.search(**search_params)
        logger.info(
            f'Tavily search completed successfully, returned {len(response.get("results", []))} results',
        )
        return response
    except Exception as e:
        logger.error(f'Error during Tavily search: {e}', exc_info=True)
        raise
