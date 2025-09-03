import os
from typing import Any
from typing import Optional

from tavily import TavilyClient


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
    client = TavilyClient(os.getenv('TAVILY_API_KEY'))

    # Build parameters dict, excluding None values
    search_params = {'query': query, 'max_results': max_results}
    if start_date is not None:
        search_params['start_date'] = start_date
    if end_date is not None:
        search_params['end_date'] = end_date

    response = client.search(**search_params)
    return response
