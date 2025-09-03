import os

from tavily import TavilyClient


def tavily_search(query: str, start_date: str = None, end_date: str = None, max_results: int = 5):
    client = TavilyClient(os.getenv('TAVILY_API_KEY'))
    response = client.search(
        query=query,
        start_date=start_date,
        end_date=end_date,
        max_results=max_results,
    )
    return response
