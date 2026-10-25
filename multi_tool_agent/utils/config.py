import os


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


class Config:
    """Configuration class for managing environment variables and settings."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        # Models: `model` drives classification, planning, reflection and the final answer, `fast_model` the
        # researchers, which make most of the calls. Splitting them also spreads the load over two per-model quotas
        # Each value can be a comma-separated chain: on a quota, rate-limit or server error the next model is used
        self.model = os.getenv(
            'MODEL', 'gemini-flash-latest,gemini-flash-lite-latest',
        )
        self.fast_model = os.getenv(
            'FAST_MODEL', 'gemini-flash-lite-latest,gemini-flash-latest',
        )

        # Research loop: researchers per round, rounds, search/read calls per researcher
        self.max_steps = _int_env('MAX_STEPS', 3)
        self.max_iterations = _int_env('MAX_ITERATIONS', 2)
        self.researcher_tool_calls = _int_env('RESEARCHER_TOOL_CALLS', 10)
        # Quick survey of the landscape before planning (skipped for simple questions); 0 disables it
        self.scout_tool_calls = _int_env('SCOUT_TOOL_CALLS', 3)
        self.passages_per_read = _int_env('PASSAGES_PER_READ', 6)

        # ArXiv
        self.arxiv_candidates = _int_env('ARXIV_CANDIDATES', 30)
        self.paper_max_chars = _int_env('PAPER_MAX_CHARS', 100_000)
        # OpenAlex: most cited papers of a topic and citation counts; works without a key on a shared daily budget
        self.cited_candidates = _int_env('CITED_CANDIDATES', 15)
        self.openalex_api_key = os.getenv('OPENALEX_API_KEY', '')

        # Web
        self.tavily_api_key = os.getenv('TAVILY_API_KEY', '')
        self.web_results = _int_env('WEB_RESULTS', 5)
        self.page_max_chars = _int_env('PAGE_MAX_CHARS', 20_000)


config = Config()
