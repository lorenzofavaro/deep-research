import os


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


class Config:
    """Configuration class for managing environment variables and settings."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        # Models: `model` drives classification, planning, reflection and the final answer,
        # `fast_model` the high-volume calls (paper selection and source extraction). Splitting them also spreads
        # the load over two per-model quotas
        self.model = os.getenv('MODEL', 'gemini-flash-latest')
        self.fast_model = os.getenv('FAST_MODEL', 'gemini-flash-lite-latest')

        # Research loop
        self.max_steps = _int_env('MAX_STEPS', 3)
        self.max_iterations = _int_env('MAX_ITERATIONS', 2)

        # ArXiv
        self.arxiv_candidates = _int_env('ARXIV_CANDIDATES', 30)
        self.papers_per_step = _int_env('PAPERS_PER_STEP', 3)
        self.paper_max_chars = _int_env('PAPER_MAX_CHARS', 100_000)
        # OpenAlex: most cited papers of a topic and citation counts; works without a key on a shared daily budget
        self.cited_candidates = _int_env('CITED_CANDIDATES', 15)
        self.openalex_api_key = os.getenv('OPENALEX_API_KEY', '')

        # Web
        self.tavily_api_key = os.getenv('TAVILY_API_KEY', '')
        self.web_results = _int_env('WEB_RESULTS', 5)
        self.page_max_chars = _int_env('PAGE_MAX_CHARS', 20_000)


config = Config()
