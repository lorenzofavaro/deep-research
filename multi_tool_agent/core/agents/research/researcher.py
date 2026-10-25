import functools
from collections.abc import Callable
from typing import Any
from typing import Literal
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.context import Context
from google.adk.models import LlmRequest
from google.adk.models import LlmResponse
from google.adk.workflow import node
from pydantic import BaseModel

from multi_tool_agent.core import llm
from multi_tool_agent.core.agents.research.store import ResearchStore
from multi_tool_agent.core.models import Finding
from multi_tool_agent.core.models import LENSES
from multi_tool_agent.core.models import ResearchTask
from multi_tool_agent.core.models import Source
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.core.tools import arxiv
from multi_tool_agent.core.tools import papers
from multi_tool_agent.core.tools import tavily
from multi_tool_agent.core.tools.passages import quote_in_text
from multi_tool_agent.core.tools.passages import top_passages
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger
from multi_tool_agent.utils.utils import today

logger = get_logger(__name__)

# Tools that stay available once the search budget is exhausted
WRAP_UP_TOOLS = {'record_finding'}
BUDGET_MESSAGE = 'Search budget exhausted: record your remaining findings, then write your final brief.'

system_prompt = """You are a Research Agent, one of several researchers working in parallel on a research question. \
You own one task: research it thoroughly with your tools, record what you find, then report back.

**Your task**
- Objective: {objective}
- Lens: {lens} — {lens_description}
- Out of scope (other researchers cover it): {boundaries}

**Tools**
{tool_guide}
- `read_source(source_id, focus)`: read the passages of a source most relevant to `focus` (what you are looking \
for). Read a source before recording findings from it; read it again with another focus if needed.
- `record_finding(source_id, claim, evidence)`: save a finding. `claim` is a self-contained statement (concrete \
names, numbers, conditions); `evidence` is a short quote copied VERBATIM from the source text you read. Findings \
are the only thing the final report can use: record everything relevant, from every useful source.
- `think(reflection)`: pause to assess what you have and decide the next move.

**Strategy**
- Start with short, broad queries, then narrow down based on what you find.
- Prefer primary and authoritative sources (papers, official docs and announcements, reputable outlets) over \
SEO content and aggregators.
- Cover the objective from several sources; look for numbers, comparisons and disagreements.
- You have {budget} search/read calls. Stop earlier when the objective is well covered.

**When you are done**, reply with a short brief (max ~150 words): what you found, how solid it is, and what is \
still missing. Do not repeat every finding: they are already recorded."""

TOOL_GUIDES = {
    'web': '- `web_search(query)`: web search (keywords, no instructions). Returns sources with id, title, date and '
           'a snippet.',
    'papers': '- `search_papers(query)`: arXiv search. The query MUST use arXiv API syntax: fields `ti:`, `abs:`, '
              '`au:`, `cat:`, `all:`, phrases in double quotes, AND/OR/ANDNOT, e.g. '
              '`abs:"speculative decoding" AND abs:LLM`; 2-3 key concepts, since plain sentences match almost '
              'anything. Returns papers with id, title, authors, date, citation count and abstract.',
}


class TaskInput(BaseModel):
    """Input of a researcher node."""
    task: ResearchTask
    question: str
    complexity: Literal['simple', 'moderate', 'complex'] = 'moderate'


def researcher_budget(complexity: str) -> int:
    """Search/read calls allowed to each researcher: simple questions need fewer."""
    return min(5, config.researcher_tool_calls) if complexity == 'simple' else config.researcher_tool_calls


def make_tools(store: ResearchStore, sources: list[str]) -> list[Callable[..., Any]]:
    """
    Create the researcher's tools, bound to its store.

    Args:
        store: Where the tools keep sources, texts and findings
        sources: Source types of the task ("web", "papers")

    Returns:
        The tool functions
    """
    async def web_search(query: str) -> dict[str, Any]:
        """Search the web.

        Args:
            query: Search engine query (keywords)

        Returns:
            Results with source id, title, url, published date and snippet
        """
        if not store.use_call():
            return {'error': BUDGET_MESSAGE}
        results = []
        for page in await tavily.tavily_search(query, max_results=config.web_results):
            source = Source(
                key=page['url'], kind='web', title=page['title'],
                url=page['url'], published=page['published'],
            )
            store.add_source(
                source, preview=page['content'][:300], text=page['content'][:config.page_max_chars],
            )
            results.append({
                'source_id': source.key, 'title': source.title, 'published': source.published,
                'snippet': page['content'][:300],
            })
        return {'results': results}

    async def search_papers(query: str) -> dict[str, Any]:
        """Search arXiv papers.

        Args:
            query: Query in arXiv API syntax

        Returns:
            Papers with source id, title, authors, date, citation count and abstract
        """
        if not store.use_call():
            return {'error': BUDGET_MESSAGE}
        results = []
        for paper in await papers.search_papers(query):
            source = Source(
                key=f'arxiv:{paper["id"]}', kind='arxiv', title=paper['title'], url=paper['url'],
                authors=paper['authors'], published=paper['published'],
            )
            store.add_source(source, preview=paper['abstract'])
            results.append({
                'source_id': source.key, 'title': source.title, 'authors': source.authors[:3],
                'published': source.published, 'citations': paper.get('citations', 'unknown'),
                'abstract': paper['abstract'][:600],
            })
        return {'results': results}

    async def read_source(source_id: str, focus: str) -> dict[str, Any]:
        """Read the passages of a source that are most relevant to a focus.

        Args:
            source_id: Id of a source returned by a search
            focus: What you are looking for in the source

        Returns:
            Title, metadata and the most relevant passages
        """
        source = store.sources.get(source_id)
        if source is None:
            return {'error': f'Unknown source_id {source_id!r}: use an id returned by a search.'}
        if not store.use_call():
            return {'error': BUDGET_MESSAGE}
        if source_id not in store.texts and source.kind == 'arxiv':
            paper_id = source_id.removeprefix('arxiv:')
            store.texts[source_id] = await arxiv.get_arxiv_paper_text(paper_id, max_chars=config.paper_max_chars)
        text = store.texts.get(source_id) or store.previews.get(source_id, '')
        passages = top_passages(text, focus, limit=config.passages_per_read)
        return {
            'title': source.title, 'authors': source.authors[:5], 'published': source.published,
            'passages': [f'[passage {position}] {passage}' for position, passage in passages],
        }

    async def record_finding(source_id: str, claim: str, evidence: str) -> dict[str, Any]:
        """Record a finding supported by a verbatim quote from a source you read.

        Args:
            source_id: Id of the source
            claim: Self-contained statement relevant to your objective
            evidence: Short quote copied verbatim from the source

        Returns:
            Whether the finding was recorded
        """
        if source_id not in store.sources:
            return {'error': f'Unknown source_id {source_id!r}.'}
        text = store.texts.get(source_id) or store.previews.get(source_id, '')
        if not quote_in_text(evidence, text):
            return {'error': 'The evidence is not a verbatim quote from the source: read it and copy the exact text.'}
        added = store.add_finding(
            source_id, Finding(claim=claim, evidence=evidence),
        )
        return {'recorded': added}

    async def think(reflection: str) -> str:
        """Write down your assessment of the research so far and your next step.

        Args:
            reflection: What you have, what is missing, what to do next

        Returns:
            An acknowledgement
        """
        return 'Noted. Searches left: ' + str(max(0, store.budget - store.calls))

    tools: list[Callable[..., Any]] = []
    if 'web' in sources:
        tools.append(web_search)
    if 'papers' in sources:
        tools.append(search_papers)
    return [*(_report_errors(tool) for tool in [*tools, read_source]), record_finding, think]


def _report_errors(tool: Callable[..., Any]) -> Callable[..., Any]:
    """Return failures of a tool (network, API) to the model as an error, so it can adapt instead of failing."""
    @functools.wraps(tool)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await tool(*args, **kwargs)
        except Exception as e:
            logger.warning(f'Tool {tool.__name__} failed: {e}')
            return {'error': f'{tool.__name__} failed ({type(e).__name__}: {e}). Try again differently or move on.'}
    return wrapper


def researcher_agent(task: ResearchTask, store: ResearchStore) -> LlmAgent:
    """
    Create a researcher for a task: an LLM agent with its own tool loop and search budget.

    Once the budget is exhausted, the search and read tools are removed from the requests, so the model can only
    record its last findings and write the brief.

    Args:
        task: The research task
        store: The researcher's store, shared with its tools

    Returns:
        The configured LlmAgent
    """
    sources = task.sources or ['web']
    instruction = system_prompt.format(
        objective=task.objective,
        lens=task.lens,
        lens_description=LENSES[task.lens],
        boundaries=task.boundaries or 'nothing',
        tool_guide='\n'.join(
            TOOL_GUIDES[source] for source in (
                'papers', 'web',
            ) if source in sources
        ),
        budget=store.budget,
    )

    def enforce_budget(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        store.model_calls += 1
        # Hard stop, in case the model keeps calling tools
        allowed = set() if store.model_calls > 3 * store.budget + 5 else WRAP_UP_TOOLS
        if store.exhausted or not allowed:
            for tool in llm_request.config.tools or []:
                if tool.function_declarations:
                    tool.function_declarations = [
                        d for d in tool.function_declarations if d.name in allowed
                    ]
            llm_request.config.tools = [
                tool for tool in llm_request.config.tools or [] if tool.function_declarations
            ]
            llm_request.tools_dict = {
                name: t for name, t in llm_request.tools_dict.items() if name in allowed
            }
            llm_request.append_instructions([BUDGET_MESSAGE])
        return None

    return LlmAgent(
        name='ResearchAgent',
        model=llm.build_model(config.fast_model),
        instruction=lambda _: instruction,
        include_contents='none',
        tools=make_tools(store, sources),
        before_model_callback=enforce_budget,
    )


def researcher_input(task: ResearchTask, question: str) -> str:
    """Build the researcher's input: the overall question and its objective."""
    return (
        f'Current date: {today()}\n'
        f'Overall research question: {question}\n'
        f'Your objective: {task.objective}\n'
        'Begin.'
    )


@node(name='researcher', rerun_on_resume=True)
async def research_task(ctx: Context, node_input: TaskInput) -> StepResult:
    """
    Run one researcher on its task. Failures are recorded in the result, which keeps the findings recorded before
    the failure, so a failing researcher does not affect the others.

    Args:
        ctx: Node context
        node_input: The task, the research question and its complexity

    Returns:
        The task result: sources, findings and brief
    """
    task = node_input.task
    store = ResearchStore(budget=researcher_budget(node_input.complexity))
    result = StepResult(step=task)
    try:
        brief = await ctx.run_node(researcher_agent(task, store), node_input=researcher_input(task, node_input.question))
        result.brief = brief or ''
    except Exception as e:
        logger.error(
            f'Researcher on `{task.objective}` failed: {e}', exc_info=True,
        )
        result.error = str(e)
    result.sources = store.cited_sources()
    result.findings = store.findings
    result.notes.append(f'{store.calls} searches/reads')
    return result
