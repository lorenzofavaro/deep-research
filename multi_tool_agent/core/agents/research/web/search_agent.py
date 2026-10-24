from google.adk.agents.context import Context

from multi_tool_agent.core.agents.research.base import extract_findings
from multi_tool_agent.core.agents.research.base import StepTask
from multi_tool_agent.core.models import Source
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.core.tools.tavily import tavily_search
from multi_tool_agent.utils.config import config


async def research_web(ctx: Context, task: StepTask, result: StepResult) -> None:
    """
    Research step on the web: search with Tavily, then extract findings from the text of each page.

    Args:
        ctx: Context of the step node
        task: The research step
        result: Step result to fill
    """
    pages = await tavily_search(task.step.query, max_results=config.web_results)
    result.notes.append(f'{len(pages)} pages')
    documents = [
        (
            Source(
                key=page['url'], kind='web', title=page['title'],
                url=page['url'], published=page['published'],
            ),
            page['content'][:config.page_max_chars],
        )
        for page in pages
        if page['content']
    ]
    result.sources, result.findings = await extract_findings(ctx, task, documents)
