from google.adk import Event
from google.adk import Workflow
from google.adk.agents.context import Context
from google.adk.workflow import node
from google.genai import types

from multi_tool_agent.core.agents.main.answer import answer_agent
from multi_tool_agent.core.agents.main.answer import answer_input
from multi_tool_agent.core.agents.main.answer import build_registry
from multi_tool_agent.core.agents.main.answer import format_references
from multi_tool_agent.core.agents.main.classify import ClassificationResult
from multi_tool_agent.core.agents.main.classify import classify_agent
from multi_tool_agent.core.agents.main.classify import classify_input
from multi_tool_agent.core.agents.main.plan import plan_agent
from multi_tool_agent.core.agents.main.plan import plan_input
from multi_tool_agent.core.agents.main.plan import ResearchPlan
from multi_tool_agent.core.agents.main.reflect import reflect_agent
from multi_tool_agent.core.agents.main.reflect import reflect_input
from multi_tool_agent.core.agents.main.reflect import Reflection
from multi_tool_agent.core.agents.main.research import run_research_round
from multi_tool_agent.core.models import ResearchTask
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)


def _task_signature(task: ResearchTask) -> str:
    return ' '.join(task.objective.lower().split())


def _format_tasks(tasks: list[ResearchTask]) -> str:
    return '\n'.join(f'- [{task.lens}, {", ".join(task.sources)}] {task.objective}' for task in tasks)


def _format_result(result: StepResult) -> str:
    found = sum(len(findings) for findings in result.findings.values())
    details = ', '.join(
        [
            *result.notes,
            f'{found} findings from {len(result.findings)} sources',
        ],
    )
    summary = f'- {result.step.objective}: {details}'
    return summary + (f' (error: {result.error})' if result.error else '')


def _format_plan(plan: ResearchPlan) -> str:
    lines = ['Outline:', *(f'- {section.title}' for section in plan.outline)]
    if plan.quality_constraints:
        lines += [
            'Quality constraints:', *
            (f'- {constraint}' for constraint in plan.quality_constraints),
        ]
    return '\n'.join(lines)


def scout_task(question: str) -> ResearchTask:
    """The task of the scout, a researcher that quickly maps the landscape before planning."""
    return ResearchTask(
        objective=f'Survey the landscape of this question: {question}',
        sources=['papers', 'web'],
        lens='scout',
        boundaries='in-depth analysis, which other researchers do after you',
    )


@node(name='research', rerun_on_resume=True)
async def research(ctx: Context, node_input: types.Content):
    """
    Execute the complete research workflow for the latest user message.

    Classify the message; let a scout survey the landscape (except for simple questions); plan the outline, the
    quality constraints and the research tasks; run rounds of parallel researchers until the reflection agent finds
    the coverage sufficient (or the round limit is reached); then write the answer with its references.

    Args:
        ctx: Node context
        node_input: The user message

    Yields:
        Progress messages, the answer and the references
    """
    logger.info('Starting research workflow')
    user_message = ''.join(part.text or '' for part in node_input.parts or [])
    # The conversation is kept in the state so the classifier can interpret follow-ups and clarifications
    conversation = [
        *ctx.state.get(
            'conversation', [],
        ), {'role': 'user', 'text': user_message},
    ]
    ctx.state['conversation'] = conversation

    classification = ClassificationResult.model_validate(
        await ctx.run_node(classify_agent(), node_input=classify_input(conversation)),
    )
    logger.info(f'Classification result: {classification.type}')
    if classification.type != 'valid' or not classification.user_intent:
        reply = classification.next_message or 'Could you tell me more about what you want to research?'
        ctx.state['conversation'] = [
            *conversation,
            {'role': 'assistant', 'text': reply},
        ]
        yield Event(message=reply)
        return

    question = classification.user_intent
    complexity = classification.complexity
    results: list[StepResult] = []

    scout = None
    if complexity != 'simple' and config.scout_tool_calls > 0:
        yield Event(message='Surveying the landscape')
        [scout] = await run_research_round(
            ctx, [scout_task(question)], question, complexity, budget=config.scout_tool_calls,
        )
        results.append(scout)

    plan = ResearchPlan.model_validate(
        await ctx.run_node(plan_agent(), node_input=plan_input(question, complexity, scout)),
    )
    yield Event(message=_format_plan(plan))
    tasks = plan.tasks[:config.max_steps]

    # Research loop: run the researchers, then let the reflection agent check the coverage
    executed: set[str] = set()
    for iteration in range(1, config.max_iterations + 1):
        tasks = [
            task for task in tasks if _task_signature(
                task,
            ) not in executed
        ]
        if not tasks:
            break
        executed.update(_task_signature(task) for task in tasks)

        logger.info(f'Research round {iteration}')
        yield Event(message=f'Research round {iteration}:\n{_format_tasks(tasks)}')
        round_results = await run_research_round(ctx, tasks, question, complexity, plan.outline)
        results += round_results
        yield Event(message='\n'.join(_format_result(result) for result in round_results))

        if iteration == config.max_iterations:
            break
        reflection = Reflection.model_validate(
            await ctx.run_node(reflect_agent(), node_input=reflect_input(question, plan, results)),
        )
        if reflection.revised_outline:
            plan = plan.model_copy(
                update={'outline': reflection.revised_outline},
            )
        if reflection.sufficient:
            break
        logger.info(f'Research incomplete: {reflection.missing}')
        tasks = reflection.follow_up_tasks[:config.max_steps]

    logger.info('Generating final answer')
    registry = build_registry(results)
    answer = await ctx.run_node(answer_agent(), node_input=answer_input(user_message, question, registry)) or ''
    references = format_references(answer, registry)

    ctx.state['conversation'] = [
        *conversation,
        {'role': 'assistant', 'text': answer},
    ]
    # The last run, for inspection and for the smoke test
    ctx.state['last_run'] = {
        'question': question,
        'plan': plan.model_dump(),
        'steps': [result.model_dump() for result in results],
        'answer': answer,
    }
    yield Event(message=references or 'No sources cited.')
    logger.info('Research workflow completed successfully')


root_agent = Workflow(name='root', edges=[('START', research)])
