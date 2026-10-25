import asyncio
from typing import Optional

from google.adk import Event
from google.adk import Workflow
from google.adk.agents.context import Context
from google.adk.workflow import node
from google.genai import types
from pydantic import BaseModel

from multi_tool_agent.core.agents.main.answer import answer_agent
from multi_tool_agent.core.agents.main.answer import answer_input
from multi_tool_agent.core.agents.main.answer import AnswerDraft
from multi_tool_agent.core.agents.main.answer import build_registry
from multi_tool_agent.core.agents.main.answer import format_references
from multi_tool_agent.core.agents.main.classify import ClassificationResult
from multi_tool_agent.core.agents.main.classify import classify_agent
from multi_tool_agent.core.agents.main.classify import classify_input
from multi_tool_agent.core.agents.main.plan import plan_agent
from multi_tool_agent.core.agents.main.plan import plan_input
from multi_tool_agent.core.agents.main.plan import ResearchPlan
from multi_tool_agent.core.agents.main.refine import refine_report
from multi_tool_agent.core.agents.main.reflect import reflect_agent
from multi_tool_agent.core.agents.main.reflect import reflect_input
from multi_tool_agent.core.agents.main.reflect import Reflection
from multi_tool_agent.core.agents.main.report import write_report
from multi_tool_agent.core.agents.main.research import run_research_round
from multi_tool_agent.core.agents.main.verify import verify_answer
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


class RunInput(BaseModel):
    """Input of a research run."""
    question: str
    complexity: str
    scout: Optional[StepResult] = None
    # Number of this run and number of runs, when several independent runs are merged (ensemble)
    variant: int = 1
    variants: int = 1


@node(name='research_run', rerun_on_resume=True)
async def research_run(ctx: Context, node_input: RunInput):
    """
    Plan the outline, the quality constraints and the research tasks, then run rounds of parallel researchers
    until the reflection agent finds the coverage sufficient (or the round limit is reached).

    Args:
        ctx: Node context
        node_input: The question, its complexity, the scout's survey and the ensemble variant

    Yields:
        Progress messages, then the output: {'plan': ResearchPlan, 'results': [StepResult]}
    """
    question, complexity = node_input.question, node_input.complexity
    prefix = f'[run {node_input.variant}/{node_input.variants}] ' if node_input.variants > 1 else ''
    plan = ResearchPlan.model_validate(
        await ctx.run_node(
            plan_agent(),
            node_input=plan_input(
                question, complexity, node_input.scout, node_input.variant, node_input.variants,
            ),
        ),
    )
    yield Event(message=prefix + _format_plan(plan))
    tasks = plan.tasks[:config.max_steps]

    results: list[StepResult] = []
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

        logger.info(f'{prefix}Research round {iteration}')
        yield Event(message=f'{prefix}Research round {iteration}:\n{_format_tasks(tasks)}')
        round_results = await run_research_round(ctx, tasks, question, complexity, plan.outline)
        results += round_results
        yield Event(message=prefix + '\n'.join(_format_result(result) for result in round_results))

        if iteration == config.max_iterations:
            break
        scouted = [node_input.scout] if node_input.scout else []
        reflection = Reflection.model_validate(
            await ctx.run_node(reflect_agent(), node_input=reflect_input(question, plan, scouted + results)),
        )
        if reflection.revised_outline:
            plan = plan.model_copy(
                update={'outline': reflection.revised_outline},
            )
        if reflection.sufficient:
            break
        logger.info(f'{prefix}Research incomplete: {reflection.missing}')
        tasks = reflection.follow_up_tasks[:config.max_steps]

    yield {'plan': plan.model_dump(), 'results': [result.model_dump() for result in results]}


@node(name='research', rerun_on_resume=True)
async def research(ctx: Context, node_input: types.Content):
    """
    Execute the complete research workflow for the latest user message.

    Classify the message; let a scout survey the landscape (except for simple questions); run the research (one
    run, or several independent runs merged when ENSEMBLE_RUNS > 1); write the answer (chat) or the report
    (optionally refined); verify its citations; append the references.

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

    variants = config.ensemble_runs
    runs = await asyncio.gather(
        *(
            ctx.run_node(
                research_run,
                node_input=RunInput(
                    question=question, complexity=complexity, scout=scout, variant=variant, variants=variants,
                ).model_dump(),
                use_sub_branch=variants > 1,
            )
            for variant in range(1, variants + 1)
        ),
        return_exceptions=True,
    )
    completed = [run for run in runs if not isinstance(run, BaseException)]
    for run in runs:
        if isinstance(run, BaseException):
            logger.warning(f'A research run failed: {run}')
    if not completed:
        raise runs[0]
    # The outline of the first completed run frames the answer; the findings of all the runs are merged
    plan = ResearchPlan.model_validate(completed[0]['plan'])
    results += [
        StepResult.model_validate(result)
        for run in completed for result in run['results']
    ]

    registry = build_registry(results)
    output = classification.output if config.output_mode == 'auto' else config.output_mode
    logger.info(f'Generating final answer ({output})')
    if output == 'report' and plan.outline:
        yield Event(message=f'Writing the report: {len(plan.outline)} sections')
        answer = await write_report(ctx, user_message, question, plan.outline, registry)
        if config.refine:
            yield Event(message='Refining the report')
            answer = await refine_report(ctx, answer, registry)
    else:
        draft = await ctx.run_node(answer_agent(), node_input=answer_input(user_message, question, registry))
        answer = AnswerDraft.model_validate(draft).answer if draft else ''

    verification = None
    if config.verify and answer:
        yield Event(message='Verifying the citations')
        answer, verification = await verify_answer(ctx, answer, registry)
        logger.info(f'Verification: {verification}')
    yield Event(message=answer)
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
        'output': output,
        'verification': verification.model_dump() | {'supported_rate': verification.supported_rate}
        if verification else None,
    }
    yield Event(message=references or 'No sources cited.')
    logger.info('Research workflow completed successfully')


root_agent = Workflow(name='root', edges=[('START', research)])
