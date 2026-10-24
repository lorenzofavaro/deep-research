from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events import EventActions

from multi_tool_agent.core.agents.main.answer import answer_agent
from multi_tool_agent.core.agents.main.answer import build_registry
from multi_tool_agent.core.agents.main.answer import format_references
from multi_tool_agent.core.agents.main.classify import classify_agent
from multi_tool_agent.core.agents.main.plan import plan_agent
from multi_tool_agent.core.agents.main.plan import ResearchPlan
from multi_tool_agent.core.agents.main.reflect import reflect_agent
from multi_tool_agent.core.agents.main.reflect import Reflection
from multi_tool_agent.core.agents.main.research import collect_results
from multi_tool_agent.core.agents.main.research import ResearchAgent
from multi_tool_agent.core.agents.research.base import message
from multi_tool_agent.core.models import ResearchStep
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.logger import get_logger
from multi_tool_agent.utils.utils import valid_uuid

logger = get_logger(__name__)


def _step_signature(step: ResearchStep) -> tuple[str, str]:
    return step.tool, ' '.join(step.query.lower().split())


def _format_steps(steps: list[ResearchStep]) -> str:
    return '\n'.join(f'- [{step.tool}] `{step.query}` — {step.goal}' for step in steps)


class RootAgent(BaseAgent):
    """Root agent that coordinates the entire research workflow."""

    def _remember(self, context: InvocationContext, role: str, text: str) -> EventActions:
        """Append a message to the conversation kept in the state, which the classifier reads."""
        conversation = list(context.session.state.get('conversation', []))
        conversation.append({'role': role, 'text': text})
        return EventActions(state_delta={'conversation': conversation})

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the complete research workflow.

        Args:
            context: Invocation context containing user query and session state

        Yields:
            Events from each stage of the research process
        """
        logger.info('Starting research workflow')
        parts = context.user_content.parts if context.user_content else None
        user_message = ''.join(part.text or '' for part in parts or [])
        yield Event(author=self.name, actions=self._remember(context, 'user', user_message))

        # Classify the user request
        logger.debug('Classifying user request')
        async for event in classify_agent().run_async(context):
            yield event

        classification = context.session.state['classification']
        logger.info(f'Classification result: {classification["type"]}')

        if classification['type'] != 'valid' or not classification.get('user_intent'):
            logger.info(
                'Request classified as invalid, returning classification message',
            )
            reply = classification.get(
                'next_message',
            ) or 'Could you tell me more about what you want to research?'
            yield message(self.name, reply, actions=self._remember(context, 'assistant', reply))
            return

        question = classification['user_intent']
        run_id = valid_uuid()

        # Generate research plan
        logger.info('Generating research plan')
        async for event in plan_agent(question=question).run_async(context):
            yield event
        steps = ResearchPlan.model_validate(
            context.session.state['research_plan'],
        ).steps[:config.max_steps]

        # Research loop: execute the steps, then let the reflection agent decide whether to dig deeper
        executed: set[tuple[str, str]] = set()
        for iteration in range(1, config.max_iterations + 1):
            steps = [
                step for step in steps if _step_signature(
                    step,
                ) not in executed
            ]
            if not steps:
                break
            executed.update(_step_signature(step) for step in steps)

            logger.info(f'Research round {iteration} with run_id: {run_id}')
            yield message(self.name, f'Research round {iteration}:\n{_format_steps(steps)}')
            research_agent = ResearchAgent(
                name=f'ResearchAgent_{iteration}', run_id=run_id, steps=steps, question=question,
            )
            async for event in research_agent.run_async(context):
                yield event

            if iteration == config.max_iterations:
                break
            reflector = reflect_agent(
                question=question, results=collect_results(
                    context.session.state, run_id,
                ),
            )
            async for event in reflector.run_async(context):
                yield event
            reflection = Reflection.model_validate(
                context.session.state['reflection'],
            )
            if reflection.sufficient:
                break
            logger.info(f'Research incomplete: {reflection.missing}')
            steps = reflection.follow_up_steps[:config.max_steps]

        # Generate final answer
        logger.info('Generating final answer')
        registry = build_registry(
            collect_results(context.session.state, run_id),
        )
        async for event in answer_agent(user_message=user_message, question=question, registry=registry).run_async(context):
            yield event

        answer = context.session.state.get('answer', '')
        references = format_references(answer, registry)
        yield message(
            self.name,
            references or 'No sources cited.',
            actions=self._remember(context, 'assistant', answer),
        )
        logger.info('Research workflow completed successfully')


root_agent = RootAgent(
    name='root',
    description='Root Agent',
)
