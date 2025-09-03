from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events import EventActions
from google.genai import types

from multi_tool_agent.core.agents.main.answer import AnswerAgent
from multi_tool_agent.core.agents.main.classify import classify_agent
from multi_tool_agent.core.agents.main.plan import plan_agent
from multi_tool_agent.core.agents.main.research import ResearchAgent
from multi_tool_agent.core.services import ServiceContainer
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.utils import valid_uuid


class RootAgent(BaseAgent):
    """Root agent that coordinates the entire research workflow."""

    classify_agent: LlmAgent
    plan_agent: LlmAgent
    services: ServiceContainer
    model_config = {'arbitrary_types_allowed': True}

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the RootAgent.

        Args:
            **kwargs: Keyword arguments including agents and services
        """
        if 'services' not in kwargs:
            kwargs['services'] = ServiceContainer(config)
        super().__init__(**kwargs)

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the complete research workflow.

        Args:
            context: Invocation context containing user query and session state

        Yields:
            Events from each stage of the research process
        """
        # Classify the user request
        async for event in self.classify_agent.run_async(context):
            yield event

        classification = context.session.state['classification']
        if classification['type'] != 'valid':
            yield Event(
                author=self.name,
                content=types.Content(
                    role='assistant',
                    parts=[types.Part(text=classification['next_message'])],
                ),
            )
            return

        # Set up query state for planning
        state_delta: dict[str, object] = {
            'query': context.session.state['classification']['user_intent'],
        }
        system_event = Event(
            invocation_id=context.invocation_id,
            author='system',
            actions=EventActions(state_delta=state_delta),
        )
        await context.session_service.append_event(context.session, system_event)

        # Generate research plan
        async for event in self.plan_agent.run_async(context):
            yield event

        # Execute research plan
        run_id = valid_uuid()
        research_agent = ResearchAgent(
            name='ResearchAgent', description='Coordinates research steps', run_id=run_id, services=self.services,
        )
        async for event in research_agent.run_async(context):
            yield event

        # Generate final answer
        answer_agent = AnswerAgent(name='AnswerAgent', run_id=run_id)
        async for event in answer_agent.run_async(context):
            yield event


root_agent = RootAgent(
    name='root',
    description='Root Agent',
    classify_agent=classify_agent,
    plan_agent=plan_agent,
    services=ServiceContainer(config),
)
