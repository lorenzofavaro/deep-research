from google.adk.agents import BaseAgent
from google.adk.agents import ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events import EventActions
from google.genai import types

from multi_tool_agent.core.agents.main.aggregate import AggregateStep
from multi_tool_agent.core.agents.research.arxiv.arx_agent import ArxivAgent
from multi_tool_agent.core.agents.research.web.search_agent import WebSearchAgent
from multi_tool_agent.core.services import ServiceContainer
from multi_tool_agent.utils.utils import valid_uuid


class ResearchAgent(BaseAgent):
    name: str = ''
    description: str = ''
    run_id: str = ''
    services: ServiceContainer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _run_async_impl(self, context: InvocationContext):
        plan = context.session.state.get('research_plan', {})
        task_delta = {}
        sub_agents = []
        for step in plan.get('steps', []):
            agent_id = valid_uuid()
            print(f'step: {step}')

            task_delta[f'query:{self.run_id}:{agent_id}'] = step.get(
                'query', '',
            )
            task_delta[f'collection_name:{self.run_id}'] = self.run_id

            if step.get('action') == 'arxiv_search':
                sub_agents.append(
                    ArxivAgent(
                        name='arxiv_search',
                        run_id=str(self.run_id),
                        agent_id=str(agent_id),
                        document_service=self.services.document_service,
                    ),
                )
            else:
                sub_agents.append(
                    WebSearchAgent(
                        name='web_search', run_id=str(self.run_id), agent_id=str(agent_id),
                    ),
                )

        yield Event(
            author=self.name,
            content=types.Content(
                role='assistant',
                parts=[
                    types.Part(
                        text='Starting parallel research execution',
                    ),
                ],
            ),
            actions=EventActions(state_delta=task_delta),
        )

        parallel = ParallelAgent(
            name='ParallelAgent',
            sub_agents=sub_agents,
        )

        async for ev in parallel.run_async(context):
            yield ev

        yield Event(
            author=self.name,
            content=types.Content(
                role='assistant',
                parts=[
                    types.Part(
                        text='Completed parallel research execution',
                    ),
                ],
            ),
        )

        aggregate = AggregateStep(
            name='aggregate', description='Aggregate the step results done in the research plan', run_id=self.run_id,
        )
        async for ev in aggregate.run_async(context):
            yield ev
