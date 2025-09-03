from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events import EventActions
from google.genai import types


class AggregateStep(BaseAgent):
    """Agent step for aggregating results from multiple research steps."""

    name: str = ''
    description: str = ''
    run_id: str = ''

    def __init__(
        self,
        name: str = 'aggregate_step',
        description: str = 'Aggregate the step results done in the research plan',
        run_id: str = '',
    ) -> None:
        """
        Initialize the AggregateStep agent.

        Args:
            name: Name of the agent step
            description: Description of what this step does
            run_id: Unique identifier for the current run
        """
        super().__init__(name=name, description=description)
        self.run_id = run_id

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Execute the aggregation step to collect and combine research results.

        Args:
            ctx: Invocation context containing session state with research results

        Yields:
            Event containing the aggregated results
        """
        results = [
            v for k, v in ctx.session.state.items(
            ) if k.startswith(f'results:{self.run_id}')
        ]
        step_delta: dict[str, object] = {f'results:{self.run_id}': results}

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=step_delta),
            content=types.Content(
                role='assistant', parts=[
                    types.Part(text='Aggregated results'),
                ],
            ),
        )
