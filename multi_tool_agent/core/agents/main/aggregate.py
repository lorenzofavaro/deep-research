from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.adk.events import EventActions
from google.genai import types


class AggregateStep(BaseAgent):
    name: str = ''
    description: str = ''
    run_id: str = ''

    def __init__(self, name: str = 'aggregate_step', description: str = 'Aggregate the step results done in the research plan', run_id: str = ''):
        super().__init__(name=name, description=description)
        self.run_id = run_id

    async def _run_async_impl(self, ctx):
        results = [
            v for k, v in ctx.session.state.items(
            ) if k.startswith(f'results:{self.run_id}')
        ]
        step_delta = {f'results:{self.run_id}': results}

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=step_delta),
            content=types.Content(
                role='assistant', parts=[
                    types.Part(text='Aggregated results'),
                ],
            ),
        )
