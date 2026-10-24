from collections.abc import AsyncGenerator
from collections.abc import Callable
from typing import Optional

from google.adk.agents import BaseAgent
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.models import BaseLlm
from google.adk.models import Gemini
from google.adk.models import LlmRequest
from google.adk.models import LlmResponse
from google.genai import types
from pydantic import BaseModel

from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)

# Gemini often answers 429/503 under load: retry with exponential backoff instead of losing the call
RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5, initial_delay=2, max_delay=60,
)


def build_model(model: str) -> BaseLlm:
    """
    Create the model client used by every agent.

    Kept as the single construction point so that tests can replace it with a scripted model.

    Args:
        model: Gemini model name

    Returns:
        The model client
    """
    return Gemini(model=model, retry_options=RETRY_OPTIONS)


def isolated_llm_agent(
    *,
    name: str,
    model: str,
    instruction: str,
    build_input: Callable[[CallbackContext], str],
    output_key: str,
    output_schema: Optional[type[BaseModel]] = None,
) -> LlmAgent:
    """
    Create an LLM agent whose only input is the text returned by `build_input`.

    ADK would otherwise send the session history (including the output of every other agent and of the
    parallel branches) to the model. Here the request contents are replaced with a single user message,
    so each call sees exactly the data it needs.

    Args:
        name: Unique name of the agent
        model: Model to use
        instruction: System instruction, used verbatim (no state templating)
        build_input: Builds the user message from the callback context
        output_key: State key the output is stored under
        output_schema: Optional schema of the structured output

    Returns:
        The configured LlmAgent
    """
    def set_input(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
        llm_request.contents = [
            types.Content(
                role='user', parts=[
                    types.Part(
                        text=build_input(callback_context),
                    ),
                ],
            ),
        ]
        return None

    return LlmAgent(
        name=name,
        model=build_model(model),
        # A provider bypasses ADK's {state} templating, which would choke on braces in the instruction
        instruction=lambda _: instruction,
        include_contents='none',
        before_model_callback=set_input,
        output_schema=output_schema,
        output_key=output_key,
        # These agents only produce output: no transfers (ADK would otherwise warn and force this itself)
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )


class GuardedAgent(BaseAgent):
    """Runs an agent and logs its failure instead of raising, so one failing branch does not abort its siblings."""

    def __init__(self, *, agent: BaseAgent) -> None:
        """
        Initialize the GuardedAgent.

        Args:
            agent: The agent to run
        """
        super().__init__(name=f'guarded_{agent.name}')
        self._agent = agent

    async def _run_async_impl(self, context: InvocationContext) -> AsyncGenerator[Event, None]:
        """
        Run the wrapped agent, swallowing its exceptions.

        Args:
            context: Invocation context

        Yields:
            Events from the wrapped agent
        """
        try:
            async for event in self._agent.run_async(context):
                yield event
        except Exception as e:
            logger.error(
                f'Agent {self._agent.name} failed: {e}', exc_info=True,
            )
