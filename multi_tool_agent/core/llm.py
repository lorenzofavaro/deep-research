from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm
from google.adk.models import FallbackModel
from google.adk.models import Gemini
from google.genai import types
from pydantic import BaseModel

# Gemini often answers 503 under load: retry the same model with exponential backoff
RETRY_STATUS_CODES = [408, 500, 502, 503, 504]
RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5, initial_delay=2, max_delay=60, http_status_codes=RETRY_STATUS_CODES,
)
# 429 (quota or rate limit) is retried only on the last model of a chain: before that, the next model takes over
LAST_MODEL_RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5, initial_delay=2, max_delay=60, http_status_codes=[429, *RETRY_STATUS_CODES],
)


def build_model(spec: str) -> BaseLlm:
    """
    Create the model client used by every agent.

    Kept as the single construction point so that tests can replace it with a scripted model.

    Args:
        spec: A Gemini model name, or a comma-separated chain of names tried in order when a model fails with a
            quota, rate-limit or server error (e.g. "gemini-flash-latest,gemini-flash-lite-latest")

    Returns:
        The model client
    """
    names = [name.strip() for name in spec.split(',') if name.strip()]
    models = [
        Gemini(
            model=name, retry_options=LAST_MODEL_RETRY_OPTIONS if i ==
            len(names) - 1 else RETRY_OPTIONS,
        )
        for i, name in enumerate(names)
    ]
    return models[0] if len(models) == 1 else FallbackModel(models=models)


def llm_agent(
    *,
    name: str,
    model: str,
    instruction: str,
    output_schema: Optional[type[BaseModel]] = None,
) -> LlmAgent:
    """
    Create an LLM agent that sees only its instruction and the input it is run with.

    The agent is run as a workflow node (`ctx.run_node(agent, node_input=...)`): with `include_contents='none'`
    the model receives the node input as its only message, never the session history or the output of other
    agents and parallel branches.

    Args:
        name: Name of the agent
        model: Model spec, see `build_model`
        instruction: System instruction, used verbatim (no state templating)
        output_schema: Optional schema of the structured output, returned by `run_node` as a dict

    Returns:
        The configured LlmAgent
    """
    return LlmAgent(
        name=name,
        model=build_model(model),
        # A provider bypasses ADK's {state} templating, which would choke on braces in the instruction
        instruction=lambda _: instruction,
        include_contents='none',
        output_schema=output_schema,
    )
