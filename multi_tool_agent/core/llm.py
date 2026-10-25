import asyncio
import re
import time
from collections.abc import AsyncGenerator
from typing import Any
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.models import BaseLlm
from google.adk.models import FallbackModel
from google.adk.models import Gemini
from google.adk.models import LlmRequest
from google.adk.models import LlmResponse
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel

from multi_tool_agent.utils.logger import get_logger

logger = get_logger(__name__)

# Gemini often answers 503 under load: retry the same model with exponential backoff. 429 is handled by
# QuotaAwareGemini, which tells a short rate limit from an exhausted daily quota
RETRY_OPTIONS = types.HttpRetryOptions(
    attempts=5, initial_delay=2, max_delay=60, http_status_codes=[408, 500, 502, 503, 504],
)
RATE_LIMIT_ATTEMPTS = 4
MAX_RATE_LIMIT_WAIT = 90.0

# Monotonic time until which a model's daily quota is known to be exhausted, by model name
_exhausted_until: dict[str, float] = {}


def _quota_details(error: ClientError) -> tuple[bool, Optional[float]]:
    """Return whether a 429 is a daily quota, and the retry delay it suggests (seconds), if any."""
    details = (error.details or {}).get('error', {}).get(
        'details', [],
    ) if isinstance(error.details, dict) else []
    daily = any(
        'PerDay' in violation.get('quotaId', '')
        for detail in details for violation in detail.get('violations', [])
    )
    delay = next(
        (
            float(match.group(1)) for detail in details
            if (match := re.fullmatch(r'([\d.]+)s', str(detail.get('retryDelay', ''))))
        ),
        None,
    )
    return daily, delay


class QuotaAwareGemini(Gemini):
    """
    Gemini client for the free tier, where 429 means one of two things.

    - A per-minute rate limit (e.g. several researchers in parallel): wait the suggested delay and retry.
    - An exhausted daily quota: fail at once, and keep failing without calling the API until the quota is back,
      so a FallbackModel moves on to the next model immediately.
    """

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        if time.monotonic() < _exhausted_until.get(self.model, 0):
            raise ClientError(
                429, {
                    'error': {
                        'code': 429, 'status': 'RESOURCE_EXHAUSTED',
                        'message': f'Daily quota of {self.model} exhausted',
                    },
                },
            )
        for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
            yielded = False
            try:
                async for response in super().generate_content_async(llm_request, stream):
                    yielded = True
                    yield response
                return
            except ClientError as e:
                if e.code != 429 or yielded:
                    raise
                daily, delay = _quota_details(e)
                if daily:
                    _exhausted_until[self.model] = time.monotonic(
                    ) + (delay or 3600)
                    logger.warning(f'Daily quota of {self.model} exhausted')
                    raise
                if attempt == RATE_LIMIT_ATTEMPTS:
                    raise
                wait = min(delay or 2 ** attempt * 5, MAX_RATE_LIMIT_WAIT)
                logger.info(
                    f'{self.model} rate limited, retrying in {wait:.0f}s',
                )
                await asyncio.sleep(wait)


def build_model(spec: str) -> BaseLlm:
    """
    Create the model client used by every agent.

    Kept as the single construction point so that tests can replace it with a scripted model.

    Args:
        spec: A Gemini model name, or a comma-separated chain of names tried in order when a model fails with a
            quota or server error (e.g. "gemini-flash-latest,gemini-flash-lite-latest")

    Returns:
        The model client
    """
    names = [name.strip() for name in spec.split(',') if name.strip()]
    models: list[Any] = [
        QuotaAwareGemini(
            model=name, retry_options=RETRY_OPTIONS,
        ) for name in names
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
