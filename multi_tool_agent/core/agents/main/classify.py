from typing import Literal
from typing import Optional

from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field

from multi_tool_agent.core.llm import llm_agent
from multi_tool_agent.utils.config import config
from multi_tool_agent.utils.utils import today


class ClassificationResult(BaseModel):
    """Result from classifying a user request."""
    type: Literal[
        'valid', 'general',
        'need-more-info',
    ] = Field(description='Classification result')
    user_intent: Optional[str] = Field(
        description="Self-contained research question for 'valid' classifications", default=None,
    )
    next_message: Optional[str] = Field(
        description="Next message to send to user for 'general' or 'need-more-info' classifications", default=None,
    )


# Characters kept from each previous message, enough to resolve follow-up questions
MESSAGE_MAX_CHARS = 2000

system_prompt = """You are a Request Classification Agent that categorizes the latest user message into three types:

**Categories:**
1. **valid** - Questions that can be answered by searching academic papers or the web: specific research requests \
with defined topics/domains, and factual or news questions that need up-to-date information
2. **need-more-info** - Too broad or ambiguous to research, but shows research intent (needs refinement)
3. **general** - Messages that need no research (e.g., "2+2?", greetings, chit-chat, requests to write code)

**Output Format:**
- **valid**: Return classification + user_intent: the research question, rewritten so it is understandable on its own \
(resolve references to earlier messages, include the clarifications the user gave)
- **need-more-info**: Return classification + next_message (ask for clarification)
- **general**: Return classification + next_message (politely decline and explain you only help with research)

Use the previous conversation to interpret the latest message. Write next_message in the user's language and \
personalize it when possible."""


def classify_input(conversation: list[dict[str, str]]) -> str:
    """
    Build the classifier input from the conversation, whose last entry is the latest user message.

    Args:
        conversation: Messages as {"role": "user" | "assistant", "text": ...}

    Returns:
        The classifier input
    """
    *previous, latest = conversation
    lines = [f'Current date: {today()}', '', 'Previous conversation:']
    for turn in previous:
        speaker = 'User' if turn['role'] == 'user' else 'Assistant'
        lines.append(f'{speaker}: {turn["text"][:MESSAGE_MAX_CHARS]}')
    if not previous:
        lines.append('(none)')
    lines += ['', 'Latest user message:', latest['text']]
    return '\n'.join(lines)


def classify_agent() -> LlmAgent:
    """Create the agent that classifies the latest message of a conversation (input: `classify_input`)."""
    return llm_agent(
        name='ClassifyAgent',
        model=config.model,
        instruction=system_prompt,
        output_schema=ClassificationResult,
    )
