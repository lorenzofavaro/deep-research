from typing import Literal
from typing import Optional

from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field


class UserRequest(BaseModel):
    """Model representing a user's research request."""
    query: str = Field(description="User's research request or query")


class ClassificationResult(BaseModel):
    """Result from classifying a user request."""
    type: Literal[
        'valid', 'general',
        'need-more-info',
    ] = Field(description='Classification result')
    user_intent: Optional[str] = Field(
        description="Precise user intent for 'valid' classifications", default=None,
    )
    next_message: Optional[str] = Field(
        description="Next message to send to user for 'general' or 'need-more-info' classifications", default=None,
    )


GEMINI_MODEL = 'gemini-1.5-flash'

system_prompt = """You are a Request Classification Agent that categorizes user requests into three types:

**Categories:**
1. **valid** - Specific, clear research requests with defined topics/domains
2. **general** - Too broad but shows research intent (needs refinement)
3. **need-more-info** - Unclear, incomplete, or non-research related

**Output Format:**
- **valid**: Return classification + user_intent (concise description of research goal)
- **general**: Return classification + next_message (explain need for more specific queries)
- **need-more-info**: Return classification + next_message (ask for clarification)

Personalize messages when possible.
"""

classify_agent = LlmAgent(
    name='ClassifyAgent',
    model=GEMINI_MODEL,
    instruction=system_prompt,
    description='Classify user requests and provide follow-up messages',
    input_schema=UserRequest,
    output_schema=ClassificationResult,
    output_key='classification',
)
