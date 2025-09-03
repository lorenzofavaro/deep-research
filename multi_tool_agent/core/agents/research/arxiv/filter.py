from google.adk.agents import LlmAgent
from pydantic import BaseModel
from pydantic import Field


class PaperMeta(BaseModel):
    id: str = Field(default_factory=str, description='Arxiv paper ID')
    title: str = Field(default_factory=str, description='Arxiv paper title')
    abstract: str = Field(
        default_factory=str,
        description='Arxiv paper abstract',
    )
    url: str = Field(default_factory=str, description='Arxiv paper URL')


class PapersMetas(BaseModel):
    infos: list[PaperMeta]


class PaperIDs(BaseModel):
    ids: list[str] = Field(
        default_factory=list,
        description='List of Arxiv paper ids',
    )


GEMINI_MODEL = 'gemini-1.5-flash'
PAPERS_NUM = 2

system_prompt = f"""You are a Research Search Agent.
Your role is to identify the most relevant academic papers for a given user query.

**Step 1 — Analyze candidates:**
- Carefully review the retrieved papers, considering both *title* and *abstract*.
- Assess their relevance to the user query with precision.

**Step 2 — Select results:**
- Choose **up to {PAPERS_NUM}** papers that best match the query.
- Prefer recent works (published in the last 3 years).
- Only include papers that provide direct, specific insights on the topic.
- If fewer than {PAPERS_NUM} are relevant, return only those (never add irrelevant ones).

Output rules (strict):
- The output must be a **valid JSON array of strings**, containing only the selected paper IDs.
- Do not include explanations, text outside the JSON, or formatting such as Markdown.
"""

filter_agent = LlmAgent(
    name='FilterAgent',
    model=GEMINI_MODEL,
    instruction=system_prompt,
    description='Filter papers',
    input_schema=PapersMetas,
    output_schema=PaperIDs,
    output_key='paper_ids',
)
