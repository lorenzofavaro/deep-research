"""End-to-end tests of the research workflow with a scripted model and stubbed search tools: no API calls."""
import asyncio
import json
from collections import Counter
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from google.adk import Workflow
from google.adk.models import BaseLlm
from google.adk.models import LlmRequest
from google.adk.models import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

from multi_tool_agent import agent as agent_module
from multi_tool_agent.core import llm
from multi_tool_agent.core.agents.research.arxiv import arx_agent
from multi_tool_agent.core.agents.research.web import search_agent
from multi_tool_agent.utils.config import config

# Each agent is recognized by a phrase of its system instruction
ROLES = {
    'Request Classification Agent': 'classify',
    'Research Planning Agent': 'plan',
    'Research Search Agent': 'select',
    'Research Extraction Agent': 'extract',
    'Research Reflection Agent': 'reflect',
    'Synthesize the research findings': 'answer',
}

PAPERS = [
    {
        'id': '2401.00001', 'title': 'Paper One', 'abstract': 'About SSMs.', 'authors': ['Ada Lovelace'],
        'published': '2024-01-01', 'url': 'https://arxiv.org/abs/2401.00001',
    },
    {
        'id': '2401.00002', 'title': 'Paper Two', 'abstract': 'Unrelated.', 'authors': ['Alan Turing'],
        'published': '2024-01-02', 'url': 'https://arxiv.org/abs/2401.00002',
    },
]
# Papers the arXiv API knows by ID, beyond the search results
ARXIV_RECORDS = {
    '2106.09685': {
        'id': '2106.09685', 'title': 'LoRA: Low-Rank Adaptation of Large Language Models', 'abstract': 'LoRA.',
        'authors': ['Edward J. Hu'], 'published': '2021-06-17', 'url': 'https://arxiv.org/abs/2106.09685',
    },
}
PAGES = [
    {
        'title': 'Page One', 'url': 'https://example.com/1',
        'published': '2026-01-01', 'content': 'Web text one.',
    },
    {
        'title': 'Page Two', 'url': 'https://example.com/2',
        'published': '', 'content': 'Web text two.',
    },
]


def _title(user_input: str) -> str:
    return next(line[len('Title: '):] for line in user_input.splitlines() if line.startswith('Title: '))


DEFAULT_RESPONSES: dict[str, Any] = {
    'classify': {'type': 'valid', 'user_intent': 'How do state space models compare to transformers?'},
    'plan': {
        'steps': [
            {
                'tool': 'arxiv', 'query': 'abs:"state space model"',
                'goal': 'Find core SSM papers',
            },
            {
                'tool': 'web', 'query': 'state space models vs transformers',
                'goal': 'Find recent coverage',
            },
        ],
    },
    # The second ID is not among the candidates and must be discarded
    'select': {'ids': ['2401.00001v2', '9999.99999']},
    'extract': lambda user_input: {'findings': [{'claim': f'Claim from {_title(user_input)}', 'evidence': 'quote'}]},
    'reflect': {'sufficient': True, 'missing': '', 'follow_up_steps': []},
    'answer': 'SSMs scale linearly [1]. The web agrees [2, 3]. Invented [9].',
}


class Script:
    """Answers each model call according to the role of the calling agent, and records the calls."""

    def __init__(self, **overrides: Any) -> None:
        self.responses = {**DEFAULT_RESPONSES, **overrides}
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_instruction: str, user_input: str) -> str:
        role = next(
            role for phrase, role in ROLES.items()
            if phrase in system_instruction
        )
        self.calls.append((role, user_input))
        response = self.responses[role]
        if callable(response):
            response = response(user_input)
        return response if isinstance(response, str) else json.dumps(response)

    def count(self) -> Counter:
        return Counter(role for role, _ in self.calls)

    def inputs(self, role: str) -> list[str]:
        return [user_input for call_role, user_input in self.calls if call_role == role]


class ScriptedLlm(BaseLlm):
    """Model that delegates every request to the active Script."""

    script: Any = None

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        system_instruction = str(llm_request.config.system_instruction or '')
        user_input = ''.join(
            part.text or '' for content in llm_request.contents for part in content.parts
        )
        text = self.script(system_instruction, user_input)
        yield LlmResponse(content=types.Content(role='model', parts=[types.Part(text=text)]))


@pytest.fixture
def script(monkeypatch):
    """Replace the model and the search tools; returns a function that installs a Script."""
    holder: dict[str, Script] = {}
    monkeypatch.setattr(
        llm, 'build_model', lambda model: ScriptedLlm(
            model=model, script=holder['script'],
        ),
    )

    async def search_arxiv(query, max_results=30):
        return [dict(paper) for paper in PAPERS]

    async def no_cited_papers(query, max_results=15):
        return []

    async def citation_counts(arxiv_ids):
        return {arxiv_id: 10 for arxiv_id in arxiv_ids}

    async def paper_text(arxiv_id, max_chars=100_000):
        return f'Full text of {arxiv_id}'

    async def tavily_search(query, max_results=5):
        return [dict(page) for page in PAGES]

    async def get_arxiv_papers(arxiv_ids):
        return [dict(ARXIV_RECORDS[arxiv_id]) for arxiv_id in arxiv_ids if arxiv_id in ARXIV_RECORDS]

    monkeypatch.setattr(arx_agent, 'search_arxiv', search_arxiv)
    monkeypatch.setattr(arx_agent, 'get_arxiv_papers', get_arxiv_papers)
    monkeypatch.setattr(
        arx_agent, 'search_cited_arxiv_papers', no_cited_papers,
    )
    monkeypatch.setattr(arx_agent, 'get_citation_counts', citation_counts)
    monkeypatch.setattr(arx_agent, 'get_arxiv_paper_text', paper_text)
    monkeypatch.setattr(search_agent, 'tavily_search', tavily_search)

    def install(**overrides: Any) -> Script:
        holder['script'] = Script(**overrides)
        return holder['script']

    return install


def run(*messages: str) -> tuple[list[tuple[str, str]], dict]:
    """Run the root agent on a conversation; returns the (author, text) of each event and the final state."""
    runner = InMemoryRunner(
        agent=Workflow(
            name='root', edges=[('START', agent_module.research)],
        ), app_name='test',
    )

    async def main():
        session = await runner.session_service.create_session(app_name='test', user_id='u')
        events = []
        for message in messages:
            content = types.Content(
                role='user', parts=[types.Part(text=message)],
            )
            async for event in runner.run_async(user_id='u', session_id=session.id, new_message=content):
                text = ''.join(
                    part.text or '' for part in (
                        event.content.parts if event.content else []
                    )
                )
                if text:
                    events.append((event.author, text))
        session = await runner.session_service.get_session(app_name='test', user_id='u', session_id=session.id)
        return events, session.state

    return asyncio.run(main())


def test_research_run_produces_answer_with_valid_references(script):
    calls = script()
    events, state = run('How do SSMs compare to transformers?')

    # One round (sufficient after reflection): 1 selected paper + 2 pages are extracted
    assert calls.count() == Counter(
        classify=1, plan=1,
        select=1, extract=3, reflect=1, answer=1,
    )
    references = events[-1][1]
    assert events[-1][0] == 'root'
    assert '[1]' in references and '[2]' in references and '[3]' in references
    assert '[9]' not in references
    assert 'Paper Two' not in references and '9999.99999' not in references
    assert state['conversation'][-1] == {
        'role': 'assistant',
        'text': DEFAULT_RESPONSES['answer'],
    }


def test_agents_receive_only_their_own_input(script):
    calls = script()
    run('How do SSMs compare to transformers?')

    [select_input] = calls.inputs('select')
    assert 'Find core SSM papers' in select_input
    assert 'Find recent coverage' not in select_input
    for extract_input in calls.inputs('extract'):
        assert extract_input.count('<source>') == 1
    [answer_input] = calls.inputs('answer')
    assert 'Claim from Paper One' in answer_input and 'Claim from Page Two' in answer_input


def test_non_research_message_gets_a_reply_without_research(script):
    calls = script(
        classify={
            'type': 'general',
            'next_message': 'I only help with research.',
        },
    )
    events, _ = run('hi!')

    assert calls.count() == Counter(classify=1)
    assert events[-1] == ('root', 'I only help with research.')


def test_follow_up_turn_sees_previous_conversation(script):
    calls = script()
    run('How do SSMs compare to transformers?', 'And what about memory usage?')

    second_classify_input = calls.inputs('classify')[1]
    assert 'User: How do SSMs compare to transformers?' in second_classify_input
    assert DEFAULT_RESPONSES['answer'] in second_classify_input
    assert second_classify_input.endswith('And what about memory usage?')


def test_reflection_runs_new_follow_up_steps_only(script):
    follow_up = {
        'sufficient': False,
        'missing': 'Recent benchmarks',
        'follow_up_steps': [
            # Already executed in round 1, up to case and spacing: must be skipped
            {
                'tool': 'web', 'query': 'State space models  vs transformers',
                'goal': 'Repeat',
            },
            {
                'tool': 'web', 'query': 'SSM long context benchmarks',
                'goal': 'Find benchmarks',
            },
        ],
    }
    calls = script(reflect=follow_up)
    events, _ = run('How do SSMs compare to transformers?')

    # Round 2 runs one web step (2 pages); no reflection after the last round
    assert calls.count() == Counter(
        classify=1, plan=1,
        select=1, extract=5, reflect=1, answer=1,
    )
    round_messages = [
        text for author, text in events if author ==
        'root' and text.startswith('Research round')
    ]
    assert len(round_messages) == 2
    assert 'SSM long context benchmarks' in round_messages[1] and 'Repeat' not in round_messages[1]


def test_plan_is_capped_to_max_steps(script, monkeypatch):
    monkeypatch.setattr(config, 'max_steps', 1)
    plan = {
        'steps': [{
            'tool': 'web', 'query': f'query {i}',
            'goal': f'goal {i}',
        } for i in range(4)],
    }
    calls = script(plan=plan)
    run('How do SSMs compare to transformers?')

    assert calls.count()['extract'] == len(PAGES)


def test_failing_tools_and_extractions_do_not_abort_the_run(script, monkeypatch):
    async def broken_search(query, max_results=30):
        raise RuntimeError('arXiv is down')
    monkeypatch.setattr(arx_agent, 'search_arxiv', broken_search)

    def extract(user_input: str) -> Any:
        if _title(user_input) == 'Page One':
            return 'not json'
        return DEFAULT_RESPONSES['extract'](user_input)

    calls = script(
        extract=extract,
        answer='Only the second page survived [1].',
    )
    events, state = run('How do SSMs compare to transformers?')

    assert calls.count()['answer'] == 1
    assert '[1] "Page Two"' in events[-1][1]
    errors = [
        step['error']
        for step in state['last_run']['steps'] if step['error']
    ]
    assert errors == ['arXiv is down']


def test_papers_found_only_by_openalex_use_arxiv_metadata(script, monkeypatch):
    async def cited_papers(query, max_results=15):
        common = {
            'abstract': 'x', 'authors': [
                'Someone',
            ], 'published': '2023-01-01', 'citations': 500,
        }
        return [
            # OpenAlex attached the title of another work to the LoRA ID
            {
                **common, 'id': '2106.09685', 'title': 'Wrong title',
                'url': 'https://arxiv.org/abs/2106.09685',
            },
            # Not confirmed by arXiv: must not become a candidate
            {
                **common, 'id': '2301.99999', 'title': 'Ghost paper',
                'url': 'https://arxiv.org/abs/2301.99999',
            },
        ]
    monkeypatch.setattr(arx_agent, 'search_cited_arxiv_papers', cited_papers)
    plan = {'steps': [DEFAULT_RESPONSES['plan']['steps'][0]]}
    calls = script(
        plan=plan, select={
            'ids': ['2106.09685'],
        }, answer='LoRA freezes the weights [1].',
    )
    events, _ = run('What is LoRA?')

    [select_input] = calls.inputs('select')
    assert 'LoRA: Low-Rank Adaptation' in select_input and '"citations": 500' in select_input
    assert 'Wrong title' not in select_input and 'Ghost paper' not in select_input
    assert '[1] Edward J. Hu, "LoRA: Low-Rank Adaptation of Large Language Models"' in events[-1][1]
