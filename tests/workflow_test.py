"""End-to-end tests of the research workflow with a scripted model and stubbed search tools: no API calls."""
import asyncio
import json
import re
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
from multi_tool_agent.core.tools import arxiv
from multi_tool_agent.core.tools import papers
from multi_tool_agent.core.tools import tavily
from multi_tool_agent.utils.config import config

# Each agent is recognized by a phrase of its system instruction
ROLES = {
    'Request Classification Agent': 'classify',
    'Research Planning Agent': 'plan',
    'You are a Research Agent': 'research',
    'Research Reflection Agent': 'reflect',
    'Synthesize the research findings': 'answer',
}

PAPERS = [
    {
        'id': '2401.00001', 'title': 'Paper One', 'abstract': 'About SSMs.', 'authors': ['Ada Lovelace'],
        'published': '2024-01-01', 'url': 'https://arxiv.org/abs/2401.00001', 'citations': 10,
    },
]
PAPER_TEXT = 'Introduction.\nState space models scale linearly with the sequence length of the input.\nConclusion.'
PAGES = [
    {
        'title': 'Page One', 'url': 'https://example.com/1', 'published': '2026-01-01',
        'content': 'Transformers remain the dominant architecture for language models in 2026.',
    },
    {
        'title': 'Page Two', 'url': 'https://example.com/2', 'published': '',
        'content': 'Hybrid models mix attention layers with state space layers.',
    },
]

PLAN = {
    'outline': [
        {
            'id': 's1', 'title': 'How SSMs work',
            'questions': ['What makes SSMs efficient?'],
        },
        {
            'id': 's2', 'title': 'Current landscape', 'questions': [
                'Which architectures dominate today?',
            ],
        },
    ],
    'quality_constraints': ['Cites primary sources'],
    'tasks': [
        {
            'objective': 'Find core SSM papers', 'sources': [
                'papers',
            ], 'lens': 'evidence', 'section_ids': ['s1'],
        },
        {
            'objective': 'Find recent coverage', 'sources': [
                'web',
            ], 'lens': 'horizon', 'section_ids': ['s2'],
        },
    ],
}
WEB_ONLY_PLAN = {**PLAN, 'tasks': [PLAN['tasks'][1]]}
SCOUT = 'Survey the landscape'


def function_responses(request: LlmRequest) -> list[dict]:
    return [
        part.function_response.response
        for content in request.contents for part in content.parts or [] if part.function_response
    ]


def tool_names(request: LlmRequest) -> set[str]:
    return {
        declaration.name
        for tool in request.config.tools or [] for declaration in tool.function_declarations or []
    }


def objective_of(request: LlmRequest) -> str:
    return request.contents[0].parts[0].text.split('Your objective: ')[1].split('\n')[0]


def sections_of(request: LlmRequest) -> list[str]:
    """The section ids listed in a researcher's instruction."""
    instruction = str(request.config.system_instruction)
    if 'Report sections you gather evidence for' not in instruction:
        return []
    block = instruction.split('Report sections you gather evidence for', 1)[
        1
    ].split('**Tools**')[0]
    return re.findall(r'`(s\d+)`', block)


def call(name: str, **args: Any) -> types.FunctionCall:
    return types.FunctionCall(name=name, args=args)


def read_source_id(request: LlmRequest) -> str:
    calls = [
        part.function_call for content in request.contents for part in content.parts or []
        if part.function_call and part.function_call.name == 'read_source'
    ]
    return calls[-1].args['source_id']


def researcher(request: LlmRequest) -> Any:
    """Search, read the first result, record a finding quoting it (for its first section), then write the brief."""
    responses = function_responses(request)
    tools = tool_names(request)
    if not responses:
        search = 'search_papers' if 'search_papers' in tools else 'web_search'
        return call(search, query='state space models')
    last = responses[-1]
    if last.get('results'):
        return call('read_source', source_id=last['results'][0]['source_id'], focus='state space models')
    if 'passages' in last:
        quote = max(
            last['passages'][0].split('] ', 1)[
                1
            ].splitlines(), key=len,
        )
        return call(
            'record_finding', source_id=read_source_id(request), claim=f'Claim from {last["title"]}', evidence=quote,
            section_id=(sections_of(request) or [''])[0],
        )
    return 'Brief: done.'


DEFAULT_RESPONSES: dict[str, Any] = {
    'classify': {
        'type': 'valid', 'complexity': 'moderate',
        'user_intent': 'How do state space models compare to transformers?',
    },
    'plan': PLAN,
    'research': researcher,
    'reflect': {'sufficient': True, 'missing': '', 'follow_up_tasks': []},
    'answer': 'SSMs scale linearly [1]. Transformers dominate [2]. Invented [9].',
}
SIMPLE = {**DEFAULT_RESPONSES['classify'], 'complexity': 'simple'}


class Script:
    """Answers each model call according to the role of the calling agent, and records the calls."""

    def __init__(self, **overrides: Any) -> None:
        self.responses = {**DEFAULT_RESPONSES, **overrides}
        self.calls: list[tuple[str, LlmRequest]] = []

    def __call__(self, request: LlmRequest) -> list[types.Part]:
        system_instruction = str(request.config.system_instruction or '')
        role = next(
            role for phrase, role in ROLES.items()
            if phrase in system_instruction
        )
        self.calls.append((role, request))
        response = self.responses[role]
        if callable(response):
            response = response(request)
        if isinstance(response, types.FunctionCall):
            return [types.Part(function_call=response)]
        if isinstance(response, list):
            return [types.Part(function_call=function_call) for function_call in response]
        return [types.Part(text=response if isinstance(response, str) else json.dumps(response))]

    def count(self) -> Counter:
        return Counter(role for role, _ in self.calls)

    def requests(self, role: str) -> list[LlmRequest]:
        return [request for call_role, request in self.calls if call_role == role]

    def inputs(self, role: str) -> list[str]:
        """The first user message of each call of a role."""
        return [request.contents[0].parts[0].text for request in self.requests(role)]


class ScriptedLlm(BaseLlm):
    """Model that delegates every request to the active Script."""

    script: Any = None

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False,
    ) -> AsyncGenerator[LlmResponse, None]:
        yield LlmResponse(content=types.Content(role='model', parts=self.script(llm_request)))


@pytest.fixture
def script(monkeypatch):
    """Replace the model and the search tools; returns a function that installs a Script."""
    holder: dict[str, Script] = {}
    monkeypatch.setattr(
        llm, 'build_model', lambda model: ScriptedLlm(
            model=model, script=holder['script'],
        ),
    )

    async def search_papers(query):
        return [dict(paper) for paper in PAPERS]

    async def paper_text(arxiv_id, max_chars=100_000):
        return PAPER_TEXT

    async def tavily_search(query, max_results=5):
        return [dict(page) for page in PAGES]

    monkeypatch.setattr(papers, 'search_papers', search_papers)
    monkeypatch.setattr(arxiv, 'get_arxiv_paper_text', paper_text)
    monkeypatch.setattr(tavily, 'tavily_search', tavily_search)

    def install(**overrides: Any) -> Script:
        holder['script'] = Script(**overrides)
        return holder['script']

    return install


def run(*messages: str) -> tuple[list[tuple[str, str]], dict]:
    """Run the workflow on a conversation; returns the (author, text) of each event and the final state."""
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


def references(events: list[tuple[str, str]]) -> str:
    author, text = events[-1]
    assert author == 'root'
    return text


def test_research_run_produces_answer_with_valid_references(script):
    calls = script()
    events, state = run('How do SSMs compare to transformers?')

    # The scout and each researcher: search, read, record, brief
    assert calls.count() == Counter(
        classify=1, plan=1,
        research=12, reflect=1, answer=1,
    )
    text = references(events)
    assert '[1]' in text and '[2]' in text and '[9]' not in text
    assert 'Ada Lovelace, "Paper One"' in text
    assert state['conversation'][-1] == {
        'role': 'assistant',
        'text': DEFAULT_RESPONSES['answer'],
    }
    steps = state['last_run']['steps']
    assert [
        step['step']['lens']
        for step in steps
    ] == ['scout', 'evidence', 'horizon']
    assert all(step['brief'] == 'Brief: done.' for step in steps)


def test_scout_surveys_before_planning(script):
    calls = script()
    run('How do SSMs compare to transformers?')

    roles = [role for role, _ in calls.calls]
    assert roles.index('plan') > max(
        i for i, role in enumerate(roles) if role == 'research' and
        SCOUT in objective_of(calls.calls[i][1])
    )
    [plan_input] = calls.inputs('plan')
    assert '# Scout survey' in plan_input and 'Brief: done.' in plan_input and 'Claim from Paper One' in plan_input
    scout_requests = [
        r for r in calls.requests(
            'research',
        ) if SCOUT in objective_of(r)
    ]
    assert tool_names(scout_requests[0]) == {
        'search_papers', 'web_search', 'read_source', 'record_finding', 'think',
    }


def test_simple_questions_skip_the_scout(script):
    calls = script(classify=SIMPLE)
    run('Who won the 2026 Nobel Prize in Physics?')

    assert not any(
        SCOUT in objective_of(r)
        for r in calls.requests('research')
    )
    assert '# Scout survey' not in calls.inputs('plan')[0]


def test_researchers_get_their_own_task_sections_and_tools(script):
    calls = script()
    run('How do SSMs compare to transformers?')

    first_calls = {
        objective_of(request): request for request in calls.requests('research')
        if not function_responses(request) and SCOUT not in objective_of(request)
    }
    assert set(first_calls) == {'Find core SSM papers', 'Find recent coverage'}
    common = {'read_source', 'record_finding', 'think'}
    assert tool_names(first_calls['Find core SSM papers']) == {
        'search_papers', *common,
    }
    assert tool_names(first_calls['Find recent coverage']) == {
        'web_search', *common,
    }
    assert sections_of(first_calls['Find core SSM papers']) == ['s1']
    assert sections_of(first_calls['Find recent coverage']) == ['s2']
    # A researcher never sees another researcher's tool calls
    for request in calls.requests('research'):
        if SCOUT not in objective_of(request):
            seen = json.dumps(function_responses(request))
            assert not ('Paper One' in seen and 'Page One' in seen)
    [answer_input] = calls.inputs('answer')
    assert 'Claim from Paper One' in answer_input and 'Claim from Page One' in answer_input
    assert 'scale linearly with the sequence length' in answer_input


def test_findings_are_tagged_with_sections_and_coverage_is_checked(script):
    follow_up = {'sufficient': False, 'missing': 'x', 'follow_up_tasks': []}
    calls = script(reflect=follow_up)
    _, state = run('How do SSMs compare to transformers?')

    steps = {step['step']['objective']: step for step in state['last_run']['steps']}
    [paper_findings] = steps['Find core SSM papers']['findings'].values()
    assert [finding['section'] for finding in paper_findings] == ['s1']
    [reflect_input] = calls.inputs('reflect')
    assert '`s1` How SSMs work: 1 findings' in reflect_input
    assert '`s2` Current landscape: 1 findings' in reflect_input
    assert 'Cites primary sources' in reflect_input


def test_unknown_sections_are_rejected(script):
    def wrong_section(request):
        response = researcher(request)
        if isinstance(response, types.FunctionCall) and response.name == 'record_finding' and sections_of(request):
            return call('record_finding', **{**response.args, 'section_id': 's9'})
        return response

    calls = script(
        classify=SIMPLE, plan=WEB_ONLY_PLAN,
        research=wrong_section, answer='None.',
    )
    _, state = run('How do SSMs compare to transformers?')

    assert not state['last_run']['steps'][0]['findings']
    errors = [
        r['error'] for request in calls.requests(
            'research',
        ) for r in function_responses(request) if 'error' in r
    ]
    assert errors and 'Unknown section_id' in errors[0]


def test_reflection_can_revise_the_outline(script):
    revised = [{'id': 's1', 'title': 'Mechanics', 'questions': ['How?']}]
    script(
        reflect={
            'sufficient': True,
            'missing': '', 'revised_outline': revised,
        },
    )
    _, state = run('How do SSMs compare to transformers?')

    assert state['last_run']['plan']['outline'] == revised


def test_findings_need_verbatim_evidence(script):
    def fabricating_researcher(request):
        response = researcher(request)
        if isinstance(response, types.FunctionCall) and response.name == 'record_finding':
            return call('record_finding', source_id=response.args['source_id'], claim='x', evidence='made up words here')
        return response

    calls = script(research=fabricating_researcher, answer='Nothing found.')
    _, state = run('How do SSMs compare to transformers?')

    assert all(not step['findings'] for step in state['last_run']['steps'])
    rejected = [
        r for request in calls.requests(
            'research',
        ) for r in function_responses(request) if 'error' in r
    ]
    assert rejected and 'not a verbatim quote' in rejected[0]['error']


def test_search_budget_is_enforced(script, monkeypatch):
    monkeypatch.setattr(config, 'researcher_tool_calls', 2)
    searches = []

    async def tavily_search(query, max_results=5):
        searches.append(query)
        return [dict(page) for page in PAGES]
    monkeypatch.setattr(tavily, 'tavily_search', tavily_search)

    def endless_searcher(request):
        if 'web_search' in tool_names(request):
            return call('web_search', query=f'query {len(function_responses(request))}')
        return 'Brief: out of budget.'

    script(
        classify=SIMPLE, plan=WEB_ONLY_PLAN,
        research=endless_searcher, answer='Nothing.',
    )
    _, state = run('How do SSMs compare to transformers?')

    assert len(searches) == 2
    assert state['last_run']['steps'][0]['brief'] == 'Brief: out of budget.'


def test_budget_holds_for_parallel_tool_calls(script, monkeypatch):
    monkeypatch.setattr(config, 'researcher_tool_calls', 2)
    searches = []

    async def tavily_search(query, max_results=5):
        searches.append(query)
        return []
    monkeypatch.setattr(tavily, 'tavily_search', tavily_search)

    def parallel_searcher(request):
        if not function_responses(request):
            return [call('web_search', query=f'query {i}') for i in range(4)]
        return 'Brief.'

    script(
        classify=SIMPLE, plan=WEB_ONLY_PLAN,
        research=parallel_searcher, answer='None.',
    )
    run('How do SSMs compare to transformers?')

    assert len(searches) == 2


def test_simple_questions_get_a_smaller_budget(script, monkeypatch):
    searches = []

    async def tavily_search(query, max_results=5):
        searches.append(query)
        return []
    monkeypatch.setattr(tavily, 'tavily_search', tavily_search)

    def endless_searcher(request):
        return call('web_search', query='q') if 'web_search' in tool_names(request) else 'Brief.'

    script(
        classify=SIMPLE, plan=WEB_ONLY_PLAN,
        research=endless_searcher, answer='None.',
    )
    run('Who won the 2026 Nobel Prize in Physics?')

    assert len(searches) == 5


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
    calls = script(classify=SIMPLE)
    run('How do SSMs compare to transformers?', 'And what about memory usage?')

    second_classify_input = calls.inputs('classify')[1]
    assert 'User: How do SSMs compare to transformers?' in second_classify_input
    assert DEFAULT_RESPONSES['answer'] in second_classify_input
    assert second_classify_input.endswith('And what about memory usage?')


def test_reflection_runs_new_follow_up_tasks_only(script):
    follow_up = {
        'sufficient': False,
        'missing': 'Recent benchmarks',
        'follow_up_tasks': [
            # Already done in round 1, up to case and spacing: must be skipped
            {'objective': 'find recent  coverage', 'sources': ['web']},
            {
                'objective': 'Find long context benchmarks',
                'sources': ['web'], 'lens': 'comparator',
            },
        ],
    }
    calls = script(reflect=follow_up)
    events, _ = run('How do SSMs compare to transformers?')

    # Round 2 runs one researcher; no reflection after the last round
    assert calls.count() == Counter(
        classify=1, plan=1,
        research=16, reflect=1, answer=1,
    )
    rounds = [
        text for author, text in events if author ==
        'root' and text.startswith('Research round')
    ]
    assert len(rounds) == 2
    assert 'Find long context benchmarks' in rounds[1] and 'recent' not in rounds[1]
    [reflect_input] = calls.inputs('reflect')
    assert 'Researcher brief: Brief: done.' in reflect_input and 'Claim from Paper One' in reflect_input


def test_plan_is_capped_to_max_steps(script, monkeypatch):
    monkeypatch.setattr(config, 'max_steps', 1)
    plan = {
        **PLAN, 'tasks': [{
            'objective': f'objective {i}',
            'sources': ['web'],
        } for i in range(4)],
    }
    calls = script(classify=SIMPLE, plan=plan)
    run('How do SSMs compare to transformers?')

    assert calls.count()['research'] == 4


def test_failing_tools_are_reported_to_the_researcher(script, monkeypatch):
    async def broken_search(query):
        raise RuntimeError('arXiv is down')
    monkeypatch.setattr(papers, 'search_papers', broken_search)

    calls = script(classify=SIMPLE, answer='Only the web survived [1].')
    events, state = run('How do SSMs compare to transformers?')

    papers_step, web_step = state['last_run']['steps']
    assert not papers_step['findings'] and papers_step['error'] is None
    assert web_step['findings']
    errors = [
        r['error'] for request in calls.requests(
            'research',
        ) for r in function_responses(request) if 'error' in r
    ]
    assert any('arXiv is down' in error for error in errors)
    assert '[1] "Page One"' in references(events)


def test_failing_researcher_keeps_its_findings(script):
    def crashing_researcher(request):
        if any(response.get('recorded') for response in function_responses(request)):
            raise RuntimeError('model crashed')
        return researcher(request)

    script(
        classify=SIMPLE, plan=WEB_ONLY_PLAN,
        research=crashing_researcher, answer='Kept [1].',
    )
    events, state = run('How do SSMs compare to transformers?')

    [step] = state['last_run']['steps']
    assert step['error'] and step['findings']
    assert '[1] "Page One"' in references(events)
