import asyncio

import pytest
from google.adk.models import BaseLlm
from google.adk.models import FallbackModel
from google.adk.models import Gemini
from google.adk.models import LlmRequest
from google.genai.errors import ClientError

from multi_tool_agent.core import llm
from multi_tool_agent.core.agents.main.answer import build_registry
from multi_tool_agent.core.agents.main.answer import format_references
from multi_tool_agent.core.agents.main.report import section_input
from multi_tool_agent.core.agents.main.verify import split_claims
from multi_tool_agent.core.agents.main.verify import strip_invalid_citations
from multi_tool_agent.core.llm import build_model
from multi_tool_agent.core.models import Finding
from multi_tool_agent.core.models import ResearchTask
from multi_tool_agent.core.models import Section
from multi_tool_agent.core.models import Source
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.core.tools import papers
from multi_tool_agent.core.tools.arxiv import _html_to_text
from multi_tool_agent.core.tools.arxiv import relax_query
from multi_tool_agent.core.tools.openalex import _abstract
from multi_tool_agent.core.tools.openalex import _arxiv_id
from multi_tool_agent.core.tools.openalex import to_boolean_search
from multi_tool_agent.core.tools.passages import quote_in_text
from multi_tool_agent.core.tools.passages import split_passages
from multi_tool_agent.core.tools.passages import top_passages

STEP = ResearchTask(objective='o', sources=['web'])


def _source(key: str) -> Source:
    return Source(key=key, kind='web', title=f'Title {key}', url=key)


def test_relax_query_drops_syntax():
    query = 'abs:"state space model" AND (cat:cs.LG OR ti:mamba)'
    assert relax_query(
        query,
    ) == 'all:state AND all:space AND all:model AND all:mamba'


def test_html_to_text_keeps_latex_and_drops_bibliography():
    html = (
        '<html><body><nav>menu</nav><article><h2>Intro</h2>'
        '<p>Loss is <math alttext="x^{2}"><mi>x</mi></math> here.</p>'
        '<section class="ltx_bibliography">[1] Some ref</section></article></body></html>'
    )
    text = _html_to_text(html)
    assert 'Intro' in text
    assert '$x^{2}$' in text
    assert 'Some ref' not in text
    assert 'menu' not in text


def test_html_without_article_returns_none():
    assert _html_to_text('<html><body>No HTML version</body></html>') is None


def test_registry_deduplicates_sources_and_claims():
    finding = Finding(claim='c1', evidence='e1')
    results = [
        StepResult(
            step=STEP, sources=[
                _source('a'), _source(
                    'b',
                ),
            ], findings={'a': [finding], 'b': [finding]},
        ),
        StepResult(
            step=STEP, sources=[_source('a')], findings={
                'a': [finding, Finding(claim='c2', evidence='e2')],
            },
        ),
    ]
    registry = build_registry(results)
    assert [source.key for source, _ in registry] == ['a', 'b']
    assert [finding.claim for finding in registry[0][1]] == ['c1', 'c2']


def test_registry_keeps_the_section_of_a_duplicate_claim():
    scout = StepResult(
        step=STEP, sources=[_source('a')], findings={
            'a': [Finding(claim='c', evidence='e')],
        },
    )
    tagged = StepResult(
        step=STEP, sources=[_source('a')], findings={'a': [Finding(claim='c', evidence='e', section='s1')]},
    )
    [(_, findings)] = build_registry([scout, tagged])
    assert [
        (finding.claim, finding.section)
        for finding in findings
    ] == [('c', 's1')]


def test_references_list_only_valid_cited_sources():
    registry = [(_source('a'), []), (_source('b'), []), (_source('c'), [])]
    references = format_references(
        'Claim [3]. Other [1, 3]. Invented [9].', registry,
    )
    assert '[1]' in references and '[3]' in references
    assert '[2]' not in references and '[9]' not in references
    assert format_references('No citations.', registry) == ''


def test_boolean_search_drops_fields_categories_and_authors():
    query = '(ti:Mamba OR abs:"state space model") AND (cat:cs.CL OR cat:cs.LG) AND au:Gu ANDNOT abs:vision'
    assert to_boolean_search(
        query,
    ) == '(Mamba OR "state space model") NOT vision'
    assert to_boolean_search('cat:cs.LG') == ''


def test_abstract_is_rebuilt_from_inverted_index():
    assert _abstract({
        'models': [1], 'Large': [0],
        'scale': [2],
    }) == 'Large models scale'
    assert _abstract(None) == ''


def test_arxiv_id_from_doi_or_locations():
    assert _arxiv_id(
        {'doi': 'https://doi.org/10.48550/arxiv.2210.17323'},
    ) == '2210.17323'
    work = {
        'doi': 'https://doi.org/10.1609/aaai.v38i12.29237',
        'locations': [{'landing_page_url': 'https://arxiv.org/abs/2306.02272v4'}],
    }
    assert _arxiv_id(work) == '2306.02272'
    assert _arxiv_id({'doi': None, 'locations': []}) is None


def test_model_chain_falls_back_on_quota_errors():
    single = build_model('gemini-flash-latest')
    assert isinstance(
        single, llm.QuotaAwareGemini,
    ) and 429 not in single.retry_options.http_status_codes

    chain = build_model('gemini-flash-latest, gemini-flash-lite-latest')
    assert isinstance(chain, FallbackModel)
    assert [model.model for model in chain.models] == [
        'gemini-flash-latest', 'gemini-flash-lite-latest',
    ]
    assert 429 in chain.retriable_status_codes


def _quota_error(quota_id: str, delay: str) -> ClientError:
    return ClientError(
        429, {
            'error': {
                'code': 429, 'details': [
                    {
                        '@type': 'type.googleapis.com/google.rpc.QuotaFailure',
                        'violations': [{'quotaId': quota_id}],
                    },
                    {
                        '@type': 'type.googleapis.com/google.rpc.RetryInfo',
                        'retryDelay': delay,
                    },
                ],
            },
        },
    )


def _drain(model: BaseLlm) -> list:
    async def main():
        return [response async for response in model.generate_content_async(LlmRequest())]
    return asyncio.run(main())


def test_rate_limits_are_retried_and_daily_quotas_skipped(monkeypatch):
    calls = []
    errors: list = [
        _quota_error(
            'GenerateRequestsPerMinutePerProjectPerModel-FreeTier', '7s',
        ),
    ]

    async def generate(self, llm_request, stream=False):
        calls.append(self.model)
        if errors:
            raise errors.pop(0)
        yield 'ok'

    async def no_sleep(seconds):
        calls.append(f'sleep {seconds:.0f}')

    monkeypatch.setattr(Gemini, 'generate_content_async', generate)
    monkeypatch.setattr(llm.asyncio, 'sleep', no_sleep)
    monkeypatch.setattr(llm, '_exhausted_until', {})
    model = build_model('model-a')

    # Per-minute limit: wait the suggested delay, then retry the same model
    assert _drain(model) == ['ok']
    assert calls == ['model-a', 'sleep 7', 'model-a']

    # Daily quota: fail at once, and do not call the API again while it lasts
    calls.clear()
    errors.append(
        _quota_error(
            'GenerateRequestsPerDayPerProjectPerModel-FreeTier', '3600s',
        ),
    )
    with pytest.raises(ClientError):
        _drain(model)
    with pytest.raises(ClientError):
        _drain(model)
    assert calls == ['model-a']


def test_passages_keep_paragraphs_and_rank_by_relevance():
    text = '\n'.join(
        ['Intro about cooking pasta.'] * 3 +
        ['Speculative decoding uses a draft model to propose tokens.']
        + ['Unrelated closing remarks.'] * 3,
    )
    passages = split_passages(text, size=60)
    assert all(len(passage) <= 120 for passage in passages)
    [(_, best)] = top_passages(
        text, 'how does speculative decoding work', limit=1, size=60,
    )
    assert 'draft model' in best


def test_quote_check_tolerates_formatting_but_not_fabrication():
    text = 'Our method reduces latency by 2.3x on A100 GPUs, compared to standard autoregressive decoding.'
    assert quote_in_text('reduces latency by 2.3x on A100 GPUs', text)
    assert quote_in_text(
        'Our method reduces latency by 2.3x on A100 GPUs, compared to standard  decoding.', text,
    )
    assert not quote_in_text('reduces latency by 5x on TPUs', text)
    assert not quote_in_text('', text)


def test_papers_found_only_by_openalex_use_arxiv_metadata(monkeypatch):
    lora = {
        'id': '2106.09685', 'title': 'LoRA: Low-Rank Adaptation of Large Language Models', 'abstract': 'LoRA.',
        'authors': ['Edward J. Hu'], 'published': '2021-06-17', 'url': 'https://arxiv.org/abs/2106.09685',
    }

    async def get_arxiv_papers(arxiv_ids):
        return [dict(lora)] if '2106.09685' in arxiv_ids else []

    async def get_citation_counts(arxiv_ids):
        return {arxiv_id: 3 for arxiv_id in arxiv_ids}

    monkeypatch.setattr(papers, 'get_arxiv_papers', get_arxiv_papers)
    monkeypatch.setattr(papers, 'get_citation_counts', get_citation_counts)
    common = {
        'abstract': 'x', 'authors': [
            'Someone',
        ], 'published': '2023-01-01', 'citations': 500,
    }
    cited = [
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
    other = {**lora, 'id': '2401.00001', 'title': 'Other'}
    merged = asyncio.run(papers.merge_with_citations([dict(other)], cited))

    assert [(paper['id'], paper['title'], paper['citations']) for paper in merged] == [
        ('2401.00001', 'Other', 3),
        ('2106.09685', 'LoRA: Low-Rank Adaptation of Large Language Models', 500),
    ]


def test_section_writer_gets_its_findings_and_the_unassigned_ones():
    registry = [
        (_source('a'), [Finding(claim='own', evidence='e', section='s1')]),
        (
            _source('b'), [
                Finding(claim='other section', evidence='e', section='s2'),
            ],
        ),
        (_source('c'), [Finding(claim='unassigned', evidence='e')]),
    ]
    section = Section(id='s1', title='T', questions=['Q?'])
    text = section_input('msg', 'question', section, registry, {'s1', 's2'})
    own, shared = text.split('# Findings for this section')[
        1
    ].split('# Other findings')
    # Citation numbers stay the global ones
    assert '[1] "Title a"' in own and 'own' in own
    assert 'other section' not in text
    assert '[3] "Title c"' in shared and 'unassigned' in shared


def test_claims_are_split_into_cited_sentences_items_and_rows():
    text = (
        '# Title [1]\n'
        'First claim [1]. Second claim [2, 3]! No citation here. Third [1].\n'
        '- An item [2]\n'
        '| Method | Speed |\n'
        '| A | 2x [3] |\n'
    )
    assert split_claims(text) == [
        'First claim [1].', 'Second claim [2, 3]!', 'Third [1].', '- An item [2]', '| A | 2x [3] |',
    ]


def test_invalid_citations_are_stripped():
    assert strip_invalid_citations(
        'A [1, 7]. B [9]. C [2].', 2,
    ) == ('A [1]. B. C [2].', 2)
