from google.adk.models import FallbackModel
from google.adk.models import Gemini

from multi_tool_agent.core.agents.main.answer import build_registry
from multi_tool_agent.core.agents.main.answer import format_references
from multi_tool_agent.core.llm import build_model
from multi_tool_agent.core.models import Finding
from multi_tool_agent.core.models import ResearchStep
from multi_tool_agent.core.models import Source
from multi_tool_agent.core.models import StepResult
from multi_tool_agent.core.tools.arxiv import _html_to_text
from multi_tool_agent.core.tools.arxiv import relax_query
from multi_tool_agent.core.tools.openalex import _abstract
from multi_tool_agent.core.tools.openalex import _arxiv_id
from multi_tool_agent.core.tools.openalex import to_boolean_search

STEP = ResearchStep(tool='web', query='q', goal='g')


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
        single, Gemini,
    ) and 429 in single.retry_options.http_status_codes

    chain = build_model('gemini-flash-latest, gemini-flash-lite-latest')
    assert isinstance(chain, FallbackModel)
    primary, backup = chain.models
    assert (primary.model, backup.model) == (
        'gemini-flash-latest', 'gemini-flash-lite-latest',
    )
    # A quota error moves to the backup right away instead of being retried on the primary
    assert 429 not in primary.retry_options.http_status_codes
    assert 429 in backup.retry_options.http_status_codes
    assert 429 in chain.retriable_status_codes
