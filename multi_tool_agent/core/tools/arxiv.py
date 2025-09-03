from io import BytesIO

import feedparser
import requests
from PyPDF2 import PdfReader


def search_arxiv(query, max_results=50, start=0):
    base_url = 'http://export.arxiv.org/api/query?'
    params = f'search_query={query}&start={start}&max_results={max_results}'
    url = base_url + params

    resp = requests.get(url)
    resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    papers = []
    for entry in feed.entries:
        title = str(entry.title).strip()
        abstract = str(entry.summary).strip()
        pdf_link = next(
            (link.href for link in entry.links if link.type == 'application/pdf'), None,
        )
        arxiv_id = str(entry.id).split('/abs/')[-1]

        papers.append({
            'id': arxiv_id,
            'title': title,
            'abstract': abstract,
            'url': pdf_link,
        })
    return papers


def get_arxiv_paper(arxiv_id: str) -> tuple[str, str, dict[str, str]]:
    """Fetch the full content of an arXiv paper (PDF text) given its ID."""
    pdf_url = f'http://arxiv.org/pdf/{arxiv_id}.pdf'
    resp = requests.get(pdf_url)
    resp.raise_for_status()
    reader = PdfReader(BytesIO(resp.content))
    text = '\n'.join(page.extract_text() or '' for page in reader.pages)

    metadata = {
        'arxiv_id': arxiv_id,
        'source': 'arxiv',
        'pdf_url': f'http://arxiv.org/pdf/{arxiv_id}.pdf',
        'abs_url': f'http://arxiv.org/abs/{arxiv_id}',
        'document_type': 'research_paper',
    }
    return arxiv_id, text, metadata
