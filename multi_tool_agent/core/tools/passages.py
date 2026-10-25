import math
import re
from collections import Counter

# Common words that carry no signal for ranking
STOPWORDS = set(
    'a an and are as at be by for from has have how in into is it its of on or that the their this to was were '
    'what when which who why with does do can not than then these those there about over under also more most'.split(),
)


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r'[a-z0-9][a-z0-9\-]*', text.lower()) if term not in STOPWORDS]


def split_passages(text: str, size: int = 1500) -> list[str]:
    """
    Split a document into passages of about `size` characters, keeping paragraphs together when possible.

    Args:
        text: Document text
        size: Target passage length in characters

    Returns:
        The passages, in document order
    """
    passages: list[str] = []
    current = ''
    for paragraph in (p.strip() for p in re.split(r'\n\s*\n|\n', text)):
        if not paragraph:
            continue
        while len(paragraph) > size:
            passages.append((current + ' ' + paragraph[:size]).strip())
            current, paragraph = '', paragraph[size:]
        if len(current) + len(paragraph) > size and current:
            passages.append(current)
            current = ''
        current = f'{current}\n{paragraph}' if current else paragraph
    if current:
        passages.append(current)
    return passages


def top_passages(text: str, focus: str, limit: int = 6, size: int = 1500) -> list[tuple[int, str]]:
    """
    Rank the passages of a document by BM25 relevance to a focus query.

    Args:
        text: Document text
        focus: What the reader is looking for
        limit: Number of passages to return
        size: Passage length in characters

    Returns:
        (position, passage) pairs of the best passages, in document order
    """
    passages = split_passages(text, size)
    query = set(_terms(focus))
    if not passages:
        return []
    if not query:
        return list(enumerate(passages[:limit]))

    tokenized = [Counter(_terms(passage)) for passage in passages]
    average_length = sum(
        sum(counts.values())
        for counts in tokenized
    ) / len(tokenized) or 1
    document_frequency = Counter(
        term for counts in tokenized for term in counts if term in query
    )
    k1, b = 1.5, 0.75

    def score(counts: Counter) -> float:
        length = sum(counts.values())
        total = 0.0
        for term in query:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = math.log(
                1 +
                (
                    len(passages) -
                    document_frequency[term] + 0.5
                ) / (document_frequency[term] + 0.5),
            )
            total += idf * frequency * \
                (k1 + 1) / (frequency + k1 * (1 - b + b * length / average_length))
        return total

    ranked = sorted(
        range(len(passages)), key=lambda i: score(
            tokenized[i],
        ), reverse=True,
    )[:limit]
    return [(i, passages[i]) for i in sorted(ranked)]


def _normalize(text: str) -> str:
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def quote_in_text(quote: str, text: str, threshold: float = 0.8) -> bool:
    """
    Check that a quote comes from a text, tolerating small differences (whitespace, punctuation, a changed word).

    Args:
        quote: The quote to check
        text: The source text
        threshold: Share of the quote's 4-word sequences that must appear in the text

    Returns:
        True if the quote is (nearly) verbatim from the text
    """
    quote_norm, text_norm = _normalize(quote), _normalize(text)
    if not quote_norm:
        return False
    if quote_norm in text_norm:
        return True
    words = quote_norm.split()
    if len(words) < 4:
        return False
    shingles = [' '.join(words[i:i + 4]) for i in range(len(words) - 3)]
    found = sum(shingle in text_norm for shingle in shingles)
    return found / len(shingles) >= threshold
