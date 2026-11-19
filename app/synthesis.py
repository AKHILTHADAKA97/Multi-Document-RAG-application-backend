"""Grounded local answer synthesis used when no LLM provider is configured.

This is deliberately extractive: it selects and orders sentences from the
retrieved passages, adds citations and a readable structure. It never invents
content beyond the uploaded documents, which keeps the same grounding contract
as the LLM path.
"""
import re
from collections import defaultdict

CONTENT_STOP = set("""a an the is are was were be been being am do does did have has had
what which who whom whose how when where why can could would should shall will may might
of for in on at to from with by about into over under between and or but not no yes
it its this that these those there their them they he she his her you your i we our
me my mine us please tell explain describe define give summarize summarise summary
document documents pdf pdfs file files text information mentioned according says say
within also more most much many few some any main key overview""".split())

ACRONYMS = {
    'ml': 'machine learning',
    'ai': 'artificial intelligence',
    'dl': 'deep learning',
    'rag': 'retrieval augmented generation',
    'nlp': 'natural language processing',
    'llm': 'large language model',
    'llms': 'large language models',
    'nn': 'neural network',
}

SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9(\["\'“])')
DEFINITION_START = re.compile(
    r'^\W*(?:the\s+)?[A-Z][^.]{2,70}?\s+(?:is|are|refers to|means|is defined as|are defined as)\b')
EXAMPLE_MARKERS = re.compile(r'\b(for example|for instance|e\.g\.|such as|example:|an? example of|examples? of|is an example)\b', re.IGNORECASE)
EMAIL_OR_URL = re.compile(r'^\s*(https?://|www\.)\S*\s*$|^\s*page\s+\d+\s*$', re.IGNORECASE)


def _stem(token: str) -> str:
    token = token.lower()
    if len(token) > 4 and token.endswith('ies'):
        return token[:-3] + 'y'
    if len(token) > 4 and token.endswith('ing'):
        return token[:-3]
    if len(token) > 3 and token.endswith('es') and token[-3] in 'sxzh':
        return token[:-2]
    if len(token) > 3 and token.endswith('s') and not token.endswith('ss'):
        return token[:-1]
    return token


def terms(text: str):
    return {_stem(t) for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", text.lower())
            if t not in CONTENT_STOP and (len(t) > 1 or t.isdigit())}


def question_terms(question: str):
    result = terms(question)
    for token in list(re.findall(r'\b[A-Za-z]+\b', question.lower())):
        if token in ACRONYMS:
            result |= terms(ACRONYMS[token])
    return result


def _clean_sentence(sentence: str) -> str:
    sentence = re.sub(r'^[\s•▪◦●\-–—*\d.)\]]+', '', sentence.strip())
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence.strip(' |')


def sentences(text: str):
    text = (text or '').replace('\r', '\n')
    parts = []
    for line in text.split('\n'):
        line = re.sub(r'[ \t]+', ' ', line).strip()
        if not line:
            continue
        parts.extend(p.strip() for p in SENTENCE_END.split(line) if p.strip())
    return parts


def _essence(text: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', text.lower())


def _heading_like(sentence: str) -> bool:
    words = sentence.split()
    if len(words) > 5 or re.search(r'\d', sentence):
        return False
    return bool(sentence.endswith('.') and not re.search(
        r'\b(is|are|was|were|means|refers|has|have|can|will|uses|includes)\b', sentence, re.IGNORECASE))


ORDINALS = {'first': 0, '1st': 0, 'second': 1, '2nd': 1, 'third': 2, '3rd': 2,
            'fourth': 3, '4th': 3, 'fifth': 4, '5th': 4, 'last': -1}


def detect_intent(question: str):
    q = (question or '').lower().strip()
    page = re.search(r'\bpage\s+(\d{1,4})\b', q)
    if re.search(r'\b(summar|overview|what is this (pdf|document|file) about|key points|tl;?dr)', q):
        return 'summary', int(page.group(1)) if page else None
    if page:
        return 'page', int(page.group(1))
    if re.search(r'\b(give|show|provide|an?|one)\s+example|\bexamples?\s+of\b', q):
        return 'example', None
    if re.search(r'\b(difference|compare|comparison|versus|vs\.?)\b', q):
        return 'comparison', None
    if re.search(r'\b(list|types|kinds|categories|what are the)\b', q):
        return 'list', None
    if re.search(r'\b(how (does|do|is|are)|steps|process|workflow|procedure)\b', q):
        return 'howto', None
    if re.search(r'\b(explain in detail|detailed|in depth|elaborate)\b', q):
        return 'detail', None
    if re.search(r'\b(what is|what are|define|definition|meaning of|explain)\b', q):
        return 'definition', None
    return 'general', None


def _topic(question: str) -> str:
    topic = re.sub(r'[?!.]+$', '', question.strip())
    topic = re.sub(r'^(what|who|which|where|when|why|how)\s+(is|are|was|were|does|do|did|can|would|should)\s+', '', topic, flags=re.IGNORECASE)
    topic = re.sub(r'^(explain|define|describe|tell me about|summarize|summarise|list)\s+(the\s+)?', '', topic, flags=re.IGNORECASE)
    topic = re.sub(r'^(a|an|the)\s+', '', topic, flags=re.IGNORECASE)
    topic = re.sub(r'\s+(in|from)\s+page\s+\d+.*$', '', topic, flags=re.IGNORECASE)
    topic = topic.strip(' :|')
    if not topic:
        return 'Document answer'
    return topic[0].upper() + topic[1:]


def _score_sentence(sentence, qterms, similarity, position, intent):
    sterms = terms(sentence)
    if not sterms:
        return 0.0, 0.0
    overlap = qterms & sterms
    coverage = len(overlap) / max(len(qterms), 1)
    score = coverage * 3.0 + min(len(overlap), 6) * 0.45 + similarity * 0.65
    if intent in {'definition', 'detail'} and DEFINITION_START.match(sentence):
        score += 1.1
    if EXAMPLE_MARKERS.search(sentence) and intent == 'example':
        score += 0.4
    score -= position * 0.012
    return score, coverage


def _collect(question, sources, intent):
    qterms = question_terms(question)
    scored = []
    seen = set()
    for source in sources:
        similarity = float(source.get('similarity') or 0)
        for position, raw in enumerate(sentences(source.get('text', ''))):
            sentence = _clean_sentence(raw)
            if len(sentence.split()) < 4 or EMAIL_OR_URL.search(sentence) or _heading_like(sentence):
                continue
            essence = _essence(sentence)
            if not essence or essence in seen:
                continue
            seen.add(essence)
            score, coverage = _score_sentence(sentence, qterms, similarity, position, intent)
            scored.append({
                'sentence': sentence,
                'source': source,
                'score': score,
                'coverage': coverage,
                'has_terms': bool(qterms & terms(sentence)),
                'position': position,
            })
    scored.sort(key=lambda item: item['score'], reverse=True)
    return scored, qterms


def _citation_items(used):
    items, seen = [], set()
    for item in used:
        source = item['source']
        key = source.get('citation') or source.get('document_id')
        if key and key not in seen:
            seen.add(key)
            items.append(source)
    return items


def _sentence_with_cite(item):
    citation = item['source'].get('citation')
    return f"{item['sentence']}" + (f" [{citation}]" if citation else '')


def _source_line(source):
    name = source.get('name') or source.get('filename') or 'document'
    location = source.get('location') or ''
    if source.get('page_number'):
        location = f"Page {source['page_number']}"
    return f"- {name} — {location}".rstrip(' —')


def _pick(scored, limit, minimum_score):
    chosen = []
    for item in scored:
        if len(chosen) >= limit:
            break
        if item['score'] >= minimum_score:
            chosen.append(item)
    return chosen


def compose(question, sources, context=''):
    """Return a structured grounded answer or None when evidence is insufficient."""
    if not sources:
        return None
    intent, page = detect_intent(question)
    if intent == 'summary':
        return _summarize(question, sources)
    if intent == 'page':
        return _page_answer(page, sources)

    scoring_question = f'{context} {question}'.strip() if context else question
    scored, qterms = _collect(scoring_question, sources, intent)
    if not scored:
        return None

    ordinal = None
    for word, index in ORDINALS.items():
        if re.search(rf'\b{re.escape(word)}\b', question, re.IGNORECASE):
            ordinal = index
            break
    if ordinal is not None and re.search(r'\b(type|kind|item|one|algorithm|category)\b', question, re.IGNORECASE):
        ordinal_result = _ordinal_answer(question, sources, scored, ordinal)
        if ordinal_result:
            return ordinal_result

    if intent == 'list':
        list_result = _list_answer(question, sources, scored, qterms=qterms)
        if list_result:
            return list_result

    if intent == 'definition':
        def_result = _definition_answer(question, sources, scored, qterms)
        if def_result:
            return def_result

    best = scored[0]
    if not best['has_terms'] and best['coverage'] == 0:
        return None
    if best['coverage'] < 0.12 and best['score'] < 1.1:
        return None

    limit = {'detail': 6, 'comparison': 5, 'howto': 5, 'page': 5, 'example': 4, 'general': 4}.get(intent, 4)
    chosen = _pick(scored, limit, max(best['score'] * 0.42, 0.35))
    if not chosen:
        chosen = [best]

    # Directly answer in the first sentence
    lead_paragraph = [_sentence_with_cite(chosen[0])]
    used_items = [chosen[0]]

    # If there is a complementary supporting sentence, combine into paragraph
    for item in chosen[1:]:
        if len(lead_paragraph) < 2 and _essence(item['sentence']) != _essence(chosen[0]['sentence']):
            lead_paragraph.append(_sentence_with_cite(item))
            used_items.append(item)

    paragraphs = [' '.join(lead_paragraph)]

    # Only multi-point detailed or comparison questions include structured supporting items
    rest = [item for item in chosen if item not in used_items]
    if rest and intent in {'detail', 'howto', 'comparison'}:
        supporting = []
        for item in rest[:3]:
            supporting.append(f"- {_sentence_with_cite(item)}")
            used_items.append(item)
        if supporting:
            paragraphs.append('\n'.join(supporting))

    examples = [item for item in scored if item not in used_items and (
        EXAMPLE_MARKERS.search(item['sentence']) or any(term in terms(item['sentence']) for term in qterms))][:1]
    if examples and intent in {'example', 'detail', 'howto'}:
        paragraphs.append(_sentence_with_cite(examples[0]))
        used_items.append(examples[0])

    used = _citation_items(used_items)
    coverage = best['coverage']
    confidence = 'high' if coverage >= 0.6 or len(used_items) >= 3 else ('medium' if coverage >= 0.3 else 'low')
    return {
        'answer': '\n\n'.join(paragraphs).strip(),
        'citations': [source['citation'] for source in used if source.get('citation')],
        'status': 'answered',
        'answer_type': 'document_based',
        'confidence': confidence,
        'usage': None,
    }


def _definition_answer(question, sources, scored, qterms):
    """Provide a clear definition in the first sentence, how it works, and one relevant example."""
    if not scored:
        return None

    best = scored[0]
    if not best['has_terms'] and best['coverage'] == 0:
        return None
    if best['coverage'] < 0.12 and best['score'] < 1.1:
        return None

    # 1. Clear definition in the first sentence
    def_item = None
    for item in scored:
        sent = item['sentence']
        if DEFINITION_START.match(sent) or re.search(r'\b(?:is a branch of|is a subset of|is the field of|is defined as|refers to|means)\b', sent, re.I):
            def_item = item
            break
    if not def_item:
        def_item = best

    # 2. Short explanation of how it works, if supported
    exp_item = None
    exp_markers = re.compile(r'\b(trains?|learns?|learning|uses?|finds?|process|operates?|functions?|neural|networks?|data|patterns?|structure)\b', re.I)
    for item in scored:
        if item is not def_item and not EXAMPLE_MARKERS.search(item['sentence']):
            if exp_markers.search(item['sentence']):
                exp_item = item
                break

    # 3. One relevant example, if supported
    ex_item = None
    for item in scored:
        if item is not def_item and item is not exp_item:
            if EXAMPLE_MARKERS.search(item['sentence']) or re.search(r'\b(example|instance|filter|recognition|translation)\b', item['sentence'], re.I):
                ex_item = item
                break

    used_items = [def_item]
    p1 = [_sentence_with_cite(def_item)]
    if exp_item and _essence(exp_item['sentence']) != _essence(def_item['sentence']):
        p1.append(_sentence_with_cite(exp_item))
        used_items.append(exp_item)

    paragraphs = [' '.join(p1)]

    if ex_item and _essence(ex_item['sentence']) != _essence(def_item['sentence']):
        paragraphs.append(_sentence_with_cite(ex_item))
        used_items.append(ex_item)

    used = _citation_items(used_items)
    coverage = def_item['coverage']
    confidence = 'high' if coverage >= 0.5 or len(used_items) >= 2 else ('medium' if coverage >= 0.25 else 'low')
    return {
        'answer': '\n\n'.join(paragraphs).strip(),
        'citations': [source['citation'] for source in used if source.get('citation')],
        'status': 'answered',
        'answer_type': 'document_based',
        'confidence': confidence,
        'usage': None,
    }


def _list_items(sources, preferred=None):
    ordered = ([preferred] if preferred else []) + [s for s in sources if s is not preferred]
    items, seen = [], set()
    for source in ordered:
        for raw in sentences(source.get('text', '')):
            sentence = _clean_sentence(raw)
            if len(sentence.split()) < 5 or EMAIL_OR_URL.search(sentence) or _heading_like(sentence):
                continue
            essence = _essence(sentence)
            if essence and essence not in seen:
                seen.add(essence)
                items.append((sentence, source))
    return items


def _best_list_source(sources, qterms=None):
    def score(source):
        sterms = terms(source.get('text', ''))
        overlap = len(sterms & qterms) if qterms else 0
        c = sum(1 for raw in sentences(source.get('text', ''))
                if len(_clean_sentence(raw).split()) >= 5 and not _heading_like(_clean_sentence(raw)))
        return (overlap, c)
    return max(sources, key=score)


def _list_answer(question, sources, scored, qterms=None):
    items = _list_items([_best_list_source(sources, qterms)])[:8]
    if len(items) < 3:
        items = _list_items(sources)[:8]
    if not items:
        return None
    used, lines = [], [f"### {_topic(question)}", '']
    for sentence, source in items:
        lines.append(f"- {sentence}" + (f" [{source['citation']}]" if source.get('citation') else ''))
        if source not in used:
            used.append(source)
    return {
        'answer': '\n'.join(lines).strip(),
        'citations': [source['citation'] for source in used if source.get('citation')],
        'status': 'answered',
        'answer_type': 'document_based',
        'confidence': 'high',
        'usage': None,
    }


def _ordinal_answer(question, sources, scored, ordinal):
    items = _list_items([_best_list_source(sources)]) or _list_items(sources)
    if not items:
        return None
    index = ordinal if ordinal >= 0 else len(items) - 1
    if index >= len(items):
        return None
    sentence, source = items[index]
    mark = f" [{source['citation']}]" if source.get('citation') else ''
    lines = [f"### {_topic(question)}", '', f"{sentence}{mark}", '', '**Sources**', '',
             _source_line(source)]
    return {
        'answer': '\n'.join(lines).strip(),
        'citations': [source['citation']] if source.get('citation') else [],
        'status': 'answered',
        'answer_type': 'document_based',
        'confidence': 'medium',
        'usage': None,
    }


def _page_answer(page, sources):
    if not sources:
        return None
    picked, seen = [], set()
    for source in sources:
        for raw in sentences(source.get('text', '')):
            sentence = _clean_sentence(raw)
            essence = _essence(sentence)
            if (len(sentence.split()) >= 5 and essence and essence not in seen
                    and not EMAIL_OR_URL.search(sentence)):
                seen.add(essence)
                picked.append((sentence, source))
            if len(picked) >= 6:
                break
        if len(picked) >= 6:
            break
    if not picked:
        return None
    used = []
    lines = [f'### Page {page}', '']
    for index, (sentence, source) in enumerate(picked):
        citation = source.get('citation')
        mark = f" [{citation}]" if citation else ''
        if index == 0:
            lines.append(f"On page {page}, the document says: {sentence}{mark}")
            lines.append('')
            lines.append('**Key details**')
            lines.append('')
        else:
            lines.append(f"- {sentence}{mark}")
        if source not in used:
            used.append(source)
    lines.append('')
    lines.append('**Sources**')
    lines.append('')
    lines.extend(_source_line(source) for source in used)
    return {
        'answer': '\n'.join(lines).strip(),
        'citations': [source['citation'] for source in used if source.get('citation')],
        'status': 'answered',
        'answer_type': 'document_based',
        'confidence': 'high',
        'usage': None,
    }


def _summarize(question, sources):
    by_document = defaultdict(list)
    for source in sources:
        by_document[source.get('document_id') or source.get('name')].append(source)

    lines = ['### Document summary', '']
    used = []
    for _, group in by_document.items():
        name = group[0].get('name') or 'Document'
        lines.append(f"**{name}**")
        lines.append('')
        picked, seen = [], set()
        for source in group:
            text = source.get('text', '')
            candidates = [s for s in sentences(text)
                          if len(s.split()) >= 5 and not _heading_like(_clean_sentence(s))]
            for raw in candidates[:2]:
                sentence = _clean_sentence(raw)
                essence = _essence(sentence)
                if essence and essence not in seen and not EMAIL_OR_URL.search(sentence):
                    seen.add(essence)
                    picked.append((sentence, source))
            if len(picked) >= 5:
                break
        if not picked:
            continue
        for sentence, source in picked[:5]:
            citation = source.get('citation')
            lines.append(f"- {sentence}" + (f" [{citation}]" if citation else ''))
            if source not in used:
                used.append(source)
        lines.append('')

    if not used:
        return None
    lines.append('**Sources**')
    lines.append('')
    lines.extend(_source_line(source) for source in used)
    return {
        'answer': '\n'.join(lines).strip(),
        'citations': [source['citation'] for source in used if source.get('citation')],
        'status': 'answered',
        'answer_type': 'document_based',
        'confidence': 'medium',
        'usage': None,
    }
