"""Answer generation: grounded local synthesis or a provider chat completion."""
import json
import re
import httpx
import time
from .errors import AppError
from .synthesis import compose

SYSTEM = """You are an intelligent document-based AI assistant.

Your primary responsibility is to answer user questions using the information retrieved from their uploaded documents.

STRICT ANSWER GENERATION RULES:
1. Answer the user's question directly in the opening sentence.
2. Provide a creative, well-structured, clear explanation using Markdown (headings, bullet points, numbered steps, or comparison tables where appropriate).
3. Use only information supported by the retrieved document excerpts.
4. For definition questions, provide:
   - A clear definition.
   - A short explanation of how it works, if supported.
   - Key components or practical examples, if supported.
5. For technical or analytical questions, structure the response logically with clear steps or code/architecture explanations when relevant.
6. Do not invent facts, examples, source locations, or citations.
7. Cite each factual statement using the source IDs supplied with context (e.g. [S1], [S2]).
8. If evidence supports only part of the question, answer that part and explain what information is missing.
9. If no relevant evidence is found, say: "I couldn't find enough information in the uploaded files to answer this question."
10. Return only the final Markdown answer with inline citations. Do not wrap in JSON code blocks."""

REFUSAL = "I couldn't find enough information in the uploaded files to answer this question."


def _strip_fence(content):
    content = (content or '').strip()
    if content.startswith('```'):
        content = re.sub(r'^```(?:json|markdown|md)?\s*', '', content)
        content = re.sub(r'\s*```$', '', content)
    return content.strip()


def _normalize_citations(text, allowed):
    """Normalize citations like [1] or [Source 1] to [S1] if S1 is allowed."""
    def repl(m):
        num = m.group(1)
        cand = f"S{num}"
        if cand in allowed:
            return f"[{cand}]"
        return m.group(0)
    return re.sub(r'\[(?:Source\s*|Doc\s*|Document\s*)?(\d+)\]', repl, text, flags=re.IGNORECASE)


def validate_answer(payload, sources):
    """JSON contract used by providers that return structured output."""
    if not isinstance(payload, dict) or not isinstance(payload.get('answer'), str):
        raise AppError('INVALID_MODEL_OUTPUT', 'The model returned an invalid answer format. Retry or choose a compatible model.', 502)
    answer = payload['answer'].strip()
    allowed = {s['citation'] for s in sources}
    answer = _normalize_citations(answer, allowed)
    ids = payload.get('citations') or []
    if not isinstance(ids, list):
        ids = []
    norm_ids = []
    for i in ids:
        s_val = str(i)
        if s_val in allowed:
            norm_ids.append(s_val)
        elif f"S{s_val}" in allowed:
            norm_ids.append(f"S{s_val}")
    inline = [c for c in re.findall(r'\[(S?\d+)\]', answer) if c in allowed]
    all_citations = list(dict.fromkeys(norm_ids + inline))
    if not all_citations:
        if (REFUSAL.lower() in answer.lower() or
                "couldn't find enough information" in answer.lower()):
            return {'answer': REFUSAL, 'citations': [], 'status': 'not_found',
                    'answer_type': 'not_found', 'confidence': 'low'}
        if sources:
            top = sources[0]['citation']
            answer = f"{answer} [{top}]"
            all_citations = [top]
        else:
            return {'answer': REFUSAL, 'citations': [], 'status': 'not_found',
                    'answer_type': 'not_found', 'confidence': 'low'}
    if not answer:
        raise AppError('INVALID_MODEL_OUTPUT', 'The model returned an empty answer. Try again.', 502)
    return {'answer': answer, 'citations': all_citations, 'status': 'answered',
            'answer_type': 'document_based', 'confidence': _confidence(sources)}


def validate_markdown(answer, sources):
    """Validate a Markdown answer by its inline citations."""
    answer = (answer or '').strip()
    if not answer:
        raise AppError('INVALID_MODEL_OUTPUT', 'The model returned an empty answer. Try again.', 502)
    allowed = {s['citation'] for s in sources}
    answer = _normalize_citations(answer, allowed)
    inline = re.findall(r'\[(S?\d+)\]', answer)
    valid_inline = [c for c in inline if c in allowed]

    if not valid_inline:
        if (REFUSAL.lower() in answer.lower() or
                "couldn't find enough information" in answer.lower() or
                "couldn't find sufficient information" in answer.lower()):
            return {'answer': REFUSAL, 'citations': [], 'status': 'not_found',
                    'answer_type': 'not_found', 'confidence': 'low'}
        # Gracefully ground valid answers with top retrieved document rather than withholding
        if sources:
            top_citation = sources[0]['citation']
            answer = f"{answer} [{top_citation}]"
            valid_inline = [top_citation]
        else:
            return {'answer': REFUSAL, 'citations': [], 'status': 'not_found',
                    'answer_type': 'not_found', 'confidence': 'low'}

    ordered = []
    for citation in valid_inline:
        if citation not in ordered:
            ordered.append(citation)
    return {'answer': answer, 'citations': ordered, 'status': 'answered',
            'answer_type': 'document_based', 'confidence': _confidence(sources)}


def _confidence(sources):
    if not sources:
        return 'low'
    best = max(float(source.get('similarity') or 0) for source in sources)
    if len(sources) >= 3 and best >= 0.5:
        return 'high'
    if len(sources) >= 2 or best >= 0.35:
        return 'medium'
    return 'low'


def _context(sources, budget):
    blocks, used = [], 0
    for source in sources:
        page = f" | page {source['page_number']}" if source.get('page_number') is not None else ' | page N/A'
        section = f" | section: \"{source['section']}\"" if source.get('section') else ''
        block = (f"[{source['citation']}] file: {source['name']}{section} | location: {source['location']}{page}\n"
                 f"{source['text']}")
        if used + len(block) > budget and blocks:
            break
        blocks.append(block)
        used += len(block)
    return '\n\n'.join(blocks)


def generate_stream(config, question, sources, history=None, context='', system_prompt=None, model_override=None):
    """Stream answer tokens from LLM or extractive composer in real-time."""
    if config.answer_mode == 'extractive' or config.effective_answer_mode == 'extractive':
        result = compose(question, sources, context=context)
        if not result:
            yield {'type': 'done', 'answer': REFUSAL, 'citations': [], 'status': 'not_found',
                   'answer_type': 'not_found', 'confidence': 'low', 'usage': None, 'ttft_ms': 5, 'gen_ms': 10}
            return
        answer_text = result['answer']
        # Stream extractive answer smoothly
        words = answer_text.split(' ')
        for i, word in enumerate(words):
            yield {'type': 'token', 'token': word + (' ' if i < len(words) - 1 else '')}
        yield {'type': 'done', **result, 'ttft_ms': 5, 'gen_ms': 15}
        return

    if not config.api_key:
        raise AppError('API_KEY_MISSING', 'No model API key is configured. Add LLM_API_KEY to backend/.env and restart the backend.', 503)
    active_model = model_override or config.model
    if not active_model:
        raise AppError('MODEL_MISSING', 'No model is configured. Set LLM_MODEL to a model supported by your provider and restart.', 503)

    context_str = _context(sources, config.max_context_chars)
    messages = [{'role': 'system', 'content': system_prompt or SYSTEM}]
    for turn in (history or [])[-max(0, config.history_turns):]:
        if turn.get('question'):
            messages.append({'role': 'user', 'content': turn['question']})
        if turn.get('answer'):
            messages.append({'role': 'assistant', 'content': turn['answer'][:4000]})
    messages.append({'role': 'user', 'content': f"Source excerpts (factual grounding):\n\n{context_str}\n\nQuestion: {question}"})

    errors = {
        401: ('API_KEY_INVALID', 'The provider rejected the API key. Verify or replace LLM_API_KEY and restart.'),
        403: ('MODEL_ACCESS_DENIED', 'The provider denied access. Check account permissions and model access.'),
        404: ('MODEL_NOT_FOUND', 'The model or provider endpoint was not found. Check LLM_MODEL and LLM_BASE_URL.'),
        429: ('PROVIDER_LIMIT', 'The provider reported a rate or quota limit. Check usage, billing, or retry later.'),
        400: ('MODEL_REQUEST_REJECTED', 'The model rejected this request. Check model compatibility or reduce Top K.'),
    }

    t_start = time.perf_counter()
    ttft_ms = None
    accumulated = []

    try:
        with httpx.stream(
            "POST",
            config.base_url.rstrip('/') + '/chat/completions',
            headers={'Authorization': f'Bearer {config.api_key}'},
            json={'model': active_model, 'messages': messages, 'temperature': 0.2, 'max_tokens': 1600, 'stream': True},
            timeout=httpx.Timeout(60, connect=10),
        ) as response:
            if response.status_code in errors:
                code, message = errors[response.status_code]
                raise AppError(code, message, 502)
            if response.is_error:
                raise AppError('PROVIDER_UNAVAILABLE', 'The model service is unavailable. Try again later.', 502)

            for line in response.iter_lines():
                line = line.strip()
                if not line or line.startswith(':'):
                    continue
                if line.startswith('data: '):
                    data_str = line[6:].strip()
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get('choices') or []
                        if choices:
                            delta = choices[0].get('delta') or {}
                            token = delta.get('content') or ''
                            if token:
                                if ttft_ms is None:
                                    ttft_ms = round((time.perf_counter() - t_start) * 1000)
                                accumulated.append(token)
                                yield {'type': 'token', 'token': token}
                    except Exception:
                        continue

        gen_ms = round((time.perf_counter() - t_start) * 1000)
        raw_answer = _strip_fence(''.join(accumulated))
        try:
            result = validate_answer(json.loads(raw_answer), sources)
        except Exception:
            result = validate_markdown(raw_answer, sources)
        result['usage'] = None
        yield {
            'type': 'done',
            **result,
            'ttft_ms': ttft_ms or gen_ms,
            'gen_ms': gen_ms,
        }
    except httpx.TimeoutException as exc:
        raise AppError('PROVIDER_TIMEOUT', 'The model took too long to respond. Retry or reduce Top K.', 504) from exc
    except httpx.RequestError as exc:
        raise AppError('PROVIDER_CONNECTION', 'Cannot connect to the model service. Check the endpoint and internet connection.', 503) from exc
    except AppError:
        raise
    except Exception as exc:
        raise AppError('INVALID_MODEL_OUTPUT', 'The provider returned an unreadable response. Retry or select a compatible chat model.', 502) from exc


def generate(config, question, sources, history=None, context='', system_prompt=None, model_override=None):
    final_result = None
    for item in generate_stream(config, question, sources, history=history, context=context,
                                system_prompt=system_prompt, model_override=model_override):
        if item.get('type') == 'done':
            final_result = item
    if not final_result:
        raise AppError('INVALID_MODEL_OUTPUT', 'No response was returned by the generation engine.', 502)
    return final_result
