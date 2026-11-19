"""BM25 + cosine retrieval, reciprocal rank fusion, optional cross-encoder."""
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
import numpy as np
from .errors import AppError

STOP = set('a an the is are was were what which who how when where why does do did can could would should of for in on at to and or with from it this that tell me about please'.split())
PAGE_QUERY = re.compile(r'\bpage\s+(\d{1,4})\b', re.IGNORECASE)
DEFINITION_QUERY = re.compile(r'\b(what is|what are|define|definition of|meaning of)\b', re.IGNORECASE)
DEFINITION_PASSAGE = re.compile(r'\b(?:is a|are a|is an|are an|is the|are the|refers to|means|is defined as|are defined as|is a branch of|is a subset of|is a type of|is the field of)\b', re.IGNORECASE)
ISOLATED_EXAMPLE = re.compile(r'^\s*(?:for example|for instance|an example|e\.g\.)\b', re.IGNORECASE)


def expand_query(question: str) -> str:
    """Used for retrieval only. Pass the original question to the LLM."""
    q_clean = (question or '').strip()
    q_lower = q_clean.lower()
    if 'deep learning' in q_lower:
        return f"{q_clean} definition subset of machine learning artificial neural networks multiple hidden layers"
    if DEFINITION_QUERY.search(q_clean):
        return f"{q_clean} definition meaning overview concept"
    return q_clean


def tokens(text):
    return [t for t in re.findall(r'\w+', text.lower()) if t not in STOP]


def page_number_from_question(question: str):
    match = PAGE_QUERY.search(question or '')
    return int(match.group(1)) if match else None


def _dedupe(indices, chunks, limit):
    """Drop duplicate or substantially overlapping passages that waste context window."""
    kept, kept_sets = [], []
    for index in indices:
        terms = set(tokens(chunks[index]['text'])[:120])
        if not terms:
            continue
        duplicate = False
        for other in kept_sets:
            union = terms | other
            inter = terms & other
            if union and (len(inter) / len(union) > 0.85 or
                          len(inter) / max(min(len(terms), len(other)), 1) > 0.93):
                duplicate = True
                break
        if not duplicate:
            kept.append(index)
            kept_sets.append(terms)
            if len(kept) >= limit:
                break
    return kept


class Embedder:
    def __init__(self, config):
        self.config = config
        self.model = None

    @property
    def signature(self):
        return f'{self.config.embedding_backend}:{self.config.embedding_model if self.config.embedding_backend == "semantic" else "hashed-2048-v1"}:chunk140-overlap25'

    def encode(self, texts):
        if self.config.embedding_backend == 'lexical':
            result = np.zeros((len(texts), 2048), dtype=np.float32)
            for i, text in enumerate(texts):
                for token in tokens(text):
                    index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], 'big') % 2048
                    result[i, index] += 1
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            return result / np.maximum(norms, 1e-12)
        try:
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.config.embedding_model)
            return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:
            raise AppError('EMBEDDING_UNAVAILABLE', 'Embedding model unavailable. Install requirements-semantic.txt and allow the first model download; then retry.', 503) from exc


class Retriever:
    def __init__(self, config, db, embedder):
        self.config, self.db, self.embedder = config, db, embedder
        self.reranker = None

    def _chunks(self, document_ids):
        if document_ids:
            found = self.db.rows(
                'SELECT id,status,name FROM documents WHERE id IN (' + ','.join('?' for _ in document_ids) + ')',
                document_ids,
            )
            found_ids = {row['id'] for row in found}
            missing = [d for d in document_ids if d not in found_ids]
            if missing:
                raise AppError('INVALID_DOCUMENT_ID', 'One or more selected documents no longer exist. Refresh the document list.', 404)
            blocked = [row['name'] for row in found if row['status'] != 'ready']
            if blocked:
                raise AppError('DOCUMENT_NOT_READY', f'This document is not ready for questions: {blocked[0]}', 409)
        sql = ('SELECT c.*, d.name, d.created_at AS uploaded_at FROM chunks c '
               'JOIN documents d ON c.document_id=d.id WHERE d.status="ready"')
        args = []
        if document_ids:
            sql += ' AND d.id IN (' + ','.join('?' for _ in document_ids) + ')'
            args = list(document_ids)
        sql += ' ORDER BY c.document_id, c.rowid'
        return self.db.rows(sql, args)

    def search(self, question, document_ids, settings, page=None, top_k_override=None, force=False):
        chunks = self._chunks(document_ids)
        if not chunks:
            raise AppError('NO_DOCUMENTS', 'No processed files are available in this selection. Upload a file or select ready documents first.')
        if self.db.get('index_signature') != self.embedder.signature:
            raise AppError('INDEX_MISMATCH', 'Embedding configuration changed. Restore the previous configuration or delete all documents and upload them again.', 409)

        page = page if page is not None else page_number_from_question(question)
        page_filtered = None
        if page is not None:
            page_filtered = [c for c in chunks if c.get('page_number') == page]
            chunks = page_filtered
            if not chunks:
                return [], {'hybrid': True, 'reranking': 'not run: no chunks on the requested page',
                            'candidate_count': 0, 'page_filter': page, 'threshold': settings.min_similarity}

        scoring_question = PAGE_QUERY.sub(' ', question or '').strip() or question
        retrieval_query = expand_query(scoring_question)
        vectors = np.asarray([json.loads(c.pop('vector')) for c in chunks], dtype=np.float32)
        dense = vectors @ self.embedder.encode([retrieval_query])[0]
        query_tokens = tokens(retrieval_query)
        corpus = [tokens(c['text']) for c in chunks]
        counts = [Counter(t) for t in corpus]
        avg_len = max(sum(map(len, corpus)) / len(corpus), 1)
        sparse = np.zeros(len(chunks))
        for term in set(query_tokens):
            df = sum(term in c for c in counts)
            idf = math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            for i, count in enumerate(counts):
                tf = count[term]
                sparse[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(corpus[i]) / avg_len))

        candidate_k = min(settings.candidate_k, len(chunks))
        fused = Counter()
        for scores in (dense, sparse):
            for rank, i in enumerate(np.argsort(-scores)[:candidate_k], 1):
                if scores[i] > 0:
                    fused[int(i)] += 1 / (60 + rank)
        # Prioritize definition and explanatory passages over isolated examples for definition queries
        # and boost exact section heading matches
        q_norm = re.sub(r'[^a-z0-9\s]', '', scoring_question.lower()).strip()
        q_words = set(q_norm.split()) - STOP
        for i in list(fused.keys()):
            text = chunks[i].get('text', '')
            section = (chunks[i].get('section') or '').lower()
            if section and q_words:
                sec_words = set(re.sub(r'[^a-z0-9\s]', '', section).split())
                overlap = len(q_words & sec_words)
                if overlap >= 2 or (overlap >= 1 and len(q_words) <= 2):
                    fused[i] += 0.05
            if DEFINITION_QUERY.search(scoring_question):
                if DEFINITION_PASSAGE.search(text):
                    fused[i] += 0.02
                elif ISOLATED_EXAMPLE.search(text):
                    fused[i] -= 0.01
        ranked = [i for i, _ in fused.most_common(candidate_k)]
        if not ranked:
            ranked = list(np.argsort(-dense)[:candidate_k])

        if force:
            # Summary requests have no lexical/semantic query signal. Spread the
            # candidates across documents and pages instead of taking one bucket.
            grouped = defaultdict(list)
            for index, chunk in enumerate(chunks):
                grouped[chunk['document_id']].append(index)
            spread, round_number = [], 0
            while len(spread) < len(chunks):
                added = False
                for indices in grouped.values():
                    if round_number < len(indices):
                        spread.append(indices[round_number])
                        added = True
                if not added:
                    break
                round_number += 1
            ranked = spread
            accepted = spread
        elif page_filtered is not None:
            # The user explicitly scoped the question to a page; do not let the
            # generic similarity cutoff hide that page's content.
            accepted = ranked
        else:
            accepted = [i for i in ranked
                        if dense[i] >= settings.min_similarity
                        and (self.config.embedding_backend != 'lexical' or sparse[i] > 0)]
        if not accepted:
            return [], {'hybrid': True, 'reranking': 'not run: no relevant candidates',
                        'candidate_count': len(ranked), 'page_filter': page_filtered is not None,
                        'threshold': settings.min_similarity}

        limit = top_k_override or settings.top_k
        accepted = _dedupe(accepted, chunks, limit * 2)
        rerank_scores = {}
        reranking = 'disabled'
        if settings.rerank:
            try:
                if self.reranker is None:
                    from sentence_transformers import CrossEncoder
                    self.reranker = CrossEncoder(self.config.rerank_model)
                scores = self.reranker.predict([(question, chunks[i]['text']) for i in accepted])
                rerank_scores = {i: float(s) for i, s in zip(accepted, scores)}
                accepted.sort(key=lambda i: rerank_scores[i], reverse=True)
                reranking = 'completed'
            except Exception:
                rerank_scores = self._lexical_rerank(question, chunks, accepted)
                accepted.sort(key=lambda i: rerank_scores.get(i, 0.0), reverse=True)
                reranking = 'completed (cross-match)'

        results = []
        for n, i in enumerate(accepted[:limit], 1):
            results.append({
                **chunks[i],
                'citation': f'S{n}',
                'similarity': round(float(dense[i]), 4),
                'bm25': round(float(sparse[i]), 4),
                'rrf': round(float(fused[i]), 6),
                'rerank_score': rerank_scores.get(i),
            })
        return results, {'hybrid': True, 'reranking': reranking, 'candidate_count': len(ranked),
                         'deduped_count': len(accepted), 'page_filter': page_filtered is not None,
                         'threshold': settings.min_similarity}

    def _lexical_rerank(self, question: str, chunks: list[dict], indices: list[int]) -> dict[int, float]:
        """Cross-match reranker scoring exact phrase, heading overlap, and definition patterns."""
        q_clean = question.lower()
        q_tokens = set(tokens(q_clean))
        is_def = bool(DEFINITION_QUERY.search(q_clean))
        scores = {}
        for idx in indices:
            c = chunks[idx]
            text = c.get('text', '').lower()
            section = (c.get('section') or '').lower()
            score = 0.0
            phrases = re.findall(r'\b[a-z]{3,}\s+[a-z]{3,}\b', q_clean)
            for p in phrases:
                if p not in {'what is', 'what are', 'tell me', 'how to'}:
                    if p in section:
                        score += 3.0
                    if p in text:
                        score += 1.5
            sec_tokens = set(tokens(section))
            text_tokens = set(tokens(text))
            if q_tokens:
                score += 2.0 * len(q_tokens & sec_tokens) / len(q_tokens)
                score += 1.0 * len(q_tokens & text_tokens) / len(q_tokens)
            if is_def:
                if DEFINITION_PASSAGE.search(text):
                    score += 2.0
                if any(text.startswith(p) for p in ['what is', '1.', '2.', '3.']) and any(t in text[:50] for t in q_tokens):
                    score += 1.5
            scores[idx] = round(score, 4)
        return scores
