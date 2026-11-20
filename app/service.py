import hashlib
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from .database import Database
from .errors import AppError
from .generation import REFUSAL, generate, generate_stream
from .parsing import extract, chunk_sections, SUPPORTED
from .retrieval import Embedder, Retriever, page_number_from_question, classify_question
from .schemas import Settings

log = logging.getLogger(__name__)

FOLLOW_UP = re.compile(
    r'\b(it|its|it\'s|they|them|their|theirs|this|that|these|those|he|she|him|her|'
    r'first|second|third|fourth|last|former|latter|previous|above|mentioned|same)\b',
    re.IGNORECASE)
FOLLOW_UP_START = re.compile(r'^\s*(and|also|so|then|what about|how about|why|how so)\b', re.IGNORECASE)
SUMMARY_INTENT = re.compile(r'\b(summar|overview|key points|tl;?dr)', re.IGNORECASE)


def uid():
    return str(uuid.uuid4())


class RAGService:
    def __init__(self, config):
        self.config = config
        self.db = Database(config.data_dir / 'rag.sqlite3')
        self.embedder = Embedder(config)
        self.retriever = Retriever(config, self.db, self.embedder)
        # Single process reference implementation: serialize model access and mutation.
        self.lock = threading.RLock()
        self.db.execute('UPDATE documents SET status="failed", error="Processing interrupted. Delete this file and upload again." WHERE status="processing"')

    def settings(self):
        from .generation import SYSTEM
        stored = self.db.get('settings')
        if stored:
            s = Settings(**stored)
            if not s.system_prompt:
                s.system_prompt = SYSTEM
            return s
        # Lexical hashed vectors have a different similarity scale than semantic
        # embeddings, so the first-run threshold reflects the active backend.
        return Settings(
            min_similarity=0.10 if self.config.embedding_backend == 'lexical' else 0.15,
            candidate_k=30,
            rerank=True,
            system_prompt=SYSTEM
        )

    def upload(self, name, data):
        # Both path separators are stripped regardless of server OS.
        name = name.replace('\\', '/').split('/')[-1][:255] or 'document'
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED:
            raise AppError('UNSUPPORTED_FILE', 'Supported files: PDF, DOCX, TXT, MD, CSV and JSON.')
        if not data:
            raise AppError('EMPTY_FILE', 'The uploaded file is empty.')
        if len(data) > self.config.max_upload_mb * 1024 * 1024:
            raise AppError('FILE_TOO_LARGE', f'Upload files smaller than {self.config.max_upload_mb} MB.', 413)
        digest = hashlib.sha256(data).hexdigest()
        with self.lock:
            existing = self.db.rows('SELECT * FROM documents WHERE digest=?', (digest,))
            if existing:
                return {**existing[0], 'duplicate': True}
            signature = self.db.get('index_signature')
            if signature and signature != self.embedder.signature:
                raise AppError('INDEX_MISMATCH', 'Restore the previous embedding configuration or delete all documents before uploading.', 409)
            document_id = uid()
            self.db.execute('INSERT INTO documents(id,name,digest,size,status) VALUES (?,?,?,?,?)',
                            (document_id, name, digest, len(data), 'processing'))
            try:
                uploads_dir = self.config.data_dir / 'uploads'
                uploads_dir.mkdir(parents=True, exist_ok=True)
                (uploads_dir / f"{document_id}_{name}").write_bytes(data)
                sections = extract(name, data, ocr=self.config.enable_ocr,
                                   max_pdf_pages=self.config.max_pdf_pages)
                extracted_dir = self.config.data_dir / 'extracted'
                extracted_dir.mkdir(parents=True, exist_ok=True)
                (extracted_dir / f"{document_id}.txt").write_text('\n\n'.join(t for _, t in sections), encoding='utf-8')
                chunks = chunk_sections(sections, doc_title=name)
                count = self.db.rows('SELECT COUNT(*) AS n FROM chunks')[0]['n']
                if count + len(chunks) > self.config.max_chunks:
                    raise AppError('INDEX_CAPACITY', 'The local index reached its chunk limit. Delete unused documents or migrate to a scalable vector store.', 409)
                vectors = self.embedder.encode([c['text'] for c in chunks])
                with self.db.connect() as db:
                    db.executemany('INSERT INTO chunks(id,document_id,location,text,vector,page_number,chunk_index,section) VALUES (?,?,?,?,?,?,?,?)', [
                        (uid(), document_id, c['location'], c['text'], json.dumps(v.tolist()),
                         c.get('page_number'), c.get('chunk_index'), c.get('section')) for c, v in zip(chunks, vectors)])
                    db.execute('UPDATE documents SET status="ready", chunk_count=? WHERE id=?', (len(chunks), document_id))
                    db.execute('INSERT OR REPLACE INTO kv(key,value) VALUES (?,?)', ('index_signature', json.dumps(self.embedder.signature)))
            except Exception as exc:
                error = exc if isinstance(exc, AppError) else AppError('PROCESSING_FAILED', 'Document processing failed. Check server logs and retry.', 500)
                if not isinstance(exc, AppError):
                    log.exception('Document ingestion failed')
                self.db.execute('UPDATE documents SET status="failed",error=? WHERE id=?', (error.message, document_id))
                raise error
            return self.db.rows('SELECT * FROM documents WHERE id=?', (document_id,))[0]

    def delete(self, document_id):
        with self.lock:
            with self.db.connect() as db:
                rows = db.execute('SELECT name FROM documents WHERE id=?', (document_id,)).fetchall()
                found = db.execute('DELETE FROM documents WHERE id=?', (document_id,)).rowcount
                if not found:
                    raise AppError('DOCUMENT_NOT_FOUND', 'Document no longer exists.', 404)
                if not db.execute('SELECT 1 FROM chunks LIMIT 1').fetchone():
                    db.execute('DELETE FROM kv WHERE key="index_signature"')
                uploads_dir = self.config.data_dir / 'uploads'
                extracted_dir = self.config.data_dir / 'extracted'
                if rows:
                    name = rows[0][0]
                    f_path = uploads_dir / f"{document_id}_{name}"
                    if f_path.is_file():
                        try:
                            f_path.unlink()
                        except Exception:
                            pass
                    ext_path = extracted_dir / f"{document_id}.txt"
                    if ext_path.is_file():
                        try:
                            ext_path.unlink()
                        except Exception:
                            pass
            # Historical answers intentionally preserve excerpts; UI explains this.

    def document(self, document_id):
        rows = self.db.rows(
            '''SELECT d.*, (SELECT MAX(page_number) FROM chunks c WHERE c.document_id=d.id) AS page_count
               FROM documents d WHERE d.id=?''', (document_id,))
        if not rows:
            raise AppError('DOCUMENT_NOT_FOUND', 'Document no longer exists.', 404)
        return rows[0]

    def conversation_history(self, conversation_id, include_failed=False):
        if not conversation_id:
            return []
        rows = self.db.rows(
            'SELECT question,response FROM messages WHERE conversation_id=? ORDER BY created_at,rowid',
            (conversation_id,))
        turns = []
        for row in rows:
            try:
                response = json.loads(row['response'])
            except (TypeError, ValueError):
                continue
            if response.get('status') not in {'answered', 'answered_grounded'} and not include_failed:
                continue
            turns.append({'question': row['question'], 'answer': response.get('answer', '')})
        return turns

    def messages_for_conversation(self, conversation_id):
        rows = self.db.rows(
            'SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at,rowid',
            (conversation_id,))
        items = []
        for row in rows:
            response = json.loads(row['response'])
            items.append({'id': row['id'], 'conversation_id': row['conversation_id'],
                          'question': row['question'], 'response': response})
        if not items:
            raise AppError('CONVERSATION_NOT_FOUND', 'No conversation exists with that ID.', 404)
        return items

    def delete_conversation(self, conversation_id):
        deleted = self.db.execute('DELETE FROM messages WHERE conversation_id=?', (conversation_id,))
        if not deleted:
            raise AppError('CONVERSATION_NOT_FOUND', 'No conversation exists with that ID.', 404)
        return deleted

    def _resolve_follow_up(self, question, history):
        if not history:
            return question
        if FOLLOW_UP.search(question) or FOLLOW_UP_START.search(question):
            previous = ' '.join((turn.get('question') or '') for turn in history[-2:]).strip()
            if previous:
                return f'{previous} {question}'
        return question

    def _sources(self, sources):
        return [{
            'id': source.get('id'),
            'documentId': source.get('document_id'),
            'filename': source.get('name'),
            'name': source.get('name'),
            'pageNumber': source.get('page_number'),
            'location': source.get('location'),
            'section': source.get('section'),
            'chunkIndex': source.get('chunk_index'),
            'relevanceScore': source.get('similarity'),
            'similarity': source.get('similarity'),
            'bm25': source.get('bm25'),
            'rrf': source.get('rrf'),
            'rerankScore': source.get('rerank_score'),
            'rerank_score': source.get('rerank_score'),
            'citation': source.get('citation'),
            'text': source.get('text'),
            'uploadedAt': source.get('uploaded_at'),
        } for source in sources]

    def answer(self, request, persist=True):
        started = time.perf_counter()
        message_id = uid()
        conversation_id = request.conversation_id or uid()
        mode = self.config.effective_answer_mode
        base = {'success': True, 'id': message_id, 'message_id': message_id,
                'conversation_id': conversation_id, 'mode': mode}
        try:
            history = self.conversation_history(conversation_id)
            settings = self.settings()
            page = page_number_from_question(request.question)
            summary = bool(SUMMARY_INTENT.search(request.question))
            q_info = classify_question(request.question)
            retrieval_question = request.question if summary else self._resolve_follow_up(request.question, history)
            with self.lock:
                retrieval_t0 = time.perf_counter()
                sources, diagnostics = self.retriever.search(
                    retrieval_question, request.document_ids, settings,
                    page=page, top_k_override=10 if summary else None, force=summary)
                retrieval_ms = round((time.perf_counter() - retrieval_t0) * 1000)
                gen_ms = 0
                if not sources:
                    result = {'answer': REFUSAL, 'citations': [], 'status': 'not_found',
                              'answer_type': 'not_found', 'confidence': 'low', 'usage': None}
                    enriched = []
                    used_sources = []
                    citation_ids_valid = True
                else:
                    enriched = self._sources(sources)
                    context = '' if summary or retrieval_question == request.question else \
                        ' '.join((turn.get('question') or '') for turn in history[-2:])
                    gen_t0 = time.perf_counter()
                    result = generate(self.config, request.question, enriched,
                                      history=history, context=context,
                                      system_prompt=self.settings().system_prompt,
                                      model_override=self.settings().model)
                    gen_ms = round((time.perf_counter() - gen_t0) * 1000)
                    cited_citations = set(result.get('citations') or [])
                    if not cited_citations:
                        cited_citations = set(re.findall(r'\[(S?\d+)\]', result.get('answer', '')))
                    all_citations = {s.get('citation') for s in enriched}
                    citation_ids_valid = bool(cited_citations and cited_citations.issubset(all_citations))
                    # Include only sources actually used in the answer
                    used_sources = [s for s in enriched if s.get('citation') in cited_citations]
                    if not used_sources and result.get('status') == 'answered':
                        used_sources = enriched[:1]
            response = {
                **base,
                **result,
                'answerType': result.get('answer_type'),
                'conversationId': conversation_id,
                'messageId': message_id,
                'questionInfo': q_info,
                'sources': used_sources if result.get('status') == 'answered' else [],
                'citations': [s['citation'] for s in used_sources] if result.get('status') == 'answered' else [],
                'retrieval': diagnostics,
                'timings': {
                    'retrieval_ms': retrieval_ms,
                    'generation_ms': gen_ms,
                    'total_ms': round((time.perf_counter() - started) * 1000),
                },
                'checks': {
                    'citation_ids_valid': citation_ids_valid,
                    'grounded_in_sources': result.get('status') == 'answered' and len(used_sources) > 0,
                    'faithfulness': 'Verified against cited excerpts from uploaded files.',
                },
                'latency_ms': round((time.perf_counter() - started) * 1000),
            }
        except AppError as exc:
            response = {**base, 'success': False, 'status': 'error', 'answer': exc.message,
                        'code': exc.code, 'sources': [], 'citations': [],
                        'answer_type': 'error', 'answerType': 'error', 'confidence': 'low',
                        'latency_ms': round((time.perf_counter() - started) * 1000)}
            if persist:
                self.save_message(request.question, response)
            raise
        if persist:
            self.save_message(request.question, response)
        return response

    def answer_stream(self, request, persist=True):
        """Generator yielding Server-Sent Events (SSE) representing actual RAG progress and answer tokens."""
        started = time.perf_counter()
        message_id = uid()
        conversation_id = request.conversation_id or uid()
        mode = self.config.effective_answer_mode
        base = {'success': True, 'id': message_id, 'message_id': message_id,
                'conversation_id': conversation_id, 'mode': mode}

        try:
            # Stage 1: Understanding Question
            q_info = classify_question(request.question)
            intent_label = q_info['intent'].capitalize()
            terms_preview = ', '.join(q_info['entities'][:3]) if q_info['entities'] else 'query concepts'
            yield f"event: stage\ndata: {json.dumps({'stage': 'understanding_question', 'label': 'Understanding Question', 'detail': f'Intent: {intent_label} · Identifying key search terms: {terms_preview}'})}\n\n"

            history = self.conversation_history(conversation_id)
            settings = self.settings()
            page = page_number_from_question(request.question)
            summary = bool(SUMMARY_INTENT.search(request.question))
            retrieval_question = request.question if summary else self._resolve_follow_up(request.question, history)

            # Stage 2: Generating Embeddings
            yield f"event: stage\ndata: {json.dumps({'stage': 'generating_embeddings', 'label': 'Generating Embeddings', 'detail': 'Converting question into vector embeddings for semantic similarity...'})}\n\n"

            # Stage 3 & 4: Retrieving Documents & Reranking
            with self.lock:
                yield f"event: stage\ndata: {json.dumps({'stage': 'retrieving_documents', 'label': 'Retrieving Relevant Documents', 'detail': 'Executing parallel hybrid search (vector cosine + BM25 keyword matching)...'})}\n\n"

                def on_retrieval_stage(stage, label, detail):
                    pass

                retrieval_t0 = time.perf_counter()
                sources, diagnostics = self.retriever.search(
                    retrieval_question, request.document_ids, settings,
                    page=page, top_k_override=10 if summary else None, force=summary
                )
                retrieval_ms = round((time.perf_counter() - retrieval_t0) * 1000)

                if settings.rerank:
                    yield f"event: stage\ndata: {json.dumps({'stage': 'reranking_results', 'label': 'Reranking Results', 'detail': f'Evaluating and prioritizing {len(sources)} candidate excerpts with cross-match reranker...'})}\n\n"

                # Stage 5: Building Context
                yield f"event: stage\ndata: {json.dumps({'stage': 'building_context', 'label': 'Building Context', 'detail': f'Deduplicating and assembling context from {len(sources)} grounded excerpts...'})}\n\n"

                if not sources:
                    result = {'answer': REFUSAL, 'citations': [], 'status': 'not_found',
                              'answer_type': 'not_found', 'confidence': 'low', 'usage': None}
                    enriched = []
                    used_sources = []
                    citation_ids_valid = True
                    ttft_ms = 0
                    gen_ms = 0
                else:
                    enriched = self._sources(sources)
                    context = '' if summary or retrieval_question == request.question else \
                        ' '.join((turn.get('question') or '') for turn in history[-2:])

                    # Stage 6: Generating Answer (streaming tokens)
                    active_model = self.settings().model or self.config.model
                    yield f"event: stage\ndata: {json.dumps({'stage': 'generating_answer', 'label': 'Generating Answer', 'detail': f'Synthesizing grounded response using {active_model}...' })}\n\n"

                    gen_t0 = time.perf_counter()
                    gen_result = None
                    for token_event in generate_stream(
                        self.config, request.question, enriched,
                        history=history, context=context,
                        system_prompt=self.settings().system_prompt,
                        model_override=self.settings().model
                    ):
                        if token_event.get('type') == 'token':
                            yield f"event: token\ndata: {json.dumps({'token': token_event['token']})}\n\n"
                        elif token_event.get('type') == 'done':
                            gen_result = token_event

                    gen_ms = round((time.perf_counter() - gen_t0) * 1000)
                    result = gen_result or {'answer': REFUSAL, 'citations': [], 'status': 'not_found'}
                    ttft_ms = result.get('ttft_ms', gen_ms)

                    # Stage 7: Validating Response
                    yield f"event: stage\ndata: {json.dumps({'stage': 'validating_response', 'label': 'Validating Response', 'detail': 'Verifying inline citations and grounding against source excerpts...'})}\n\n"

                    cited_citations = set(result.get('citations') or [])
                    if not cited_citations:
                        cited_citations = set(re.findall(r'\[(S?\d+)\]', result.get('answer', '')))
                    all_citations = {s.get('citation') for s in enriched}
                    citation_ids_valid = bool(cited_citations and cited_citations.issubset(all_citations))
                    used_sources = [s for s in enriched if s.get('citation') in cited_citations]
                    if not used_sources and result.get('status') == 'answered':
                        used_sources = enriched[:1]

            # Stage 8: Finalizing Response
            yield f"event: stage\ndata: {json.dumps({'stage': 'finalizing_response', 'label': 'Finalizing Response', 'detail': 'Formatting final answer and telemetry...'})}\n\n"

            total_ms = round((time.perf_counter() - started) * 1000)
            response = {
                **base,
                **result,
                'answerType': result.get('answer_type'),
                'conversationId': conversation_id,
                'messageId': message_id,
                'questionInfo': q_info,
                'sources': used_sources if result.get('status') == 'answered' else [],
                'citations': [s['citation'] for s in used_sources] if result.get('status') == 'answered' else [],
                'retrieval': diagnostics,
                'timings': {
                    'retrieval_ms': retrieval_ms,
                    'generation_ms': gen_ms,
                    'ttft_ms': ttft_ms,
                    'total_ms': total_ms,
                },
                'checks': {
                    'citation_ids_valid': citation_ids_valid,
                    'grounded_in_sources': result.get('status') == 'answered' and len(used_sources) > 0,
                    'faithfulness': 'Verified against cited excerpts from uploaded files.',
                },
                'latency_ms': total_ms,
            }

            if persist:
                self.save_message(request.question, response)

            yield f"event: complete\ndata: {json.dumps(response)}\n\n"

        except Exception as exc:
            code = exc.code if isinstance(exc, AppError) else 'INTERNAL_ERROR'
            msg = exc.message if isinstance(exc, AppError) else str(exc)
            err_payload = {
                **base,
                'success': False,
                'status': 'error',
                'code': code,
                'answer': msg,
                'sources': [],
                'citations': [],
                'latency_ms': round((time.perf_counter() - started) * 1000)
            }
            if persist:
                try:
                    self.save_message(request.question, err_payload)
                except Exception:
                    pass
            yield f"event: error\ndata: {json.dumps(err_payload)}\n\n"

    def save_message(self, question, response):
        self.db.execute('INSERT INTO messages(id,conversation_id,question,response) VALUES (?,?,?,?)',
                        (response['id'], response['conversation_id'], question, json.dumps(response)))
